"""
Combined approach: local-search optimization of greedy decoder for MIS.
Structural idea from current program (SA + multiple move types + rich initial heuristic).
Refinement machinery from inspiration: faster simulation via Python sets,
and a final high-precision polish pass with tighter acceptance criteria.
Added: low-discrepancy sequence-based initialization for better vertex spread.
"""
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    DEADLINE = 143.0  # Safety margin before 150s limit

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetry and remove self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)

    # Convert to boolean for faster neighbor lookups
    A_bool = A > 0.5

    # Precompute neighbor lists as Python lists for fast set operations
    neighbors_list = [np.where(A_bool[i])[0].tolist() for i in range(N)]
    deg = A.sum(axis=1)

    # --- 1. Rich initial heuristic ---
    neighbor_deg_sum = A.dot(deg)
    safe_deg = deg + 1e-6
    avg_neighbor_deg = neighbor_deg_sum / safe_deg

    # Second-order: sum of neighbor degrees of neighbors
    neighbor2_deg_sum = A.dot(neighbor_deg_sum)
    avg_neighbor2_deg = neighbor2_deg_sum / safe_deg

    # Clustering coefficient proxy
    A2_diag = np.diag(A @ A)
    denom = deg * (deg - 1)
    denom = np.where(denom > 0, denom, 1e-6)
    clustering = A2_diag / denom

    # Eigenvector centrality via power iteration
    eigvec = np.ones(N) / np.sqrt(N)
    for _ in range(50):
        new_eigvec = A @ eigvec
        norm = np.linalg.norm(new_eigvec)
        if norm < 1e-10:
            break
        eigvec = new_eigvec / norm

    # Combine features
    w1, w2, w3, w4, w5 = 1.0, 0.5, 0.3, 0.2, 0.15
    raw_scores = -(w1 * deg + w2 * avg_neighbor_deg + w3 * clustering + w4 * avg_neighbor2_deg + w5 * eigvec)

    # Normalize to [0, 1]
    if raw_scores.max() > raw_scores.min():
        scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        scores = np.ones(N) / N

    # --- Low-discrepancy initialization: Halton-based spread ---
    def halton_sequence(n, base):
        """Generate n Halton numbers in the given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(1, n + 1):
            result[i - 1] = f
            f /= base
        return result

    # Map structural features to [0, 1] range for Halton-based assignment
    def normalize_to_01(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-10:
            return np.ones(N) / N
        return (x - mn) / (mx - mn)

    # Create a 2D feature space: degree and clustering
    feat_deg = normalize_to_01(deg)
    feat_clust = normalize_to_01(clustering)

    # Generate Halton sequence for 2 dimensions
    n_halton = N
    h1 = halton_sequence(n_halton, 2)
    h2 = halton_sequence(n_halton, 3)

    # Blend structural scores with Halton sequence for uniform spread
    blend = 0.7
    halton_scores = blend * scores + (1 - blend) * (0.5 * h1 + 0.5 * h2)
    scores_ld = halton_scores

    # Use the better of original and low-discrepancy initialization
    # (we'll let the search decide, but start from a mix)
    scores = 0.6 * scores + 0.4 * scores_ld

    # --- Fast simulation using Python sets ---
    def simulate_fast(s):
        """
        Fast greedy simulation using vectorized operations and Python sets.
        Returns the size of the independent set.
        """
        order = np.argsort(-s, kind='stable')
        active_set = set(range(N))
        count = 0

        for v in order:
            v = int(v)
            if v in active_set:
                count += 1
                active_set.remove(v)
                for u in neighbors_list[v]:
                    if u in active_set:
                        active_set.remove(u)
        return count

    # --- 2. Baseline ---
    best_val = simulate_fast(scores)
    best_scores = scores.copy()

    # --- 3. Local Search with Simulated Annealing ---
    rng = np.random.default_rng(42)

    num_restarts = 20
    current_temp = 1.0
    cooling = 0.95

    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 2:
            break

        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Perturb the best score found so far
            noise_scale = 0.08 + 0.03 * restart
            noise = rng.normal(0, noise_scale, N)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)

        # Local search with simulated annealing
        no_improve = 0
        max_no_improve = 12000
        temp = current_temp

        while time.time() - start_time < DEADLINE - 3 and no_improve < max_no_improve:
            # Choose move type based on temperature
            r = rng.random()
            if r < 0.4:
                # Random swap
                i, j = rng.integers(0, N, size=2)
                if i == j:
                    continue
            elif r < 0.7:
                # Rank-adjacent swap
                order = np.argsort(-current_scores, kind='stable')
                i_idx = rng.integers(0, N)
                offset = rng.integers(1, min(15, N))
                j_idx = (i_idx + offset) % N
                i, j = order[i_idx], order[j_idx]
                if i == j:
                    continue
            else:
                # Score perturbation: add small noise to a random vertex
                i = rng.integers(0, N)
                j = rng.integers(0, N)
                if i == j:
                    continue
                # Perturb scores rather than swap
                delta = rng.normal(0, 0.1)
                current_scores[i] += delta
                new_val = simulate_fast(current_scores)
                if new_val > current_val:
                    current_val = new_val
                    no_improve = 0
                    if current_val > best_val:
                        best_val = current_val
                        best_scores = current_scores.copy()
                else:
                    current_scores[i] -= delta
                    no_improve += 1
                continue

            # Swap-based move
            si, sj = current_scores[i], current_scores[j]
            current_scores[i], current_scores[j] = sj, si
            new_val = simulate_fast(current_scores)

            delta_val = new_val - current_val
            if delta_val > 0 or (temp > 0.01 and rng.random() < np.exp(delta_val / temp)):
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
            else:
                current_scores[i], current_scores[j] = si, sj
                no_improve += 1

            # Cool down
            temp *= cooling

        # Reset temperature for next restart
        current_temp = max(0.1, current_temp * 0.8)

    # --- 4. Final high-precision polish pass ---
    # Use a stricter acceptance criterion and more iterations to fine-tune
    if time.time() - start_time < DEADLINE - 5:
        polish_scores = best_scores.copy()
        polish_val = best_val
        rng_polish = np.random.default_rng(123)
        no_improve_polish = 0
        max_no_improve_polish = 5000

        while time.time() - start_time < DEADLINE - 5 and no_improve_polish < max_no_improve_polish:
            # Try random swaps with strict improvement only
            i, j = rng_polish.integers(0, N, size=2)
            if i == j:
                continue

            si, sj = polish_scores[i], polish_scores[j]
            polish_scores[i], polish_scores[j] = sj, si
            new_val = simulate_fast(polish_scores)

            if new_val > polish_val:
                polish_val = new_val
                no_improve_polish = 0
                if polish_val > best_val:
                    best_val = polish_val
                    best_scores = polish_scores.copy()
            else:
                polish_scores[i], polish_scores[j] = si, sj
                no_improve_polish += 1

    return best_scores
