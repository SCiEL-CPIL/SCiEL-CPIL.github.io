"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Convert to numpy int matrix for fast vectorized ops
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    # Deadline guard: kill after ~135s to be safe under 150s limit
    start_time = time.time()
    DEADLINE = start_time + 135.0

    def time_left():
        return DEADLINE - time.time()

    E = int(np.sum(adj)) // 2
    degrees = adj.sum(axis=1)
    neighbors = [np.nonzero(adj[v])[0] for v in range(n)]

    def cut_size_v2(labels):
        """Compute cut size using x^T A x formulation."""
        x = 2 * labels.astype(np.float64) - 1
        xTadjx = x @ adj @ x
        return 0.5 * E - 0.25 * xTadjx

    def compute_flip_gains(labels):
        """Vectorized computation of flip gains for all vertices.
        Gain for flipping v = (# same-side neighbors) - (# opposite-side neighbors)
        """
        x = 2 * labels.astype(np.float64) - 1  # +1 or -1
        # For each vertex v, gain = sum_u A[v,u] * (1 if x[u]==x[v] else -1)
        # = sum_u A[v,u] * x[v] * x[u] = x[v] * (A @ x)[v]
        Ax = adj @ x
        gains = x * Ax
        return gains

    def local_search_fast(labels, max_rounds=200):
        """Fast hill-climbing local search with vectorized gain updates. Returns best labels found."""
        best_labels = labels.copy()
        best_cut = cut_size_v2(best_labels)
        current_cut = best_cut
        
        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break
            improved = False
            for v in range(n):
                if time_left() < 1.0:
                    break
                # Compute delta from flipping v
                x_v = 1 if labels[v] == 0 else -1
                # neighbors on same side contribute +1, opposite contribute -1
                # delta = sum_{u in N(v)} (1 if labels[u]==labels[v] else -1)
                # = x_v * sum_{u in N(v)} x_u = x_v * (A @ x)[v]
                # But we need to be careful: when we flip v, the gain changes
                # Let's use the direct formula
                same_side = 0
                opp_side = 0
                for u in neighbors[v]:
                    if labels[u] == labels[v]:
                        same_side += 1
                    else:
                        opp_side += 1
                delta = same_side - opp_side
                if delta > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += delta
                    improved = True
            if not improved:
                break
        
        final_cut = cut_size_v2(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        return best_labels, best_cut

    def vectorized_local_search(labels, max_rounds=200):
        """Local search using vectorized gain computation for speed."""
        best_labels = labels.copy()
        best_cut = cut_size_v2(best_labels)
        current_cut = best_cut
        
        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break
            improved = False
            # Compute all flip gains at once
            gains = compute_flip_gains(labels)
            # Find vertices with positive gain
            positive = np.where(gains > 0)[0]
            if len(positive) == 0:
                break
            # Flip all vertices with positive gain simultaneously
            # But this can be problematic; do sequential within a round
            for v in positive:
                if time_left() < 1.0:
                    break
                delta = int(gains[v])
                if delta > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += delta
                    improved = True
            if not improved:
                break
        
        final_cut = cut_size_v2(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        return best_labels, best_cut

    def quantum_annealing(labels, max_iters=100000, gamma_start=3.0, gamma_end=0.05, cooling_rate=0.99998):
        """Quantum Annealing-inspired search using transverse field tunneling.
        
        Uses a transverse field parameter gamma that controls the probability
        of accepting worse moves, mimicking quantum tunneling effects.
        """
        best_labels = labels.copy()
        best_cut = cut_size_v2(best_labels)
        current_cut = best_cut
        gamma = gamma_start

        for it in range(max_iters):
            if time_left() < 3.0:
                break
            
            # Pick random vertex
            v = np.random.randint(0, n)
            
            # Compute gain from flipping v
            x_v = 1 if labels[v] == 0 else -1
            # delta = sum_{u in N(v)} (1 if labels[u]==labels[v] else -1)
            # = x_v * sum_{u in N(v)} x_u
            same_side = 0
            opp_side = 0
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_side += 1
                else:
                    opp_side += 1
            delta = same_side - opp_side  # change in cut if we flip v

            if delta >= 0:
                # Accept if it improves or is neutral
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                # Quantum tunneling: accept with probability based on transverse field
                if gamma > 0.01:
                    # Smooth acceptance: use sigmoid-like function
                    accept_prob = 1.0 / (1.0 + np.exp(abs(delta) / gamma))
                    if np.random.random() < accept_prob:
                        labels[v] = 1 - labels[v]
                        current_cut += delta

            # Cool down the transverse field
            gamma = max(gamma * cooling_rate, gamma_end)

        return best_labels, best_cut

    def spectral_init():
        """Initialize using the leading eigenvector of the adjacency matrix."""
        try:
            # Use power iteration to find leading eigenvector
            x = np.random.randn(n)
            x /= np.linalg.norm(x)
            for _ in range(50):
                x_new = adj @ x
                norm = np.linalg.norm(x_new)
                if norm < 1e-10:
                    break
                x = x_new / norm
            # Threshold at zero
            labels = (x >= 0).astype(np.int8)
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n)

    def multi_start(num_starts, labels_init=None, use_qa=False, use_fast_ls=True):
        """Try multiple random starts and return the best result."""
        best_labels = None
        best_cut = -1
        for s in range(num_starts):
            if time_left() < 3.0:
                break
            if s == 0 and labels_init is not None:
                labels = labels_init.copy()
            elif s == 0:
                # Use spectral init for first random start
                labels = spectral_init()
            else:
                labels = np.random.randint(0, 2, size=n)

            if use_qa:
                labels, cut = quantum_annealing(labels)
            else:
                labels, cut = vectorized_local_search(labels)

            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        return best_labels, best_cut

    def high_precision_polish(labels, max_rounds=300):
        """High-precision polish with more iterations and tighter convergence."""
        best_labels = labels.copy()
        best_cut = cut_size_v2(best_labels)
        current_cut = best_cut
        
        for rnd in range(max_rounds):
            if time_left() < 1.0:
                break
            improved = False
            # Try all possible single flips
            for v in range(n):
                same_side = 0
                opp_side = 0
                for u in neighbors[v]:
                    if labels[u] == labels[v]:
                        same_side += 1
                    else:
                        opp_side += 1
                delta = same_side - opp_side
                if delta > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += delta
                    improved = True
            
            # Try pairwise swaps (flip two vertices at once)
            if not improved and time_left() > 2.0:
                for i in range(n):
                    if time_left() < 1.5:
                        break
                    for j in range(i+1, n):
                        if time_left() < 1.0:
                            break
                        # Compute delta for flipping both i and j
                        delta_ij = 0
                        for u in neighbors[i]:
                            if u == j:
                                continue
                            if labels[u] == labels[i]:
                                delta_ij += 1
                            else:
                                delta_ij -= 1
                        for u in neighbors[j]:
                            if u == i:
                                continue
                            if labels[u] == labels[j]:
                                delta_ij += 1
                            else:
                                delta_ij -= 1
                        # Account for edge between i and j
                        if adj[i, j]:
                            if labels[i] == labels[j]:
                                delta_ij -= 2
                            else:
                                delta_ij += 2
                        
                        if delta_ij > 0:
                            labels[i] = 1 - labels[i]
                            labels[j] = 1 - labels[j]
                            current_cut += delta_ij
                            improved = True
            
            if not improved:
                break
        
        final_cut = cut_size_v2(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        return best_labels, best_cut

    # Strategy:
    # 1. Start with spectral init and alternating, run fast local search
    # 2. Try several random restarts with fast local search
    # 3. If time permits, try quantum annealing with different transverse field schedules
    # 4. Final high-precision polish with pairwise swaps

    # Initial: spectral
    labels_spec = spectral_init()
    best_labels, best_cut = vectorized_local_search(labels_spec, max_rounds=200)

    # Also try alternating
    labels_alt = np.arange(n) % 2
    labels_alt_ls, cut_alt = vectorized_local_search(labels_alt, max_rounds=200)
    if cut_alt > best_cut:
        best_labels = labels_alt_ls
        best_cut = cut_alt

    # Random restarts with fast local search
    num_restarts = 30
    labels_rs, cut_rs = multi_start(num_restarts, use_fast_ls=True)
    if cut_rs > best_cut:
        best_labels = labels_rs
        best_cut = cut_rs

    # If time permits, try quantum annealing with different transverse field schedules
    if time_left() > 20.0:
        # Try QA from best found so far with different gamma schedules
        for gamma_start in [2.0, 5.0, 8.0]:
            if time_left() < 5.0:
                break
            labels_qa, cut_qa = quantum_annealing(best_labels, max_iters=100000, gamma_start=gamma_start, gamma_end=0.05)
            if cut_qa > best_cut:
                best_cut = cut_qa
                best_labels = labels_qa

        # More random restarts with QA
        if time_left() > 15.0:
            labels_qa2, cut_qa2 = multi_start(20, use_qa=True)
            if cut_qa2 > best_cut:
                best_cut = cut_qa2
                best_labels = labels_qa2

    # Final high-precision polish with pairwise swaps
    if time_left() > 3.0:
        labels_polish, cut_polish = high_precision_polish(best_labels, max_rounds=300)
        if cut_polish > best_cut:
            best_cut = cut_polish
            best_labels = labels_polish

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
