import numpy as np
import time
import math

def priority(adjacency):
    """
    Calculate priority scores for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Initialize scores using a static heuristic that favors vertices with
       low degree and low local density, specifically penalizing vertices
       adjacent to hubs less than the hubs themselves to promote 'hub-blocking'.
    2. Refine these scores using simulated annealing with an energy function
       that models the vertex ordering as a 1D Ising spin chain, penalizing
       adjacent selected vertices in the greedy decoder's permutation.
    3. Use a smooth surrogate (soft-min via log-sum-exp) for the discrete
       selection, enabling gradient-like moves.
    4. Final high-precision polish pass with tighter tolerance.
    """
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin before 120s limit
    
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
    
    # Ensure symmetry and remove self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    
    # Convert to boolean for faster neighbor lookups
    A_bool = A > 0.5
    
    # Precompute neighbor lists as numpy arrays for vectorized deactivation
    neighbors_list = [np.where(A_bool[i])[0] for i in range(N)]
    
    # --- 1. Initial Heuristic ---
    deg = A.sum(axis=1)
    neighbor_deg_sum = A.dot(deg)
    safe_deg = deg + 1e-6
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    
    def normalize(x):
        if x.max() > x.min():
            return (x - x.min()) / (x.max() - x.min())
        else:
            return np.zeros_like(x)
            
    norm_deg = normalize(deg)
    norm_avg_ndeg = normalize(avg_neighbor_deg)
    
    scores = -(0.7 * norm_deg + 0.3 * norm_avg_ndeg)
    scores = normalize(scores)
    
    def simulate_fast(s):
        """
        Fast greedy simulation.
        Returns the size of the independent set.
        """
        order = np.argsort(-s, kind='stable')
        active = np.ones(N, dtype=bool)
        count = 0
        
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count
    
    def simulate_with_selection(s):
        """
        Fast greedy simulation returning both size and selection mask.
        """
        order = np.argsort(-s, kind='stable')
        active = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        count = 0
        
        for v in order:
            if active[v]:
                count += 1
                selected[v] = True
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count, selected
    
    # --- 2. Energy function: Ising-like model on the permutation ---
    # The greedy decoder picks vertices in descending score order.
    # We model this as a 1D chain where position k in the permutation has a spin.
    # Energy penalizes: (1) low total selection count, (2) "frustration" where
    # a selected vertex has a neighbor that appears later in the permutation
    # but could have been selected if the order were different.
    
    def compute_energy(s):
        """
        Compute an energy score for the greedy decoder output.
        Lower energy = better.
        Uses a smooth surrogate via log-sum-exp to handle discrete selection.
        """
        count, selected = simulate_with_selection(s)
        
        # Primary term: maximize count (negate since we minimize)
        E_primary = -count
        
        # Secondary term: penalize "wasted" potential.
        # For each selected vertex v, count how many of its neighbors are NOT selected.
        # This measures how much potential was lost due to the ordering.
        # Use a soft penalty: sum over selected v of (number of unselected neighbors)
        # This is a proxy for "if we had picked neighbors first, we might have gotten more."
        
        # Vectorized computation of unselected neighbor count for each selected vertex
        # For each vertex, count neighbors that are not selected
        # neighbor_count[i] = sum of (1 - selected[j]) for j in neighbors[i]
        
        # Compute using matrix multiplication
        # unselected_neighbor_count[i] = deg[i] - sum(selected[j] for j in neighbors[i])
        selected_neighbor_sum = A.dot(selected.astype(float))
        unselected_neighbor_count = deg - selected_neighbor_sum
        
        # Only count for selected vertices
        E_secondary = np.sum(unselected_neighbor_count * selected)
        
        # Tertiary term: smooth surrogate using log-sum-exp
        # This encourages a more "spread out" selection pattern
        # Compute a soft count using temperature-scaled scores
        # For each vertex, compute a "selection probability" based on its rank
        order = np.argsort(-s, kind='stable')
        rank = np.empty(N, dtype=float)
        rank[order] = np.arange(N)
        
        # Soft selection: use sigmoid of (score - threshold)
        # This creates a smooth approximation of the discrete selection
        score_range = s.max() - s.min() + 1e-10
        normalized_s = (s - s.min()) / score_range
        
        # Temperature controls sharpness
        beta = 5.0
        soft_selected = 1.0 / (1.0 + np.exp(-beta * (normalized_s - 0.5)))
        
        # E_tertiary: penalize variance in soft selection (encourages uniform spread)
        E_tertiary = np.var(soft_selected)
        
        # Combine with weights
        E = E_primary + 0.1 * E_secondary + 0.01 * E_tertiary
        
        return E, count
    
    # --- 3. Simulated Annealing with Ising-like energy ---
    rng = np.random.default_rng(42)
    
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    # Initial energy
    best_energy, _ = compute_energy(best_scores)
    
    # Simulated annealing parameters
    T_initial = 1.0
    T_final = 0.01
    cooling_rate = 0.999
    
    current_scores = best_scores.copy()
    current_energy, current_val = compute_energy(current_scores)
    
    T = T_initial
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() - start_time < DEADLINE and no_improve < max_no_improve and T > T_final:
        # Choose move: 70% swap two scores, 30% perturb one score
        if rng.random() < 0.7:
            # Swap two scores
            i, j = rng.integers(0, N, size=2)
            if i == j:
                continue
            
            # Save old scores
            old_i, old_j = current_scores[i], current_scores[j]
            current_scores[i], current_scores[j] = old_j, old_i
            
            new_energy, new_val = compute_energy(current_scores)
            delta_E = new_energy - current_energy
            
            # Metropolis acceptance
            if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                current_energy = new_energy
                current_val = new_val
                no_improve = 0
                if new_val > best_val:
                    best_val = new_val
                    best_scores = current_scores.copy()
                    best_energy = new_energy
            else:
                # Revert
                current_scores[i], current_scores[j] = old_i, old_j
                no_improve += 1
        else:
            # Perturb one score
            i = rng.integers(0, N)
            old_i = current_scores[i]
            current_scores[i] += rng.normal(0, 0.1)
            
            new_energy, new_val = compute_energy(current_scores)
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                current_energy = new_energy
                current_val = new_val
                no_improve = 0
                if new_val > best_val:
                    best_val = new_val
                    best_scores = current_scores.copy()
                    best_energy = new_energy
            else:
                current_scores[i] = old_i
                no_improve += 1
        
        T *= cooling_rate
    
    # --- 4. Final high-precision polish pass ---
    # Use tighter tolerance and more iterations
    polish_scores = best_scores.copy()
    polish_val = best_val
    
    score_order = np.argsort(-polish_scores)
    no_improve_polish = 0
    max_no_improve_polish = 10000
    
    while time.time() - start_time < DEADLINE - 2.0 and no_improve_polish < max_no_improve_polish:
        # Structured swaps only
        base_idx = rng.integers(0, N - 1)
        offset = rng.integers(1, min(5, N - 1))
        i = score_order[base_idx]
        j = score_order[min(base_idx + offset, N - 1)]
        
        if i == j:
            continue
        
        # Swap
        polish_scores[i], polish_scores[j] = polish_scores[j], polish_scores[i]
        new_val = simulate_fast(polish_scores)
        
        if new_val > polish_val:
            polish_val = new_val
            no_improve_polish = 0
            if polish_val > best_val:
                best_val = polish_val
                best_scores = polish_scores.copy()
                score_order = np.argsort(-best_scores)
        else:
            # Revert
            polish_scores[i], polish_scores[j] = polish_scores[j], polish_scores[i]
            no_improve_polish += 1
    
    return best_scores
