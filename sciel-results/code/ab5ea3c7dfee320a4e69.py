import time
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin before 120s limit

def _fiedler_vector(adjacency, n):
    """
    Compute the Fiedler vector (eigenvector of the second-smallest eigenvalue
    of the graph Laplacian). Returns an array of length n with the Fiedler
    values, or None on failure.
    """
    if n < 3:
        return np.zeros(n)
    
    degree = adjacency.sum(axis=1).astype(np.float64)
    A = csr_matrix(adjacency.astype(np.float64))
    D = diags(degree)
    L = D - A
    
    try:
        k = min(2, n - 1)
        if k < 2:
            return np.zeros(n)
        
        # Use 'SA' (smallest algebraic) for better stability with Laplacians
        vals, vecs = eigsh(L, k=k, which='SA', maxiter=n*10)
        
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        if len(vals) >= 2:
            fiedler = vecs[:, 1]
            # Remove mean to ensure orthogonality to all-ones
            fiedler = fiedler - fiedler.mean()
            norm = np.linalg.norm(fiedler)
            if norm > 1e-12:
                fiedler = fiedler / norm
            return fiedler
        else:
            return np.zeros(n)
            
    except Exception:
        return None

def priority(adjacency):
    """
    Compute priority scores for each vertex.
    
    Strategy:
    1. Base Heuristic: "2-Hop Survival Probability"
       score_base = 1 / (1 + degree + 0.5 * avg_neighbor_degree)
       This prioritizes vertices that are isolated from dense cores.
       
    2. Spectral Regularization:
       Add a small component based on the Fiedler vector to encourage
       selection spread across the graph's natural cuts, preventing
       the greedy decoder from saturating one cluster and ignoring others.
       
    3. Refinement: Simulated Annealing with Directed Conflict Swaps.
       - Random swaps for exploration.
       - Conflict swaps: Identify a rejected vertex v blocked by a selected 
         neighbor u. Swap their scores to see if v can be selected instead,
         potentially unlocking more selections later.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    A = np.asarray(adjacency, dtype=bool)
    
    # Precompute neighbor lists for fast simulation
    neighbors_list = [np.where(A[i])[0] for i in range(n)]
    deg = A.sum(axis=1).astype(float)
    
    # 1. Base Heuristic: 2-Hop Survival
    # Sum of degrees of neighbors
    neighbor_deg_sum = A.dot(deg)
    safe_deg = deg + 1e-6
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # Higher is better. 
    # Vertices with low degree and low neighbor degree are "safe" to pick early.
    raw_survival = 1.0 / (1.0 + deg + 0.5 * avg_neighbor_deg)
    
    # 2. Spectral Component
    fiedler = _fiedler_vector(A, n)
    if fiedler is None:
        fiedler = np.zeros(n)
    
    # Normalize Fiedler to [0, 1]
    f_min, f_max = fiedler.min(), fiedler.max()
    if f_max - f_min > 1e-12:
        fiedler_norm = (fiedler - f_min) / (f_max - f_min)
    else:
        fiedler_norm = np.zeros(n)
    
    # Normalize survival to [0, 1]
    s_min, s_max = raw_survival.min(), raw_survival.max()
    if s_max - s_min > 1e-12:
        surv_norm = (raw_survival - s_min) / (s_max - s_min)
    else:
        surv_norm = np.ones(n)
    
    # Combine: Dominant survival, small spectral spread
    # Weight 0.9 on survival, 0.1 on spectral
    scores = 0.9 * surv_norm + 0.1 * fiedler_norm
    
    # Simulation Function
    def simulate_fast(s):
        """
        Fast greedy simulation.
        Returns the size of the independent set and the set itself (as boolean mask).
        """
        order = np.argsort(-s, kind='stable')
        active = np.ones(n, dtype=bool)
        count = 0
        picked_mask = np.zeros(n, dtype=bool)
        
        for v in order:
            if active[v]:
                count += 1
                picked_mask[v] = True
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    active[nbrs] = False
        return count, picked_mask, order

    # Initial Simulation
    best_val, best_mask, best_order = simulate_fast(scores)
    best_scores = scores.copy()
    
    # Local Search: Simulated Annealing
    rng = np.random.default_rng(42)
    
    # Temperature schedule
    # Start high, decrease linearly
    t_start = time.time()
    duration = _DEADLINE - t_start
    
    # We don't know the exact number of steps, so we loop until time is up.
    # We track "temperature" based on elapsed time.
    
    current_scores = best_scores.copy()
    current_val, current_mask, _ = simulate_fast(current_scores)
    
    # Identify blocked vertices for conflict swaps
    # A vertex v is blocked if it is not picked, but has a neighbor u that IS picked 
    # and u appears earlier in the order than v would have?
    # Actually, in the greedy decoder, v is blocked if any neighbor u is picked before v.
    # We want to find pairs (v, u) where v is rejected, u is picked, and u is a neighbor of v.
    # Swapping scores of v and u might allow v to be picked, removing u.
    # This is a "1-for-1" swap in the picked set, but it might change the availability
    # of other vertices.
    
    while time.time() < _DEADLINE:
        # Calculate temperature (1.0 at start, 0.0 at end)
        elapsed = time.time() - t_start
        temp = max(0.0, 1.0 - (elapsed / duration))
        
        # Choose move type
        # 50% Random Swap, 50% Conflict Swap
        if rng.random() < 0.5:
            # Random Swap
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            
            # Swap
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_val, new_mask, _ = simulate_fast(current_scores)
            
            if new_val > current_val:
                current_val = new_val
                if new_val > best_val:
                    best_val = new_val
                    best_scores = current_scores.copy()
                    best_mask = new_mask
            else:
                # Metropolis criterion: accept with probability exp(-delta/T)
                delta = current_val - new_val
                if delta > 0 and temp > 0.01:
                    if rng.random() < np.exp(-delta / (temp + 1e-6)):
                        current_val = new_val
                        current_mask = new_mask
                    else:
                        # Revert
                        current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                else:
                    # Revert
                    current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        
        else:
            # Conflict Swap
            # Find a rejected vertex v and a picked neighbor u.
            # We need to be careful: we need v to be rejected and u to be picked.
            # And u should be "responsible" for blocking v (i.e., u is picked before v in the order).
            
            # Get current order
            order = np.argsort(-current_scores, kind='stable')
            
            # Find a random rejected vertex
            rejected_indices = np.where(~current_mask)[0]
            if len(rejected_indices) == 0:
                # All picked? Unlikely for MIS, but handle it
                continue
                
            v = rejected_indices[rng.integers(0, len(rejected_indices))]
            
            # Find picked neighbors of v
            nbrs = neighbors_list[v]
            picked_nbrs = [u for u in nbrs if current_mask[u]]
            
            if len(picked_nbrs) == 0:
                # v was rejected not because of a picked neighbor? 
                # In greedy MIS, if v is active when its turn comes, it is picked.
                # So v must have been deactivated by a neighbor picked earlier.
                # So there MUST be a picked neighbor.
                # Unless v was never active? No, all start active.
                # So this case shouldn't happen if logic is correct.
                continue
            
            # Pick a random picked neighbor
            u = picked_nbrs[rng.integers(0, len(picked_nbrs))]
            
            # Swap scores of v and u
            current_scores[v], current_scores[u] = current_scores[u], current_scores[v]
            new_val, new_mask, _ = simulate_fast(current_scores)
            
            if new_val > current_val:
                current_val = new_val
                if new_val > best_val:
                    best_val = new_val
                    best_scores = current_scores.copy()
                    best_mask = new_mask
            else:
                delta = current_val - new_val
                if delta > 0 and temp > 0.01:
                    if rng.random() < np.exp(-delta / (temp + 1e-6)):
                        current_val = new_val
                        current_mask = new_mask
                    else:
                        # Revert
                        current_scores[v], current_scores[u] = current_scores[u], current_scores[v]
                else:
                    # Revert
                    current_scores[v], current_scores[u] = current_scores[u], current_scores[v]

    return best_scores
