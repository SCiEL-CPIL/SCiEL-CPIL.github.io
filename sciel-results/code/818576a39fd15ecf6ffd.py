# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)

def _spectral_init(adjacency):
    """Use spectral method (eigenvectors of Laplacian) for initial partition."""
    n = len(adjacency)
    if n < 2:
        return np.zeros(n, dtype=np.int8)
        
    A = adjacency.astype(np.float64)
    degrees = A.sum(axis=1)
    # Laplacian L = D - A
    L = np.diag(degrees) - A
    
    try:
        # Use sparse eigsh for efficiency
        L_sparse = csr_matrix(L)
        # We want the Fiedler vector (2nd smallest eigenvalue)
        # which='SM' is slow, 'SA' (smallest algebraic) or 'LR' (largest regular) might be better depending on library
        # For Laplacian, smallest non-zero is key. 
        # eigsh with k=2, which='SA' usually works for connected graphs.
        if n > 2:
            vals, vecs = eigsh(L_sparse, k=2, which='SA')
            # Sort by eigenvalue
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]
        else:
            fiedler = np.random.randn(n)
    except Exception:
        # Fallback to dense if sparse fails
        try:
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]
        except Exception:
            fiedler = np.random.randn(n)
    
    # Partition based on sign of Fiedler vector
    labels = (fiedler >= 0).astype(np.int8)
    
    # Ensure not all 0 or all 1
    if labels.sum() == 0 or labels.sum() == n:
         labels = (fiedler >= np.median(fiedler)).astype(np.int8)
    if labels.sum() == 0 or labels.sum() == n:
         labels = np.arange(n) % 2
         
    return labels

def _fast_simulated_annealing(adjacency, labels, deadline, max_iters=200000):
    """
    Simulated annealing with O(N) update per step.
    Maintains a 'gains' array where gains[v] is the change in cut size if v is flipped.
    gain[v] = deg(v) - 2 * (number of neighbors with same label as v)
    Actually, standard definition:
    Current contribution of v to cut: cut_v = sum(A[v,u] * (1 if label[u]!=label[v] else 0))
    If we flip v, new contribution: new_cut_v = sum(A[v,u] * (1 if label[u]==label[v] else 0))
    Gain = new_cut_v - cut_v
    
    Let S_v = sum(A[v,u] where label[u] == label[v])
    Let D_v = deg(v)
    Cut_v = D_v - S_v
    New_Cut_v = S_v
    Gain = S_v - (D_v - S_v) = 2*S_v - D_v
    
    When we flip vertex u:
    For each neighbor v of u:
    If label[v] == old_label[u] (which is new_label[u] after flip? No, old_label[u] != new_label[u])
    Let's trace carefully.
    Initially:
    gains[v] = 2 * S_v - D_v
    
    Flip vertex u.
    For any neighbor v of u:
    S_v changes.
    If label[v] == label[u] (before flip):
       Before: u contributed 1 to S_v.
       After: u contributes 0 to S_v (because labels differ).
       S_v decreases by 1.
       gains[v] decreases by 2.
    If label[v] != label[u] (before flip):
       Before: u contributed 0 to S_v.
       After: u contributes 1 to S_v (because labels become same).
       S_v increases by 1.
       gains[v] increases by 2.
       
    So, for each neighbor v of u:
    if label[v] == label[u]: gains[v] -= 2
    else: gains[v] += 2
    
    Also, the gain for u itself changes.
    Old Gain[u] = 2*S_u - D_u.
    New S_u (after flip) = D_u - S_u (old).
    New Gain[u] = 2*(D_u - S_u) - D_u = 2*D_u - 2*S_u = -(2*S_u - D_u) = -Old Gain[u].
    So gains[u] = -gains[u].
    """
    n = len(labels)
    labels = labels.copy()
    
    # Precompute degrees
    degrees = adjacency.sum(axis=1).astype(np.int64)
    
    # Initialize gains
    # S_v = number of neighbors with same label
    # We can compute this efficiently
    # same_mask[v, u] = (labels[v] == labels[u])
    # S_v = sum(adjacency[v, :] & same_mask[v, :])
    
    # Vectorized init
    labels_col = labels[:, None]
    labels_row = labels[None, :]
    same_mask = (labels_col == labels_row)
    # S_v for all v
    S = (adjacency & same_mask).sum(axis=1)
    gains = 2 * S - degrees
    
    current_cut = _cut_size(adjacency, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Annealing parameters
    # Start temp high enough to accept bad moves, cool down
    # Max possible gain is roughly max_degree. 
    # Let's start T such that exp(-gain/T) is reasonable.
    max_degree = np.max(degrees)
    current_temp = max_degree * 2.0 if max_degree > 0 else 5.0
    start_temp = current_temp
    # We want to end up with T ~ 0.1 or so
    # Cooling rate
    # If we run max_iters, we want T_end = T_start * (cooling_rate)^max_iters
    # Let's aim for T_end = 0.1
    if max_iters > 0:
        cooling_rate = (0.1 / start_temp) ** (1.0 / max_iters)
    else:
        cooling_rate = 0.999
        
    # Precompute neighbor lists for fast updates
    # adjacency is bool matrix
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Choose random vertex
        v = np.random.randint(n)
        
        gain = gains[v]
        
        # Acceptance probability
        if gain > 0 or np.random.rand() < np.exp(gain / max(current_temp, 1e-10)):
            # Flip v
            old_label = labels[v]
            labels[v] = 1 - old_label
            current_cut += gain
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                
            # Update gains for neighbors
            for u in neighbors[v]:
                # u is a neighbor of v
                # Check if label[u] == old_label (which is opposite to new_label)
                # If label[u] == old_label, they were same, now diff. S_u decreases by 1. gain[u] -= 2
                # If label[u] != old_label, they were diff, now same. S_u increases by 1. gain[u] += 2
                if labels[u] == old_label:
                    gains[u] -= 2
                else:
                    gains[u] += 2
                    
            # Update gain for v itself
            gains[v] = -gain
            
        current_temp *= cooling_rate
        
    return best_labels, best_cut

def _local_search_polish(adjacency, labels, deadline):
    """
    Fast 1-flip local search to polish the result.
    Uses the same O(N) gain update logic but only accepts positive gains.
    """
    n = len(labels)
    labels = labels.copy()
    
    degrees = adjacency.sum(axis=1).astype(np.int64)
    
    labels_col = labels[:, None]
    labels_row = labels[None, :]
    same_mask = (labels_col == labels_row)
    S = (adjacency & same_mask).sum(axis=1)
    gains = 2 * S - degrees
    
    current_cut = _cut_size(adjacency, labels)
    
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        # Find vertex with max positive gain
        max_gain = np.max(gains)
        if max_gain > 0:
            v = np.argmax(gains)
            old_label = labels[v]
            labels[v] = 1 - old_label
            current_cut += max_gain
            
            # Update gains
            for u in neighbors[v]:
                if labels[u] == old_label:
                    gains[u] -= 2
                else:
                    gains[u] += 2
            gains[v] = -max_gain
            improved = True
        else:
            break
            
    return labels, current_cut

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 145  # Safety margin: 145s out of 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Strategy:
    # 1. Spectral Init
    # 2. Fast SA
    # 3. Local Search Polish
    # 4. If time remains, repeat with different random seeds for SA
    
    # Initial Spectral
    spec_labels = _spectral_init(adjacency)
    
    # Run SA
    sa_labels, sa_cut = _fast_simulated_annealing(adjacency, spec_labels, deadline, max_iters=100000)
    
    # Polish
    pol_labels, pol_cut = _local_search_polish(adjacency, sa_labels, deadline)
    
    if pol_cut > best_cut:
        best_cut = pol_cut
        best_labels = pol_labels.copy()
        
    # If we have time, do more restarts
    # Each SA run takes some time. Let's estimate time per run.
    # We have ~145s total.
    # Let's try to fit as many high-quality SA runs as possible.
    
    num_extra_restarts = 5
    base_seed = 42
    
    for r in range(num_extra_restarts):
        if time.time() > deadline - 10: # Leave 10s for final polish and safety
            break
            
        # Perturb the best found so far or use random?
        # Perturbing best found is often better for convergence
        if best_labels is not None:
            # Randomly flip 10% of vertices
            perturb_idx = np.random.choice(n, size=max(1, n//10), replace=False)
            curr_labels = best_labels.copy()
            curr_labels[perturb_idx] = 1 - curr_labels[perturb_idx]
        else:
            curr_labels = np.random.randint(0, 2, n)
            
        # Run SA with a tighter deadline for this specific run
        # We want to ensure we don't overshoot the main deadline
        run_deadline = deadline - 10 
        # Estimate time: if we have 100s left and 5 runs, give each 15s?
        # Or just use the main deadline and let it break out?
        # Using main deadline is safer but might waste time if one run is slow.
        # Let's use a sub-deadline.
        sub_deadline = time.time() + min(20, deadline - 10 - time.time())
        
        sa_labels, sa_cut = _fast_simulated_annealing(adjacency, curr_labels, sub_deadline, max_iters=50000)
        
        # Polish
        pol_labels, pol_cut = _local_search_polish(adjacency, sa_labels, sub_deadline)
        
        if pol_cut > best_cut:
            best_cut = pol_cut
            best_labels = pol_labels.copy()
            
    # Final polish on the very best
    if best_labels is not None and time.time() < deadline - 2:
        final_labels, final_cut = _local_search_polish(adjacency, best_labels, deadline)
        if final_cut > best_cut:
            best_labels = final_labels
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
        
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
