"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as sla

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = labels[None, :] != labels[:, None]
    return int((adjacency & diff).sum() // 2)

def _spectral_init(adjacency, n):
    """
    Generate an initial partition using the Fiedler vector (second smallest eigenvector)
    of the graph Laplacian. This provides a high-quality starting point that is
    structurally informed, unlike pure random initialization.
    """
    # Construct Laplacian L = D - A
    # A is the adjacency matrix (bool, symmetric)
    # D is the degree matrix
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    A = sp.csr_matrix(adjacency)
    L = D - A
    
    try:
        # We want the eigenvector corresponding to the second smallest eigenvalue.
        # sla.eigsh finds eigenvalues near sigma. sigma=0 finds the smallest.
        # Since L is PSD, the smallest is 0 (constant vector). The next is the Fiedler vector.
        # k=2 gives the two smallest.
        vals, vecs = sla.eigsh(L, k=2, sigma=0, which='LM')
        
        # The eigenvectors are returned in order of eigenvalues.
        # vecs[:, 0] corresponds to the smallest eigenvalue (0).
        # vecs[:, 1] corresponds to the second smallest (Fiedler vector).
        fiedler = vecs[:, 1]
        
        # Partition based on the sign of the Fiedler vector
        # If value > 0, label 1; else label 0
        labels = (fiedler > 0).astype(np.int8)
        
        # Ensure both partitions are non-empty (though unlikely for connected graphs)
        if labels.sum() == 0 or labels.sum() == n:
             # Fallback to random if spectral fails to split
             labels = np.random.randint(0, 2, size=n)
             
        return labels
        
    except Exception:
        # Fallback to random if eigenvalue computation fails
        return np.random.randint(0, 2, size=n)

def _local_search(adjacency, labels, max_iters=10000, time_limit=None):
    """
    Fast local search using the FLIP rule.
    A move is accepted if it strictly increases the cut size.
    Uses gain vectors for O(deg) update per step.
    """
    n = len(labels)
    labels = labels.copy()
    
    # Precompute neighbors for faster access? 
    # For n=100, iterating over np.where is fine, but let's keep it simple.
    
    # Calculate initial cut
    cut = _cut_size(adjacency, labels)
    
    # Initialize gain array
    # gain[i] = (# neighbors with different label) - (# neighbors with same label)
    # If gain[i] > 0, flipping i increases cut by gain[i].
    
    # Vectorized calculation of gains
    # diff[i,j] = 1 if labels[i] != labels[j]
    diff_mat = labels[None, :] != labels[:, None]
    # other_counts[i] = sum(adjacency[i,j] * diff[i,j])
    other_counts = (adjacency & diff_mat).sum(axis=1)
    # same_counts[i] = sum(adjacency[i,j] * (1 - diff[i,j]))
    # Note: adjacency[i,i] is 0, so diagonal doesn't matter for same_counts via this method if we are careful,
    # but diff[i,i] is False (0), so 1-diff[i,i] is 1. 
    # same_counts = (adjacency & ~diff_mat).sum(axis=1)
    same_counts = (adjacency & ~diff_mat).sum(axis=1)
    
    gains = other_counts - same_counts
    
    # Local search loop
    improved = True
    iters = 0
    start_time = time.time()
    
    while improved:
        improved = False
        iters += 1
        
        if max_iters and iters > max_iters:
            break
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        # Find the vertex with the maximum positive gain
        max_gain = gains.max()
        
        if max_gain <= 0:
            break # No improving move exists
            
        # Pick one of the vertices with max_gain
        # Using argmax is deterministic, which is fine for local search convergence.
        # Randomizing selection among max gain vertices can help escape plateaus, 
        # but since we only accept strict improvements, plateaus don't matter for the loop condition.
        v = np.argmax(gains)
        
        # Apply flip
        cut += gains[v]
        old_label = labels[v]
        labels[v] = 1 - old_label
        
        # Update gains
        # 1. Gain of v becomes -gain[v]
        gains[v] = -gains[v]
        
        # 2. Update gains of neighbors
        # For each neighbor u of v:
        # If labels[u] == old_label (same as v before flip):
        #    u and v were same. Now they are different.
        #    u's 'other' count increases by 1, 'same' decreases by 1.
        #    gain[u] increases by 2.
        # If labels[u] != old_label (different from v before flip):
        #    u and v were different. Now they are same.
        #    u's 'other' count decreases by 1, 'same' increases by 1.
        #    gain[u] decreases by 2.
        
        neighbors = np.where(adjacency[v])[0]
        for u in neighbors:
            if labels[u] == 1 - old_label:
                # labels[u] is same as NEW label of v.
                # This means labels[u] was DIFFERENT from OLD label of v.
                # Wait, let's re-verify.
                # old_label = L_v. new_label = 1-L_v.
                # Condition: labels[u] == new_label.
                # So labels[u] != old_label.
                # Before flip: u and v were different.
                # After flip: u and v are same.
                # So u loses an 'other' neighbor and gains a 'same' neighbor.
                # gain[u] decreases by 2.
                gains[u] -= 2
            else:
                # labels[u] != new_label.
                # So labels[u] == old_label.
                # Before flip: u and v were same.
                # After flip: u and v are different.
                # So u gains an 'other' neighbor and loses a 'same' neighbor.
                # gain[u] increases by 2.
                gains[u] += 2
                
        improved = True
        
    return labels, cut

def partition(adjacency):
    """
    Main partition function using Multi-Start Local Search with Spectral Initialization.
    """
    n = len(adjacency)
    global_deadline = time.time() + 145.0 # Safety margin for 150s limit
    
    best_labels = np.zeros(n, dtype=np.int8)
    best_cut = -1
    
    # We will run multiple independent local searches.
    # Each search starts from a spectral initialization.
    # We can vary the spectral init by slightly perturbing the graph or just relying on the fact
    # that spectral init is deterministic for a given graph, so we need diversity.
    # Actually, spectral init is deterministic. To get diversity, we can:
    # 1. Use spectral init for the first few.
    # 2. Then use random inits.
    # Or, better: Use spectral init, but then perform a "kick" (random flips) before local search.
    
    # Strategy:
    # 1. Get spectral init.
    # 2. Run local search from it.
    # 3. For subsequent runs: Start from best found so far, apply random "kick" (flip K random nodes), then local search.
    #    This is a form of "iterated local search".
    
    # Let's implement Iterated Local Search (ILS) with Spectral Warm Start.
    
    # Initial spectral partition
    current_labels = _spectral_init(adjacency, n)
    
    # First pass: Pure local search from spectral
    current_labels, current_cut = _local_search(adjacency, current_labels)
    
    if current_cut > best_cut:
        best_cut = current_cut
        best_labels = current_labels.copy()
    
    # Iterated Local Search loop
    # We estimate how many iterations we can afford.
    # Each iteration involves a "kick" (O(K)) and local search (O(n * deg * iters_ls)).
    # For n=100, local search is very fast (< 1ms usually).
    # Let's say we can do ~10,000 iterations in 140s.
    
    kick_size = 10 # Number of random flips to apply before each LS
    
    while time.time() < global_deadline:
        # Create a perturbed version of the best solution
        # Flip 'kick_size' random vertices
        perturbed_labels = best_labels.copy()
        if n > kick_size:
            flip_indices = np.random.choice(n, size=kick_size, replace=False)
            perturbed_labels[flip_indices] = 1 - perturbed_labels[flip_indices]
        else:
            perturbed_labels = np.random.randint(0, 2, size=n)
            
        # Run local search from perturbed state
        # Limit local search iterations per restart to ensure we do many restarts
        # If we allow unlimited LS, we might get stuck in one long LS.
        # But LS converges fast. Let's allow it to converge.
        new_labels, new_cut = _local_search(adjacency, perturbed_labels)
        
        if new_cut > best_cut:
            best_cut = new_cut
            best_labels = new_labels.copy()
            
            # If we found a new best, we might want to reduce kick size or keep it same.
            # Keeping it same is standard.
            
        # Optional: If we haven't improved in a long time, try a completely random restart?
        # For now, ILS from best is robust.
        
        # Small sleep to avoid 100% CPU? No, we want max speed.
        
    return best_labels
# EVOLVE-BLOCK-END
