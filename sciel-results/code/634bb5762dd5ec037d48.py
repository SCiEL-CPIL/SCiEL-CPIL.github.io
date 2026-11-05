"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE_MARGIN = 5.0

def partition(adjacency):
    start_time = time.time()
    deadline = start_time + 120.0 - DEADLINE_MARGIN
    
    # Ensure adjacency is a numpy array for fast operations
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    
    # Precompute degrees for potential use, though gain calc is direct
    # degrees = adj.sum(axis=1)
    
    def calc_cut(labels):
        # Vectorized cut calculation
        # labels: 0/1 array of length n
        # Cut = sum of edges (u,v) where labels[u] != labels[v]
        # Using matrix multiplication:
        # Let s = 2*labels - 1 (values -1, 1)
        # (s @ s.T) gives 1 if same, -1 if different? No.
        # s_i * s_j is 1 if same, -1 if diff.
        # So diff_mask is (1 - s_i*s_j)/2 ?
        # Simpler:
        # diff = (labels[:, None] != labels[None, :])
        # return int(np.sum(adj * diff))
        # This creates an n x n boolean matrix. For n=100, this is 10k elements. Very fast.
        diff = (labels[:, None] != labels[None, :])
        return int(np.sum(adj * diff))
    
    def local_search(labels, time_limit):
        """
        Asynchronous greedy local search.
        Repeatedly finds the vertex with the maximum positive gain and flips it.
        Stops when no positive gain exists or time limit is reached.
        Returns the best labels found.
        """
        labels = labels.copy()
        current_cut = calc_cut(labels)
        best_cut = current_cut
        best_labels = labels.copy()
        
        # Precompute adjacency rows for faster access? adj is already numpy.
        
        while time.time() < time_limit:
            # Calculate gain for flipping each vertex v
            # Gain(v) = (neighbors with opposite label) - (neighbors with same label)
            # If Gain(v) > 0, flipping v increases the cut by Gain(v).
            
            # Vectorized gain calculation:
            # same_side_neighbors[v] = sum(adj[v, labels == labels[v]])
            # other_side_neighbors[v] = sum(adj[v, labels != labels[v]])
            
            # Create a mask of same-side neighbors for each vertex
            # match_mask[i, j] is True if labels[i] == labels[j]
            match_mask = (labels[:, None] == labels[None, :])
            
            # same_side[v] = sum of adj[v, :] where match_mask[v, :] is True
            same_side = (adj * match_mask).sum(axis=1)
            other_side = (adj * (~match_mask)).sum(axis=1)
            
            gains = other_side - same_side
            
            # Find the vertex with the maximum positive gain
            max_gain = np.max(gains)
            
            if max_gain <= 0:
                break
            
            # Flip the vertex with max gain
            # If multiple vertices have the same max gain, np.argmax picks the first one.
            # This is a valid greedy strategy.
            flip_idx = np.argmax(gains)
            labels[flip_idx] = 1 - labels[flip_idx]
            
            # Update cut
            current_cut += max_gain
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                
        return best_labels, best_cut

    best_global_labels = None
    best_global_cut = -1
    
    # Initial set of starting partitions
    starts = []
    
    # 1. Spectral Initialization (Fiedler Vector)
    try:
        from scipy.sparse.csgraph import laplacian
        from scipy.sparse.linalg import eigsh
        
        L = laplacian(adj)
        # We want the Fiedler vector (eigenvector for the 2nd smallest eigenvalue)
        # eigsh with k=2, which='SM'
        w, v = eigsh(L, k=2, which='SM')
        idx = np.argsort(w)
        fiedler_vec = v[:, idx[1]]
        
        # Partition based on sign
        labels_spec = (fiedler_vec >= 0).astype(int)
        
        # Ensure non-trivial partition if possible
        if 0 < labels_spec.sum() < n:
            starts.append(labels_spec)
        else:
            # Fallback if spectral gives trivial partition
            labels_spec = np.arange(n) % 2
            starts.append(labels_spec)
    except Exception:
        pass # If spectral fails, we still have random starts
    
    # 2. Random Initializations
    # Add a few random starts to escape potential local optima of spectral
    for _ in range(10):
        labels_rand = np.random.randint(0, 2, size=n)
        # Ensure non-trivial
        if 0 < labels_rand.sum() < n:
            starts.append(labels_rand)
        else:
            # Force non-trivial
            labels_rand = np.arange(n) % 2
            starts.append(labels_rand)
    
    # Run local search on each start
    # We have ~11 starts. 
    # Each local search step is O(n^2) for gain calc. n=100 -> 10k ops.
    # Python overhead is significant. 
    # Let's estimate: 10k ops in numpy is ~10-50us. Loop overhead ~10us. 
    # So ~100-200us per step. 
    # If a local search takes 100 steps, that's 20ms.
    # 11 starts * 20ms = 220ms. 
    # We have 115s. So we can afford MANY more starts or longer searches.
    
    # Strategy: 
    # Run local search on the spectral start first.
    # Then run on random starts.
    # If time permits, generate more random starts or do "restarts" from perturbed best solutions.
    
    # Let's structure it to use the full time budget.
    # 1. Process the initial list of starts.
    for labels_start in starts:
        if time.time() >= deadline:
            break
        
        # Allow a fraction of the remaining time for this local search?
        # Or just let it run until convergence?
        # Convergence is usually fast for n=100.
        # Let's give it a generous limit, e.g., 10s per start, but stop early if converged.
        # Actually, just passing 'deadline' is safer, but if one start takes 100s, others get skipped.
        # Better: give each start a time slice, or just let it run and rely on the outer loop check.
        # Since local search is fast, let's just run it to convergence.
        
        labels_res, cut_res = local_search(labels_start, deadline)
        
        if cut_res > best_global_cut:
            best_global_cut = cut_res
            best_global_labels = labels_res.copy()
            
    # If we have time left, do more random starts or perturbations
    # This is where the "multi-start" really shines.
    # We can generate new random starts and run local search on them.
    
    # Let's estimate how many more we can do.
    # If local search takes ~10ms on average, we can do ~10,000 starts in 100s.
    # Let's try to fill the remaining time with random restarts.
    
    # To avoid infinite loop if local search is very slow, we check time in the outer loop.
    # We will generate random starts in batches.
    
    while time.time() < deadline - 1.0: # Leave 1s safety margin for final return
        # Generate a random start
        labels_rand = np.random.randint(0, 2, size=n)
        if labels_rand.sum() == 0 or labels_rand.sum() == n:
             labels_rand = np.arange(n) % 2
             
        # Run local search
        # To prevent a single pathological case from eating all time, 
        # we could limit the number of iterations, but for n=100 it's usually fine.
        # However, if the graph is dense, gains might oscillate? No, greedy monotonic increase ensures termination.
        # Max cut size is bounded by E. Cut increases by integer amounts.
        # So it MUST terminate.
        
        labels_res, cut_res = local_search(labels_rand, deadline)
        
        if cut_res > best_global_cut:
            best_global_cut = cut_res
            best_global_labels = labels_res.copy()
            
        # Optional: If we found a new best, we could also try perturbing it.
        # But random restarts are simpler and often sufficient for n=100.
        
    if best_global_labels is None:
        # Fallback if something went wrong
        best_global_labels = np.arange(n) % 2
        
    return best_global_labels
# EVOLVE-BLOCK-END
