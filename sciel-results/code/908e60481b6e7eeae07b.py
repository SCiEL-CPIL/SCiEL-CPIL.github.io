import numpy as np
import time
import itertools
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Generate candidate vectors with norm squared 4
    # Types:
    # A: (±2, 0, ..., 0)
    # B: (±1, ±1, ±1, ±1, 0, ..., 0)
    
    candidates = []
    
    # Type A
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type B
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    if n_cand == 0:
        return np.array([], dtype=np.int64).reshape(0, D)

    # 2. Compute pairwise squared distances
    # dist_sq[i, j] = ||c_i - c_j||^2
    # We need dist_sq >= 4 for all selected pairs.
    # Conflict if dist_sq < 4.
    
    # Memory check: n_cand ~ 5300. 5300x5300 float64 is ~224MB.
    # Might be tight, but let's try. If it fails, use block processing.
    try:
        diff = candidates[:, np.newaxis, :] - candidates[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
    except MemoryError:
        # Fallback to block processing if memory is an issue
        # But for 5300, it should fit in 256MB+ usually.
        # If not, we can just use random greedy without spectral.
        dist_sq = None
        spectral_order = np.random.permutation(n_cand)
    else:
        # Build Conflict Graph Adjacency
        # Conflict if dist_sq < 4 and i != j
        # Note: dist_sq[i,i] = 0, so i!=j is handled by <4 check if we mask diagonal?
        # 0 < 4 is True, so we must mask diagonal.
        conflict_mask = (dist_sq < 4)
        np.fill_diagonal(conflict_mask, False)
        
        # Spectral Partitioning
        # Create sparse adjacency matrix
        adj_matrix = lil_matrix(conflict_mask, dtype=np.float64)
        
        # Compute Laplacian
        L = laplacian(adj_matrix)
        
        # Get Fiedler vector (smallest non-zero eigenvalue)
        # eigh for sparse matrices
        try:
            # We need the second smallest eigenvalue's vector
            # scipy.linalg.eigh doesn't work directly on sparse.
            # Use scipy.sparse.linalg.eigsh
            from scipy.sparse.linalg import eigsh
            # k=2 to get lambda_1 and lambda_2. lambda_1 is 0 (for connected comp)
            # If graph is disconnected, lambda_1=0, lambda_2=0.
            # We want the vector corresponding to the smallest positive eigenvalue.
            # eigsh with sigma=0 might be slow.
            # Just get the smallest 2.
            vals, vecs = eigsh(L, k=2, which='SM')
            # Sort by value
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]
        except Exception:
            fiedler_vec = np.random.randn(n_cand)
            
        # Sort candidates by Fiedler vector
        spectral_order = np.argsort(fiedler_vec)
        
    # 3. Greedy Selection
    # We will run greedy selection multiple times with different orderings
    # 1. Spectral order
    # 2. Random orders
    
    best_set = []
    best_size = 0
    
    def greedy_select(order):
        selected_indices = []
        selected_vectors = []
        
        for idx in order:
            v = candidates[idx]
            # Check against all selected
            if not selected_vectors:
                selected_indices.append(idx)
                selected_vectors.append(v)
                continue
                
            # Vectorized distance check
            # diff = np.array(selected_vectors) - v
            # dists = np.sum(diff**2, axis=1)
            # if np.min(dists) >= 4:
            
            # Optimization: Early exit if we are close to deadline
            if time.time() > deadline - 5.0:
                break
                
            # Use numpy for speed
            # selected_vectors is a list, convert to array for batch op?
            # Better to keep a running array
            pass
            
        return selected_indices

    # Refactored Greedy for speed
    def run_greedy(order):
        selected_indices = []
        selected_vecs = [] # List of numpy arrays
        
        for idx in order:
            if time.time() > deadline - 5.0:
                break
                
            v = candidates[idx]
            
            if not selected_vecs:
                selected_indices.append(idx)
                selected_vecs.append(v)
                continue
                
            # Check distances
            # Stack selected_vecs into a matrix?
            # If len(selected_vecs) is large, this is expensive.
            # But max size is ~500. 500x11 is small.
            
            # Optimization: Only check if v is "compatible"
            # dist_sq >= 4
            
            # Convert list to array for vectorized op
            if len(selected_vecs) < 10:
                sel_arr = np.array(selected_vecs)
            else:
                # Cache the array?
                pass
                
            # Simple loop might be faster for small K?
            # No, vectorized is better.
            
            # Let's maintain a numpy array for selected vectors
            # This requires resizing.
            
            # Alternative:
            # dists = np.sum((np.array(selected_vecs) - v)**2, axis=1)
            # if np.min(dists) >= 4:
            
            # This creates a new array each time.
            # For 500 items, it's fine.
            
            try:
                sel_arr = np.array(selected_vecs)
                dists = np.sum((sel_arr - v)**2, axis=1)
                if np.min(dists) >= 4:
                    selected_indices.append(idx)
                    selected_vecs.append(v)
            except Exception:
                pass
                
        return selected_indices

    # Run 1: Spectral
    if dist_sq is not None:
        res = run_greedy(spectral_order)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res

    # Run 2-N: Random
    # Budget for random restarts
    t_start = time.time()
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        np.random.seed(i) # Reproducibility? No, let it be random or fixed.
        order = np.random.permutation(n_cand)
        res = run_greedy(order)
        
        if len(res) > best_size:
            best_size = len(res)
            best_set = res

    if not best_set:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return candidates[best_set]
# EVOLVE-BLOCK-END
