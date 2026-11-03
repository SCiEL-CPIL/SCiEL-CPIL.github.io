import numpy as np
import time
import itertools
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit (allowing for overhead)
    
    # 1. Generate all weight-4 vectors (four ±1 entries)
    # Norm squared is 4. Constraint: dot product <= 2
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # 2. Compute dot product matrix and conflict matrix
    # n_cand is 5280. 5280x5280 matrix is ~28M entries.
    # Using int8 for conflict to save memory and speed up operations.
    dots = candidates @ candidates.T
    # Conflict if dot > 2. Note: diagonal is 4, so it would be a conflict with itself, but we ignore self.
    conflict = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # 3. Spectral Initialization
    # We want to find dense clusters in the COMPATIBILITY graph (non-conflicts).
    # Or, find sparse clusters in the CONFLICT graph.
    # Let's use the conflict graph. Independent sets are clusters of nodes with no edges between them.
    # The Fiedler vector of the conflict graph might help identify partitions.
    # However, computing eigenvectors of a 5280x5280 sparse matrix is fast enough.
    
    # Build sparse conflict matrix
    conflict_sparse = csr_matrix(conflict)
    
    try:
        # Compute Laplacian of the conflict graph
        L = laplacian(conflict_sparse, normed=True)
        
        # Get the Fiedler vector (2nd smallest eigenvalue)
        # eigsh finds a few eigenvalues. We want k=2, smallest.
        # sigma=0 to find smallest.
        vals, vecs = eigsh(L, k=2, sigma=0, which='SM')
        # The Fiedler vector is the eigenvector corresponding to the 2nd smallest eigenvalue
        # eigsh returns eigenvalues in ascending order if which='SM' and sigma=0? 
        # Actually, eigsh with sigma=0 shifts the spectrum. It's safer to just get the smallest few.
        # Let's just use the vector with the largest variance or simply use the 2nd vector returned.
        # For a connected graph, the first eigenvector is constant. The second is the Fiedler vector.
        fiedler = vecs[:, 1]
    except Exception:
        # Fallback to random or degree-based ordering if spectral fails
        fiedler = np.random.rand(n_cand)
        
    # 4. Greedy Construction with Spectral Ordering
    # Sort nodes by Fiedler vector value. 
    # Nodes with similar values are likely in the same "side" of the graph.
    # We want to pick an independent set.
    
    def greedy_construct(order):
        in_set = np.zeros(n_cand, dtype=bool)
        selected = []
        
        for i in order:
            if time.time() > deadline - 20:
                break
            if in_set[i]:
                continue
            # Check if i conflicts with any selected node
            # conflict[i] is a row of the conflict matrix
            # We need to check if any selected node j has conflict[i][j] == 1
            # This is equivalent to checking if the sum of conflicts with selected is 0
            if np.sum(conflict[i][in_set]) == 0:
                in_set[i] = True
                selected.append(i)
                
        return in_set, selected

    def local_search(in_set, selected):
        # Try to improve the set
        # 1. Add any node that doesn't conflict
        improved = True
        while improved:
            if time.time() > deadline - 10:
                return in_set, selected
            improved = False
            unselected = np.where(~in_set)[0]
            for i in unselected:
                if np.sum(conflict[i][in_set]) == 0:
                    in_set[i] = True
                    selected.append(i)
                    improved = True
                    break # Restart loop to check new additions
            
        # 2. Swap moves: Remove one node u, then greedily add as many as possible
        # This is expensive, so we limit the number of swaps or randomize the order of u
        max_swaps = 50
        for _ in range(max_swaps):
            if time.time() > deadline - 5:
                break
            
            # Try to find a beneficial swap
            # We can't just iterate all u and do a full greedy refill (too slow).
            # Heuristic: For each u in selected (random order), remove it, 
            # then try to add any single node that was previously blocked only by u.
            # Or simpler: Remove u, then run a few steps of "add if possible".
            
            # Let's try a simpler local search: 
            # For each unselected node i, count its conflicts with selected.
            # If count is 1, say with u, check if removing u allows adding i AND if we can add another node j that was blocked by i or u.
            # This is complex. Let's stick to "Remove u, then add any node that is now free".
            
            # To keep it fast, we'll just do a limited number of "remove and refill" attempts.
            # Pick a random selected node to remove.
            if not selected:
                break
                
            u_idx = random.choice(selected)
            u_pos = selected.index(u_idx)
            
            # Temporarily remove u
            in_set[u_idx] = False
            selected.pop(u_pos)
            
            # Try to add nodes
            added = 0
            unselected = np.where(~in_set)[0]
            # Sort unselected by number of conflicts? No, just iterate.
            # To be fast, we only try to add nodes that had exactly 1 conflict (which was u).
            # We can precompute degrees or just check on the fly.
            
            for i in unselected:
                if time.time() > deadline - 5:
                    break
                # Check conflict with current set
                if np.sum(conflict[i][in_set]) == 0:
                    in_set[i] = True
                    selected.append(i)
                    added += 1
                    # We stop adding if we've added more than we removed (1)
                    # Actually, we want to maximize size. 
                    # If we add 1, size is same. If we add 2, size increases.
                    # We should continue adding as long as possible.
            
            # If we didn't end up with a larger set, revert?
            # No, the greedy refill might have added nodes that block others.
            # A simple "improve or revert" is hard to implement efficiently without backtracking.
            # However, the "add if possible" loop above is monotonic in terms of "filling gaps".
            # If the new set size is larger, we keep it. If not, we might be stuck in a local minimum.
            # Given the time constraints, we accept the result of the refill.
            
        return in_set, selected

    best_in_set = None
    best_k = 0
    
    # Run multiple trials with different orderings
    # Trial 1: Spectral ordering
    order_spec = np.argsort(fiedler)
    in_set, selected = greedy_construct(order_spec)
    in_set, selected = local_search(in_set, selected)
    if len(selected) > best_k:
        best_k = len(selected)
        best_in_set = in_set.copy()
        
    # Trial 2-10: Random orderings
    for trial in range(10):
        if time.time() > deadline - 20:
            break
            
        order_rand = np.random.permutation(n_cand)
        in_set, selected = greedy_construct(order_rand)
        in_set, selected = local_search(in_set, selected)
        
        if len(selected) > best_k:
            best_k = len(selected)
            best_in_set = in_set.copy()
            
    # Final result
    if best_in_set is None:
        # Fallback: just return first few non-conflicting
        in_set = np.zeros(n_cand, dtype=bool)
        selected = []
        for i in range(n_cand):
            if np.sum(conflict[i][in_set]) == 0:
                in_set[i] = True
                selected.append(i)
        best_in_set = in_set
        
    result = candidates[best_in_set]
    return result.astype(np.int64)
