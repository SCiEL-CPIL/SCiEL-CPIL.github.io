"""Rewritten program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill

    # Generate all weight-4 vectors with entries in {-1, 0, 1}
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)

    vectors_v4 = np.array(v4_list, dtype=np.int64)
    N4 = len(vectors_v4)

    # Generate weight-1 vectors with entries in {-2, 0, 2}
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    vectors_v1 = np.array(v1_list, dtype=np.int64)
    N1 = len(vectors_v1)

    if N4 == 0:
        return vectors_v1

    # Compute pairwise dot products among v4 vectors
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute adjacency lists as lists of integers for faster iteration in local search
    adj = [np.where(conflicts[i])[0].tolist() for i in range(N4)]
    adj_sizes = np.array([len(a) for a in adj], dtype=np.int32)
    
    # Convert conflict matrix to list of sets for fast intersection checks?
    # Or just use the boolean matrix for vectorized operations.
    
    rng = np.random.default_rng(42)

    def randomized_greedy():
        """
        Randomized greedy: pick vertices in random order, add if no conflict with already picked.
        """
        in_set = np.zeros(N4, dtype=bool)
        order = rng.permutation(N4)
        
        for v in order:
            if not in_set[v]:
                # Check if v conflicts with any in_set
                # conflicts[v] is a boolean array
                if not np.any(conflicts[v] & in_set):
                    in_set[v] = True
        return in_set

    def local_search(in_set, max_time=5.0):
        """
        Local search with Add/Remove moves.
        """
        in_set = in_set.copy()
        current_size = in_set.sum()
        best_size = current_size
        best_in_set = in_set.copy()
        
        # conflict_count[i] = number of neighbors of i in in_set
        conflict_count = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[adj[i]] += 1
        
        ls_deadline = time.time() + max_time
        T = 1.0
        cooling = 0.995
        min_T = 0.05
        
        while time.time() < ls_deadline:
            # Try Add Move
            not_in = np.where(~in_set)[0]
            if len(not_in) > 0:
                # Prioritize nodes with low conflict count
                cc = conflict_count[not_in]
                # Sample a few candidates with lowest conflict counts
                k = min(10, len(not_in))
                # Get indices of k smallest conflict counts
                # argsort is O(N log N), for N=5280 it's fine
                idxs = np.argsort(cc)[:k]
                v = not_in[idxs[rng.integers(k)]]
                
                k_conf = conflict_count[v]
                delta = 1 - k_conf
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    # Execute Add
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            conflict_count[adj[u]] -= 1
                    
                    in_set[v] = True
                    current_size += delta
                    conflict_count[adj[v]] += 1
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
            
            # Try Remove Move (to escape local optima)
            in_set_indices = np.where(in_set)[0]
            if len(in_set_indices) > 0:
                u = in_set_indices[rng.integers(len(in_set_indices))]
                
                # Check if removing u allows adding a neighbor w that was previously blocked
                # This is complex. Simple remove:
                # Just remove u if it's "bad"? No, removing decreases size.
                # Only remove if we can immediately add back more?
                # Let's try: Remove u, then try to add a random neighbor of u that is not in set.
                # If we can add 1 neighbor, size stays same. If we can add 2, size increases.
                
                # For simplicity and speed, we will mostly rely on Add moves.
                # But sometimes we need to remove to allow a better Add.
                # Let's do a simple random removal with low probability.
                if rng.random() < 0.05:
                    in_set[u] = False
                    current_size -= 1
                    conflict_count[adj[u]] -= 1
                    # Try to add back u or a neighbor?
                    # Let's just let the next Add move handle it.
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_in_set, best_size

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    n_restarts = 0
    while time.time() < deadline - 5:
        n_restarts += 1
        
        # Initialize with randomized greedy
        in_set = randomized_greedy()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # Local Search
        remaining = deadline - 5 - time.time()
        if remaining < 2:
            break
        
        # Use a portion of remaining time for local search
        ls_time = min(remaining * 0.7, 10.0)
        ls_result, ls_size = local_search(in_set, max_time=ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
            
        # If we have time, do another pass on the best solution found so far
        remaining = deadline - 2 - time.time()
        if remaining > 5 and n_restarts % 5 == 0:
            polish_time = min(remaining * 0.8, 15.0)
            p_result, p_size = local_search(best_in_set, max_time=polish_time)
            if p_size > best_size:
                best_size = p_size
                best_in_set = p_result.copy()

    # Final: combine best v4 independent set with all v1 vectors
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    if len(result_v4) > 0:
        final_vectors = np.vstack([result_v4, vectors_v1])
    else:
        final_vectors = vectors_v1

    return final_vectors

# EVOLVE-BLOCK-END
