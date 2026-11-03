import math
import time
import random
import numpy as np

def score(U):
    """Compute the score for a finite set U containing 0."""
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    n_sums = len(sums)
    n_diffs = len(diffs)
    
    if n_sums == 0:
        return 0
        
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
        
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def _fast_score_fft(U):
    """
    Compute score using FFT for speed.
    U: list of non-negative integers.
    """
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Length of convolution needs to be at least 2*max_u + 1
    # We use a power of 2 for FFT efficiency
    L = 1
    while L < 2 * max_u + 1:
        L *= 2
        
    # Indicator vector for U
    A = np.zeros(L, dtype=np.float64)
    for u in U:
        A[u] = 1.0
        
    # Sumset: Convolution of A with A
    # The result at index k is the number of pairs (a,b) such that a+b=k
    conv_sum = np.fft.irfft(np.fft.rfft(A) * np.fft.rfft(A))
    # Only consider indices 0 to 2*max_u
    n_sums = np.sum(conv_sum[:2 * max_u + 1] > 0.5) # Threshold to handle floating point errors
    
    # Difference set:
    # We want counts of a - b.
    # Let B be A reversed. B[i] = A[L-1-i].
    # Convolution A * B at index k corresponds to sum a + b' where b' = L-1-b.
    # So a + (L-1-b) = a - b + L - 1.
    # So index k in conv_diff corresponds to difference d = k - (L - 1).
    # We want d in range [-max_u, max_u].
    # So k in [L-1-max_u, L-1+max_u].
    
    B = A[::-1]
    conv_diff = np.fft.irfft(np.fft.rfft(A) * np.fft.rfft(B))
    
    # The peak is at L-1 (difference 0).
    # We count non-zero entries in the relevant range.
    # Range of indices: L-1-max_u to L-1+max_u
    start_idx = L - 1 - max_u
    end_idx = L - 1 + max_u + 1
    n_diffs = np.sum(conv_diff[start_idx:end_idx] > 0.5)
    
    if n_sums == 0:
        return 0.0
        
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
        
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def _local_search_fft(U, max_iters=500, deadline=None, temp=1.0):
    """
    Local search using FFT scoring.
    Allows adding, removing, and moving elements.
    """
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
    best_score = _fast_score_fft(U)
    best_U = U[:]
    
    # Estimate a reasonable upper bound for new values
    # Start with max_u + 10, expand if needed
    current_max = max(U)
    
    for i in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        # Generate neighbors
        neighbors = []
        
        # 1. Move an element
        for idx, val in enumerate(U):
            if val == 0: continue
            # Try moving to nearby values
            for delta in [-2, -1, 1, 2, 3, 5, 10]:
                new_val = val + delta
                if new_val > 0 and new_val != val:
                    new_U = U[:]
                    new_U[idx] = new_val
                    if len(set(new_U)) == len(new_U):
                        neighbors.append(new_U)
                        
        # 2. Add an element
        # Try adding near existing elements or at the boundary
        for val in U:
            for delta in [1, 2, 3, 5, 10]:
                new_val = val + delta
                if new_val not in U:
                    new_U = U[:] + [new_val]
                    neighbors.append(new_U)
                new_val = val - delta
                if new_val > 0 and new_val not in U:
                    new_U = U[:] + [new_val]
                    neighbors.append(new_U)
                    
        # 3. Remove an element (if size > 2)
        if len(U) > 2:
            for idx in range(1, len(U)):
                new_U = U[:idx] + U[idx+1:]
                neighbors.append(new_U)

        if not neighbors:
            break

        # Shuffle neighbors to avoid bias
        random.shuffle(neighbors)
        
        best_neighbor_score = best_score
        best_neighbor = best_U
        
        for nU in neighbors:
            s = _fast_score_fft(nU)
            if s > best_neighbor_score:
                best_neighbor_score = s
                best_neighbor = nU
        
        if best_neighbor_score > best_score + 1e-9:
            best_score = best_neighbor_score
            best_U = best_neighbor
            improved = True
            current_max = max(best_U)
        else:
            # If stuck, maybe expand the search space slightly?
            # Or break if we've been stuck for a while.
            # For simplicity, just break or do a random jump.
            if random.random() < 0.1:
                 # Random jump: add a random element
                 new_val = random.randint(1, current_max + 20)
                 if new_val not in best_U:
                     best_U = best_U + [new_val]
                     best_score = _fast_score_fft(best_U)
            else:
                 break

    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # 1. Try some heuristic initializations
    initial_sets = []
    
    # Small exhaustive sets (fast to check with FFT too, but keep small)
    # We'll just use a few hardcoded good small sets or generate them
    # [0, 1, 3] is a Golomb ruler
    initial_sets.append([0, 1, 3])
    initial_sets.append([0, 1, 4, 10]) # Another Golomb ruler
    initial_sets.append([0, 1, 2, 5, 10, 12]) # Larger Golomb
    
    # Symmetric core sets
    # U = {0, M} U {x, M-x}
    for M in [20, 50, 100, 200]:
        # Pick random subset of {1, ..., M/2}
        S = random.sample(range(1, M//2 + 1), min(10, M//2))
        U = [0, M] + [x for x in S] + [M-x for x in S]
        U = sorted(set(U))
        initial_sets.append(U)
        
    # Random sets with varying max
    for max_val in [50, 100, 200, 500]:
        n = random.randint(10, 30)
        # Ensure 0 is included
        S = random.sample(range(1, max_val + 1), n - 1)
        U = [0] + S
        initial_sets.append(U)
        
    # 2. Local search on each initial set
    for U_init in initial_sets:
        if time.time() > deadline:
            break
        U_opt, s_opt = _local_search_fft(U_init, max_iters=100, deadline=deadline)
        if s_opt > best_score:
            best_score = s_opt
            best_set = U_opt
            
    # 3. Final polish with a tighter local search on the best found
    if time.time() < deadline:
         best_set, best_score = _local_search_fft(best_set, max_iters=200, deadline=deadline)
         
    return sorted(set(best_set))
