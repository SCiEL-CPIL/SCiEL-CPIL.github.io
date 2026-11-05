import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    N = 1000
    max_rep = 2
    MAX_SUM = 2 * N

    best_size = 0
    best_set = []

    def vectorized_greedy(seed):
        """
        Constructs a B2[2] set using vectorized NumPy operations.
        At each step, selects the candidate x that minimizes the number of sums 
        that would become saturated (count 1 -> 2), subject to validity.
        """
        rng = np.random.default_rng(seed)
        
        # counts[s] = number of representations a+b=s with a<=b in S
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        
        S = []
        in_set = np.zeros(N + 1, dtype=bool)
        
        # Candidates are 1..N
        # We will maintain a list of active candidates, or just check all each time?
        # Checking all 1..N each time is O(N) candidates.
        # Vectorized check for all N candidates is O(N*K).
        
        for _ in range(N):
            if time.time() > deadline:
                break
            
            # Get current set as array for vectorized ops
            if not S:
                # First element: any is valid. Pick random to break symmetry?
                # Or pick 1?
                # Let's pick a random one to allow different structures.
                x = rng.integers(1, N + 1)
                S.append(x)
                in_set[x] = True
                counts[2 * x] += 1
                continue
            
            S_arr = np.array(S, dtype=np.int32)
            K = len(S_arr)
            
            # Candidates: all 1..N not in S
            # We can just check all 1..N and mask out in_set
            all_candidates = np.arange(1, N + 1, dtype=np.int32)
            valid_mask = ~in_set[all_candidates]
            
            if not np.any(valid_mask):
                break
                
            # Filter candidates
            candidates = all_candidates[valid_mask]
            M = len(candidates)
            
            # Compute sums: candidates[:, None] + S_arr[None, :]
            # Shape: (M, K)
            sums = candidates[:, None] + S_arr[None, :]
            
            # Get counts for these sums
            # counts is 1D, so counts[sums] works
            c_sums = counts[sums]
            
            # Validity:
            # 1. No sum in S has count >= 2
            # 2. 2*x has count < 2
            valid_pair_mask = np.all(c_sums < max_rep, axis=1)
            
            # Check 2*x
            double_counts = counts[2 * candidates]
            valid_double_mask = double_counts < max_rep
            
            is_valid = valid_pair_mask & valid_double_mask
            
            if not np.any(is_valid):
                break
            
            # Cost: number of sums that are currently 1 (will become 2)
            # We only care about valid candidates
            costs = np.sum(c_sums == 1, axis=1)
            
            # Set invalid candidates to high cost
            costs[~is_valid] = 10000
            
            # Pick min cost
            min_idx = np.argmin(costs)
            x = candidates[min_idx]
            
            # Add x
            S.append(x)
            in_set[x] = True
            
            # Update counts
            # x + y for y in S (excluding x itself? No, x is new)
            # The sums are x + S_arr
            new_sums = x + S_arr
            counts[new_sums] += 1
            counts[2 * x] += 1
        
        return S

    # Run multiple restarts with different seeds
    # Each run is fast (~0.1-0.5s)
    num_restarts = 100
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        S = vectorized_greedy(i)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
        # Optional: Simple local search on the best found so far?
        # Given the speed, we can just do many restarts.
        
    return best_set
