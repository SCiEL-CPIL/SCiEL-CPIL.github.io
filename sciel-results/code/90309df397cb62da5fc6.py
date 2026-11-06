import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 4096
    MAX_SUM = 3 * N  # 12288
    
    def is_valid_set(S):
        """Check if all unordered triple sums of S are unique."""
        k = len(S)
        if k == 0: 
            return True
        S_arr = np.array(S, dtype=np.int32)
        # Generate indices for i <= j <= l
        i_idx, j_idx, l_idx = np.indices((k, k, k))
        mask = (i_idx <= j_idx) & (j_idx <= l_idx)
        sums = S_arr[i_idx[mask]] + S_arr[j_idx[mask]] + S_arr[l_idx[mask]]
        # Check for duplicates
        if len(np.unique(sums)) != len(sums):
            return False
        return True

    def run_greedy(order):
        """Fast greedy using NumPy vectorized collision checking."""
        S = []
        S_arr = np.array([], dtype=np.int32)
        used_sums = np.zeros(MAX_SUM + 1, dtype=bool)
        
        for x in order:
            if time.time() > deadline:
                break
            x = int(x)
            
            if not S:
                S.append(x)
                S_arr = np.array([x], dtype=np.int32)
                used_sums[3 * x] = True
                continue
            
            k = len(S)
            
            # Generate all new triple sums involving x
            # Type 1: x + x + x
            s1 = 3 * x
            
            # Type 2: x + x + s_i for s_i in S
            s2 = 2 * x + S_arr
            
            # Type 3: x + s_i + s_j for i <= j
            i_idx, j_idx = np.triu_indices(k)
            s3 = x + S_arr[i_idx] + S_arr[j_idx]
            
            all_new_sums = np.concatenate(([s1], s2, s3))
            
            # Check for duplicates within all_new_sums
            # If there are duplicates, adding x would create internal conflicts
            if len(np.unique(all_new_sums)) != len(all_new_sums):
                continue
            
            # Check against used_sums
            # Filter to valid range
            valid_mask = (all_new_sums >= 0) & (all_new_sums <= MAX_SUM)
            if not np.all(valid_mask):
                continue
            if np.any(used_sums[all_new_sums]):
                continue
            
            # Valid: add x
            S.append(x)
            S_arr = np.append(S_arr, x)
            used_sums[all_new_sums] = True
        
        return S
    
    def local_improvement(S, remaining_time):
        """Try to add more elements or swap to improve set S."""
        if not S:
            return S
        
        S_list = list(S)
        S_set = set(S_list)
        
        # Limit number of iterations to avoid infinite loops
        max_iterations = 500
        
        for _ in range(max_iterations):
            if time.time() > start_time + remaining_time:
                break
            
            improved = False
            # Scan candidates in decreasing order (larger numbers are better)
            for x in range(N, 0, -1):
                if time.time() > start_time + remaining_time:
                    break
                
                if x in S_set:
                    continue
                
                # 1. Try to add x
                if is_valid_set(S_list + [x]):
                    S_list.append(x)
                    S_set.add(x)
                    improved = True
                    break # Restart outer loop with new set
                
                # 2. Try to swap x with y (where y < x)
                # This "upgrades" the set to larger numbers, which may allow more additions later
                for y in S_list:
                    if y >= x:
                        continue
                    
                    new_S_list = [z for z in S_list if z != y]
                    new_S_list.append(x)
                    
                    if is_valid_set(new_S_list):
                        S_list = new_S_list
                        S_set = set(S_list)
                        improved = True
                        break # Restart outer loop
                if improved:
                    break
            
            if not improved:
                break
                
        return S_list
    
    # Strategy 1: Greedy with natural increasing order
    S_inc = run_greedy(range(1, N + 1))
    best_set = S_inc
    best_len = len(best_set)
    
    # Strategy 2: Greedy with decreasing order
    S_dec = run_greedy(range(N, 0, -1))
    if len(S_dec) > best_len:
        best_set = S_dec
        best_len = len(S_dec)
    
    # Strategy 3: Randomized greedy with multiple restarts
    random.seed(42)
    
    while time.time() < deadline - 10.0:
        # Different random strategies
        r = random.random()
        if r < 0.4:
            # Full random shuffle
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif r < 0.7:
            # Biased towards larger numbers (they have fewer collisions)
            weights = np.arange(1, N + 1, dtype=np.float64)
            weights = weights ** 2  # Quadratic bias towards larger
            order = np.random.choice(N, size=N, replace=False, p=weights / weights.sum()) + 1
        elif r < 0.85:
            # Mixed: half random, half increasing
            first_half = list(range(1, N // 2 + 1))
            second_half = list(range(N // 2 + 1, N + 1))
            random.shuffle(first_half)
            random.shuffle(second_half)
            order = first_half + second_half
        else:
            # Reverse random
            order = list(range(1, N + 1))
            random.shuffle(order)
            order = order[::-1]
        
        S_try = run_greedy(order)
        
        if len(S_try) > best_len:
            best_set = S_try
            best_len = len(S_try)
    
    # Strategy 4: Local improvement on best set
    remaining = deadline - time.time()
    if remaining > 5.0:
        S_improved = local_improvement(best_set, remaining - 3.0)
        if len(S_improved) > best_len:
            best_set = S_improved
            best_len = len(S_improved)
    
    # Strategy 5: Try a few more randomized passes if time permits
    remaining = deadline - time.time()
    while remaining > 5.0:
        order = list(range(1, N + 1))
        random.shuffle(order)
        S_try = run_greedy(order)
        if len(S_try) > best_len:
            best_set = S_try
            best_len = len(S_try)
        
        # Also try local improvement on this result
        if len(S_try) >= best_len - 2:
            S_improved = local_improvement(S_try, deadline - time.time() - 3.0)
            if len(S_improved) > best_len:
                best_set = S_improved
                best_len = len(S_improved)
        
        remaining = deadline - time.time()
    
    return best_set
