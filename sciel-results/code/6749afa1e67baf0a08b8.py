"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0
    
    N = 4096
    MAX_SIZE = 80
    
    def triple_sums_with_x(S_arr, x):
        """Compute all unordered triple sums involving x, given sorted S_arr."""
        n = len(S_arr)
        if n == 0:
            return np.array([3 * x], dtype=np.int64)
        
        # (x, x, x)
        s1 = np.array([3 * x], dtype=np.int64)
        
        # (x, x, y) for y in S
        s2 = 2 * x + S_arr
        
        # (x, y, y) for y in S
        s3 = x + 2 * S_arr
        
        # (x, y, z) for y < z in S
        if n >= 2:
            # Build all pairs y < z
            i_idx = np.repeat(np.arange(n), np.arange(n - 1, 0, -1))
            j_idx = np.tile(np.arange(1, n), np.arange(n - 1, 0, -1))
            s4 = x + S_arr[i_idx] + S_arr[j_idx]
        else:
            s4 = np.array([], dtype=np.int64)
        
        return np.concatenate([s1, s2, s3, s4])
    
    def can_add(S_list, x, sums_set):
        """Check if x can be added to S_list without conflict."""
        S_arr = np.array(S_list, dtype=np.int64)
        new_sums = triple_sums_with_x(S_arr, x)
        
        # Check internal uniqueness
        if len(np.unique(new_sums)) != len(new_sums):
            return False
        
        # Check against existing sums
        new_set = set(new_sums.tolist())
        if not new_set.isdisjoint(sums_set):
            return False
        
        return True
    
    # Phase 1: Greedy construction
    S = []
    sums_set = set()
    
    for x in range(1, N + 1):
        if time.time() > DEADLINE - 5.0:
            break
        if len(S) >= MAX_SIZE:
            break
        
        if can_add(S, x, sums_set):
            S.append(x)
            S_arr = np.array(S, dtype=np.int64)
            new_sums = triple_sums_with_x(S_arr[:-1], x)
            sums_set.update(new_sums.tolist())
    
    # Phase 2: Local search polish - try to swap elements to potentially fit more
    if len(S) < MAX_SIZE and time.time() < DEADLINE - 3.0:
        S_sorted = sorted(S)
        changed = True
        while changed and time.time() < DEADLINE - 3.0:
            changed = False
            for i in range(len(S_sorted)):
                if time.time() > DEADLINE - 3.0:
                    break
                old = S_sorted[i]
                # Try replacing old with a larger candidate
                for new_val in range(old + 1, N + 1):
                    if time.time() > DEADLINE - 3.0:
                        break
                    S_test = S_sorted[:i] + S_sorted[i+1:]
                    S_test = sorted(S_test + [new_val])
                    
                    # Check if S_test is valid
                    valid = True
                    all_sums = set()
                    for k, a in enumerate(S_test):
                        S_sub = np.array(S_test[:k] + S_test[k+1:], dtype=np.int64)
                        sums_a = triple_sums_with_x(S_sub, a)
                        if len(np.unique(sums_a)) != len(sums_a):
                            valid = False
                            break
                        if not all_sums.isdisjoint(set(sums_a.tolist())):
                            valid = False
                            break
                        all_sums.update(sums_a.tolist())
                    
                    if valid:
                        S_sorted = S_test
                        changed = True
                        break
    
    return sorted(S_sorted if len(S_sorted) >= len(S) else S)
# EVOLVE-BLOCK-END
