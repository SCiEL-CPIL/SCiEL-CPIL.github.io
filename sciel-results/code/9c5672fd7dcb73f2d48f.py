"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
import sys
import numpy as np

def _check_candidate(x, S, forbidden):
    """
    Check if x can be added to set S (sorted) without triple-sum collisions.
    forbidden is a boolean numpy array where forbidden[s] is True if sum s is already used.
    Returns True if valid, False otherwise.
    """
    # Check 3x
    if forbidden[3 * x]:
        return False
    
    # Check 2x+y and x+2y for all y in S
    if S:
        S_arr = np.array(S, dtype=np.int64)
        sums_2x_y = 2 * x + S_arr
        sums_x_2y = x + 2 * S_arr
        
        if np.any(forbidden[sums_2x_y]) or np.any(forbidden[sums_x_2y]):
            return False
        
        # Check internal collisions between 2x+y_i and x+2y_j
        # 2x + y_i == x + 2y_j => x + y_i == 2y_j
        # This means x == 2y_j - y_i for some i,j
        # We need to check if any such collision exists
        # Also check 2x+y_i == 2x+y_j => y_i == y_j (always distinct since S has distinct elements)
        # And x+2y_i == x+2y_j => y_i == y_j (same)
        
        # Check if 2x+y_i collides with x+2y_j for i != j
        # 2x + S[i] == x + 2*S[j] => x == 2*S[j] - S[i]
        # For each pair (i,j) with i != j, check if x == 2*S[j] - S[i]
        # This is equivalent to checking if x is in {2*S[j] - S[i] : i != j}
        
        # More efficient: for each j, check if 2*S[j] - x is in S and != S[j]
        # i.e., check if (2*S[j] - x) is in S and (2*S[j] - x) != S[j]
        # 2*S[j] - x != S[j] => S[j] != x, which is true since x not in S
        # So check if (2*S[j] - x) is in S for any j
        
        # Actually, let's just check all pairs for small S
        n = len(S)
        if n < 50:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if 2 * x + S[i] == x + 2 * S[j]:
                        return False
        else:
            # Use a set for S for O(1) lookup
            S_set = set(S)
            for j in range(n):
                val = 2 * S[j] - x
                if val in S_set and val != S[j]:
                    return False
    
    # Check x+y+z for y<=z in S
    if len(S) >= 2:
        S_arr = np.array(S, dtype=np.int64)
        n = len(S_arr)
        i_idx, j_idx = np.triu_indices(n, k=0)
        pair_sums = S_arr[i_idx] + S_arr[j_idx]
        triple_sums = x + pair_sums
        
        if np.any(forbidden[triple_sums]):
            return False
        
        # Check internal collisions among triple_sums
        # x + S[i] + S[j] == x + S[k] + S[l] => S[i]+S[j] == S[k]+S[l]
        # This can happen even with distinct i,j,k,l
        # Check if any pair sums are equal
        if len(pair_sums) > 1:
            if len(set(pair_sums.tolist())) < len(pair_sums):
                return False
        
        # Check collision between triple_sums and 2x+y / x+2y
        if S:
            S_arr2 = np.array(S, dtype=np.int64)
            sums_2x_y = 2 * x + S_arr2
            sums_x_2y = x + 2 * S_arr2
            # Check if any triple_sum equals any 2x+y or x+2y
            # x + S[i] + S[j] == 2x + S[k] => S[i] + S[j] == x + S[k]
            # x + S[i] + S[j] == x + 2*S[k] => S[i] + S[j] == 2*S[k]
            
            # Check x+S[i]+S[j] == 2x+S[k] => S[i]+S[j]-S[k] == x
            # Check x+S[i]+S[j] == x+2*S[k] => S[i]+S[j]-2*S[k] == 0 => S[i]+S[j] == 2*S[k]
            
            # For efficiency, check the second condition first (sum of two equals double of another)
            S_set = set(S)
            for i in range(n):
                for j in range(i, n):
                    pair_sum = S[i] + S[j]
                    # Check if pair_sum == 2*S[k] for some k
                    half = pair_sum / 2
                    if half == int(half):
                        half = int(half)
                        if half in S_set:
                            return False
                    # Check if pair_sum - S[k] == x for some k
                    for k in range(n):
                        if pair_sum - S[k] == x:
                            return False
    
    return True

def _greedy_increasing(deadline):
    """Simple greedy in increasing order with vectorized checks."""
    S = []
    max_sum = 3 * 4096
    forbidden = np.zeros(max_sum + 1, dtype=bool)
    
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        
        if _check_candidate(x, S, forbidden):
            S.append(x)
            # Update forbidden
            forbidden[3 * x] = True
            for y in S[:-1]:
                forbidden[2 * x + y] = True
                forbidden[x + 2 * y] = True
            for i in range(len(S) - 1):
                for j in range(i + 1, len(S) - 1):
                    forbidden[x + S[i] + S[j]] = True
    
    return S

def _greedy_randomized(deadline, seed=0):
    """Greedy with randomized candidate ordering."""
    rng = random.Random(seed)
    S = []
    max_sum = 3 * 4096
    forbidden = np.zeros(max_sum + 1, dtype=bool)
    
    candidates = list(range(1, 4097))
    rng.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        if _check_candidate(x, S, forbidden):
            S.append(x)
            forbidden[3 * x] = True
            for y in S[:-1]:
                forbidden[2 * x + y] = True
                forbidden[x + 2 * y] = True
            for i in range(len(S) - 1):
                for j in range(i + 1, len(S) - 1):
                    forbidden[x + S[i] + S[j]] = True
    
    return sorted(S)

def _greedy_with_backtracking(deadline, seed=0):
    """
    Greedy with limited backtracking: when stuck, try removing a recent element
    and exploring alternatives.
    """
    rng = random.Random(seed)
    S = []
    max_sum = 3 * 4096
    forbidden = np.zeros(max_sum + 1, dtype=bool)
    best = []
    best_size = 0
    
    def backtrack(depth, max_depth=3):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        
        if len(S) > best_size:
            best = S[:]
            best_size = len(S)
        
        if depth > max_depth:
            return
        
        # Try to add next candidate
        found = False
        for x in range(1, 4097):
            if x in S:
                continue
            if _check_candidate(x, S, forbidden):
                S.append(x)
                forbidden[3 * x] = True
                for y in S[:-1]:
                    forbidden[2 * x + y] = True
                    forbidden[x + 2 * y] = True
                for i in range(len(S) - 1):
                    for j in range(i + 1, len(S) - 1):
                        forbidden[x + S[i] + S[j]] = True
                found = True
                break
        
        if not found:
            # Try backtracking: remove last element and try different
            if S and depth < max_depth:
                removed = S.pop()
                # Remove its sums from forbidden
                for y in S:
                    forbidden[2 * removed + y] = False
                    forbidden[removed + 2 * y] = False
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        forbidden[removed + S[i] + S[j]] = False
                forbidden[3 * removed] = False
                
                # Try next alternative
                backtrack(depth + 1, max_depth)
    
    backtrack(0)
    return best

def _local_search(S, deadline, seed=0):
    """
    Local search: try to improve the set by removing one element and adding another.
    """
    rng = random.Random(seed)
    S = sorted(S)
    best = S[:]
    best_size = len(S)
    
    max_sum = 3 * 4096
    
    def build_forbidden(arr):
        forbidden = np.zeros(max_sum + 1, dtype=bool)
        for i in range(len(arr)):
            for j in range(i, len(arr)):
                for k in range(j, len(arr)):
                    s = arr[i] + arr[j] + arr[k]
                    forbidden[s] = True
        return forbidden
    
    forbidden = build_forbidden(S)
    
    while time.time() < deadline:
        improved = False
        
        # Try removing each element and finding a replacement
        for idx in range(len(S)):
            if time.time() > deadline:
                break
            
            removed = S[idx]
            S_new = S[:idx] + S[idx+1:]
            
            # Rebuild forbidden without the removed element
            forbidden_new = build_forbidden(S_new)
            
            # Try to add a better element
            candidates = [x for x in range(1, 4097) if x not in S_new]
            rng.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    break
                
                if _check_candidate(x, S_new, forbidden_new):
                    S_final = sorted(S_new + [x])
                    if len(S_final) > best_size:
                        best = S_final
                        best_size = len(S_final)
                        S = S_final
                        forbidden = build_forbidden(S)
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best = []
    
    # Strategy 1: Simple greedy in increasing order
    S1 = _greedy_increasing(deadline)
    if len(S1) > len(best):
        best = S1[:]
    
    # Strategy 2: Randomized greedy with multiple seeds
    for seed in [42, 123, 456, 789]:
        if time.time() > deadline:
            break
        S2 = _greedy_randomized(deadline, seed=seed)
        if len(S2) > len(best):
            best = S2[:]
    
    # Strategy 3: Backtracking
    if time.time() < deadline:
        S3 = _greedy_with_backtracking(deadline, seed=0)
        if len(S3) > len(best):
            best = S3[:]
    
    # Strategy 4: Local search from best
    if time.time() < deadline and best:
        S4 = _local_search(best, deadline, seed=123)
        if len(S4) > len(best):
            best = S4[:]
    
    return best
