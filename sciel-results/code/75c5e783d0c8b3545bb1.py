# EVOLVE-BLOCK-START
import sys
import time

def construct():
    n = 192
    best_S = []
    best_size = 0
    
    # Precompute divisors for numbers up to 3*192 = 576
    max_sum = 3 * n
    divisors = [[] for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(i, max_sum + 1, i):
            divisors[j].append(i)

    def is_valid_addition(a, S, S_set):
        """
        Check if adding a to S maintains the constraint.
        S is a list, S_set is a set of S.
        """
        # 1. Check constraint for a:
        if a == 1:
            if len(S) >= 3:
                return False
        else:
            len_S = len(S)
            for i in range(len_S):
                x = S[i]
                for j in range(i + 1, len_S):
                    y = S[j]
                    target_res = (-x - y) % a
                    # Check if any z in S, z != x, z != y, has z % a == target_res
                    for k in range(len_S):
                        z = S[k]
                        if z == x or z == y:
                            continue
                        if z % a == target_res:
                            return False
                            
        # 2. Check constraint for existing elements b in S:
        len_S = len(S)
        for i in range(len_S):
            x = S[i]
            for j in range(i + 1, len_S):
                y = S[j]
                sum_val = a + x + y
                # Check divisors of sum_val
                for b in divisors[sum_val]:
                    if b == x or b == y:
                        continue
                    if b in S_set:
                        return False
                        
        return True

    # Greedy initialization
    S_greedy = []
    S_greedy_set = set()
    for a in range(n, 0, -1):
        if is_valid_addition(a, S_greedy, S_greedy_set):
            S_greedy.append(a)
            S_greedy_set.add(a)
    best_S = S_greedy[:]
    best_size = len(S_greedy)
    
    # DFS
    # Order: 1 to n. Try Skip first.
    candidates = list(range(1, n + 1))
    
    sys.setrecursionlimit(10000)
    start_time = time.time()
    
    def dfs(idx, current_S, current_S_set, current_size):
        nonlocal best_S, best_size
        
        if time.time() - start_time > 55:
            return False
            
        # Pruning
        if current_size + (n - idx) <= best_size:
            return True
            
        if idx == n:
            if current_size > best_size:
                best_S = current_S[:]
                best_size = current_size
            return True
        
        a = candidates[idx]
        
        # Try Skip first (for small numbers, this is likely better to keep set clean)
        if not dfs(idx + 1, current_S, current_S_set, current_size):
            return False
        
        # Try Include
        if is_valid_addition(a, current_S, current_S_set):
            current_S.append(a)
            current_S_set.add(a)
            if not dfs(idx + 1, current_S, current_S_set, current_size + 1):
                current_S.pop()
                current_S_set.remove(a)
                return False
            current_S.pop()
            current_S_set.remove(a)
        
        return True

    dfs(0, [], set(), 0)
    
    return best_S
# EVOLVE-BLOCK-END
