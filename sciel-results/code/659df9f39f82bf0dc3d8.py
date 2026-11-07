import time
from math import gcd

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
LCM_ALL = 30030
ALL_MASK = (1 << N) - 1

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    
    # Pre-compute divisors for quick lookup?
    # Or just check divisibility.
    
    # We will iterate L from 1 to LCM_ALL.
    # For each L, we expand R until we find a valid window or R-L >= best_W.
    
    best_W = float('inf')
    best_L = -1
    best_assignment = None
    
    # Helper to check if a matching exists for a given window [L, R]
    # Returns the assignment (list of values) if found, else None.
    def find_matching(L, R):
        # Collect candidates: dict divisor_index -> list of values
        # Or better: list of (value, mask_of_divisors_it_satisfies)
        # Since we need distinct values, and values are unique in the window,
        # we can just collect all values in [L, R] that are divisible by any d_i.
        
        candidates = [] # list of (value, mask)
        for v in range(L, R + 1):
            mask = 0
            for i in range(N):
                if v % DIVISORS[i] == 0:
                    mask |= (1 << i)
            if mask != 0:
                candidates.append((v, mask))
        
        if not candidates:
            return None
            
        # If the union of all masks is not ALL_MASK, no solution.
        union_mask = 0
        for _, m in candidates:
            union_mask |= m
        if union_mask != ALL_MASK:
            return None
            
        # Now solve the matching problem.
        # We have a set of items (values), each with a mask.
        # We need to assign a distinct item to each bit in ALL_MASK.
        # This is equivalent to: Can we select a subset of items such that 
        # we can assign each selected item to a unique bit it covers, 
        # and all bits are covered?
        
        # Since N=10, we can use DP over subsets of divisors.
        # Let dp[mask] be the minimum number of items used to cover 'mask'?
        # No, we just need to know if it's possible.
        # Actually, we can use a recursive backtracking with memoization on (index_in_candidates, current_mask).
        # But candidates can be up to ~50-100 items.
        # 2^10 * 100 states = 100,000 states. This is fast.
        
        # However, a simpler approach:
        # Sort candidates by mask popcount (descending) or just use the list.
        # We try to assign values to divisors.
        
        # Let's use a DFS that assigns divisors one by one.
        # Order divisors by the number of available candidates (MRV).
        
        # For each divisor i, available candidates are those with bit i set.
        
        # To speed up, we can pre-filter candidates.
        # If a candidate only has 1 bit set, it MUST be used for that bit.
        
        # Let's stick to a simple DFS with pruning.
        # State: (div_index, used_values_mask) -- wait, used_values is a set of values.
        # Since values are up to 10000, we can't bitmask them easily.
        # But we only care about the candidates in the current window.
        # We can map each candidate value to an index 0..M-1.
        
        M = len(candidates)
        # Map value to index
        val_to_idx = {v: i for i, (v, _) in enumerate(candidates)}
        
        # For each divisor i, list of candidate indices that cover it.
        div_to_cands = [[] for _ in range(N)]
        for idx, (v, m) in enumerate(candidates):
            for i in range(N):
                if (m >> i) & 1:
                    div_to_cands[i].append(idx)
        
        # Sort divisors by number of candidates (MRV)
        div_indices = list(range(N))
        div_indices.sort(key=lambda i: len(div_to_cands[i]))
        
        used_cands = [False] * M
        assignment = [0] * N # assignment[i] = value assigned to divisor i
        
        def dfs(idx):
            if idx == N:
                return True
            i = div_indices[idx]
            for c_idx in div_to_cands[i]:
                if not used_cands[c_idx]:
                    used_cands[c_idx] = True
                    assignment[i] = candidates[c_idx][0]
                    if dfs(idx + 1):
                        return True
                    used_cands[c_idx] = False
            return False
            
        if dfs(0):
            return assignment
        return None

    # Main Loop
    for L in range(1, LCM_ALL + 1):
        if time.time() - start_time > 85.0:
            break
            
        # We only need to check R up to L + best_W - 1
        # If best_W is inf, we need a cap. Let's cap at 500 initially, or just expand.
        # Given the density, W should be small.
        max_R = L + (best_W if best_W < float('inf') else 500)
        
        # We can optimize by only checking R where a new candidate is added.
        # But the window is small, so iterating R is fine.
        
        for R in range(L, max_R + 1):
            if R - L >= best_W:
                break
                
            # Check if matching exists in [L, R]
            res = find_matching(L, R)
            if res is not None:
                best_W = R - L
                best_L = L
                best_assignment = res
                break # Found min W for this L, move to next L
                
    if best_assignment is not None:
        return [int(x) for x in best_assignment]
        
    # Fallback
    return [int(d * (i + 1) * 100) for i, d in enumerate(DIVISORS)]
