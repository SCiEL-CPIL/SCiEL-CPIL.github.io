"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    n = 12
    period = 27720  # lcm(1,2,...,12)
    max_L = 144
    
    # Precompute div_mask[x-1] for x in 1..period
    # div_mask[x-1] = bitmask where bit (i-1) is set if i divides x
    div_mask = [0] * period
    for i in range(1, n + 1):
        for x in range(i, period + 1, i):
            div_mask[x - 1] |= (1 << (i - 1))
    
    def solve_for_r(r):
        """Find f(12, r) = smallest L such that matching exists."""
        # Precompute the mask sequence for this r
        mask_seq = [0] * max_L
        for k in range(max_L):
            idx = (r + k) % period
            mask_seq[k] = div_mask[idx]
        
        # For each i, precompute the list of positions k where mask_seq[k] has bit i
        full_lists = [[] for _ in range(n)]
        for k in range(max_L):
            m = mask_seq[k]
            for i in range(n):
                if m & (1 << i):
                    full_lists[i].append(k)
        
        # For each L, check matching using Hopcroft-Karp style BFS/DFS
        for L in range(1, max_L + 1):
            # Quick check: if any i has no candidates in [0, L), fail
            valid = True
            for i in range(n):
                if not full_lists[i] or full_lists[i][0] >= L:
                    valid = False
                    break
            if not valid:
                continue
            
            # Use DFS-based bipartite matching (Kuhn's algorithm)
            # matchR[k] = left node matched to position k, or -1
            matchR = [-1] * L
            
            # Order left nodes by scarcity (fewest candidates first)
            order = list(range(n))
            cand_counts = []
            for i in range(n):
                cnt = 0
                for k in full_lists[i]:
                    if k < L:
                        cnt += 1
                    else:
                        break
                cand_counts.append(cnt)
            order.sort(key=lambda i: cand_counts[i])
            
            # DFS: assign left nodes in order
            used = [False] * L
            
            def dfs(idx_in_order):
                if idx_in_order == n:
                    return True
                i = order[idx_in_order]
                for k in full_lists[i]:
                    if k >= L:
                        break
                    if not used[k]:
                        used[k] = True
                        if dfs(idx_in_order + 1):
                            matchR[k] = i
                            return True
                        used[k] = False
                return False
            
            if dfs(0):
                return L
        
        return max_L + 1  # Should not happen
    
    best_L = 0
    best_r = 0
    
    for r in range(period):
        if time.time() > deadline - 2:
            break
        
        L = solve_for_r(r)
        if L <= max_L and L > best_L:
            best_L = L
            best_r = r
            if best_L >= max_L:
                break
    
    # Find m in [0, 1000000] with m % period == best_r
    if best_r <= 1000000:
        m = best_r
    else:
        m = best_r % (1000000 // period + 1)
    
    return int(m)
# EVOLVE-BLOCK-END
