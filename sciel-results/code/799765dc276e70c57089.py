"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    max_m = 1000000
    
    def can_assign(m, L):
        """Check if there's a perfect matching: assign distinct b_i in (m, m+L] with i | b_i."""
        # Build adjacency: for each index i (1..12), which positions (values m+1..m+L) are divisible by i
        adj = [[] for _ in range(n)]
        for i in range(n):
            idx = i + 1  # actual divisor value
            # First multiple of idx in (m, m+L]
            first = ((m + 1 + idx - 1) // idx) * idx
            if first > m + L:
                return False  # No candidate for this index
            for v in range(first, m + L + 1, idx):
                pos = v - (m + 1)
                if 0 <= pos < L:
                    adj[i].append(pos)
        
        # Hall's theorem pre-check: for every subset of indices, |N(S)| >= |S|
        # For small n=12, check all subsets is 2^12=4096, which is feasible
        # But we can do a lighter check: for each index, it must have at least 1 neighbor
        for i in range(n):
            if len(adj[i]) == 0:
                return False
        
        # Use bitmask DP for maximum bipartite matching
        # dp[mask] = max number of matched indices considering only positions in mask
        # Actually, let's use a different approach: 
        # For each position, which indices can use it
        # Then do DFS matching
        
        # Build reverse adjacency: for each position, which indices can use it
        rev_adj = [[] for _ in range(L)]
        for i in range(n):
            for pos in adj[i]:
                rev_adj[pos].append(i)
        
        # Try to find perfect matching using backtracking with pruning
        # Assign indices in order of fewest options first (MRV heuristic)
        order = sorted(range(n), key=lambda i: len(adj[i]))
        
        # Use memoization on (index_in_order, used_positions_mask)
        # But L can be up to 144, so mask is too large for memo
        # Instead, use DFS with backtracking
        
        used = [False] * L
        assignment = [-1] * n  # assignment[i] = position assigned to index i
        
        def dfs(idx_in_order):
            if idx_in_order == n:
                return True
            i = order[idx_in_order]
            for pos in adj[i]:
                if not used[pos]:
                    used[pos] = True
                    assignment[i] = pos
                    if dfs(idx_in_order + 1):
                        return True
                    used[pos] = False
            return False
        
        return dfs(0)
    
    def f_of_m(m):
        """Find minimum L such that can_assign(m, L) is True."""
        # L must be at least 12 (need 12 distinct values)
        # Upper bound: at most 144 per problem statement
        lo, hi = 12, 144
        if not can_assign(m, hi):
            return -1  # shouldn't happen for valid m
        while lo < hi:
            mid = (lo + hi) // 2
            if can_assign(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Strategy: search for m values that maximize f(12,m)
    # Key insight: f(12,m) is large when the interval (m, m+L] is "hard" to match
    # This happens when many indices have few multiples in the interval
    # Try to find m where the pattern of multiples is unfavorable
    
    best_m = 0
    best_f = -1
    
    # Phase 1: Coarse search over structured candidates
    # Try m values that are multiples of LCM(1..12) or related patterns
    # LCM(1..12) = 27720
    lcm_val = 27720
    
    candidates = set()
    
    # Try m near multiples of lcm and offsets
    for k in range(0, max_m // lcm_val + 1):
        base = k * lcm_val
        for offset in range(-50, 51):
            m_val = base + offset
            if 0 <= m_val <= max_m:
                candidates.add(m_val)
    
    # Also try random samples
    random.seed(42)
    for _ in range(3000):
        candidates.add(random.randint(0, max_m))
    
    # Try some small m values
    for m_val in range(0, 200):
        candidates.add(m_val)
    
    # Sort candidates and evaluate
    candidate_list = sorted(candidates)
    
    for m in candidate_list:
        if time.time() - start_time > deadline:
            break
        f_val = f_of_m(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # Phase 2: Local search around best_m
    if time.time() - start_time < deadline - 10:
        # Try neighbors of best_m
        for delta in range(-100, 101):
            if time.time() - start_time > deadline:
                break
            m_val = best_m + delta
            if 0 <= m_val <= max_m:
                f_val = f_of_m(m_val)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m_val
    
    return int(best_m)
# EVOLVE-BLOCK-END
