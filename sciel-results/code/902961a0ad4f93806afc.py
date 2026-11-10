"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    max_m = 1000000
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = 1
    for i in range(1, 13):
        period = lcm(period, i)
    # period = 27720
    
    def _can_match_fast(m, L):
        """
        Fast check if there exist distinct b_1,...,b_12 in (m, m+L] with i | b_i.
        Uses precomputed availability and greedy matching with backtracking.
        """
        if L < 12:
            return False
        
        # Precompute for each divisor i (1..12), which positions in [0, L-1] are available
        # pos j corresponds to value m+1+j
        available = [[] for _ in range(n)]
        for i in range(1, n + 1):
            # Find all positions j where (m+1+j) % i == 0
            # (m+1+j) % i == 0  =>  j % i == (-m-1) % i
            target = (-m - 1) % i
            for j in range(target, L, i):
                available[i-1].append(j)
        
        # Check if any index has no available positions
        for i in range(n):
            if len(available[i]) == 0:
                return False
        
        # Greedy matching with backtracking (DFS)
        match_pos = [-1] * n  # match_pos[i] = position assigned to index i, or -1
        match_idx = [-1] * L  # match_idx[pos] = index assigned to position pos, or -1
        
        def dfs(idx, visited):
            # Try to assign a position to index idx
            for pos in available[idx]:
                if visited[pos]:
                    continue
                visited[pos] = True
                if match_idx[pos] == -1 or dfs(match_idx[pos], visited):
                    match_idx[pos] = idx
                    match_pos[idx] = pos
                    return True
            return False
        
        # Process indices in order of scarcity (fewest available positions first)
        order = sorted(range(n), key=lambda i: len(available[i]))
        
        for idx in order:
            visited = [False] * L
            if not dfs(idx, visited):
                return False
        
        return True
    
    def _f_of_m(m):
        """Compute f(12, m): the least positive L such that matching exists."""
        # Binary search for the smallest L where matching exists
        lo, hi = 1, 144
        while lo < hi:
            mid = (lo + hi) // 2
            if _can_match_fast(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    def find_best_m():
        """
        Find m in [0, period) that maximizes f(12, m).
        Strategy: 
        1. Sample candidate m values
        2. For each, compute f(12, m)
        3. Return the m with maximum f
        """
        best_m = 0
        best_f = 0
        
        # Phase 1: Coarse grid search
        step = max(1, period // 2000)
        for m in range(0, period, step):
            if time.time() > deadline:
                break
            f_val = _f_of_m(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
        
        # Phase 2: Local search around best found
        if time.time() < deadline - 5:
            search_radius = max(1, step * 2)
            for dm in range(-search_radius, search_radius + 1):
                m_candidate = (best_m + dm) % period
                if m_candidate == best_m:
                    continue
                if time.time() > deadline:
                    break
                f_val = _f_of_m(m_candidate)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m_candidate
        
        # Phase 3: Random sampling
        if time.time() < deadline - 2:
            attempts = 0
            while time.time() < deadline - 2 and attempts < 500:
                m_candidate = random.randint(0, period - 1)
                attempts += 1
                f_val = _f_of_m(m_candidate)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m_candidate
        
        return best_m, best_f
    
    result = find_best_m()
    m_best, f_best = result
    
    # Verify m_best is in range
    if m_best < 0 or m_best > max_m:
        m_best = m_best % (max_m + 1)
        
    return int(m_best)

# EVOLVE-BLOCK-END
