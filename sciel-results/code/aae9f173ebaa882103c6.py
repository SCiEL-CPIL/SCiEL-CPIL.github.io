"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    max_m = 1000000
    LCM_12 = 27720  # LCM(1,2,...,12)
    
    def can_assign(m, L):
        """Check if there's a perfect matching using bitmask DP."""
        # For each index i (1..12), find which positions in (m, m+L] are divisible by i
        # Position pos (0-indexed) corresponds to value m+1+pos
        # Build for each position, which indices can use it
        # Then use bitmask DP: dp[mask] = True if we can assign indices in mask to some subset of positions
        
        # Actually, better approach: for each index i, compute the set of valid positions
        # Then use inclusion-exclusion or bitmask DP
        
        # For n=12, use DP over subsets of indices
        # dp[S] = set of positions used? No, too large.
        # Better: dp[S] = True if indices in S can be matched to distinct valid positions
        # This is still complex. Let's use the standard bipartite matching with DFS but optimized.
        
        # Alternative: since L <= 144 and n=12, use a simple approach:
        # For each position j in 0..L-1, compute which indices i can use it
        # Then do matching: assign positions to indices
        
        # Build: for each position pos (0..L-1), which indices can use it
        # pos corresponds to value v = m + 1 + pos
        # Index i (1..12) can use pos if v % i == 0
        
        # Let's build adj: adj[i] = list of positions that index i can use
        adj = [[] for _ in range(n)]
        for i in range(n):
            idx = i + 1
            # Find first multiple of idx in (m, m+L]
            first = ((m + idx) // idx) * idx  # smallest multiple of idx > m
            if first > m + L:
                adj[i] = []
                continue
            for v in range(first, m + L + 1, idx):
                pos = v - m - 1
                adj[i].append(pos)
        
        # Quick check: any index with no candidates
        for i in range(n):
            if not adj[i]:
                return False
        
        # Hall's theorem quick check for small subsets
        # For each subset of indices, check if union of their adj has size >= subset size
        # Only check subsets of size 2 and 3 for speed
        for i in range(n):
            for j in range(i+1, n):
                union = set(adj[i]) | set(adj[j])
                if len(union) < 2:
                    return False
        # Check triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    union = set(adj[i]) | set(adj[j]) | set(adj[k])
                    if len(union) < 3:
                        return False
        
        # Now do bipartite matching using DFS (Kuhn's algorithm)
        # match[pos] = index assigned to pos, or -1
        match = [-1] * L
        
        def dfs(i, visited):
            for pos in adj[i]:
                if visited[pos]:
                    continue
                visited[pos] = True
                if match[pos] == -1 or dfs(match[pos], visited):
                    match[pos] = i
                    return True
            return False
        
        for i in range(n):
            visited = [False] * L
            if not dfs(i, visited):
                return False
        return True
    
    def f_of_m(m):
        """Find minimum L such that can_assign(m, L) is True."""
        lo, hi = 12, 144
        if not can_assign(m, hi):
            return 0
        while lo < hi:
            mid = (lo + hi) // 2
            if can_assign(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Search strategy:
    # 1. First, try m values that are structurally "bad" for matching
    # 2. Use the fact that f(12,m) is periodic-ish with period LCM(1..12)=27720
    # 3. Focus on m values where residues create conflicts
    
    best_m = 0
    best_f = 0
    
    # Phase 1: Try structured candidates
    # m values where m mod i creates sparse candidates for many i
    # Try m = k * LCM_12 + offset for various offsets
    candidates = set()
    
    # Random samples
    random.seed(12345)
    for _ in range(3000):
        candidates.add(random.randint(0, max_m))
    
    # Structured: m near multiples of LCM
    for k in range(0, max_m // LCM_12 + 1):
        base = k * LCM_12
        for offset in range(0, min(200, LCM_12)):
            m_val = base + offset
            if 0 <= m_val <= max_m:
                candidates.add(m_val)
        if time.time() - start_time > deadline - 5:
            break
    
    # Also try m values where m+1 is divisible by many small numbers (creating dense candidates)
    # vs m values where m+1 is coprime to many numbers (sparse candidates)
    
    # Sort candidates and evaluate
    candidates = list(candidates)
    random.shuffle(candidates)
    
    for m in candidates:
        if time.time() - start_time > deadline:
            break
        f_val = f_of_m(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            # If we found a very high value, we might be close to optimal
            if best_f >= 140:
                break
    
    # Phase 2: Local search around best_m
    if time.time() - start_time < deadline - 5 and best_f < 144:
        for delta in range(-50, 51):
            m_try = best_m + delta
            if 0 <= m_try <= max_m:
                if time.time() - start_time > deadline:
                    break
                f_val = f_of_m(m_try)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m_try
    
    return int(best_m)
# EVOLVE-BLOCK-END
