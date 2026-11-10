"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    max_m = 1000000
    LCM_12 = 27720  # LCM(1,2,...,12)
    
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
        
        # Quick check: any index with no candidates
        for i in range(n):
            if len(adj[i]) == 0:
                return False
        
        # Use Kuhn's algorithm (DFS-based bipartite matching)
        # match[pos] = which index is matched to position pos, or -1
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
        
        # Try to match all indices
        for i in range(n):
            visited = [False] * L
            if not dfs(i, visited):
                return False
        return True
    
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
    
    # Strategy: Use periodicity via LCM(1..12)=27720
    # f(12, m) depends only on m mod 27720 for the pattern of divisibility
    # So we can exhaustively search all residues mod 27720 to find the best pattern
    # Then pick the best m in [0, 1000000] with that residue
    
    best_f = -1
    best_residue = 0
    best_m = 0
    
    # Phase 1: Exhaustive search over all residues mod LCM_12
    # For each residue r in 0..LCM_12-1, compute f(12, r)
    # Since the pattern repeats every LCM_12, f(12, m) = f(12, m mod LCM_12)
    # Wait, that's not quite right. The interval (m, m+L] shifts, so the pattern
    # of multiples depends on m mod i for each i, which is determined by m mod LCM.
    # But the absolute values in the interval matter for which multiples are present.
    # Actually, for a fixed L, the set of multiples of i in (m, m+L] depends on
    # m mod i and L. Since i divides LCM, m mod i is determined by m mod LCM.
    # So for fixed L, can_assign(m, L) depends only on m mod LCM.
    # Therefore f(12, m) depends only on m mod LCM.
    
    # Let's verify: if m1 ≡ m2 (mod LCM), then for any i in 1..12,
    # m1 mod i = m2 mod i. The multiples of i in (m, m+L] are determined by
    # the first multiple > m, which is ceil((m+1)/i)*i. This depends on m mod i.
    # So yes, can_assign(m, L) is periodic in m with period LCM.
    
    # Therefore, we can find the best residue r in 0..LCM_12-1,
    # then pick any m in [0, max_m] with m ≡ r (mod LCM_12).
    
    # Evaluate all residues
    for r in range(LCM_12):
        if time.time() - start_time > deadline:
            break
        f_val = f_of_m(r)
        if f_val > best_f:
            best_f = f_val
            best_residue = r
    
    # Find the best m in [0, max_m] with m ≡ best_residue (mod LCM_12)
    # The smallest such m is best_residue itself (if <= max_m)
    # We can also check neighbors for safety, but the periodicity should hold
    if best_residue <= max_m:
        best_m = best_residue
    else:
        # Should not happen since LCM_12 = 27720 < max_m = 1000000
        best_m = best_residue % (max_m + 1)
    
    # Phase 2: Local search around best_m to verify and potentially improve
    # (In case my periodicity argument has a subtle flaw)
    if time.time() - start_time < deadline - 10:
        for delta in range(-50, 51):
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
