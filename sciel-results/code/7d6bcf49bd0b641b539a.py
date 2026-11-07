"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import lru_cache

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n = 12
    P = 1  # LCM of 1..12
    for i in range(1, n + 1):
        P = P * i // math.gcd(P, i)
    # P = 27720

    # For a given m, f(12,m) = least L>0 s.t. there exist distinct b_1..b_12
    # in (m, m+L] with i | b_i.
    # Equivalently: positions p_i in [1,L], distinct, with (m + p_i) % i == 0.
    # i.e. p_i % i == (-m) % i.

    # We'll compute f for each residue r = m % P (since structure is periodic).
    # For each r, we need the minimal L.

    # Precompute: for each i, for each residue r, the valid p mod i is ((-r) % i).
    # In window [1, L], valid p for i are: p = ((-r)%i) + k*i, but p must be in [1,L].
    # Note: ((-r)%i) is in [0, i-1]. If 0, then p must be i, 2i, ... (since p>=1).

    # Bipartite matching: left = {1..12}, right = {1..L}. Edge if p valid for i.
    # We binary search L.

    def can_match(m_res, L):
        # m_res = m % P
        # For each i in 1..n, valid p in [1,L] with p % i == ((-m_res) % i)
        # Build adjacency and do bipartite matching (left side = i's, right = p's)
        # But L can be up to 144, so right side is at most 144 nodes.
        # Left side is 12 nodes. Use DFS augmenting path.
        
        # For each i, compute the set of valid p values
        adj = [set() for _ in range(n)]
        for i in range(1, n + 1):
            target = ((-m_res) % i)
            # p in [1, L] with p % i == target
            if target == 0:
                # p must be multiple of i, >= 1: i, 2i, ...
                first = i
            else:
                first = target
                if first < 1:
                    first += i
            p = first
            while p <= L:
                adj[i - 1].add(p)
                p += i
        
        # Bipartite matching: left nodes 0..n-1 (representing i=1..n), right nodes 1..L
        # Use standard DFS-based matching
        match_r = {}  # right node -> left node
        
        def dfs(u, visited):
            for p in adj[u]:
                if p in visited:
                    continue
                visited.add(p)
                if p not in match_r or dfs(match_r[p], visited):
                    match_r[p] = u
                    return True
            return False
        
        count = 0
        for u in range(n):
            visited = set()
            if dfs(u, visited):
                count += 1
        return count == n

    def compute_f(m_res):
        # Binary search for least L > 0 such that can_match(m_res, L)
        # Upper bound: by pigeonhole and structure, L <= 144 (given in task)
        # Actually let's use a safe upper bound
        lo, hi = 1, 200
        # First check if hi works
        while hi < 500 and not can_match(m_res, hi):
            hi *= 2
        if not can_match(m_res, hi):
            return 0  # shouldn't happen
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m_res, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    # Compute f for each residue r in [0, P)
    # But P = 27720, and for each we do binary search with matching.
    # This might be slow. Let's estimate: ~27720 residues * ~8 binary steps * matching cost.
    # Matching cost: 12 left nodes, each with up to L/i neighbors. For L~144, total edges ~ sum(L/i) ~ L*H(12) ~ 144*3.1 ~ 450.
    # DFS matching is O(E*V) worst case but in practice fast for 12 left nodes.
    # 27720 * 8 * (small constant) should be manageable.

    # But wait - we only need m in [0, 1000000]. The residues that appear are r = m % P for m in [0, 1000000].
    # Since 1000000 > 27720, all residues 0..27719 appear (actually 0..min(27719, 1000000%27720) and full cycles).
    # 1000000 // 27720 = 36 full cycles, remainder 1000000 - 36*27720 = 1000000 - 997920 = 2080.
    # So residues 0..2080 appear in the last partial cycle, and 0..27719 all appear in full cycles.
    # So all residues 0..27719 are achievable.

    # To save time, let's compute f for all residues and pick the best one that's achievable.
    # All are achievable, so just pick argmax.

    best_f = 0
    best_r = 0
    start_time = time.time()
    
    for r in range(P):
        if time.time() > deadline:
            break
        f_val = compute_f(r)
        if f_val > best_f:
            best_f = f_val
            best_r = r
    
    # Now find m in [0, 1000000] with m % P == best_r
    # Smallest such m is best_r (if best_r <= 1000000)
    m = best_r
    if m > 1000000:
        # This shouldn't happen since best_r < P = 27720 < 1000000
        m = 0
    
    return int(m)

# EVOLVE-BLOCK-END
