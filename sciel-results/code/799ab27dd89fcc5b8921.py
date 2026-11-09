import time
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
P = 27720  # LCM(1..12)

def _precompute_residues():
    """Precompute for each residue r in [0,P) the list of (r0_i, divisor_i) for i=1..12."""
    # For residue r (where base = r, i.e., m+1 ≡ r mod P), divisor i+1 has
    # valid positions j where j ≡ (-r) mod (i+1)
    # Precompute the offset for each divisor
    offsets = np.zeros((P, 12), dtype=np.int32)
    for r in range(P):
        for i in range(12):
            divisor = i + 1
            offsets[r, i] = (-r) % divisor
    return offsets

OFFSETS = _precompute_residues()

def _hopcroft_karp(adj, n_left, n_right):
    """Hopcroft-Karp bipartite matching. adj[i] = list of right nodes for left node i."""
    # pairU[i] = right node matched to left node i, or -1
    # pairV[j] = left node matched to right node j, or -1
    pairU = [-1] * n_left
    pairV = [-1] * n_right
    dist = [0] * n_left
    
    def bfs():
        # Build distance layers from free left nodes
        q = deque()
        for u in range(n_left):
            if pairU[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pairV[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found
    
    def dfs(u):
        for v in adj[u]:
            pu = pairV[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pairU[u] = v
                pairV[v] = u
                return True
        dist[u] = -1
        return False
    
    matching = 0
    while bfs():
        for u in range(n_left):
            if pairU[u] == -1:
                if dfs(u):
                    matching += 1
    return matching

def _has_matching_fast(m, L):
    """Check if interval (m, m+L] contains distinct b_1..b_12 with i | b_i."""
    if L < 12:
        return False
    base_mod = (m + 1) % P
    
    # Build adjacency using precomputed offsets
    adj = [[] for _ in range(12)]
    for i in range(12):
        divisor = i + 1
        r0 = OFFSETS[base_mod, i]
        if r0 < L:
            # positions: r0, r0+divisor, r0+2*divisor, ... < L
            adj[i] = list(range(int(r0), L, divisor))
        else:
            adj[i] = []
    
    # Early exit: if any divisor has no candidates, fail
    for i in range(12):
        if not adj[i]:
            return False
    
    # Hall's condition quick check: for each subset of left nodes, 
    # |N(S)| >= |S|. Check singletons (done above) and pairs.
    # For pairs: for each pair (i,j), |adj[i] ∪ adj[j]| >= 2
    # This is expensive for all pairs, but 12 choose 2 = 66 pairs, manageable
    # Actually, let's just run the matching - it's fast enough with Hopcroft-Karp
    
    matching = _hopcroft_karp(adj, 12, L)
    return matching == 12

def _compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has valid matching."""
    for L in range(12, 200):
        if _has_matching_fast(m, L):
            return L
    # Fallback: should not reach here
    for L in range(200, P + 1):
        if _has_matching_fast(m, L):
            return L
    return P

def construct():
    deadline = time.time() + 85.0
    best_m = 0
    best_f = 0
    
    # Precompute all f values
    for r in range(P):
        if time.time() > deadline:
            break
        f_val = _compute_f(r)
        if f_val > best_f:
            best_f = f_val
            best_m = r
    
    return int(best_m)
# EVOLVE-BLOCK-END
