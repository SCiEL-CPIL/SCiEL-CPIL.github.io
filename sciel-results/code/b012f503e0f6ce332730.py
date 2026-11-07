"""Dense odd-order graphs avoiding equal-degree path endpoints."""
import time

# EVOLVE-BLOCK-START
_N = 15

def _is_valid_adj(adj, n):
    """Check: no simple 3-edge path s-u-v-t with deg(s)==deg(t).
    adj: list of int bitmasks. Returns True if valid."""
    deg = [bin(adj[i]).count('1') for i in range(n)]
    # Precompute: for each degree value, bitmask of vertices with that degree
    deg_to_mask = {}
    for i in range(n):
        d = deg[i]
        deg_to_mask[d] = deg_to_mask.get(d, 0) | (1 << i)
    
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if not (au & (1 << v)):
                continue
            au_minus_v = au & ~(1 << v)
            av_minus_u = adj[v] & ~(1 << u)
            s_mask = au_minus_v
            while s_mask:
                s_bit = s_mask & -s_mask
                s = s_bit.bit_length() - 1
                s_mask ^= s_bit
                # t must be in N(v)\{u}, t != s, and deg(t) == deg(s)
                t_mask = av_minus_u & ~(1 << s) & deg_to_mask[deg[s]]
                if t_mask:
                    return False
    return True

def _count_edges(adj, n):
    total = 0
    for i in range(n):
        total += bin(adj[i]).count('1')
    return total // 2

def _try_add_edge_fast(adj, n, a, b):
    """Try adding edge (a,b). Returns new adj if valid, else None."""
    if a == b or (adj[a] & (1 << b)):
        return None
    new_adj = adj[:]
    new_adj[a] |= (1 << b)
    new_adj[b] |= (1 << a)
    if _is_valid_adj(new_adj, n):
        return new_adj
    return None

def _build_k78_adj(n):
    """Build K_{7,8} as adjacency bitmasks."""
    adj = [0] * n
    for i in range(7):
        for j in range(7, 15):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    return adj

def _local_search_fast(adj, n, deadline):
    """Deterministic local search: try all missing edges, add if valid."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() >= deadline:
                    return adj
                if adj[u] & (1 << v):
                    continue
                new_adj = _try_add_edge_fast(adj, n, u, v)
                if new_adj is not None:
                    adj = new_adj
                    improved = True
    return adj

def _swap_search(adj, n, deadline):
    """Try 1-for-1 edge swaps to reach better configurations."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        existing = []
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    existing.append((u, v))
                else:
                    missing.append((u, v))
        
        for eu, ev in existing:
            if time.time() >= deadline:
                return adj
            for mu, mv in missing:
                if time.time() >= deadline:
                    return adj
                new_adj = adj[:]
                new_adj[eu] &= ~(1 << ev)
                new_adj[ev] &= ~(1 << eu)
                new_adj[mu] |= (1 << mv)
                new_adj[mv] |= (1 << mu)
                if _is_valid_adj(new_adj, n):
                    adj2 = _local_search_fast(new_adj, n, deadline)
                    if _count_edges(adj2, n) > _count_edges(adj, n):
                        adj = adj2
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    return adj

def construct():
    start = time.time()
    deadline = start + 50
    
    n = _N
    adj = _build_k78_adj(n)
    
    if not _is_valid_adj(adj, n):
        adj = [0] * n
        for v in range(1, n):
            adj[0] |= (1 << v)
            adj[v] |= (1 << 0)
    
    if time.time() < deadline:
        adj = _local_search_fast(adj, n, deadline)
    
    if time.time() < deadline - 5:
        adj = _swap_search(adj, n, deadline - 5)
    
    if time.time() < deadline:
        adj = _local_search_fast(adj, n, deadline)
    
    edges = []
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if ai & (1 << j):
                edges.append([int(i), int(j)])
    
    return edges
# EVOLVE-BLOCK-END
