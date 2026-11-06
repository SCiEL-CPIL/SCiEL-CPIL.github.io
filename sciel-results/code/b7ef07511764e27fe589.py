"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def _has_k222(adj, n):
    """Check if graph with adjacency bitmasks adj on n vertices contains K2,2,2 as a subgraph."""
    for a in range(n):
        for b in range(a + 1, n):
            if not ((adj[a] >> b) & 1):
                continue
            cn_mask = adj[a] & adj[b]
            cn_mask &= ~(1 << a)
            cn_mask &= ~(1 << b)
            if cn_mask.bit_count() < 4:
                continue
            # Get list of common neighbors
            cn_list = []
            m = cn_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            # Check if any two vertices in cn_list have >= 2 common neighbors in cn_mask
            L = len(cn_list)
            for i in range(L):
                u = cn_list[i]
                for j in range(i + 1, L):
                    v = cn_list[j]
                    common_uv = adj[u] & adj[v] & cn_mask
                    if common_uv.bit_count() >= 2:
                        return True
    return False

def _is_valid(adj, n):
    """Check both color classes are K2,2,2-free."""
    if _has_k222(adj, n):
        return False
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    if _has_k222(comp, n):
        return False
    return True

def _build_paley(q):
    """Build Paley graph on q vertices (q prime, q ≡ 1 mod 4)."""
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _paley_colors(adj, n):
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # Try Paley graph on q=29 first (largest candidate)
    q = 29
    adj = _build_paley(q)
    if _is_valid(adj, q):
        colors = _paley_colors(adj, q)
        return {'n': q, 'colors': colors}

    # Fallback: Paley on 17
    q = 17
    adj = _build_paley(q)
    colors = _paley_colors(adj, q)
    return {'n': q, 'colors': colors}
# EVOLVE-BLOCK-END
