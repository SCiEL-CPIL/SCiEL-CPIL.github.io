import random
import time
import math

_START = time.time()
DEADLINE = _START + 115.0

def _time_left():
    return DEADLINE - time.time()

def _popcount(x):
    return bin(x).count('1')

def _is_k4_free(adj, n):
    """Check K4-free using bitmasks."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
            m = common
            while m:
                lsb = m & -m
                c = lsb.bit_length() - 1
                m ^= lsb
                rest = common & ~(1 << c)
                if adj[c] & rest:
                    return False
    return True

def _max_triangle_free(adj, n, deadline):
    """Find max number of vertices inducing a triangle-free subgraph."""
    full_mask = (1 << n) - 1
    best = [0]
    
    def has_edge(mask):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            if adj[v] & mask:
                return True
        return False
    
    def branch(candidates, current_set, current_size):
        if time.time() > deadline:
            return
        if current_size + _popcount(candidates) <= best[0]:
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        v_idx = -1
        best_deg = -1
        m = candidates
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = _popcount(adj[v] & candidates)
            if deg > best_deg:
                best_deg = deg
                v_idx = v
        
        v = v_idx
        v_bit = 1 << v
        rest = candidates & ~v_bit
        
        N_v = adj[v] & current_set
        can_add = True
        if N_v and has_edge(N_v):
            can_add = False
        
        if can_add:
            new_set = current_set | v_bit
            forbidden = 0
            if N_v:
                m3 = N_v
                while m3:
                    lsb3 = m3 & -m3
                    u = lsb3.bit_length() - 1
                    m3 ^= lsb3
                    forbidden |= (adj[v] & adj[u])
            new_cands = rest & ~forbidden
            branch(new_cands, new_set, current_size + 1)
        
        branch(rest, current_set, current_size)
    
    branch(full_mask, 0, 0)
    return best[0]

def _is_3_regular(adj, n):
    for i in range(n):
        if _popcount(adj[i]) != 3:
            return False
    return True

def _make_complement(adj_h, n):
    adj_g = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj_h[i] >> j) & 1:
                adj_g[i] |= (1 << j)
                adj_g[j] |= (1 << i)
    return adj_g

def _two_switch(adj_h, n, rng):
    """Perform a 2-switch on H to maintain 3-regularity. Returns (success, revert_func)."""
    edges = []
    for i in range(n):
        m = adj_h[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                edges.append((i, j))
            m ^= lsb
    
    if len(edges) < 2:
        return False, None
    
    for _ in range(50):
        e1 = rng.choice(edges)
        e2 = rng.choice(edges)
        a, b = e1
        c, d = e2
        if len({a, b, c, d}) < 4:
            continue
        if (adj_h[a] >> c) & 1 or (adj_h[b] >> d) & 1:
            continue
        
        # Save state for revert
        state = adj_h[:]
        
        adj_h[a] ^= (1 << b)
        adj_h[b] ^= (1 << a)
        adj_h[c] ^= (1 << d)
        adj_h[d] ^= (1 << c)
        adj_h[a] |= (1 << c)
        adj_h[c] |= (1 << a)
        adj_h[b] |= (1 << d)
        adj_h[d] |= (1 << b)
        
        def revert():
            for i in range(n):
                adj_h[i] = state[i]
        
        return True, revert
    return False, None

def _random_3_regular(n, rng):
    half_edges = []
    for i in range(n):
        half_edges.extend([i, i, i])
    rng.shuffle(half_edges)
    adj = [0] * n
    used = set()
    for i in range(0, len(half_edges), 2):
        u = half_edges[i]
        v = half_edges[i + 1]
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in used:
            continue
        used.add(key)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    for _ in range(200):
        if _is_3_regular(adj, n):
            return adj
        degs = [_popcount(adj[i]) for i in range(n)]
        low = [i for i in range(n) if degs[i] < 3]
        high = [i for i in range(n) if degs[i] > 3]
        if not low or not high:
            break
        u = rng.choice(low)
        v = rng.choice(high)
        if (adj[u] >> v) & 1:
            continue
        m = adj[v]
        if not m:
            continue
        lsb = m & -m
        w = lsb.bit_length() - 1
        adj[v] &= ~(1 << w)
        adj[w] &= ~(1 << v)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _is_3_regular(adj, n):
        return adj
    return None

def _q4_graph(n):
    adj = [0] * n
    for i in range(16):
        for bit in range(4):
            j = i ^ (1 << bit)
            if j > i:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    # Modify to include vertices 16, 17
    adj[0] &= ~(1 << 1)
    adj[1] &= ~(1 << 0)
    adj[2] &= ~(1 << 3)
    adj[3] &= ~(1 << 2)
    adj[0] |= (1 << 16)
    adj[16] |= (1 << 0)
    adj[1] |= (1 << 17)
    adj[17] |= (1 << 1)
    adj[2] |= (1 << 16)
    adj[16] |= (1 << 2)
    adj[3] |= (1 << 17)
    adj[17] |= (1 << 3)
    adj[16] |= (1 << 17)
    adj[17] |= (1 << 16)
    return adj

def construct():
    n = 18
    deadline = DEADLINE
    rng = random.Random(42)
    
    best_score = n + 1
    best_adj_g = None
    
    def evaluate(adj_g):
        nonlocal best_score, best_adj_g
        if time.time() > deadline:
            return
        s = _max_triangle_free(adj_g, n, deadline)
        if s < best_score:
            best_score = s
            best_adj_g = adj_g[:]
    
    candidates_h = []
    
    q4 = _q4_graph(n)
    if _is_3_regular(q4, n):
        candidates_h.append(q4)
    
    for _ in range(80):
        if time.time() > deadline - 25:
            break
        h = _random_3_regular(n, rng)
        if h is not None:
            candidates_h.append(h)
    
    adj_turan = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj_turan[u] |= (1 << v)
                    adj_turan[v] |= (1 << u)
    
    for h in candidates_h:
        if time.time() > deadline - 15:
            break
        g = _make_complement(h, n)
        evaluate(g)
    
    if time.time() < deadline - 15:
        evaluate(adj_turan)
    
    # Local search on 3-regular H
    if best_adj_g is not None and time.time() < deadline - 10:
        # Find a 3-regular H that gives best_adj_g
        # Or just do local search on general K4-free graphs
        current_g = best_adj_g[:]
        current_score = _max_triangle_free(current_g, n, deadline)
        
        T = 1.0
        for step in range(1500):
            if time.time() > deadline - 3:
                break
            
            i, j = rng.sample(range(n), 2)
            current_g[i] ^= (1 << j)
            current_g[j] ^= (1 << i)
            
            if not _is_k4_free(current_g, n):
                current_g[i] ^= (1 << j)
                current_g[j] ^= (1 << i)
                continue
            
            new_score = _max_triangle_free(current_g, n, deadline)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj_g = current_g[:]
            elif new_score > current_score:
                delta = new_score - current_score
                if rng.random() < math.exp(-delta / T):
                    current_score = new_score
                else:
                    current_g[i] ^= (1 << j)
                    current_g[j] ^= (1 << i)
            
            T *= 0.999
    
    if best_adj_g is None:
        best_adj_g = adj_turan
    
    edges = []
    for i in range(n):
        m = best_adj_g[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                edges.append([i, j])
            m ^= lsb
    
    return edges
