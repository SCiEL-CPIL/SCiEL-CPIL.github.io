"""Rewritten program: Simulated Annealing with fast bitmask triangle-free evaluation."""
import random
import time
import itertools

N = 18
ALL_MASK = (1 << N) - 1

def _edge_bit(u, v):
    if u > v:
        u, v = v, u
    return u * N + v

def _edge_exists(mask, u, v):
    if u > v:
        u, v = v, u
    return (mask >> (u * N + v)) & 1

def _toggle_edge(mask, u, v):
    if u > v:
        u, v = v, u
    return mask ^ (1 << (u * N + v))

def _is_k4_free(mask):
    """Check K4-free. Returns True if no K4 exists."""
    for a in range(N):
        for b in range(a+1, N):
            if not _edge_exists(mask, a, b):
                continue
            for c in range(b+1, N):
                if not _edge_exists(mask, a, c) or not _edge_exists(mask, b, c):
                    continue
                for d in range(c+1, N):
                    if (_edge_exists(mask, a, d) and _edge_exists(mask, b, d) and 
                        _edge_exists(mask, c, d)):
                        return False
    return True

def _max_triangle_free_size(mask):
    """Exact branch and bound: largest induced subgraph that is triangle-free."""
    best = [0]
    
    adj_mask = [0] * N
    for v in range(N):
        for u in range(v+1, N):
            if _edge_exists(mask, u, v):
                adj_mask[v] |= (1 << u)
                adj_mask[u] |= (1 << v)
    
    def _can_add(selected_mask, v):
        common = selected_mask & adj_mask[v]
        for a_bit in range(N):
            if not (common & (1 << a_bit)):
                continue
            a = a_bit
            rest = common & ~(1 << a)
            if rest & adj_mask[a]:
                return False
        return True
    
    def _branch(avail_mask, selected_mask):
        upper = bin(selected_mask).count('1') + bin(avail_mask).count('1')
        if upper <= best[0]:
            return
        if avail_mask == 0:
            sz = bin(selected_mask).count('1')
            if sz > best[0]:
                best[0] = sz
            return
        lsb = avail_mask & (-avail_mask)
        v = lsb.bit_length() - 1
        rest_avail = avail_mask ^ lsb
        if _can_add(selected_mask, v):
            _branch(rest_avail, selected_mask | lsb)
        _branch(rest_avail, selected_mask)
    
    _branch(ALL_MASK, 0)
    return best[0]

def _greedy_triangle_free(mask, trials=30):
    best = 0
    adj_mask = [0] * N
    for v in range(N):
        for u in range(v+1, N):
            if _edge_exists(mask, u, v):
                adj_mask[v] |= (1 << u)
                adj_mask[u] |= (1 << v)
    for _ in range(trials):
        order = list(range(N))
        random.shuffle(order)
        selected = 0
        for v in order:
            common = selected & adj_mask[v]
            ok = True
            for a_bit in range(N):
                if not (common & (1 << a_bit)):
                    continue
                a = a_bit
                rest = common & ~(1 << a)
                if rest & adj_mask[a]:
                    ok = False
                    break
            if ok:
                selected |= (1 << v)
        size = bin(selected).count('1')
        if size > best:
            best = size
    return best

def _mask_to_edges(mask):
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if _edge_exists(mask, u, v):
                edges.append([u, v])
    return edges

def construct():
    deadline = time.time() + 140
    
    # Start with random graph, make K4-free
    mask = 0
    for u in range(N):
        for v in range(u+1, N):
            if random.random() < 0.5:
                mask |= (1 << (u * N + v))
    
    # Make K4-free by removing edges from K4s
    guard = 0
    while not _is_k4_free(mask) and guard < 1000:
        guard += 1
        found = False
        for a in range(N):
            for b in range(a+1, N):
                if not _edge_exists(mask, a, b):
                    continue
                for c in range(b+1, N):
                    if not _edge_exists(mask, a, c) or not _edge_exists(mask, b, c):
                        continue
                    for d in range(c+1, N):
                        if (_edge_exists(mask, a, d) and _edge_exists(mask, b, d) and 
                            _edge_exists(mask, c, d)):
                            edges_in_k4 = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                            e = random.choice(edges_in_k4)
                            mask = _toggle_edge(mask, e[0], e[1])
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
    
    current_mask = mask
    current_score = _max_triangle_free_size(mask)
    best_mask = mask
    best_score = current_score
    
    # Simulated Annealing
    T = 2.0
    T_min = 0.01
    cooling = 0.9995
    
    all_pairs = list(itertools.combinations(range(N), 2))
    
    while time.time() < deadline and T > T_min:
        u, v = random.choice(all_pairs)
        new_mask = _toggle_edge(current_mask, u, v)
        
        if not _is_k4_free(new_mask):
            T *= cooling
            continue
        
        greedy_score = _greedy_triangle_free(new_mask, trials=20)
        if greedy_score >= best_score:
            T *= cooling
            continue
        
        new_score = _max_triangle_free_size(new_mask)
        
        if new_score < current_score:
            current_mask = new_mask
            current_score = new_score
            if new_score < best_score:
                best_mask = new_mask
                best_score = new_score
        else:
            delta = new_score - current_score
            if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 0.001)):
                current_mask = new_mask
                current_score = new_score
        
        T *= cooling
    
    return _mask_to_edges(best_mask)
