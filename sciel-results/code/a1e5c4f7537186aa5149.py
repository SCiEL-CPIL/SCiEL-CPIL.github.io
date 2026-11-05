import time
import random
import sys

_START = time.time()
_LIMIT = 145.0  # Leave 5s margin for evaluator overhead

def _time_left():
    return _LIMIT - (time.time() - _START)

def _validate(n, colors):
    """Check that no colour class contains a 4-cycle (C4)."""
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    
    # Build neighbour sets per colour
    nbrs = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            nbrs[c][i].add(j)
            nbrs[c][j].add(i)
            idx += 1
            
    # Check for C4: two vertices with >= 2 common neighbours in one colour
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                if len(nbrs[c][i] & nbrs[c][j]) >= 2:
                    return False
    return True

def _edge_idx(i, j, n):
    if i > j:
        i, j = j, i
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _local_search(n, max_iters=200000):
    """
    Local search to find a 5-coloring of K_n with no monochromatic C4.
    Returns (n, colors) if found, else None.
    """
    m = n * (n - 1) // 2
    colors = [random.randint(0, 4) for _ in range(m)]
    
    # Build initial neighbor sets
    nbrs = [[set() for _ in range(n)] for _ in range(5)]
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[_edge_idx(i, j, n)]
            nbrs[c][i].add(j)
            nbrs[c][j].add(i)
    
    # Calculate initial cost
    cost = 0
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                common = len(nbrs[c][i] & nbrs[c][j])
                cost += common * (common - 1) // 2
    
    if cost == 0:
        return n, colors

    best_cost = cost
    best_colors = colors[:]
    
    # Precompute list of edges for random selection
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for _ in range(max_iters):
        if _time_left() < 10:
            break
            
        # Pick a random edge
        u, v = random.choice(edges)
        idx = _edge_idx(u, v, n)
        c_old = colors[idx]
        
        # Try all other colors
        best_delta = float('inf')
        best_new_c = None
        
        for c_new in range(5):
            if c_new == c_old:
                continue
            
            delta = 0
            
            # 1. Remove edge (u,v) from c_old
            # Affected pairs: (u, x) and (v, x) for x != u, v
            # Change in cost for pair (a, b) when C changes by -k:
            # Old: C(C-1)/2, New: (C-k)(C-k-1)/2
            # Diff: C(C-1)/2 - (C-k)(C-k-1)/2
            
            # For (u, x):
            # C_old(u, x) = |N_cold(u) ∩ N_cold(x)|
            # v is in N_cold(u).
            # If v is in N_cold(x), then v is in the intersection.
            # So C_new(u, x) = C_old(u, x) - 1 if v in N_cold(x), else C_old(u, x).
            
            for x in range(n):
                if x == u or x == v:
                    continue
                
                # Pair (u, x)
                if v in nbrs[c_old][x]:
                    c_u_x = len(nbrs[c_old][u] & nbrs[c_old][x])
                    # c_u_x includes v? Yes, because v in N(u) and v in N(x).
                    # So new count is c_u_x - 1.
                    old_term = c_u_x * (c_u_x - 1) // 2
                    new_term = (c_u_x - 1) * (c_u_x - 2) // 2
                    delta += new_term - old_term
                # else: no change for (u,x) in c_old?
                # Wait, N_cold(u) loses v.
                # Intersection N(u) ∩ N(x) loses v ONLY IF v was in N(x).
                # If v not in N(x), v wasn't in intersection.
                # So yes, no change.
                
                # Pair (v, x)
                if u in nbrs[c_old][x]:
                    c_v_x = len(nbrs[c_old][v] & nbrs[c_old][x])
                    old_term = c_v_x * (c_v_x - 1) // 2
                    new_term = (c_v_x - 1) * (c_v_x - 2) // 2
                    delta += new_term - old_term
            
            # 2. Add edge (u,v) to c_new
            # Affected pairs: (u, x) and (v, x) for x != u, v
            # C_new(u, x) = C_old(u, x) + 1 if v in N_cnew(x), else C_old(u, x).
            
            for x in range(n):
                if x == u or x == v:
                    continue
                
                # Pair (u, x)
                if v in nbrs[c_new][x]:
                    c_u_x = len(nbrs[c_new][u] & nbrs[c_new][x])
                    # c_u_x does NOT include v yet.
                    # New count is c_u_x + 1.
                    old_term = c_u_x * (c_u_x - 1) // 2
                    new_term = (c_u_x + 1) * c_u_x // 2
                    delta += new_term - old_term
                
                # Pair (v, x)
                if u in nbrs[c_new][x]:
                    c_v_x = len(nbrs[c_new][v] & nbrs[c_new][x])
                    old_term = c_v_x * (c_v_x - 1) // 2
                    new_term = (c_v_x + 1) * c_v_x // 2
                    delta += new_term - old_term
            
            if delta < best_delta:
                best_delta = delta
                best_new_c = c_new
        
        if best_new_c is not None and best_delta < 0:
            # Apply change
            colors[idx] = best_new_c
            nbrs[c_old][u].remove(v)
            nbrs[c_old][v].remove(u)
            nbrs[best_new_c][u].add(v)
            nbrs[best_new_c][v].add(u)
            cost += best_delta
            best_cost = cost
            best_colors = colors[:]
            
            if cost == 0:
                return n, colors
        else:
            # Stuck? Maybe random restart or just continue with random moves
            # To avoid getting stuck in local minima, we can allow random moves with low prob
            if random.random() < 0.1:
                # Random move
                c_rand = random.choice([c for c in range(5) if c != c_old])
                colors[idx] = c_rand
                nbrs[c_old][u].remove(v)
                nbrs[c_old][v].remove(u)
                nbrs[c_rand][u].add(v)
                nbrs[c_rand][v].add(u)
                # Recalculate cost fully? Too slow.
                # Just break and restart if stuck too long?
                # For simplicity, let's just continue. The best_colors is saved.
    
    if best_cost == 0:
        return n, best_colors
    return None

def _cyclic_construction(n):
    """Cyclic 5-colouring using circular distance mapped to colour (d-1)%5."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            d = min(d, n - d)
            c = (d - 1) % 5
            colors.append(c)
    return {'n': n, 'colors': colors}

def _seed_construction():
    """Original K11 seed."""
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def construct():
    """Entry point: return the best 5-colouring found."""
    if _time_left() <= 0:
        return _seed_construction()
    
    best = _seed_construction()
    
    # Try local search for n=26
    if _time_left() > 20:
        res = _local_search(26, max_iters=100000)
        if res:
            n, colors = res
            if _validate(n, colors):
                best = {'n': n, 'colors': colors}
    
    # If 26 failed, try 25
    if best['n'] < 26 and _time_left() > 20:
        res = _local_search(25, max_iters=100000)
        if res:
            n, colors = res
            if _validate(n, colors):
                best = {'n': n, 'colors': colors}
                
    # If 25 failed, try 20
    if best['n'] < 25 and _time_left() > 10:
        res = _local_search(20, max_iters=50000)
        if res:
            n, colors = res
            if _validate(n, colors):
                best = {'n': n, 'colors': colors}

    # Fallback to cyclic if needed
    if best['n'] < 11:
        best = _seed_construction()

    return best

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, valid={_validate(res['n'], res['colors'])}")
