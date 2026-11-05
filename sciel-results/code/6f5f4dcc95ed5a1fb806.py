import time
import random
import numpy as np

_START = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _time_left():
    return _LIMIT - _MARGIN - (time.time() - _START)

def _validate(n, colors):
    """Check that no colour class contains a 4-cycle (C4)."""
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    adj = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i, j] = 1
            adj[c][j, i] = 1
            idx += 1
    for c in range(5):
        A2 = adj[c] @ adj[c]
        np.fill_diagonal(A2, 0)
        if np.any(A2 >= 2):
            return False
    return True

def _seed_construction():
    """Original K11 seed: split into five Hamiltonian cycles by circular distance 1..5."""
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def _construct_gf25():
    """
    Lazebnik-Woldar style construction for n=26 using GF(25).
    """
    p = 5
    # GF(25) elements as (a, b) = a + b*alpha, alpha^2 = 3 (since 3 is non-square mod 5)
    elements = [(a, b) for a in range(p) for b in range(p)]
    
    def gf_mul(x, y):
        a1, b1 = x
        a2, b2 = y
        a = (a1 * a2 + 3 * b1 * b2) % p
        b = (a1 * b2 + a2 * b1) % p
        return (a, b)
    
    def gf_pow(x, n):
        result = (1, 0)
        base = x
        while n > 0:
            if n % 2 == 1:
                result = gf_mul(result, base)
            base = gf_mul(base, base)
            n //= 2
        return result
    
    # Find primitive element (order 24)
    primitive = None
    for elem in elements:
        if elem == (0, 0):
            continue
        if gf_pow(elem, 24) == (1, 0):
            is_primitive = True
            for d in [1, 2, 3, 4, 6, 8, 12]:
                if gf_pow(elem, d) == (1, 0):
                    is_primitive = False
                    break
            if is_primitive:
                primitive = elem
                break
                
    if primitive is None:
        return None

    # Generate powers
    powers = []
    current = (1, 0)
    for _ in range(24):
        powers.append(current)
        current = gf_mul(current, primitive)
        
    log_map = {elem: i for i, elem in enumerate(powers)}
    
    n = 26
    colors = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if i == 25 or j == 25:
                other = j if i == 25 else i
                if other == 0:
                    colors.append(0)
                else:
                    c = log_map[elements[other]] % 4
                    colors.append(c)
            else:
                elem_i = elements[i]
                elem_j = elements[j]
                if elem_i == (0, 0) or elem_j == (0, 0):
                    colors.append(0)
                else:
                    log_i = log_map[elem_i]
                    log_j = log_map[elem_j]
                    c = (log_i - log_j) % 4
                    colors.append(c)
                    
    return {'n': n, 'colors': colors}

def _try_extend(n_base, base_colors, n_target, time_budget=25.0):
    """Try to extend coloring by adding vertices greedily."""
    if n_target <= n_base:
        return None
    
    colors = list(base_colors)
    n = n_base
    
    def build_adj_bitmasks(n, colors):
        adj = [0] * (5 * n)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c * n + i] |= (1 << j)
                adj[c * n + j] |= (1 << i)
                idx += 1
        return adj
    
    def check_valid(n, adj):
        for c in range(5):
            for i in range(n):
                mi = adj[c * n + i]
                for j in range(i + 1, n):
                    mj = adj[c * n + j]
                    common = mi & mj
                    if common & (common - 1):
                        return False
        return True
    
    def add_vertex(n, adj, colors):
        """Try to add vertex n. Returns (new_adj, new_colors) or None."""
        n_new = n + 1
        S = [set() for _ in range(5)]
        
        random.seed(12345 + n)
        order = list(range(n))
        random.shuffle(order)
        
        for v in order:
            placed = False
            for c in range(5):
                ok = True
                for u in S[c]:
                    if adj[c * n + u] & adj[c * n + v]:
                        ok = False
                        break
                if ok:
                    S[c].add(v)
                    placed = True
                    break
            if not placed:
                return None
        
        new_adj = [0] * (5 * n_new)
        for c in range(5):
            for i in range(n):
                new_adj[c * n_new + i] = adj[c * n + i]
        
        new_colors = list(colors)
        for c in range(5):
            for v in sorted(S[c]):
                new_adj[c * n_new + v] |= (1 << n)
                new_adj[c * n_new + n] |= (1 << v)
                new_colors.append(c)
        
        if not check_valid(n_new, new_adj):
            return None
        return (new_adj, new_colors)
    
    adj = build_adj_bitmasks(n, colors)
    
    while n < n_target and _time_left() > time_budget:
        success = False
        for attempt in range(20):
            random.seed(attempt * 7919 + n * 31)
            result = add_vertex(n, adj, colors)
            if result is not None:
                adj, colors = result
                n += 1
                success = True
                break
        if not success:
            break
    
    if n == n_target:
        return colors
    return None

def construct():
    """Entry point: return the best 5-colouring found."""
    if _time_left() <= 0:
        return _seed_construction()
    
    # 1. Try the known valid n=26 construction
    res_26 = _construct_gf25()
    if res_26 and _validate(26, res_26['colors']):
        best = res_26
    else:
        best = _seed_construction()
        
    # 2. If we have n=26, try to extend to n=27
    if best['n'] == 26 and _time_left() > 20:
        extended = _try_extend(26, best['colors'], 27, time_budget=15.0)
        if extended is not None and _validate(27, extended):
            return {'n': 27, 'colors': extended}
    
    return best

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, valid={_validate(res['n'], res['colors'])}")
