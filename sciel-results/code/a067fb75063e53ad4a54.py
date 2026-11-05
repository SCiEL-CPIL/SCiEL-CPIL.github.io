import time
import random

_START = time.time()
_LIMIT = 140.0

def _time_left():
    return _LIMIT - (time.time() - _START)

def _validate(n, colors):
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    nbrs = [[0]*n for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            nbrs[c][i] |= (1 << j)
            nbrs[c][j] |= (1 << i)
            idx += 1
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                common = nbrs[c][i] & nbrs[c][j]
                if bin(common).count('1') >= 2:
                    return False
    return True

def _count_c4s(n, colors):
    nbrs = [[0]*n for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            nbrs[c][i] |= (1 << j)
            nbrs[c][j] |= (1 << i)
            idx += 1
    total = 0
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                common = nbrs[c][i] & nbrs[c][j]
                cnt = bin(common).count('1')
                if cnt >= 2:
                    total += cnt * (cnt-1) // 2
    return total

def _seed_construction():
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def _edge_idx(a, b, n):
    if a > b:
        a, b = b, a
    return a * (n - 1) - a * (a + 1) // 2 + (b - a - 1)

def _simulated_annealing(n, max_time=60.0, seed=42):
    """Simulated annealing to find C4-free 5-coloring of K_n."""
    random.seed(seed)
    m = n * (n - 1) // 2
    
    # Initial coloring: random
    colors = [random.randint(0, 4) for _ in range(m)]
    
    current_energy = _count_c4s(n, colors)
    best_colors = colors[:]
    best_energy = current_energy
    
    T = 10.0
    cooling = 0.999
    
    start_time = time.time()
    while _time_left() > max_time and T > 0.01:
        # Pick random edge
        e = random.randint(0, m - 1)
        old_c = colors[e]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue
        
        colors[e] = new_c
        new_energy = _count_c4s(n, colors)
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_colors = colors[:]
        else:
            colors[e] = old_c
        
        T *= cooling
    
    if best_energy == 0:
        return {'n': n, 'colors': best_colors}
    return None

def _lazebnik_woldar():
    """Try the Lazebnik-Woldar construction with various ∞-edge colorings."""
    p = 5
    
    def gf25_mul(x, y):
        a1, b1 = x
        a2, b2 = y
        a = (a1 * a2 + 3 * b1 * b2) % p
        b = (a1 * b2 + a2 * b1) % p
        return (a, b)
    
    def gf25_add(x, y):
        return ((x[0]+y[0])%p, (x[1]+y[1])%p)
    
    def gf25_sub(x, y):
        return ((x[0]-y[0])%p, (x[1]-y[1])%p)
    
    def gf25_pow(x, n):
        result = (1, 0)
        base = x
        while n > 0:
            if n % 2 == 1:
                result = gf25_mul(result, base)
            base = gf25_mul(base, base)
            n //= 2
        return result
    
    gf25_elements = [(a, b) for a in range(p) for b in range(p)]
    zero = (0, 0)
    one = (1, 0)
    
    primitive = None
    for elem in gf25_elements:
        if elem == zero:
            continue
        if gf25_pow(elem, 24) == one and gf25_pow(elem, 12) != one and gf25_pow(elem, 8) != one and gf25_pow(elem, 6) != one:
            primitive = elem
            break
    
    if primitive is None:
        return None
    
    powers = []
    current = one
    for i in range(24):
        powers.append(current)
        current = gf25_mul(current, primitive)
    
    log_map = {elem: i for i, elem in enumerate(powers)}
    gf5_elements = set((a, 0) for a in range(p))
    
    n = 26
    
    # Try different ∞-edge coloring strategies
    inf_strategies = [
        # Strategy 0: ∞-x color based on x's log mod 5
        lambda x_idx: 0 if x_idx == 0 else (log_map[gf25_elements[x_idx]] % 5) if gf25_elements[x_idx] not in gf5_elements else 0,
        # Strategy 1: ∞-x color based on x's log mod 4 + 1 for non-GF5
        lambda x_idx: 0 if x_idx == 0 else (1 + log_map[gf25_elements[x_idx]] % 4) if gf25_elements[x_idx] not in gf5_elements else 0,
        # Strategy 2: ∞-x all color 0
        lambda x_idx: 0,
        # Strategy 3: ∞-x color based on b-coordinate
        lambda x_idx: gf25_elements[x_idx][1] % 5,
        # Strategy 4: ∞-x color based on a-coordinate  
        lambda x_idx: gf25_elements[x_idx][0] % 5,
    ]
    
    for inf_fn in inf_strategies:
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i == 25 or j == 25:
                    other = j if i == 25 else i
                    colors.append(inf_fn(other))
                else:
                    elem_i = gf25_elements[i]
                    elem_j = gf25_elements[j]
                    diff = gf25_sub(elem_i, elem_j)
                    if diff == zero:
                        colors.append(0)
                    elif diff in gf5_elements:
                        colors.append(0)
                    else:
                        k = log_map[diff]
                        colors.append(1 + (k % 4))
        
        if _validate(n, colors):
            return {'n': n, 'colors': colors}
    
    return None

def construct():
    if _time_left() <= 0:
        return _seed_construction()
    
    best = _seed_construction()
    
    # Try Lazebnik-Woldar
    lw = _lazebnik_woldar()
    if lw is not None:
        return lw
    
    # Try simulated annealing for n=26
    if _time_left() > 30:
        for seed in [42, 123, 456, 789, 1000]:
            if _time_left() < 25:
                break
            res = _simulated_annealing(26, max_time=min(20.0, _time_left() - 20.0), seed=seed)
            if res is not None:
                return res
    
    # Try smaller n
    for n in [25, 24, 23, 22, 21, 20]:
        if _time_left() < 15:
            break
        if n <= best['n']:
            break
        for seed in [42, 123]:
            if _time_left() < 12:
                break
            res = _simulated_annealing(n, max_time=min(10.0, _time_left() - 10.0), seed=seed)
            if res is not None:
                best = res
                break
    
    return best
