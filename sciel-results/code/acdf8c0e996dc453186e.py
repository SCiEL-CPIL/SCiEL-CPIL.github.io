import time
import random

_START = time.time()
_LIMIT = 140.0  # Safety margin before 150s kill

def _time_left():
    return _LIMIT - (time.time() - _START)

def _validate(n, colors):
    """Check that no colour class contains a 4-cycle (C4).
    Equivalent: for some colour c, two vertices have >=2 common neighbours in c.
    """
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
            
    # Check for C4
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                # Optimization: if degree is low, intersection is small
                # But sets are small enough for n=26
                if len(nbrs[c][i] & nbrs[c][j]) >= 2:
                    return False
    return True

def _lazebnik_woldar():
    """
    Lazebnik-Woldar algebraic construction for R_5(C4) >= 26.
    Uses GF(25) = GF(5^2) with alpha^2 = 3.
    Vertices: 25 elements of GF(25) + infinity = 26 vertices.
    """
    p = 5
    
    # Precompute GF(25) operations
    # Element is (a, b) representing a + b*alpha
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

    # Find primitive element
    # GF(25)^* has order 24. Primitive element has order 24.
    primitive = None
    for a in range(p):
        for b in range(p):
            if a == 0 and b == 0:
                continue
            elem = (a, b)
            # Check if order is 24
            # 24 = 2^3 * 3. Must not be 1 for divisors 12, 8, 6, 4, 3, 2
            if gf_pow(elem, 12) == (1, 0): continue
            if gf_pow(elem, 8) == (1, 0): continue
            if gf_pow(elem, 6) == (1, 0): continue
            if gf_pow(elem, 4) == (1, 0): continue
            if gf_pow(elem, 3) == (1, 0): continue
            if gf_pow(elem, 2) == (1, 0): continue
            primitive = elem
            break
        if primitive:
            break
            
    if primitive is None:
        return None

    # Generate all powers of primitive
    powers = []
    current = (1, 0)
    for i in range(24):
        powers.append(current)
        current = gf_mul(current, primitive)
        
    # Map element to log (index in powers)
    log_map = {}
    for i, elem in enumerate(powers):
        log_map[elem] = i
        
    # List of all 25 elements of GF(25)
    # 0 is (0,0). Non-zero are powers[0..23]
    # We need a stable ordering for vertices 0..24
    # Let vertex 0 be (0,0)
    # Let vertices 1..24 be powers[0..23]
    gf_elements = [(0,0)] + powers
    
    n = 26
    colors = []
    
    # Helper to get color for pair (i, j)
    # i, j are indices 0..25
    # 0..24 correspond to gf_elements[0..24]
    # 25 corresponds to infinity
    
    def get_color(i, j):
        # Handle infinity
        if i == 25 or j == 25:
            other = j if i == 25 else i
            if other == 0: # Edge (0, infinity)
                return 0
            else:
                elem = gf_elements[other]
                log_val = log_map[elem]
                return (log_val % 4) + 1
        
        # Both finite
        elem_i = gf_elements[i]
        elem_j = gf_elements[j]
        
        # If either is 0
        if elem_i == (0,0) or elem_j == (0,0):
            return 0
            
        # Both non-zero
        # Calculate ratio elem_i / elem_j
        # In GF(25), division is multiplication by inverse.
        # Inverse of x is x^(23). Or use log: log(x/y) = (log(x) - log(y)) % 24
        log_i = log_map[elem_i]
        log_j = log_map[elem_j]
        diff = (log_i - log_j) % 24
        return (diff % 4) + 1

    for i in range(n):
        for j in range(i + 1, n):
            colors.append(get_color(i, j))
    
    if _validate(n, colors):
        return {'n': n, 'colors': colors}
    return None

def _greedy_with_repair(n, max_iter=20000):
    """
    Greedy construction with local search to find a valid 5-colouring of K_n.
    """
    random.seed(42)
    
    m = n * (n - 1) // 2
    # Start with cyclic initialisation
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = min(j - i, n - j + i)
            colors.append((d - 1) % 5)
            
    def edge_idx(a, b):
        if a > b:
            a, b = b, a
        return a * (n - 1) - a * (a + 1) // 2 + (b - a - 1)
        
    for _ in range(max_iter):
        if _time_left() < 15:
            break
            
        # Find a violation
        nbrs = [[set() for _ in range(n)] for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                nbrs[c][i].add(j)
                nbrs[c][j].add(i)
                idx += 1
                
        violation = None
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = nbrs[c][i] & nbrs[c][j]
                    if len(common) >= 2:
                        violation = (c, i, j, list(common)[:2])
                        break
                if violation:
                    break
            if violation:
                break
                
        if violation is None:
            return {'n': n, 'colors': colors}
            
        c, i, j, common = violation
        edges_in_c4 = [(i, common[0]), (i, common[1]), (j, common[0]), (j, common[1])]
        random.shuffle(edges_in_c4)
        for a, b in edges_in_c4:
            idx = edge_idx(a, b)
            new_c = (colors[idx] + 1 + random.randint(0, 3)) % 5
            colors[idx] = new_c
            break
            
    if _validate(n, colors):
        return {'n': n, 'colors': colors}
    return None

def construct():
    """Entry point: return the best 5-colouring found."""
    if _time_left() <= 0:
        return {'n': 1, 'colors': [0]}
        
    # Try Lazebnik-Woldar construction for n=26
    lw = _lazebnik_woldar()
    if lw is not None:
        return lw
        
    # Fallback: try greedy for n=25, 24, etc.
    for n in [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if _time_left() < 10:
            break
        res = _greedy_with_repair(n, max_iter=5000)
        if res is not None:
            return res
            
    # Absolute fallback: K1
    return {'n': 1, 'colors': []}
