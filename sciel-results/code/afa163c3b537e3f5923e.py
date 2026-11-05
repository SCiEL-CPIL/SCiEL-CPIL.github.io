import time
import random

_START = time.time()
_LIMIT = 150.0

def _time_left():
    return _LIMIT - (time.time() - _START)

def _validate(n, colors):
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    nbrs = [[0] * n for _ in range(5)]
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

def _seed_construction():
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def _edge_idx(a, b, n):
    if a > b:
        a, b = b, a
    return a * (n - 1) - a * (a + 1) // 2 + (b - a - 1)

class _C4Counter:
    """Fast C4 counter with O(n) delta updates per edge recolor."""
    def __init__(self, n, colors):
        self.n = n
        self.nbrs = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                self.nbrs[c][i] |= (1 << j)
                self.nbrs[c][j] |= (1 << i)
                idx += 1
        self.count = self._total_c4s()

    def _total_c4s(self):
        total = 0
        n = self.n
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = self.nbrs[c][i] & self.nbrs[c][j]
                    cnt = bin(common).count('1')
                    if cnt >= 2:
                        total += cnt * (cnt - 1) // 2
        return total

    def recolor(self, e, old_c, new_c, n):
        """Recolor edge e from old_c to new_c. Returns delta in C4 count."""
        # Decode edge
        a = 0
        while a * (n - 1) - a * (a + 1) // 2 <= e:
            if a * (n - 1) - a * (a + 1) // 2 + (n - 1 - a - 1) >= e:
                b = e - (a * (n - 1) - a * (a + 1) // 2) + a + 1
                break
            a += 1
        if a > b:
            a, b = b, a

        # Remove from old_c
        self.nbrs[old_c][a] &= ~(1 << b)
        self.nbrs[old_c][b] &= ~(1 << a)
        delta_remove = self._delta_for_pair(a, b, old_c, -1)

        # Add to new_c
        self.nbrs[new_c][a] |= (1 << b)
        self.nbrs[new_c][b] |= (1 << a)
        delta_add = self._delta_for_pair(a, b, new_c, +1)

        self.count += delta_remove + delta_add
        return delta_remove + delta_add

    def _delta_for_pair(self, a, b, c, sign):
        """Compute change in C4 count when edge (a,b) is added (+1) or removed (-1) from color c."""
        delta = 0
        n = self.n
        # For each other vertex v, check if adding/removing edge (a,b) creates/destroys C4s
        # A C4 involves two pairs of vertices with two common neighbors.
        # When we add edge (a,b) to color c: for each v not in {a,b}, if v is neighbor of both a and b in c,
        # then (a,b,v) forms a triangle... no, C4 needs 4 vertices.
        # Actually: C4 is a-b-x-y-a. When we add edge (a,b), new C4s are formed with any pair (x,y)
        # where x is neighbor of a and b in c, and y is neighbor of a and b in c... no.
        # Let me think differently. The C4 count for color c is sum over all pairs (i,j) of C(common(i,j), 2).
        # When we add edge (a,b) to color c, for each vertex v != a,b:
        #   - a gains neighbor b, b gains neighbor a
        #   - For pair (a,v): common(a,v) gains b if v was neighbor of b in c
        #   - For pair (b,v): common(b,v) gains a if v was neighbor of a in c
        #   - For pair (x,y) where x!=a,b and y!=a,b: no change
        # So we need to check pairs involving a or b.
        
        # Pairs (a, v) for v != a, b:
        for v in range(n):
            if v == a or v == b:
                continue
            # Pair (a, v): was common = nbrs[c][a] & nbrs[c][v] (before change, but we already modified)
            # After adding: nbrs[c][a] includes b. So common_after includes b if v was neighbor of b.
            # common_before = (nbrs[c][a] without b) & nbrs[c][v]
            # common_after = (nbrs[c][a] with b) & nbrs[c][v] = common_before | {b if v~b}
            
            # For pair (a,v):
            if sign == +1:  # adding edge (a,b)
                # b is now in nbrs[c][a]. Check if b was already in common(a,v)
                # b in common(a,v) iff b in nbrs[c][a] and b in nbrs[c][v]
                # b in nbrs[c][a]: yes (just added)
                # b in nbrs[c][v]: check
                if self.nbrs[c][v] & (1 << b):
                    # b was not in common before (since a didn't have b as neighbor before)
                    # Now b is in common. C4s change: C(k,2) -> C(k+1,2) where k was old count
                    # We need old count of common(a,v)
                    old_common = (self.nbrs[c][a] & ~(1 << b)) & self.nbrs[c][v]
                    old_k = bin(old_common).count('1')
                    new_k = old_k + 1
                    delta += sign * (new_k * (new_k - 1) // 2 - old_k * (old_k - 1) // 2)
            else:  # removing edge (a,b)
                if self.nbrs[c][v] & (1 << b):
                    # b was in common, now removed
                    new_common = self.nbrs[c][a] & self.nbrs[c][v]  # a no longer has b
                    new_k = bin(new_common).count('1')
                    old_k = new_k + 1
                    delta += sign * (new_k * (new_k - 1) // 2 - old_k * (old_k - 1) // 2)
            
            # Pair (b, v):
            if sign == +1:
                if self.nbrs[c][v] & (1 << a):
                    old_common = (self.nbrs[c][b] & ~(1 << a)) & self.nbrs[c][v]
                    old_k = bin(old_common).count('1')
                    new_k = old_k + 1
                    delta += sign * (new_k * (new_k - 1) // 2 - old_k * (old_k - 1) // 2)
            else:
                if self.nbrs[c][v] & (1 << a):
                    new_common = self.nbrs[c][b] & self.nbrs[c][v]
                    new_k = bin(new_common).count('1')
                    old_k = new_k + 1
                    delta += sign * (new_k * (new_k - 1) // 2 - old_k * (old_k - 1) // 2)
        
        # Pair (a, b) itself:
        # common(a,b) = nbrs[c][a] & nbrs[c][b]
        if sign == +1:
            # After adding, a has b and b has a, but we don't count self-loops
            # common(a,b) doesn't include a or b themselves (no self-loops in our representation)
            # Actually nbrs[c][a] includes b now, and nbrs[c][b] includes a now.
            # common(a,b) = (nbrs[c][a] & nbrs[c][b]) - we should exclude a and b from consideration
            # But a is not in nbrs[c][a] (no self-loop), and b is not in nbrs[c][b].
            # However, b IS in nbrs[c][a] and a IS in nbrs[c][b].
            # So common(a,b) includes a and b? No: common = nbrs[c][a] & nbrs[c][b]
            # nbrs[c][a] has b, nbrs[c][b] has a. b is in nbrs[c][a] but is b in nbrs[c][b]? No.
            # a is in nbrs[c][b] but is a in nbrs[c][a]? No.
            # So a and b are NOT in common(a,b). Good.
            common = self.nbrs[c][a] & self.nbrs[c][b]
            # But wait, we need to exclude a and b from common for C4 counting?
            # A C4 is a-b-x-y-a. The common neighbors of a and b are x,y.
            # x and y should be distinct from a and b. Since no self-loops, a not in nbrs[c][a], etc.
            # So common already excludes a and b.
            k = bin(common).count('1')
            delta += sign * (k * (k - 1) // 2)
        else:
            common = self.nbrs[c][a] & self.nbrs[c][b]
            k = bin(common).count('1')
            delta += sign * (k * (k - 1) // 2)
        
        return delta

def _repair_search(n, max_time=30.0, seed=42):
    """Constraint-driven repair: find C4 violations and fix them."""
    random.seed(seed)
    m = n * (n - 1) // 2
    
    # Initial coloring: cyclic
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            d = min(d, n - d)
            colors.append((d - 1) % 5)
    
    counter = _C4Counter(n, colors)
    best_count = counter.count
    best_colors = colors[:]
    
    start_time = time.time()
    max_iter = 500000
    
    for it in range(max_iter):
        if time.time() - start_time > max_time:
            break
        if _time_left() < 20:
            break
        
        if counter.count == 0:
            return {'n': n, 'colors': colors}
        
        if counter.count < best_count:
            best_count = counter.count
            best_colors = colors[:]
        
        # Find a C4 violation
        violation = _find_violation(n, colors)
        if violation is None:
            return {'n': n, 'colors': colors}
        
        c, i, j, common = violation
        # Pick an edge from the C4 to recolor
        edges_in_c4 = [(i, common[0]), (i, common[1]), (j, common[0]), (j, common[1])]
        random.shuffle(edges_in_c4)
        
        for a, b in edges_in_c4:
            if a > b:
                a, b = b, a
            e = _edge_idx(a, b, n)
            old_c = colors[e]
            # Try all 4 other colors
            new_cs = [cc for cc in range(5) if cc != old_c]
            random.shuffle(new_cs)
            for new_c in new_cs:
                delta = counter.recolor(e, old_c, new_c, n)
                if delta < 0:
                    colors[e] = new_c
                    break
                else:
                    counter.recolor(e, new_c, old_c, n)  # undo
            else:
                # No improvement found, accept worst move with probability
                new_c = new_cs[0]
                delta = counter.recolor(e, old_c, new_c, n)
                colors[e] = new_c
                if random.random() > 0.3:
                    counter.recolor(e, new_c, old_c, n)  # undo
                    colors[e] = old_c
    
    if best_count == 0:
        return {'n': n, 'colors': best_colors}
    return None

def _find_violation(n, colors):
    """Find one C4 violation. Returns (color, i, j, [v1, v2]) or None."""
    nbrs = [[0] * n for _ in range(5)]
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
                    # Find two vertices in common
                    bits = []
                    mask = common
                    while mask and len(bits) < 2:
                        lsb = mask & (-mask)
                        bits.append(lsb.bit_length() - 1)
                        mask &= mask - 1
                    return (c, i, j, bits)
    return None

def _try_z2z13_construction():
    """Try Z_2 x Z_13 based construction for n=26."""
    n = 26
    # Quadratic residues mod 13
    qr = {1, 3, 4, 9, 10, 12}
    
    # Try multiple colorings
    strategies = []
    
    # Strategy: use 5 characters of Z_13
    # Z_13 has a cyclic group of order 12. We need to partition {1..12} into 5 sets
    # such that no C4 forms.
    
    # Try: color based on (dy mod 13) mapped through a function
    for perm_seed in range(20):
        random.seed(perm_seed)
        # Create a random permutation of {1,...,12}
        elems = list(range(1, 13))
        random.shuffle(elems)
        # Map to 5 colors: first 2-3 elements get each color
        # 12 elements into 5 colors: sizes like [3,3,2,2,2]
        color_map = {}
        sizes = [3, 3, 2, 2, 2]
        idx = 0
        for c in range(5):
            for k in range(sizes[c]):
                color_map[elems[idx]] = c
                idx += 1
        
        colors = []
        for i in range(n):
            x1 = i % 2
            y1 = i // 2
            for j in range(i + 1, n):
                x2 = j % 2
                y2 = j // 2
                dx = (x2 - x1) % 2
                dy = (y2 - y1) % 13
                if dy == 0:
                    # Same y, different x: color based on... use a fixed color
                    colors.append(0)
                elif dx == 0:
                    # Same layer
                    colors.append(color_map[dy])
                else:
                    # Different layers: use a different mapping
                    # Color based on dy with offset
                    colors.append(color_map[dy])
        
        if _validate(n, colors):
            return {'n': n, 'colors': colors}
    
    return None

def _try_gf25_construction():
    """Try GF(25) based construction for n=26."""
    p = 5
    
    def gf25_mul(x, y):
        a1, b1 = x
        a2, b2 = y
        a = (a1 * a2 + 3 * b1 * b2) % p
        b = (a1 * b2 + a2 * b1) % p
        return (a, b)
    
    def gf25_add(x, y):
        return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)
    
    def gf25_sub(x, y):
        return ((x[0] - y[0]) % p, (x[1] - y[1]) % p)
    
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
    
    # Try multiple strategies for coloring
    for strat in range(30):
        random.seed(strat)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i == 25 or j == 25:
                    other = j if i == 25 else i
                    elem = gf25_elements[other]
                    if elem in gf5_elements:
                        c = 0
                    else:
                        k = log_map[elem]
                        c = 1 + (k % 4)
                    colors.append(c)
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
                        # Try different mappings
                        if strat < 10:
                            c = 1 + (k % 4)
                        elif strat < 20:
                            c = 1 + ((k // 5) % 4)
                        else:
                            c = 1 + (k % 4)
                            if c == 0:
                                c = 1
                        colors.append(c)
        
        if _validate(n, colors):
            return {'n': n, 'colors': colors}
    
    return None

def _try_cyclic_variants(n):
    """Try various cyclic-like constructions."""
    for variant in range(50):
        random.seed(variant)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                d = min(d, n - d)
                if variant < 25:
                    c = (d - 1) % 5
                else:
                    # Perturbed cyclic
                    c = (d - 1 + (i % 5) * (variant - 25) // 5) % 5
                colors.append(c)
        if _validate(n, colors):
            return {'n': n, 'colors': colors}
    return None

def construct():
    if _time_left() <= 5:
        return _seed_construction()
    
    best = _seed_construction()
    
    # Try algebraic constructions for n=26
    if _time_left() > 30:
        lw = _try_gf25_construction()
        if lw is not None:
            return lw
    
    if _time_left() > 25:
        z2z = _try_z2z13_construction()
        if z2z is not None:
            return z2z
    
    # Try repair search for n=26
    if _time_left() > 40:
        for seed in [42, 123, 777, 999]:
            if _time_left() < 25:
                break
            res = _repair_search(26, max_time=min(15.0, _time_left() - 15.0), seed=seed)
            if res is not None:
                return res
    
    # Try cyclic variants for various n
    for n in [25, 20, 15, 11]:
        if _time_left() < 10:
            break
        if n <= best['n']:
            break
        res = _try_cyclic_variants(n)
        if res is not None:
            best = res
    
    # Try repair search for smaller n
    for n in [25, 24, 23, 22, 21, 20]:
        if _time_left() < 15:
            break
        if n <= best['n']:
            break
        for seed in [42, 123]:
            if _time_left() < 12:
                break
            res = _repair_search(n, max_time=min(8.0, _time_left() - 10.0), seed=seed)
            if res is not None:
                best = res
                break
    
    return best
