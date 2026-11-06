"""R_4(C6): maximize n with 4-colouring of K_n with no monochromatic C6.
Uses cyclic group construction over Z_p for prime p.
"""
import itertools
import random
import time

def _c6_in_circulant(residues, n):
    """Check if circulant graph on Z_n with connection set residues has a C6."""
    if len(residues) < 2:
        return False
    rlist = list(set(residues))
    for steps in itertools.product(rlist, repeat=6):
        if sum(steps) % n != 0:
            continue
        acc = 0
        partials = []
        valid = True
        for s in steps:
            acc = (acc + s) % n
            if acc == 0:
                valid = False
                break
            partials.append(acc)
        if valid and len(set(partials)) == 6:
            return True
    return False

def _find_circulant_coloring(n, time_limit=100):
    """Find a 4-coloring of Z_p edges (by difference) with no mono C6."""
    start = time.time()
    
    if (n - 1) % 2 != 0:
        return None
    
    num_pairs = (n - 1) // 2
    pairs = [(r, n - r) for r in range(1, num_pairs + 1)]
    
    for attempt in range(5000):
        if time.time() - start > time_limit:
            break
        
        assignment = [random.randint(0, 3) for _ in range(num_pairs)]
        
        color_sets = [set() for _ in range(4)]
        for i, (r, nr) in enumerate(pairs):
            c = assignment[i]
            color_sets[c].add(r)
            color_sets[c].add(nr)
        
        valid = True
        for c in range(4):
            if _c6_in_circulant(color_sets[c], n):
                valid = False
                break
        
        if valid:
            color_of = [0] * n
            for i, (r, nr) in enumerate(pairs):
                color_of[r] = assignment[i]
                color_of[nr] = assignment[i]
            
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = (j - i) % n
                    colors.append(color_of[d])
            return colors
        
        if not valid:
            for _ in range(20):
                if time.time() - start > time_limit:
                    break
                inv_c = None
                for c in range(4):
                    if _c6_in_circulant(color_sets[c], n):
                        inv_c = c
                        break
                if inv_c is None:
                    break
                
                improved = False
                for other in range(4):
                    if other == inv_c:
                        continue
                    for i in range(num_pairs):
                        if assignment[i] != inv_c:
                            continue
                        for j in range(num_pairs):
                            if assignment[j] != other:
                                continue
                            assignment[i], assignment[j] = other, inv_c
                            color_sets = [set() for _ in range(4)]
                            for k, (r, nr) in enumerate(pairs):
                                color_sets[assignment[k]].add(r)
                                color_sets[assignment[k]].add(nr)
                            ok = True
                            for c in range(4):
                                if _c6_in_circulant(color_sets[c], n):
                                    ok = False
                                    break
                            if ok:
                                color_of = [0] * n
                                for k, (r, nr) in enumerate(pairs):
                                    color_of[r] = assignment[k]
                                    color_of[nr] = assignment[k]
                                colors = []
                                for i in range(n):
                                    for j in range(i + 1, n):
                                        d = (j - i) % n
                                        colors.append(color_of[d])
                                return colors
                            assignment[i], assignment[j] = inv_c, other
                            improved = True
                if not improved:
                    break
    
    return None

def _verify(colors, n):
    """Verify no monochromatic C6."""
    adj = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add((i, j))
            idx += 1
    
    for c in range(4):
        if _has_c6(adj[c], n):
            return False
    return True

def _has_c6(adj, n):
    """Check if graph contains a C6."""
    if len(adj) < 6:
        return False
    adjmat = [[False] * n for _ in range(n)]
    for (u, v) in adj:
        adjmat[u][v] = True
        adjmat[v][u] = True
    
    from itertools import combinations, permutations
    for subset in combinations(range(n), 6):
        v0 = subset[0]
        rest = list(subset[1:])
        for perm in permutations(rest):
            v = [v0] + list(perm)
            ok = True
            for i in range(6):
                if not adjmat[v[i]][v[(i+1) % 6]]:
                    ok = False
                    break
            if ok:
                return True
    return False

def construct():
    """Find the largest n with a valid 4-colouring of K_n with no monochromatic C6."""
    
    primes = [17, 13, 11, 7, 5, 3]
    
    for n in primes:
        if n <= 2:
            continue
        
        start = time.time()
        remaining = 115 - (time.time() - start)
        
        random.seed(42)
        colors = _find_circulant_coloring(n, time_limit=remaining)
        
        if colors is not None:
            if _verify(colors, n):
                return {'n': n, 'colors': colors}
    
    for n in [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        start = time.time()
        for attempt in range(200):
            if time.time() - start > 5:
                break
            random.seed(attempt)
            colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
            if _verify(colors, n):
                return {'n': n, 'colors': colors}
    
    return {'n': 1, 'colors': []}
