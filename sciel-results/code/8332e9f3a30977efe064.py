"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _build_index(n):
    """Build mapping from triple (a,b,c) to index in lexicographic order."""
    triples = list(combinations(range(n), 3))
    idx = {}
    for i, t in enumerate(triples):
        idx[t] = i
    return triples, idx

def _count_red_k4(colors, n, triples, idx):
    """Count number of red K4^(3) violations."""
    count = 0
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        if (colors[idx[(a,b,c)]] == 0 and 
            colors[idx[(a,b,d)]] == 0 and 
            colors[idx[(a,c,d)]] == 0 and 
            colors[idx[(b,c,d)]] == 0):
            count += 1
    return count

def _count_blue_k5(colors, n, triples, idx):
    """Count number of blue K5^(3) violations."""
    count = 0
    for quint in combinations(range(n), 5):
        all_blue = True
        for tri in combinations(quint, 3):
            if colors[idx[tri]] != 1:
                all_blue = False
                break
        if all_blue:
            count += 1
    return count

def _is_valid(colors, n):
    """Check if coloring is valid (no red K4, no blue K5)."""
    triples, idx = _build_index(n)
    if _count_red_k4(colors, n, triples, idx) > 0:
        return False
    if _count_blue_k5(colors, n, triples, idx) > 0:
        return False
    return True

def _local_search(n, max_iter=50000, deadline=None):
    """Local search to find valid coloring for n vertices."""
    triples, idx = _build_index(n)
    m = len(triples)
    
    # Start with random coloring
    colors = bytearray(random.choice([0, 1]) for _ in range(m))
    
    def violations():
        r = _count_red_k4(colors, n, triples, idx)
        b = _count_blue_k5(colors, n, triples, idx)
        return r, b
    
    best_colors = colors.copy()
    best_viol = sum(violations())
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        r, b = violations()
        total = r + b
        if total == 0:
            best_colors = colors.copy()
            break
        
        if total < best_viol:
            best_colors = colors.copy()
            best_viol = total
        
        # Pick a random triple to flip
        i = random.randint(0, m - 1)
        colors[i] = 1 - colors[i]
    
    if best_viol == 0:
        return best_colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Try increasing n values
    for n in range(34, 8, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for this n
        found = None
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            random.seed(42 + n * 100 + restart)
            result = _local_search(n, max_iter=20000, deadline=deadline)
            if result is not None:
                found = (n, result)
                break
        
        if found:
            n_found, colors_found = found
            triples, _ = _build_index(n_found)
            return {'n': n_found, 'colors': list(colors_found)}
    
    # Fallback: use the simple modular coloring for small n
    n = 9
    triples = list(combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
