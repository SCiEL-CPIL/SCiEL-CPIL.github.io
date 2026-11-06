"""Improved program for edge-colourings of K16 with >=5 colours per K4."""
import time
import itertools

def _validate(coloring, n=16):
    """Check that every K4 spans at least 5 distinct colours."""
    # Build adjacency colour matrix
    col = [[-1]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            col[i][j] = col[j][i] = coloring[idx]
            idx += 1
    # Check all K4s
    for a, b, c, d in itertools.combinations(range(n), 4):
        colours = set()
        colours.add(col[a][b])
        colours.add(col[a][c])
        colours.add(col[a][d])
        colours.add(col[b][c])
        colours.add(col[b][d])
        colours.add(col[c][d])
        if len(colours) < 5:
            return False
    return True

def _count_colours(coloring):
    """Count distinct colours used."""
    return len(set(coloring))

def _try_modular_k(k, n=16):
    """Try (i + j) % k and check validity."""
    coloring = [(i + j) % k for i in range(n) for j in range(i + 1, n)]
    if _validate(coloring, n):
        return coloring, _count_colours(coloring)
    return None, None

def _try_modular_variant(a, b, k, n=16):
    """Try (a*i + b*j) % k."""
    coloring = [(a * i + b * j) % k for i in range(n) for j in range(i + 1, n)]
    if _validate(coloring, n):
        return coloring, _count_colours(coloring)
    return None, None

def _try_product_mod(k, n=16):
    """Try (i * j) % k."""
    coloring = [(i * j) % k for i in range(n) for j in range(i + 1, n)]
    if _validate(coloring, n):
        return coloring, _count_colours(coloring)
    return None, None

def _try_sum_diff_mod(k, n=16):
    """Try (i + j + abs(i-j)) % k = (2*max(i,j)) % k."""
    coloring = [(2 * max(i, j)) % k for i in range(n) for j in range(i + 1, n)]
    if _validate(coloring, n):
        return coloring, _count_colours(coloring)
    return None, None

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    best_coloring = None
    best_count = 17  # current baseline
    
    # Try to improve from the baseline
    # First, try the original construction
    baseline = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    if _validate(baseline):
        best_coloring = baseline
        best_count = 17
    
    # Try smaller k values with (i+j) % k
    for k in range(13, 17):
        if time.time() > deadline - 10:
            break
        coloring, count = _try_modular_k(k)
        if coloring is not None and count < best_count:
            best_coloring = coloring
            best_count = count
    
    # Try various linear combinations
    if time.time() < deadline - 20:
        for k in range(13, 16):
            if time.time() > deadline - 15:
                break
            for a in range(1, 5):
                for b in range(1, 5):
                    if time.time() > deadline - 12:
                        break
                    coloring, count = _try_modular_variant(a, b, k)
                    if coloring is not None and count < best_count:
                        best_coloring = coloring
                        best_count = count
    
    # Try product-based
    if time.time() < deadline - 10:
        for k in range(13, 16):
            if time.time() > deadline - 8:
                break
            coloring, count = _try_product_mod(k)
            if coloring is not None and count < best_count:
                best_coloring = coloring
                best_count = count
    
    # Try sum-diff based
    if time.time() < deadline - 5:
        for k in range(13, 16):
            if time.time() > deadline - 3:
                break
            coloring, count = _try_sum_diff_mod(k)
            if coloring is not None and count < best_count:
                best_coloring = coloring
                best_count = count
    
    if best_coloring is None:
        # Fallback to original
        best_coloring = baseline
    
    return best_coloring
# EVOLVE-BLOCK-END
