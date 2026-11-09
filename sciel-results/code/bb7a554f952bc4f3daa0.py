"""Program for finding colourings of K_n without monochromatic K2,2,2."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _popcount(x):
    """Count set bits in integer x."""
    return bin(x).count('1')

def _get_vertices(mask, n):
    """Get list of vertices from a bitmask."""
    verts = []
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
    return verts

def _contains_K222(neigh, n):
    """
    Check if the graph defined by neighbor bitmasks contains K2,2,2.
    neigh[v] is a bitmask of neighbors of vertex v.
    Uses bit operations for efficiency.
    """
    for a in range(n):
        na = neigh[a]
        for b in range(a + 1, n):
            S = na & neigh[b]
            # Need at least 4 common neighbors
            if _popcount(S) < 4:
                continue
            # Get vertices in S
            verts_in_S = _get_vertices(S, n)
            L = len(verts_in_S)
            # For each pair (c, d) in S, check if N(c) & N(d) & S has >= 2 bits
            for i_c in range(L):
                c = verts_in_S[i_c]
                nc = neigh[c] & S
                for i_d in range(i_c + 1, L):
                    d = verts_in_S[i_d]
                    T = nc & neigh[d]
                    if _popcount(T) >= 2:
                        return True
    return False

def _build_paley(p):
    """
    Build the Paley 2-colouring of K_p for prime p ≡ 1 (mod 4).
    Returns (colors, neigh0, neigh1) where:
      - colors is list of 0/1 for each edge (i,j) i<j in lex order
      - neigh0 is list of bitmasks for color-0 graph
      - neigh1 is list of bitmasks for color-1 graph
    """
    squares = set(x * x % p for x in range(1, p))
    colors = []
    neigh0 = [0] * p
    neigh1 = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            d = (j - i) % p
            if d in squares:
                c = 0
                # Edge in color 0
                neigh0[i] |= (1 << j)
                neigh0[j] |= (1 << i)
            else:
                c = 1
                # Edge in color 1
                neigh1[i] |= (1 << j)
                neigh1[j] |= (1 << i)
            colors.append(c)
    return colors, neigh0, neigh1

def _random_coloring(n, seed=None):
    """Generate a random 2-coloring of K_n."""
    if seed is not None:
        random.seed(seed)
    colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
    return colors

def _colors_to_neigh(colors, n):
    """Convert color list to neighbor bitmasks for both color classes."""
    neigh0 = [0] * n
    neigh1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                neigh0[i] |= (1 << j)
                neigh0[j] |= (1 << i)
            else:
                neigh1[i] |= (1 << j)
                neigh1[j] |= (1 << i)
            idx += 1
    return neigh0, neigh1

def _is_valid_coloring(colors, n):
    """Check if a coloring is valid (no K2,2,2 in either color class)."""
    neigh0, neigh1 = _colors_to_neigh(colors, n)
    if _contains_K222(neigh0, n):
        return False
    if _contains_K222(neigh1, n):
        return False
    return True

def construct():
    start_time = time.time()
    DEADLINE = start_time + 135.0  # 15s safety margin for 150s limit
    
    best_result = None
    best_n = 0
    
    # Try primes p ≡ 1 (mod 4) in decreasing order: 29, 17, 13, 5
    primes_to_try = [29, 17, 13, 5]
    
    for p in primes_to_try:
        if time.time() > DEADLINE:
            break
        colors, neigh0, neigh1 = _build_paley(p)
        # Check color-1 graph
        if _contains_K222(neigh1, p):
            continue
        # Check color-0 graph
        if _contains_K222(neigh0, p):
            continue
        # Valid!
        if p > best_n:
            best_n = p
            best_result = {'n': p, 'colors': colors}
        break  # Found valid coloring, no need to check smaller primes
    
    # If we found n=29, try random search for n=30 (proves R=31)
    if best_n == 29:
        remaining_time = DEADLINE - time.time()
        if remaining_time > 5.0:
            # Try a few random colorings of K_30
            num_attempts = int(remaining_time * 10)  # ~10 attempts per second
            for attempt in range(num_attempts):
                if time.time() > DEADLINE:
                    break
                colors30 = _random_coloring(30, seed=start_time * 1000 + attempt)
                if _is_valid_coloring(colors30, 30):
                    return {'n': 30, 'colors': colors30}
    
    # If we didn't find n=29, try random search for n=17, 18, etc.
    if best_n < 29:
        remaining_time = DEADLINE - time.time()
        if remaining_time > 3.0:
            # Try random colorings for n = best_n+1 up to 20
            target_n = min(best_n + 1, 20)
            num_attempts = int(remaining_time * 5)
            for attempt in range(num_attempts):
                if time.time() > DEADLINE:
                    break
                colors_target = _random_coloring(target_n, seed=start_time * 1000 + attempt)
                if _is_valid_coloring(colors_target, target_n):
                    if target_n > best_n:
                        best_n = target_n
                        best_result = {'n': target_n, 'colors': colors_target}
                    break
    
    # Return best result found
    if best_result is not None:
        return best_result
    
    # Fallback: n=1
    return {'n': 1, 'colors': []}

# EVOLVE-BLOCK-END
