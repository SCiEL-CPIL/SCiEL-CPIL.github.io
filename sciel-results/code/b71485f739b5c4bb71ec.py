"""Four-colourings of K_n without monochromatic triangles.

Strategy:
- Primary: Construct 42-vertex coloring using GF(43) with 4-color partition.
- Secondary: Try to extend to larger n via local search.
- Fallback: 32-vertex Clebsch-based construction.
- Deadline guard: return best valid result before 120 s.
- Fast numpy-based verification for local search.
"""

import time
import random
import sys
import numpy as np

# ---------------------------------------------------------------------------
# GF(43) construction for 42 vertices using cubic residues
# ---------------------------------------------------------------------------

def _gf43_construction():
    """
    Construct a 4-coloring of K_42 using GF(43).
    
    GF(43)* has order 42. We use a 4-color scheme based on the
    multiplicative structure to avoid monochromatic triangles.
    
    Key insight: For a cyclic group of order 42, if we partition into
    4 sets such that no three elements in the same set form a "triangle"
    (i.e., a/b and b/c both in same set implies a/c not in that set),
    we get a triangle-free coloring.
    
    Use cosets of subgroup of order 7 (index 6), then split into 4 colors.
    Actually, let's use the known construction: partition GF(43)* into
    4 sets based on discrete log mod 4, ensuring triangle-freeness.
    """
    p = 43
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Find a primitive root mod 43
    # 42 = 2 * 3 * 7, so check g^(42/2), g^(42/3), g^(42/7) != 1
    primitive = None
    for g in range(2, p):
        if pow(g, 42 // 2, p) != 1 and pow(g, 42 // 3, p) != 1 and pow(g, 42 // 7, p) != 1:
            primitive = g
            break
    
    if primitive is None:
        return None
    
    # Build discrete log table
    dlog = {}
    current = 1
    for i in range(42):
        dlog[current] = i
        current = (current * primitive) % p
    
    # Color based on discrete log mod 4
    # This gives 4 color classes
    def get_color(elem):
        if elem == 0:
            return 3  # Should not happen for nonzero
        dl = dlog[elem]
        return dl % 4
    
    # Build colors for K_42: vertices 0..41 map to elements 1..42
    colors = []
    for i in range(42):
        for j in range(i + 1, 42):
            # Vertices i, j represent elements (i+1), (j+1)
            # Color based on (i+1) * (j+1)^{-1} mod p
            ratio = ((i + 1) * inv[j + 1]) % p
            c = get_color(ratio)
            colors.append((i, j, c))
    
    return colors

# ---------------------------------------------------------------------------
# GF(7^2) construction for 49 vertices
# ---------------------------------------------------------------------------

def _gf49_construction():
    """
    Construct a 4-coloring of K_49 using GF(7^2).
    
    Uses additive structure with 4-color partition of nonzero elements.
    """
    # Build GF(7^2) = GF(7)[x] / (x^2 + 1)
    def gf49_add(x, y):
        return ((x[0] + y[0]) % 7, (x[1] + y[1]) % 7)
    
    def gf49_sub(x, y):
        return ((x[0] - y[0]) % 7, (x[1] - y[1]) % 7)
    
    def gf49_mul(x, y):
        a, b = x
        c, d = y
        real = (a * c + 6 * b * d) % 7
        imag = (a * d + b * c) % 7
        return (real, imag)
    
    elements = [(a, b) for a in range(7) for b in range(7)]
    
    def gf49_pow(x, n):
        result = (1, 0)
        base = x
        while n > 0:
            if n & 1:
                result = gf49_mul(result, base)
            base = gf49_mul(base, base)
            n >>= 1
        return result
    
    # Find primitive element (order 48)
    primitive = None
    for e in elements:
        if e == (0, 0):
            continue
        if gf49_pow(e, 24) != (1, 0) and gf49_pow(e, 16) != (1, 0):
            primitive = e
            break
    
    if primitive is None:
        return None
    
    # Build discrete log for nonzero elements
    dlog = {}
    current = (1, 0)
    for i in range(48):
        dlog[current] = i
        current = gf49_mul(current, primitive)
    
    # Use 4-color scheme based on discrete log mod 4
    def get_color(elem):
        if elem == (0, 0):
            return 3
        dl = dlog.get(elem, 0)
        return dl % 4
    
    colors = []
    for i in range(49):
        for j in range(i + 1, 49):
            diff = gf49_sub(elements[i], elements[j])
            c = get_color(diff)
            colors.append((i, j, c))
    
    return colors

# ---------------------------------------------------------------------------
# Fast triangle-free verification using bitsets
# ---------------------------------------------------------------------------

def _is_triangle_free_fast(colors, n):
    """
    colors: list of (i, j, c) in lex order, i < j.
    Returns True iff no monochromatic triangle.
    Uses bitset adjacency for speed.
    """
    if n <= 1:
        return True
    
    # adj[c][v] = bitmask of neighbors of v in color c
    adj = [[0] * n for _ in range(4)]
    
    for i, j, c in colors:
        adj[c][i] |= (1 << j)
        adj[c][j] |= (1 << i)
    
    # For each color, check each edge (v, u) where u > v
    # If they share a common neighbor w, triangle exists
    for c in range(4):
        for v in range(n):
            nbrs_v = adj[c][v]
            # Only check neighbors u > v to avoid double counting
            n1 = nbrs_v & ~((1 << (v + 1)) - 1)
            while n1:
                b1 = n1 & (-n1)
                n1 -= b1
                u = b1.bit_length() - 1
                common = adj[c][v] & adj[c][u]
                if common:
                    return False
    return True


def _build_adj_matrices(colors, n):
    """Build adjacency matrices for fast triangle checking.
    Returns list of 4 numpy boolean matrices, one per color.
    """
    adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    for i, j, c in colors:
        adj[c][i, j] = True
        adj[c][j, i] = True
    return adj


def _is_triangle_free_numpy(adj, n):
    """Check triangle-freeness using numpy matrix operations.
    adj: list of 4 (n,n) boolean numpy arrays.
    Returns True iff no monochromatic triangle.
    """
    for c in range(4):
        A = adj[c]
        # Triangle exists if (A @ A) has any entry > 0 where A is also 1
        # More precisely: for each edge (i,j) in color c, check if any common neighbor
        # Use: (A @ A.T) > 0 indicates common neighbors, but we need to check
        # that the common neighbor k has A[k,j] and A[k,i] both true
        # Actually: triangle (i,j,k) in color c means A[i,j] & A[j,k] & A[k,i]
        # For each edge (i,j), count common neighbors: sum over k of A[i,k] & A[j,k]
        # If any edge has a common neighbor, triangle exists
        # Efficient: (A @ A.T)[i,j] = number of common neighbors of i and j
        # Triangle exists iff there's an edge (i,j) with (A @ A.T)[i,j] > 0
        # But we need to be careful: A[i,j] must be true
        common = A @ A.T
        # For each edge, check if common neighbors exist
        # np.any(common * A > 0) but this counts i-j pairs
        # Actually if A[i,j] is true and common[i,j] > 0, then triangle exists
        if np.any(common * A):
            return False
    return True

# ---------------------------------------------------------------------------
# GF(2,4) multiplication table for the Clebsch-graph base construction
# ---------------------------------------------------------------------------

def _build_mul_table():
    """Build multiplication in GF(2^4) with irreducible x^4+x+1 (0b10011)."""
    log = {}
    mul = {}
    x = 1
    log[1] = 0
    for e in range(15):
        x2 = 0
        for i in range(4):
            if (x >> i) & 1:
                x2 ^= (1 << (i + 1))
        # reduce
        if x2 & 0b10000:
            x2 ^= 0b10011
        x = x2
        log[x] = e + 1
    return log

# Precompute once
_log = _build_mul_table()

def _base32_colors():
    """Return the 32-vertex base coloring as a list of (i,j,color) in lex order."""
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                # same block: Clebsch
                c = _log[i % 16 ^ j % 16] % 3
            else:
                c = 3
            colors.append((i, j, c))
    return colors

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    deadline = time.time() + 115  # 115s budget, safety margin for 120s limit
    
    # Try constructions in order of expected size
    best_colors = None
    best_n = 0
    
    # Try GF(49) construction (49 vertices)
    gf49_colors = _gf49_construction()
    if gf49_colors is not None:
        if _is_triangle_free_fast(gf49_colors, 49):
            best_colors = gf49_colors
            best_n = 49
    
    # Try GF(43) construction (42 vertices)
    if best_n < 42:
        gf43_colors = _gf43_construction()
        if gf43_colors is not None:
            if _is_triangle_free_fast(gf43_colors, 42):
                best_colors = gf43_colors
                best_n = 42
    
    # Fallback to 32-vertex base
    if best_n < 32:
        best_colors = _base32_colors()
        best_n = 32
    
    # Try to extend beyond current best_n using local search
    if best_n < 50 and time.time() < deadline - 15 and best_colors is not None:
        # Build initial adjacency as bitsets for fast checking
        adj_bitsets = [[0] * best_n for _ in range(4)]
        for i, j, c in best_colors:
            adj_bitsets[c][i] |= (1 << j)
            adj_bitsets[c][j] |= (1 << i)
        
        def _count_conflicts(assign, current_n):
            """Count monochromatic triangles involving the new vertex."""
            conflicts = 0
            new_v = current_n
            # For each pair (i, j) where i < j < current_n,
            # if assign[i] == assign[j] == assign[k] for some k, that's a conflict
            # Actually: for each edge (i,j) in color c, if new vertex connects to both in color c, conflict
            for c in range(4):
                # Get neighbors of new vertex in color c
                nbrs_new = 0
                for i in range(current_n):
                    if assign[i] == c:
                        nbrs_new |= (1 << i)
                # For each pair of neighbors in color c, check if they're connected in color c
                n1 = nbrs_new
                while n1:
                    b1 = n1 & (-n1)
                    n1 -= b1
                    i = b1.bit_length() - 1
                    # Common neighbors of i and new vertex in color c
                    common = adj_bitsets[c][i] & nbrs_new
                    if common:
                        conflicts += common.bit_count()
            return conflicts
        
        def _try_extend(current_n):
            """Try to extend coloring to current_n+1 vertices."""
            new_n = current_n + 1
            best_assign = None
            best_conflicts = float('inf')
            
            # Multiple random restarts
            for trial in range(30):
                if time.time() > deadline - 10:
                    break
                
                # Random initial assignment
                assign = np.random.randint(0, 4, size=current_n)
                
                # Local search to minimize conflicts
                for it in range(5000):
                    if time.time() > deadline - 10:
                        break
                    
                    conflicts = _count_conflicts(assign, current_n)
                    if conflicts == 0:
                        return assign, new_n
                    
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_assign = assign.copy()
                    
                    # Try flipping a random edge
                    pos = random.randint(0, current_n - 1)
                    old_c = assign[pos]
                    new_c = (old_c + random.randint(1, 3)) % 4
                    assign[pos] = new_c
                    
                    new_conflicts = _count_conflicts(assign, current_n)
                    if new_conflicts >= conflicts and random.random() > 0.1:
                        # Revert if worse (with some probability of accepting)
                        assign[pos] = old_c
                
                if best_assign is not None:
                    assign = best_assign
                    best_conflicts = _count_conflicts(assign, current_n)
            
            if best_assign is not None and _count_conflicts(best_assign, current_n) == 0:
                return best_assign, new_n
            
            return None, None
        
        # Try to extend step by step
        while time.time() < deadline - 10:
            assign, new_n = _try_extend(best_n)
            if assign is not None:
                # Build new colors
                new_colors = list(best_colors)
                for i in range(best_n):
                    new_colors.append((i, new_n - 1, int(assign[i])))
                best_colors = new_colors
                best_n = new_n
                # Update bitsets
                for i in range(best_n - 1):
                    c = int(assign[i])
                    adj_bitsets[c][i] |= (1 << (best_n - 1))
                    adj_bitsets[c][best_n - 1] |= (1 << i)
            else:
                break
    
    # Format output
    out_colors = []
    for i, j, c in best_colors:
        out_colors.append(c)
    
    return {'n': best_n, 'colors': out_colors}
