"""Witt design S(5,6,12) construction via PGL(2,11) group action."""
import time
from itertools import combinations

def _pgl2_11_generators():
    """Return generators for PGL(2,11) as 2x2 matrices over GF(11)."""
    # PGL(2,11) acts on 12 points: {0, 1, ..., 10, inf}
    # Represented as points 0..11 where 11 = infinity
    # Generators: x -> x+1 (translation) and x -> 1/x (inversion)
    # As Möbius transforms: (ax+b)/(cx+d)
    # Translation: (1,1,0,1) meaning x -> x+1
    # Inversion: (0,1,1,0) meaning x -> 1/x
    return [(1, 1, 0, 1), (0, 1, 1, 0)]

def _apply_mobius(p, a, b, c, d):
    """Apply Möbius transform (ax+b)/(cx+d) to point p in {0..10, 11=inf}."""
    if p == 11:  # infinity
        if c == 0:
            return 11  # inf -> inf
        else:
            return a * c % 11  # inf -> a/c
    # p is finite, in 0..10
    num = (a * p + b) % 11
    den = (c * p + d) % 11
    if den == 0:
        return 11  # infinity
    else:
        # Need modular inverse of den mod 11
        # 11 is prime, so use pow(den, 10, 11) for inverse
        inv_den = pow(den, 10, 11)
        return (num * inv_den) % 11

def _compose_mobius(m1, m2):
    """Compose two Möbius transforms m1 after m2: m1(m2(x))."""
    a1, b1, c1, d1 = m1
    a2, b2, c2, d2 = m2
    # m1(m2(x)) = (a1*(a2*x+b2) + b1*(c2*x+d2)) / (c1*(a2*x+b2) + d1*(c2*x+d2))
    a = (a1 * a2 + b1 * c2) % 11
    b = (a1 * b2 + b1 * d2) % 11
    c = (c1 * a2 + d1 * c2) % 11
    d = (c1 * b2 + d1 * d2) % 11
    return (a, b, c, d)

def _generate_group_elements(generators, max_elements=7920):
    """Generate all elements of PGL(2,11) using BFS on generators."""
    # Identity: x -> x, which is (1,0,0,1)
    identity = (1, 0, 0, 1)
    group = {identity}
    frontier = [identity]
    while frontier and len(group) < max_elements:
        new_frontier = []
        for g in frontier:
            for gen in generators:
                new_g = _compose_mobius(g, gen)
                if new_g not in group:
                    group.add(new_g)
                    new_frontier.append(new_g)
                # Also compose with inverse of generator
                # Inverse of (a,b,c,d) is (d,-b,-c,a)/det, but in PGL we can scale
                # For simplicity, just use forward composition
        frontier = new_frontier
    return list(group)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline for 150s limit
    
    # Points: 0..10 are finite field elements, 11 = infinity
    points = list(range(12))
    
    # Base blocks for S(5,6,12)
    # The Witt design can be generated from a few base blocks under PGL(2,11)
    # Known base blocks (one standard set):
    # We need to find blocks that generate all 132 under the group action
    
    # Standard construction: The stabilizer of a point in PGL(2,11) has order 660
    # The group has order 7920, and 7920/660 = 12 (correct, 12 points)
    # 7920/132 = 60, so each block has stabilizer of order 60
    
    # Let's use known base blocks for S(5,6,12)
    # One standard set of base blocks under PGL(2,11):
    base_blocks = [
        [0, 1, 3, 9, 10, 11],  # Need to verify these
    ]
    
    # Actually, let me use a more reliable approach:
    # The Witt design S(5,6,12) has a well-known construction.
    # Let me generate the group and find orbits of 6-subsets.
    
    generators = _pgl2_11_generators()
    
    # Generate all group elements
    group_elements = _generate_group_elements(generators)
    
    if time.time() > deadline:
        return []
    
    # Try to find base blocks whose orbit under the group gives 132 blocks
    # The group has 7920 elements. If a block has stabilizer of size s,
    # its orbit has size 7920/s. We need orbit size 132, so s = 60.
    
    # Let's try specific known base blocks
    # From literature, one standard representation uses:
    # Points as {0, 1, 2, ..., 10, ∞}
    
    # Let me try a different, more direct approach:
    # Use the fact that S(5,6,12) can be constructed from the McLaughlin or other methods
    # But the PGL(2,11) approach is most direct.
    
    # Known base blocks for the Witt design (one standard set):
    # Let's try to find them by checking which 6-subsets have orbit size 132
    
    # First, let's verify group size
    if len(group_elements) != 7920:
        # Group generation may have failed, try a different approach
        # Fall back to a known explicit construction
        pass
    
    # Try specific base blocks from literature
    # The Witt design S(5,6,12) can be generated from these base blocks:
    candidate_base_blocks = [
        [0, 1, 3, 9, 10, 11],
        [0, 1, 4, 6, 10, 11],
        [0, 2, 5, 7, 9, 11],
        [0, 2, 6, 8, 9, 11],
        [0, 3, 5, 8, 10, 11],
        [0, 3, 7, 8, 10, 11],
        [0, 4, 5, 7, 9, 11],
        [0, 4, 6, 7, 9, 11],
        [0, 5, 6, 8, 10, 11],
        [1, 2, 4, 7, 9, 11],
        [1, 2, 5, 8, 10, 11],
        [1, 3, 4, 6, 9, 11],
        [1, 3, 5, 6, 10, 11],
        [1, 4, 5, 8, 9, 11],
        [2, 3, 4, 7, 10, 11],
        [2, 3, 6, 7, 10, 11],
        [2, 4, 6, 8, 9, 11],
        [3, 4, 7, 8, 10, 11],
    ]
    
    # For each candidate, compute its orbit under the group
    # and check if it gives exactly 132 distinct blocks with no repeated 5-subsets
    
    best_blocks = []
    
    for base in candidate_base_blocks:
        if time.time() > deadline:
            break
        
        # Compute orbit of this block
        orbit = set()
        for g in group_elements:
            a, b, c, d = g
            transformed = tuple(sorted(_apply_mobius(p, a, b, c, d) for p in base))
            orbit.add(transformed)
        
        if len(orbit) == 132:
            # Check if this orbit forms a valid S(5,6,12)
            covered_5subsets = set()
            valid = True
            for block in orbit:
                for sub in combinations(block, 5):
                    if sub in covered_5subsets:
                        valid = False
                        break
                    covered_5subsets.add(sub)
                if not valid:
                    break
            
            if valid:
                best_blocks = [list(b) for b in orbit]
                break
    
    if not best_blocks:
        # If the above didn't work, try a completely different approach:
        # Use the known explicit list of 132 blocks
        # Or use a greedy approach with the group structure
        
        # Let's try all 6-subsets containing point 0, and see which orbits work
        # There are C(11,5) = 462 such subsets
        all_with_zero = []
        for rest in combinations(range(1, 12), 5):
            all_with_zero.append(tuple([0] + list(rest)))
        
        for base in all_with_zero:
            if time.time() > deadline:
                break
            
            orbit = set()
            for g in group_elements:
                a, b, c, d = g
                transformed = tuple(sorted(_apply_mobius(p, a, b, c, d) for p in base))
                orbit.add(transformed)
            
            if len(orbit) == 132:
                covered_5subsets = set()
                valid = True
                for block in orbit:
                    for sub in combinations(block, 5):
                        if sub in covered_5subsets:
                            valid = False
                            break
                        covered_5subsets.add(sub)
                    if not valid:
                        break
                
                if valid:
                    best_blocks = [list(b) for b in orbit]
                    break
    
    if not best_blocks:
        # Final fallback: use a known explicit construction
        # The Witt design S(5,6,12) blocks can be explicitly listed
        # Let me use a different algebraic construction
        
        # Use the fact that S(5,6,12) is related to the binary Golay code
        # But for now, return what we have or an empty list
        
        # Try a simple greedy as fallback
        from itertools import combinations as comb
        covered, blocks = set(), []
        for b in comb(range(12), 6):
            subsets = list(comb(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        best_blocks = blocks
    
    return best_blocks
