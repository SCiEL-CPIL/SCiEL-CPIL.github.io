"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_latin(sq, n):
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _are_orthogonal(sq1, sq2, n):
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (sq1[i][j], sq2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def _pair_to_int(a, b):
    """Convert (a in Z_2, b in Z_5) to integer 0..9"""
    return a * 5 + b

def _int_to_pair(x):
    """Convert integer 0..9 to (a in Z_2, b in Z_5)"""
    return (x // 5, x % 5)

def construct():
    n = 10
    start = time.time()
    deadline = start + 50.0  # safety margin before 60s limit
    
    # Use Z_2 x Z_5 structure for order 10
    # Represent symbols as (a, b) where a in Z_2, b in Z_5
    # a = symbol // 5, b = symbol % 5
    
    # For position (i, j):
    #   i -> (i2, i5) where i2 = i//5, i5 = i%5
    #   j -> (j2, j5) where j2 = j//5, j5 = j%5
    
    # Construction of Latin squares using linear maps:
    # Square type (c2, c5): 
    #   a_out = (c2 * i2 + j2) % 2
    #   b_out = (c5 * i5 + j5) % 5
    #   symbol = _pair_to_int(a_out, b_out)
    
    # For this to be Latin:
    # - Row i fixed: as j varies, (j2, j5) covers all 10 values
    #   a_out = (c2*i2 + j2) % 2, b_out = (c5*i5 + j5) % 5
    #   As j goes 0..9, j2 goes 0,0,0,0,0,1,1,1,1,1 and j5 goes 0,1,2,3,4,0,1,2,3,4
    #   So (j2, j5) takes all 10 pairs exactly once
    #   a_out takes values (c2*i2 + 0)%2 and (c2*i2+1)%2, each 5 times
    #   b_out takes values (c5*i5 + 0..4)%5 = all of Z_5, repeated twice
    #   So (a_out, b_out) covers all 10 pairs -> Latin in rows ✓
    # - Column j fixed: as i varies, similar argument -> Latin in columns ✓
    
    # Orthogonality of squares (c2_1, c5_1) and (c2_2, c5_2):
    # We need all 100 pairs to be distinct
    # For positions (i,j), the pair is:
    #   ((c2_1*i2 + j2)%2, (c5_1*i5 + j5)%5, (c2_2*i2 + j2)%2, (c5_2*i5 + j5)%5)
    # 
    # Case 1: c5_1 != c5_2 (different Z_5 multipliers)
    #   Fix the Z_5 components: (c5_1*i5 + j5, c5_2*i5 + j5) 
    #   For fixed (i5, j5), as we vary (i2, j2), we get 4 combinations of the Z_2 parts
    #   The Z_5 pair (c5_1*i5+j5, c5_2*i5+j5) depends on (i5, j5)
    #   For the full pair to repeat, we need same (i5,j5) AND same Z_2 parts
    #   Same Z_2 parts means: (c2_1*i2+j2, c2_2*i2+j2) = (c2_1'*i2'+j2', c2_2'*i2'+j2')
    #   This requires i2=i2', j2=j2' if the system is injective
    #   Actually, let's think more carefully...
    #   
    #   The map (i2,j2) -> ((c2_1*i2+j2)%2, (c2_2*i2+j2)%2) 
    #   If c2_1 == c2_2, this maps to (c2_1*i2+j2, c2_1*i2+j2) which has only 2 distinct values
    #   If c2_1 != c2_2, the map is injective (4 distinct outputs for 4 inputs)
    #   
    #   So for orthogonality when c5_1 != c5_2:
    #   We need c2_1 != c2_2 to make the Z_2 part injective
    #   Then each (i5,j5) gives 4 distinct Z_2 pairs, and different (i5,j5) give different Z_5 pairs
    #   Total: 25 * 4 = 100 distinct pairs ✓
    
    # Case 2: c5_1 == c5_2 (same Z_5 multiplier)
    #   Then Z_5 parts are identical for both squares
    #   We need the Z_2 parts to distinguish all 100 positions
    #   But Z_2 part has only 4 possible values, and Z_5 part has 25
    #   For fixed (i5,j5), we have 4 positions with same Z_5 pair but different Z_2 pairs
    #   If c2_1 != c2_2, the 4 Z_2 pairs are distinct, so all 100 are distinct ✓
    #   Wait, but if c5_1 == c5_2, then for two different positions (i,j) and (i',j')
    #   with same (i5,j5) but different (i2,j2), the Z_5 parts match
    #   The Z_2 parts differ iff (c2_1*i2+j2, c2_2*i2+j2) != (c2_1*i2'+j2', c2_2*i2'+j2')
    #   This is injective iff c2_1 != c2_2
    #   So even with c5_1 == c5_2, if c2_1 != c2_2, we get orthogonality!
    
    # Actually wait, I need to be more careful. Let me just enumerate and check.
    
    candidates = []
    for c2 in range(2):
        for c5 in range(5):
            sq = []
            for i in range(10):
                i2, i5 = i // 5, i % 5
                row = []
                for j in range(10):
                    j2, j5 = j // 5, j % 5
                    a_out = (c2 * i2 + j2) % 2
                    b_out = (c5 * i5 + j5) % 5
                    row.append(_pair_to_int(a_out, b_out))
                sq.append(row)
            if _is_latin(sq, 10):
                candidates.append((c2, c5, sq))
    
    # Find maximum set of mutually orthogonal squares
    best_set = []
    for cand in candidates:
        c2, c5, sq = cand
        ok = True
        for _, _, sq2 in best_set:
            if not _are_orthogonal(sq, sq2, 10):
                ok = False
                break
        if ok:
            best_set.append(cand)
    
    if len(best_set) >= 1:
        result = [sq for _, _, sq in best_set]
    else:
        # Fallback: single trivial square
        result = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    
    # Ensure proper types
    result = [[list(map(int, row)) for row in sq] for sq in result]
    
    return result
# EVOLVE-BLOCK-END
