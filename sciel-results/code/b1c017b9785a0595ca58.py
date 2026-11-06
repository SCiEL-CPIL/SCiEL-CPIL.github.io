"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _validate(color_map):
    """Check every K4 spans at least 5 distinct colours."""
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        cols = set()
        for x, y in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
            if x > y:
                x, y = y, x
            cols.add(color_map[(x, y)])
        if len(cols) < 5:
            return False
    return True

def _check_merge_validity(color_map, c1, c2):
    """Check if merging c1 and c2 keeps all K4s valid."""
    # Create a temporary map with c2 mapped to c1
    # But we only need to check K4s that contain edges of c1 or c2
    # Optimization: Iterate K4s, check if they contain c1 and c2.
    # If they do, check if the remaining colors count is >= 3 (so total >= 5 after merge)
    # Actually, if a K4 has colors {c1, c2, c3, c4, c5}, merging c1,c2 gives {c1, c3, c4, c5} (4 colors). Invalid.
    # So we need: No K4 has exactly 5 distinct colors where 2 of them are c1 and c2.
    # Or more generally, if a K4 contains edges of both c1 and c2, let S be the set of colors in that K4.
    # If |S| == 5 and c1 in S and c2 in S, then invalid.
    # If |S| > 5, merging reduces size by 1, so |S|-1 >= 5 is valid.
    # If |S| < 5, the original coloring was already invalid (but 1-factorization is valid, so |S|>=5 always).
    # So we just need to ensure no K4 has |S| == 5 and contains both c1 and c2.
    
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        cols = set()
        has_c1 = False
        has_c2 = False
        for x, y in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
            if x > y:
                x, y = y, x
            col = color_map[(x, y)]
            if col == c1:
                has_c1 = True
            if col == c2:
                has_c2 = True
            cols.add(col)
        
        if has_c1 and has_c2:
            if len(cols) == 5:
                return False
    return True

def construct():
    t0 = time.time()
    deadline = 110

    # 1-factorization of K16
    color_map = {}
    for i in range(15):
        color_map[(15, i)] = i
        color_map[(i, 15)] = i
        for j in range(1, 8):
            a = (i + j) % 15
            b = (i - j) % 15
            if a > b:
                a, b = b, a
            color_map[(a, b)] = i

    # Try to merge two colors to get 14
    for c1 in range(15):
        for c2 in range(c1 + 1, 15):
            if time.time() - t0 > deadline - 5:
                break
            if _check_merge_validity(color_map, c1, c2):
                # Merge c2 into c1
                for key, val in color_map.items():
                    if val == c2:
                        color_map[key] = c1
                # Remap colors to 0..13
                used_colors = sorted(set(color_map.values()))
                color_remap = {old: new for new, old in enumerate(used_colors)}
                for key in color_map:
                    color_map[key] = color_remap[color_map[key]]
                
                if _validate(color_map):
                    result = []
                    for i in range(16):
                        for j in range(i + 1, 16):
                            result.append(color_map[(i, j)])
                    return result

    # Fallback: 15 colors
    result = []
    for i in range(16):
        for j in range(i + 1, 16):
            result.append(color_map[(i, j)])
    return result
# EVOLVE-BLOCK-END
