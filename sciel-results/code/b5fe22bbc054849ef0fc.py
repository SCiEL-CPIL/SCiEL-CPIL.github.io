"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def _check_validity(colors, n=128):
    """Check if coloring satisfies: every nonconstant 4-term AP has >=3 distinct colors."""
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            c0 = colors[a]
            c1 = colors[a + d]
            c2 = colors[a + 2*d]
            c3 = colors[a + 3*d]
            if len({c0, c1, c2, c3}) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _search_periodic():
    """Search for valid periodic colorings with small periods and few colors."""
    best = None
    best_count = 999
    deadline = time.time() + 70  # safety margin

    # Try periods from small to larger
    for p in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 48, 56, 60, 64, 72, 80, 96, 120, 128]:
        if time.time() > deadline:
            break
        
        # Try with 2 colors first, then 3
        for num_colors in [2, 3]:
            if time.time() > deadline:
                    break
            
            if p * num_colors**p > 200000:  # too many patterns, skip
                continue
            
            # Generate all possible base patterns of length p using num_colors colors
            # Use itertools.product
            for pattern in itertools.product(range(num_colors), repeat=p):
                # Check if this pattern extended to 128 is valid
                colors = [pattern[i % p] for i in range(128)]
                
                # Quick check: count distinct colors actually used
                actual_colors = set(pattern)
                if len(actual_colors) > num_colors:
                    continue
                if len(actual_colors) < 2:
                    continue  # need at least 2 colors for non-constant APs to work... actually need >=3 in APs
                
                # Validate
                if _check_validity(colors):
                    cnt = len(actual_colors)
                    if cnt < best_count:
                        best_count = cnt
                        best = list(colors)
                    if best_count == 2:
                        return best  # can't do better than 2
    
    return best

def _search_direct():
    """Direct greedy/backtracking search for minimal colors."""
    # Try to construct with 2 colors first using backtracking
    n = 128
    colors = [-1] * n
    
    # Precompute all 4-term APs
    aps = []
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a, a+d, a+2*d, a+3*d))
    
    # Greedy with backtracking limited depth
    # Actually, let's try a smarter approach: try all 2-color assignments for first k positions
    # and check if extendable
    
    # For 2 colors: 0 and 1
    # A 4-term AP is invalid if all 4 same, or if 3 same + 1 different (that's 2 colors, invalid)
    # So with 2 colors, every 4-term AP must have exactly 2 of one color and 2 of the other
    # (i.e., pattern must be 2-2 split)
    
    # This is very restrictive. Let's check if it's possible.
    # For d=1: positions a, a+1, a+2, a+3 must have 2 zeros and 2 ones
    # This means in any 4 consecutive positions, exactly 2 are 0 and 2 are 1
    # This implies a periodic pattern with period 4: e.g., 0011, 0101, 0110, 1001, 1010, 1100
    # Let's check which period-4 patterns work for ALL APs
    
    for pattern in itertools.product([0, 1], repeat=4):
        # Check if this 4-periodic pattern works for all 4-term APs up to 128
        colors = [pattern[i % 4] for i in range(n)]
        valid = True
        for (a, b, c, d) in aps:
            vals = {colors[a], colors[b], colors[c], colors[d]}
            if len(vals) < 3:
                valid = False
                break
        if valid:
            return colors
    
    # Try period 6 with 2 colors
    for pattern in itertools.product([0, 1], repeat=6):
        colors = [pattern[i % 6] for i in range(n)]
        valid = True
        for (a, b, c, d) in aps:
            vals = {colors[a], colors[b], colors[c], colors[d]}
            if len(vals) < 3:
                valid = False
                break
        if valid:
            return colors
    
    # Try period 8 with 2 colors
    for pattern in itertools.product([0, 1], repeat=8):
        colors = [pattern[i % 8] for i in range(n)]
        valid = True
        for (a, b, c, d) in aps:
            vals = {colors[a], colors[b], colors[c], colors[d]}
            if len(vals) < 3:
                valid = False
                break
        if valid:
            return colors
    
    # If 2 colors not found, try 3 colors with small periods
    for p in [4, 6, 8, 10, 12]:
        for pattern in itertools.product([0, 1, 2], repeat=p):
            actual = set(pattern)
            if len(actual) < 2:
                continue
            colors = [pattern[i % p] for i in range(n)]
            valid = True
            for (a, b, c, d) in aps:
                vals = {colors[a], colors[b], colors[c], colors[d]}
                if len(vals) < 3:
                    valid = False
                    break
            if valid:
                return colors
    
    return None

def construct():
    """Return n=128 integer color labels in 0..127. Every nonconstant four-term progression
    of positions must use AT LEAST three distinct colors. Minimize number of labels used."""
    
    # First try to find a 2-color solution
    result = _search_direct()
    if result is not None:
        # Verify
        if _check_validity(result):
            return result
    
    # Try periodic search
    result = _search_periodic()
    if result is not None:
        if _check_validity(result):
            return result
    
    # Fallback: use the original greedy approach as safety net
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3 + 1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2 + 1))):
                colors.append(color)
                break
    
    return colors
# EVOLVE-BLOCK-END
