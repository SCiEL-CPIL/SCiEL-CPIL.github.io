"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    n = 512
    
    # Fast greedy: maintain in_set array, check 4-AP conflicts in O(n) per candidate
    in_set = [False] * (n + 1)
    selected = []
    
    def can_add(x, in_set):
        # Check all 4-APs that could include x
        # x as 4th term: x-3d, x-2d, x-d, x
        for d in range(1, (x // 3) + 1):
            if in_set[x-d] and in_set[x-2*d] and in_set[x-3*d]:
                return False
        # x as 1st term: x, x+d, x+2d, x+3d
        for d in range(1, (n - x) // 3 + 1):
            if in_set[x+d] and in_set[x+2*d] and in_set[x+3*d]:
                return False
        # x as 2nd term: x-d, x, x+d, x+2d
        max_d = min(x, (n - x) // 2)
        for d in range(1, max_d + 1):
            if in_set[x-d] and in_set[x+d] and in_set[x+2*d]:
                return False
        # x as 3rd term: x-2d, x-d, x, x+d
        max_d = min(x // 2, n - x)
        for d in range(1, max_d + 1):
            if in_set[x-2*d] and in_set[x-d] and in_set[x+d]:
                return False
        return True
    
    # Greedy in natural order
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if can_add(x, in_set):
            in_set[x] = True
            selected.append(x)
    
    # Try to improve with a second pass using different ordering
    # Numbers with low base-3 digit sum tend to be in large AP-free sets
    def base3_sum(x):
        s = 0
        while x > 0:
            s += x % 3
            x //= 3
        return s
    
    order = sorted(range(1, n + 1), key=lambda x: (base3_sum(x), x))
    
    in_set2 = [False] * (n + 1)
    selected2 = []
    
    for x in order:
        if time.time() > deadline:
            break
        if can_add(x, in_set2):
            in_set2[x] = True
            selected2.append(x)
    
    # Return the larger set
    if len(selected2) > len(selected):
        return sorted(selected2)
    else:
        return sorted(selected)
# EVOLVE-BLOCK-END
