"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s deadline to stay under 90s wall-clock
    
    n = 256
    elements = list(range(1, n + 1))
    weights = [Fraction(1, x) for x in elements]
    
    # Build conflict graph: i and j conflict if they form a 3-AP with some k in [1,256]
    # A 3-AP is (a, b, c) with a < b < c and b - a == c - b, i.e., a + c == 2*b
    # So for pair (i, j), they conflict if there exists k such that:
    #   i + j == 2*k (k is the middle) -> k = (i+j)/2, need i+j even and 1<=k<=256
    #   j - i == d, and we need a = j - d = 2*i - j >= 1, or c = j + d = 2*j - i <= 256
    # Actually, pair (i,j) is in a 3-AP if:
    #   (i,j) as (a,b): c = 2*j - i, need 1 <= c <= 256 and c != i, c != j
    #   (i,j) as (a,c): b = (i+j)/2, need i+j even, 1 <= b <= 256, b != i, b != j
    #   (i,j) as (b,c): a = 2*i - j, need 1 <= a <= 256, a != i, a != j
    
    # Precompute conflicts as bitsets for speed
    # Use a set of frozensets or just a 2D boolean array
    conflict = [[False] * (n + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            # Check if (i, j) can be part of any 3-AP
            # Case 1: i=a, j=b, c=2*j-i
            c = 2 * j - i
            if 1 <= c <= n and c != i and c != j:
                conflict[i][j] = True
                conflict[j][i] = True
                continue
            # Case 2: i=a, j=c, b=(i+j)/2
            if (i + j) % 2 == 0:
                b = (i + j) // 2
                if 1 <= b <= n and b != i and b != j:
                    conflict[i][j] = True
                    conflict[j][i] = True
                    continue
            # Case 3: i=b, j=c, a=2*i-j
            a = 2 * i - j
            if 1 <= a <= n and a != i and a != j:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # For efficiency, store conflicts as bitsets using integers
    # element index e (1-based) -> position e-1 in the bitset
    conflict_bits = [0] * (n + 1)
    for i in range(1, n + 1):
        bits = 0
        for j in range(1, n + 1):
            if i != j and conflict[i][j]:
                bits |= (1 << (j - 1))
        conflict_bits[i] = bits
    
    # Greedy initial solution: pick elements in order of decreasing weight (i.e., increasing value)
    # But we need to avoid conflicts. Use a simple greedy: process elements by weight desc,
    # add if no conflict with already selected.
    order = sorted(elements, key=lambda x: -weights[x-1])  # decreasing weight = increasing x is wrong
    # weights[x-1] = 1/x, so higher weight = smaller x
    # Actually sorted by -weight means largest weight first, which is smallest x first
    # Let's just sort by x ascending (smallest x = largest reciprocal)
    order = list(range(1, n + 1))  # already in order of decreasing weight
    
    selected = set()
    for x in order:
        if time.time() > deadline - 5.0:
            break
        # Check if x conflicts with any selected element
        x_bits = 1 << (x - 1)
        can_add = True
        for y in selected:
            if conflict[x][y]:
                can_add = False
                break
        if can_add:
            selected.add(x)
    
    current_weight = sum(weights[x-1] for x in selected)
    
    # Local search: try to improve by swapping or adding/removing
    # Phase 1: Try adding elements (if possible)
    # Phase 2: Try removing one and adding one or more
    # Phase 3: Try removing one and adding a different one (swap)
    
    def check_conflict_with_selected(x, sel):
        """Check if x conflicts with any element in sel (a set)."""
        for y in sel:
            if conflict[x][y]:
                return True
        return False
    
    def try_add(x, sel):
        """Try to add x to sel, return (success, new_sel)."""
        if x in sel:
            return False, sel
        if check_conflict_with_selected(x, sel):
            return False, sel
        new_sel = sel | {x}
        return True, new_sel
    
    def try_remove(x, sel):
        """Remove x from sel."""
        return sel - {x}
    
    # Iterative local search
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        
        # Try adding any single element
        for x in range(1, n + 1):
            if time.time() > deadline - 3.0:
                break
            if x in selected:
                continue
            success, new_sel = try_add(x, selected)
            if success:
                new_weight = sum(weights[y-1] for y in new_sel)
                if new_weight > current_weight:
                    selected = new_sel
                    current_weight = new_weight
                    improved = True
                    break
        
        if time.time() > deadline - 3.0:
            break
        
        # Try swapping: remove one element, add one different element
        sel_list = list(selected)
        for x in sel_list:
            if time.time() > deadline - 3.0:
                break
            sel_without_x = selected - {x}
            # Try adding each y not in sel_without_x
            for y in range(1, n + 1):
                if time.time() > deadline - 3.0:
                    break
                if y in sel_without_x:
                    continue
                if check_conflict_with_selected(y, sel_without_x):
                    continue
                new_sel = sel_without_x | {y}
                new_weight = sum(weights[z-1] for z in new_sel)
                if new_weight > current_weight:
                    selected = new_sel
                    current_weight = new_weight
                    improved = True
                    break
            if improved:
                break
        
        if time.time() > deadline - 3.0:
            break
        
        # Try removing one element and adding two
        for x in list(selected):
            if time.time() > deadline - 3.0:
                break
            sel_without_x = selected - {x}
            # Try adding pairs
            candidates = [y for y in range(1, n+1) if y not in sel_without_x and not check_conflict_with_selected(y, sel_without_x)]
            for i, y in enumerate(candidates):
                if time.time() > deadline - 3.0:
                    break
                sel_with_y = sel_without_x | {y}
                for z in candidates[i+1:]:
                    if time.time() > deadline - 3.0:
                        break
                    if z in sel_with_y:
                        continue
                    if check_conflict_with_selected(z, sel_with_y):
                        continue
                    new_sel = sel_with_y | {z}
                    new_weight = sum(weights[w-1] for w in new_sel)
                    if new_weight > current_weight:
                        selected = new_sel
                        current_weight = new_weight
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return list(selected)
# EVOLVE-BLOCK-END
