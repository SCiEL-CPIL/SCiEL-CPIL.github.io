import time
import random
import math
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 256
    # Build compatibility graph using bitmasks
    # compat[i] is a bitmask where bit j is set if i is compatible with j
    # Compatible means (i*j) % (i+j) != 0
    compat = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) != 0:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
    
    # Initial solution: All odd numbers (no conflicts among odds)
    # For odd a, b: a+b is even, a*b is odd. Even can't divide odd.
    # So all odds are mutually compatible.
    best_S = set(range(1, n + 1, 2))
    best_size = len(best_S)
    
    # Tomita-style branch and bound for Maximum Clique in the compatibility graph
    # A clique in the compatibility graph is an independent set in the conflict graph.
    
    def color_bound(candidates):
        """
        Compute greedy coloring upper bound on candidates bitmask.
        This is the standard upper bound for Max Clique algorithms.
        We assign colors to vertices in candidates such that no two adjacent vertices have the same color.
        The number of colors is an upper bound on the size of the maximum clique in candidates.
        """
        color_classes = []
        remaining = candidates
        while remaining:
            # Start a new color class
            color_class = 0
            # We need to pick vertices from remaining that are not adjacent to any vertex in this color class
            # Standard greedy: iterate through vertices, assign to first available color
            # For bitmask, we can do:
            # 1. Pick a vertex v from remaining.
            # 2. Add v to color_class.
            # 3. Remove v and all its neighbors (in the induced subgraph of remaining) from remaining?
            #    No, that's independent set. For clique coloring, we remove v from remaining, 
            #    and we must ensure subsequent vertices in this color class are NOT adjacent to v.
            #    So, candidates for this color class = remaining & ~compat[v] ? No.
            #    If we put v in color C, the next vertex u in color C must NOT be adjacent to v.
            #    So u must be in remaining AND NOT in compat[v].
            
            # Efficient Greedy Coloring for Max Clique Bound:
            # We want to cover the vertices in `remaining` with minimum colors.
            # Iteration:
            #   current_color_set = 0
            #   available = remaining
            #   while available:
            #       pick v from available (e.g. lowest bit)
            #       current_color_set |= (1 << v)
            #       # Remove v from available
            #       # Remove all neighbors of v from available (they can't have this color)
            #       available &= ~(compat[v] | (1 << v))
            #   color_classes.append(current_color_set)
            #   remaining &= ~current_color_set
            
            # Wait, the logic above is:
            # 1. Start with `available` = all remaining vertices.
            # 2. Pick v.
            # 3. Mark v as used for this color.
            # 4. Vertices adjacent to v CANNOT use this color. So remove them from `available`.
            # 5. Repeat until `available` is empty.
            # 6. This forms one color class.
            # 7. Remove this color class from `remaining`.
            # 8. Repeat for next color.
            
            available = remaining
            color_class_mask = 0
            while available:
                # Pick lowest set bit
                v_bit = available & (-available)
                v = v_bit.bit_length() - 1
                color_class_mask |= v_bit
                # Remove v and its neighbors from available
                available &= ~(compat[v] | v_bit)
            
            color_classes.append(color_class_mask)
            remaining &= ~color_class_mask
            
        return len(color_classes)

    def expand(candidates, current_size, current_set):
        """
        Branch and bound for Max Clique.
        candidates: bitmask of vertices to consider
        current_size: size of current clique
        current_set: list of vertices in current clique
        """
        nonlocal best_size, best_S
        if time.time() > deadline:
            return
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_S = set(current_set)
            return
        
        # Upper bound check
        ub = current_size + color_bound(candidates)
        if ub <= best_size:
            return
        
        # Heuristic: Order candidates by degree in the candidate subgraph?
        # Tomita usually orders vertices by degree to improve coloring bound and branching.
        # For simplicity and speed in Python, we pick the vertex with highest degree in `candidates`.
        
        # Find vertex with max degree in candidates
        best_v = -1
        best_deg = -1
        temp = candidates
        while temp:
            v_bit = temp & (-temp)
            v = v_bit.bit_length() - 1
            deg = (candidates & compat[v]).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= v_bit
        
        if best_v < 0:
            return
        
        # Branch 1: Include best_v
        current_set.append(best_v)
        # New candidates must be compatible with best_v
        new_candidates = candidates & compat[best_v]
        expand(new_candidates, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        new_candidates = candidates & ~(1 << best_v)
        expand(new_candidates, current_size, current_set)
    
    # Run BnB with the full candidate set
    full_candidates = 0
    for i in range(1, n + 1):
        full_candidates |= (1 << i)
    
    expand(full_candidates, 0, [])
    
    # Local search polish if time permits
    if time.time() < deadline - 5:
        S = set(best_S)
        all_nums = list(range(1, n + 1))
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in all_nums:
                if x not in S:
                    # Check if x is compatible with all in S
                    # Use bitmask for speed?
                    # Let's just check directly, it's small enough
                    is_compatible = True
                    for y in S:
                        if (x * y) % (x + y) == 0:
                            is_compatible = False
                            break
                    if is_compatible:
                        S.add(x)
                        improved = True
            if len(S) > best_size:
                best_size = len(S)
                best_S = S
                improved = True
    
    return list(best_S)
