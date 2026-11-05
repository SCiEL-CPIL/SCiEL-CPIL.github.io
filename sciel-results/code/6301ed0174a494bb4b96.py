"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    import random
    from collections import defaultdict
    
    DEADLINE = time.time() + 80
    
    N = 512
    
    # Build conflict bitsets
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if i * j // math.gcd(i, j) > N:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    def is_valid(s):
        s_list = sorted(s)
        for idx, a in enumerate(s_list):
            for b in s_list[idx+1:]:
                if (conflict_mask[a] >> b) & 1:
                    return False
        return True
    
    # Greedy coloring for upper bound in branch-and-bound
    def greedy_coloring(vertices, conflict_mask):
        """
        Greedy coloring of the subgraph induced by vertices.
        Returns list of color classes (lists of vertices).
        """
        colors = []
        uncolored = set(vertices)
        while uncolored:
            color_class = []
            remaining = set(uncolored)
            for v in sorted(remaining):
                can_add = True
                for u in color_class:
                    if (conflict_mask[v] >> u) & 1:
                        can_add = False
                        break
                if can_add:
                    color_class.append(v)
                    remaining.discard(v)
            colors.append(color_class)
            uncolored = remaining
        return colors
    
    # Branch and bound for maximum independent set
    def branch_and_bound(initial_set, deadline):
        """
        Branch and bound to find maximum independent set.
        Uses greedy coloring for upper bound.
        """
        best_size = len(initial_set)
        best_set = set(initial_set)
        
        # Build adjacency list for faster lookup
        adj = [set() for _ in range(N + 1)]
        for i in range(1, N + 1):
            mask = conflict_mask[i]
            while mask:
                lsb = mask & (-mask)
                j = lsb.bit_length() - 1
                if j >= 1:
                    adj[i].add(j)
                mask &= mask - 1
        
        # Recursive branch and bound
        def bb(candidates, current_set, depth):
            nonlocal best_size, best_set
            
            if time.time() >= deadline:
                return
            
            # Upper bound: greedy coloring
            if not candidates:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                return
            
            color_classes = greedy_coloring(candidates, conflict_mask)
            upper_bound = len(current_set) + len(color_classes)
            
            if upper_bound <= best_size:
                return
            
            # Pick vertex with highest degree in candidates for branching
            v = max(candidates, key=lambda x: len([y for y in candidates if y != x and (conflict_mask[x] >> y) & 1]))
            
            # Branch 1: include v
            new_candidates = set(candidates)
            new_candidates.discard(v)
            # Remove all neighbors of v from candidates
            for u in list(new_candidates):
                if (conflict_mask[v] >> u) & 1:
                    new_candidates.discard(u)
            current_set.add(v)
            bb(new_candidates, current_set, depth + 1)
            current_set.remove(v)
            
            # Branch 2: exclude v
            new_candidates = set(candidates)
            new_candidates.discard(v)
            bb(new_candidates, current_set, depth + 1)
        
        # Start with candidates = all vertices not in initial_set that don't conflict with initial_set
        candidates = set()
        for v in range(1, N + 1):
            if v in initial_set:
                continue
            ok = True
            for u in initial_set:
                if (conflict_mask[v] >> u) & 1:
                    ok = False
                    break
            if ok:
                candidates.add(v)
        
        bb(candidates, set(initial_set), 0)
        
        return best_set
    
    # Strong initial solution
    def strong_initial():
        s = set()
        # Divisors of 360
        for d in range(1, N + 1):
            if 360 % d == 0:
                s.add(d)
        # Divisors of other highly composite numbers
        for base in [120, 240, 480, 72, 144, 288]:
            for d in range(1, N + 1):
                if base % d == 0 and d not in s:
                    s.add(d)
        return s
    
    def add_nonconflicting(s):
        s = set(s)
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for x in range(1, N + 1):
                if x in s:
                    continue
                ok = True
                for y in s:
                    if (conflict_mask[x] >> y) & 1:
                        ok = False
                        break
                if ok:
                    s.add(x)
                    changed = True
        return s
    
    # Get initial best
    best = strong_initial()
    best = add_nonconflicting(best)
    
    # Run branch and bound with the initial set
    if time.time() < DEADLINE - 5:
        improved = branch_and_bound(best, DEADLINE - 5)
        if len(improved) > len(best):
            best = improved
    
    # Final polish
    best = add_nonconflicting(best)
    
    result = sorted(best)
    if not is_valid(set(result)):
        return list(range(1, math.isqrt(N) + 1))
    
    return result
# EVOLVE-BLOCK-END
