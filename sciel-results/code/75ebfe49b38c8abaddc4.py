"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
DEADLINE = 58.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def is_valid_sidon(selected):
        """Check if selected is a valid Sidon set (all pair sums distinct including diagonals)."""
        sums = set()
        for i, a in enumerate(selected):
            for j in range(i, len(selected)):
                b = selected[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def is_maximal(selected):
        """Check if selected is inclusion-maximal: no missing element can be added."""
        selected_set = set(selected)
        sums = set()
        for i, a in enumerate(selected):
            for j in range(i, len(selected)):
                b = selected[j]
                sums.add(a + b)
        
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            # Check if adding x would create a conflict
            conflict = False
            for a in selected:
                s = x + a
                if s in sums:
                    conflict = True
                    break
            if not conflict:
                if 2 * x in sums:
                    conflict = True
            if not conflict:
                return False
        return True
    
    def build_conflict_graph():
        """Build conflict graph: edge between a,b if they can't coexist in a Sidon set.
        
        Two numbers a,b conflict if there exist c,d in [1,N] such that:
        - a+b = c+d (with {a,b} != {c,d})
        This means if we pick both a and b, we need to ensure no other pair sums to a+b.
        
        For MIS approach: we want an independent set (no two selected numbers conflict directly).
        But actually the conflict is more complex - it's about all pair sums being distinct.
        
        Simpler model: vertices are 1..N. Edge (a,b) exists if a+b equals some other pair's sum.
        But that's not quite right either. Let me think differently.
        
        Actually for a Sidon set, the constraint is: for all a<=b, c<=d in the set, 
        if a+b = c+d then {a,b} = {c,d}.
        
        A simpler sufficient condition for conflict: if a+b = c+d where (a,b) != (c,d),
        then we can't have all four in the set (unless pairs are identical).
        
        For the MIS formulation, let's define: two elements x,y "conflict" if there exists
        some other pair (u,v) with u+v = x+y and {u,v} != {x,y}. But this depends on what
        else is in the set.
        
        Alternative: Use a different model. Let's use the fact that for small sets, we can
        try greedy elimination.
        
        Actually, let me use a cleaner approach: 
        - Start with all elements 1..N
        - Iteratively remove the element that "blocks" the most other elements
        - An element x blocks y if x+y creates a conflict with some existing pair
        """
        pass
    
    def greedy_conflict_elimination(order=None):
        """Greedy MIS-style: start with all elements, iteratively remove the most 'restrictive' one.
        
        Strategy: 
        1. Start with candidate set = {1, ..., N}
        2. While not valid Sidon: find a conflicting pair of sums, remove one of the involved elements
        3. Prefer removing elements that are involved in more potential conflicts
        
        Actually, better approach:
        1. Start empty, add elements in order, but when conflict arises, remove the element 
           that causes the most future conflicts (i.e., has highest "degree" in conflict graph)
        """
        if order is None:
            order = list(range(1, N + 1))
        
        selected = []
        sums = {}  # sum -> (a, b) pair that produced it
        
        for x in order:
            # Check if x can be added
            can_add = True
            for a in selected:
                s = x + a
                if s in sums:
                    can_add = False
                    break
            if can_add:
                s = 2 * x
                if s in sums:
                    can_add = False
        
            if can_add:
                selected.append(x)
                for a in selected[:-1]:
                    sums[x + a] = (x, a)
                sums[2 * x] = (x, x)
            else:
                # Conflict: try to resolve by removing the most "restrictive" element
                # Find which existing element, if removed, would allow x to be added
                # Or just skip x for now (greedy)
                pass
        
        return selected
    
    def mis_vertex_elimination():
        """MIS-based approach: 
        1. Build a graph where vertices are 1..N
        2. Edge (u,v) if u and v cannot both be in a Sidon set with some "witness"
        3. Find a maximal independent set using greedy vertex elimination
        
        For the conflict definition: u and v conflict if there exist a,b with u+v = a+b 
        and {u,v} != {a,b}. But this is pairwise and depends on what's in the set.
        
        Simpler: Let's use the "degree" heuristic. For each element x, compute how many 
        other elements y would conflict with x (i.e., x+y equals 2*z or z+w for some z,w).
        Elements with high conflict degree are "expensive" to include.
        """
        # Precompute conflict degrees
        conflict_degree = [0] * (N + 1)
        for x in range(1, N + 1):
            for y in range(x + 1, N + 1):
                s = x + y
                # Check if s can be formed by another pair
                for z in range(1, min(y, s // 2) + 1):
                    w = s - z
                    if w >= z and w <= N and (z, w) != (x, y):
                        conflict_degree[x] += 1
                        conflict_degree[y] += 1
                        break
        
        # Greedy: pick elements in order of lowest conflict degree (least restrictive)
        order = sorted(range(1, N + 1), key=lambda x: conflict_degree[x])
        
        selected = []
        sums = set()
        for x in order:
            added_sums = [x + a for a in selected] + [2 * x]
            if len(set(added_sums)) == len(added_sums) and not sums.intersection(added_sums):
                selected.append(x)
                sums.update(added_sums)
        
        return selected
    
    def local_search_reduce(selected, max_iters=500):
        """Try to reduce size while maintaining maximality."""
        current = list(selected)
        improved = True
        iters = 0
        while improved and iters < max_iters and elapsed() < DEADLINE - 5:
            improved = False
            iters += 1
            indices = list(range(len(current)))
            random.shuffle(indices)
            for idx in indices:
                x = current[idx]
                new_sel = [a for a in current if a != x]
                if not new_sel:
                    continue
                if is_valid_sidon(new_sel) and is_maximal(new_sel):
                    current = new_sel
                    improved = True
                    break
        return current
    
    # Strategy: Multiple approaches
    best_result = None
    best_size = N + 1
    
    # Approach 1: MIS vertex elimination
    if elapsed() < DEADLINE - 10:
        mis_result = mis_vertex_elimination()
        if is_valid_sidon(mis_result) and is_maximal(mis_result):
            best_result = mis_result
            best_size = len(mis_result)
    
    # Approach 2: Greedy with random orderings
    max_restarts = 300
    restart = 0
    while restart < max_restarts and elapsed() < DEADLINE - 10:
        restart += 1
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        selected = []
        sums = set()
        for x in order:
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        
        if not is_valid_sidon(selected) or not is_maximal(selected):
            continue
        
        # Local search to reduce
        minimized = local_search_reduce(selected, max_iters=200)
        if len(minimized) < best_size:
            best_size = len(minimized)
            best_result = minimized
    
    # Final refinement
    if best_result is not None and elapsed() < DEADLINE - 3:
        improved = local_search_reduce(best_result, max_iters=1000)
        if len(improved) < best_size:
            best_result = improved
    
    if best_result is None:
        # Fallback: simple ascending greedy
        selected = []
        sums = set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_result = selected
    
    return best_result
# EVOLVE-BLOCK-END
