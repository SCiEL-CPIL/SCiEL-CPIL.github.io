import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd table for speed
    gcd_table = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gcd_table[(a, b)] = gcd(a, b)
    
    # Precompute all bad triples: triples (a,b,c) where all pairwise gcds are equal
    # A triple is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                g_ab = gcd_table[(a, b)]
                g_ac = gcd_table[(a, c)]
                g_bc = gcd_table[(b, c)]
                if g_ab == g_ac == g_bc:
                    bad_triples.add((a, b, c))
    
    # For each number, precompute which pairs it conflicts with
    # conflict[x] = set of pairs (u,v) such that {x,u,v} is a bad triple
    conflict = {x: set() for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflict[a].add((b, c))
        conflict[b].add((a, c))
        conflict[c].add((a, b))
    
    def can_add(x, current_set):
        """Check if x can be added without forming a bad triple"""
        # For each pair in current_set, check if it's a conflict with x
        for pair in conflict[x]:
            if pair[0] in current_set and pair[1] in current_set:
                return False
        return True
    
    def is_valid(s):
        """Check if set s has no bad triple"""
        for (a, b, c) in bad_triples:
            if a in s and b in s and c in s:
                return False
        return True
    
    # Strategy: Use a different structural approach
    # Key insight: The constraint is about triples with equal pairwise gcds.
    # If we group numbers by their "gcd structure", we can exploit this.
    
    # Alternative approach: Build the set using a maximum independent set heuristic
    # on the 3-uniform hypergraph of bad triples.
    
    # First, let's try a smarter greedy: order by how many conflicts each number has
    # Numbers with fewer conflicts are "safer" to include
    
    conflict_count = {x: len(conflict[x]) for x in range(1, N + 1)}
    
    # Sort by ascending conflict count (safest first)
    priority_order = sorted(elements, key=lambda x: conflict_count[x])
    
    def greedy_build(order):
        s = set()
        for x in order:
            if can_add(x, s):
                s.add(x)
        return s
    
    best_set = greedy_build(priority_order)
    
    # Local search: try to add missing elements, then try swaps
    def local_search(s, max_iters=100):
        s = s.copy()
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try to add any missing element
            missing = [x for x in elements if x not in s]
            # Try in order of least conflicts
            for x in sorted(missing, key=lambda x: conflict_count[x]):
                if can_add(x, s):
                    s.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try single swaps: remove one, add another
            s_list = list(s)
            for u in s_list:
                if time.time() > deadline:
                    return s
                s_temp = s - {u}
                for x in sorted(missing, key=lambda x: conflict_count[x]):
                    if can_add(x, s_temp):
                        s_temp.add(x)
                        if len(s_temp) > len(s):
                            s = s_temp
                            improved = True
                            break
                if improved:
                    break
        
        return s
    
    best_set = local_search(best_set)
    
    # Try multiple random restarts with different orderings
    import random
    rng = random.Random(42)
    
    for restart in range(200):
        if time.time() > deadline - 2.0:
            break
        
        # Create a perturbed ordering
        order = elements.copy()
        rng.shuffle(order)
        # But bias towards low-conflict numbers
        order.sort(key=lambda x: (conflict_count[x] * 0.3 + rng.random() * 0.7))
        
        s = greedy_build(order)
        s = local_search(s, max_iters=50)
        
        if len(s) > len(best_set):
            best_set = s.copy()
    
    # Try a completely different approach: start from all elements and remove
    # those that are in the most bad triples
    if time.time() < deadline - 5.0:
        # Count how many bad triples each number is in
        triple_count = {x: 0 for x in range(1, N + 1)}
        for (a, b, c) in bad_triples:
            triple_count[a] += 1
            triple_count[b] += 1
            triple_count[c] += 1
        
        # Start with all elements, remove the most "conflicting" ones
        s_all = set(elements)
        # Sort by descending triple count
        removal_order = sorted(elements, key=lambda x: triple_count[x], reverse=True)
        
        # Remove elements until valid
        for x in removal_order:
            if not is_valid(s_all):
                s_all.discard(x)
        
        if len(s_all) > len(best_set):
            best_set = s_all.copy()
        
        # Local search on this
        s_all = local_search(s_all, max_iters=100)
        if len(s_all) > len(best_set):
            best_set = s_all.copy()
    
    # Try another structural approach: group by square-free kernel
    # Numbers with the same square-free kernel have related gcd structures
    def square_free_kernel(n):
        """Compute the square-free kernel of n"""
        kernel = 1
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                count = 0
                while temp % d == 0:
                    temp //= d
                    count += 1
                if count % 2 == 1:
                    kernel *= d
            d += 1
        if temp > 1:
            kernel *= temp
        return kernel
    
    kernels = {x: square_free_kernel(x) for x in range(1, N + 1)}
    
    # Group by kernel
    from collections import defaultdict
    kernel_groups = defaultdict(list)
    for x in range(1, N + 1):
        kernel_groups[kernels[x]].append(x)
    
    # For each kernel group, we can include at most a limited number
    # Try to build by selecting from each group strategically
    if time.time() < deadline - 3.0:
        s_kernel = set()
        # Process groups by size (larger groups first, as they have more constraints)
        groups_sorted = sorted(kernel_groups.items(), key=lambda kv: len(kv[1]), reverse=True)
        
        for kernel, group in groups_sorted:
            # Try to add as many from this group as possible
            for x in sorted(group, key=lambda x: conflict_count[x]):
                if can_add(x, s_kernel):
                    s_kernel.add(x)
        
        s_kernel = local_search(s_kernel, max_iters=50)
        if len(s_kernel) > len(best_set):
            best_set = s_kernel.copy()
    
    return list(best_set)
