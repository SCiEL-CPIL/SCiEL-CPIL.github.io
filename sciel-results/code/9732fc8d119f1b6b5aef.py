# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    N = 256
    nums = list(range(1, N + 1))

    # Precompute conflict: conflict[a][b] = True if (a*b)%(a+b)==0
    conflict_set = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict_set.add((a, b))

    # Build adjacency bitmasks (256 bits each, fits in Python int)
    adj = [0] * (N + 1)
    for (a, b) in conflict_set:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Greedy coloring to get upper bounds for branch-and-bound
    # Color classes are independent sets; max color used = upper bound on clique size
    # We want max independent set = max clique in complement, but easier:
    # Use branch-and-bound on the conflict graph directly for max independent set.
    
    # Tomita's algorithm for maximum clique (we'll adapt for max independent set)
    # Max independent set in G = Max clique in complement of G
    # But we can also do branch-and-bound directly on G for independent set.
    
    # Let's use the standard approach: find max clique in the complement graph.
    # Complement adjacency: comp_adj[i] has bits set where there is NO conflict.
    comp_adj = [0] * (N + 1)
    for i in range(1, N + 1):
        # All other nodes
        all_others = ((1 << (N + 1)) - 1) ^ (1 << i)  # all bits set except i
        # Remove i itself and set bits where no conflict
        comp_adj[i] = all_others & ~adj[i]
        # Actually, comp_adj[i] should have bit j set iff i != j and (i,j) not in conflict
        # adj[i] has bit j set iff i != j and (i,j) in conflict
        # So comp_adj[i] = all_others & ~adj[i] is correct (since adj[i] doesn't have bit i set anyway)

    # Greedy coloring of the complement graph to get upper bounds
    # Colors assigned in order; number of colors used is upper bound on clique size
    # We use a simple greedy: process vertices in order, assign smallest available color
    
    # Actually, for Tomita's algorithm, we need to color the candidate set P at each node.
    # Let's implement a proper branch-and-bound for maximum clique in the complement graph.
    
    # Tomita's MCS algorithm:
    # We maintain a stack of (candidate_set, current_clique_size)
    # At each node, we color the candidate set to get an upper bound.
    # If upper bound <= best found, prune.
    
    # For efficiency with 256 vertices, we use bitsets and a greedy coloring.
    
    # Let's implement a recursive branch-and-bound with memoization-free approach
    # but with good ordering and coloring.
    
    # Since 256 is manageable, let's use a greedy initialization first, then branch-and-bound.
    
    # Greedy max independent set on the conflict graph:
    # Process vertices in order, add if no conflict with current set.
    # Multiple orderings to get good initial solution.
    
    def greedy_independent(order):
        s = set()
        for v in order:
            if all((v, u) not in conflict_set and (u, v) not in conflict_set for u in s):
                s.add(v)
        return s
    
    # Try several orderings
    best_size = 0
    best_set = set()
    
    # Order 1: natural
    s = greedy_independent(nums)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
    
    # Order 2: reverse
    s = greedy_independent(reversed(nums))
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
    
    # Order 3: odds first, then evens
    odds = [x for x in nums if x % 2 == 1]
    evens = [x for x in nums if x % 2 == 0]
    s = greedy_independent(odds + evens)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
    
    # Order 4: evens first, then odds
    s = greedy_independent(evens + odds)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
    
    # Now branch-and-bound to find the true maximum
    # We'll use the complement graph and find maximum clique.
    # Maximum clique in complement = Maximum independent set in original.
    
    # Tomita's algorithm with bitsets
    # We represent sets as Python ints (bitmasks)
    
    # Greedy coloring for upper bound:
    # Given a set P (bitmask), assign colors greedily.
    # Returns list of color classes (each a bitmask) and the number of colors used.
    
    def color_bound(P):
        """Greedy coloring of vertices in P. Returns (num_colors, color_classes)."""
        if P == 0:
            return 0, []
        
        colors = []
        remaining = P
        
        while remaining != 0:
            # Pick a vertex from remaining (lowest set bit for determinism)
            # Actually, for better bounds, we should order vertices by degree or something.
            # For simplicity, use natural order.
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1
            
            color_class = 0
            # This vertex gets the current color
            # We need to find all vertices in remaining that are not adjacent to v in the complement
            # i.e., they have a conflict with v in the original graph
            # Wait, in the complement graph, an edge means no conflict in original.
            # For coloring, we want independent sets in the complement = cliques in original?
            # No. In the complement graph, we're finding maximum clique.
            # For coloring to get upper bound on clique size, we color the complement graph.
            # A color class in the complement is an independent set in the complement,
            # which is a clique in the original graph.
            # But we want the clique to be in the complement graph.
            # So we color the complement graph. A color class is an independent set in the complement,
            # meaning all pairs in the class have no edge in the complement, i.e., they have an edge in the original.
            # That's not right for upper bound on clique in complement.
            
            # Let me reconsider. To upper bound the maximum clique in a graph H,
            # we color H. The chromatic number chi(H) >= omega(H) (clique number).
            # So we color the complement graph H = complement of G.
            # A proper coloring of H assigns colors such that adjacent vertices in H have different colors.
            # Adjacent in H means no conflict in G.
            # So a color class in H is a set of vertices no two of which are adjacent in H,
            # i.e., every pair has a conflict in G.
            # That's a clique in G, not helpful for upper bounding clique in H.
            
            # I'm confusing myself. Let me use the standard approach:
            # We want max clique in H (complement of G).
            # We color H properly. The number of colors is an upper bound on the clique number of H.
            # Proper coloring of H: adjacent vertices in H get different colors.
            # Adjacent in H = no conflict in G.
            # So if two vertices have no conflict in G, they must have different colors.
            # A color class is a set of vertices where no two are adjacent in H,
            # i.e., every pair has a conflict in G.
            # The size of the largest color class is NOT an upper bound on clique size in H.
            # The NUMBER of colors used is the upper bound.
            
            # So the algorithm is:
            # 1. Pick a vertex v from remaining.
            # 2. Assign it the current color.
            # 3. Remove from remaining all vertices that are adjacent to v in H
            #    (i.e., no conflict in G with v).
            # 4. Also remove v itself.
            # 5. The current color class is {v} plus... wait, no.
            #    In greedy coloring, we assign the smallest available color.
            #    For upper bound, we can use a simpler approach:
            #    Process vertices in some order, assign each to the first color class
            #    where it doesn't conflict (in H) with existing members.
            
            # Let me restart the coloring function.
            break
        
        # Simpler greedy coloring for upper bound:
        # Order vertices, assign each to the first color class where it has no H-adjacency.
        # H-adjacency means no conflict in G.
        # So a vertex can join a color class if it has a conflict in G with all members of that class.
        
        # Actually, for the purpose of upper bounding the clique in H,
        # we can use the following: the number of colors in a proper coloring of H
        # is >= omega(H). So we want to minimize colors (or just use any proper coloring).
        
        # Let me implement a proper greedy coloring of H.
        
        # Convert P to list of vertices
        verts = []
        temp = P
        while temp:
            bit = temp & (-temp)
            v = bit.bit_length() - 1
            verts.append(v)
            temp ^= bit
        
        # Order by degree in H (descending) for better coloring
        # Degree in H = number of non-neighbors in G = (len(verts) - 1) - (number of conflicts in G with other verts in P)
        # For simplicity, use natural order for now.
        
        color_classes = []  # list of bitmasks
        assigned = 0  # bitmask of assigned vertices
        
        for v in verts:
            v_bit = 1 << v
            if assigned & v_bit:
                continue
            # Try to assign to an existing color class
            placed = False
            for ci, cc in enumerate(color_classes):
                # Check if v can join color class ci
                # v must have a conflict in G with all vertices in cc
                # i.e., for all u in cc, (v,u) is in conflict_set
                # Equivalent: in H, v is not adjacent to any u in cc
                # H-adjacency: no conflict in G
                # So we need: for all u in cc, (v,u) in conflict_set
                # In bitmask terms: the set of H-neighbors of v within P should not intersect cc
                # H-neighbors of v = comp_adj[v] & P
                # We need (comp_adj[v] & P) & cc == 0
                h_neighbors = comp_adj[v] & P
                if (h_neighbors & cc) == 0:
                    color_classes[ci] |= v_bit
                    assigned |= v_bit
                    placed = True
                    break
            
            if not placed:
                color_classes.append(v_bit)
                assigned |= v_bit
        
        return len(color_classes), color_classes
    
    # Branch and bound for maximum clique in H (complement of G)
    # We use a recursive approach with bitsets
    
    best_clique_size = best_size  # from greedy
    best_clique = set(best_set)
    
    # For efficiency, let's use an iterative deepening or a proper stack-based approach
    # Given the time constraint, let's try a limited branch-and-bound
    
    # Actually, with 256 vertices, even a good B&B might be too slow if the graph is dense.
    # Let's check: how many conflicts are there?
    
    num_conflicts = len(conflict_set)
    # If the graph is sparse, the complement is dense, and max clique in complement is large.
    # If the graph is dense, the complement is sparse, and max clique in complement is small.
    
    # Let's just try the B&B with a time limit and see.
    
    # Tomita's algorithm uses a specific vertex ordering and coloring.
    # For simplicity, let's use a recursive B&B with the following:
    # - Order vertices by degree in H (descending)
    # - At each node, color the remaining candidates to get an upper bound
    # - If upper bound <= best, prune
    
    # Given the complexity, let's first check if our greedy solution is likely optimal
    # by trying a few local improvements.
    
    # Local search: try to swap elements
    current_set = set(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing one element and adding another
        for rem in list(current_set):
            if time.time() > deadline:
                break
            new_set = current_set - {rem}
            for c in range(1, N + 1):
                if c in new_set:
                    continue
                if all((c, u) not in conflict_set and (u, c) not in conflict_set for u in new_set):
                    if len(new_set | {c}) > len(current_set):
                        current_set = new_set | {c}
                        improved = True
                        break
            if improved:
                break
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # Now try a more systematic approach: use the fact that odds never conflict
    # (since for odd a,b, a+b is even, a*b is odd, so (a*b)%(a+b) != 0)
    # Wait, is that true? Let's verify.
    # If a and b are both odd, a+b is even, a*b is odd.
    # An odd number cannot be divisible by an even number (unless the even number is 0, which it's not).
    # So yes, all odds are mutually non-conflicting.
    # So we can always include all 128 odds.
    
    # The question is how many evens we can add.
    # Let's focus on finding the max set of evens that can be added to the odds.
    
    odds_set = set(x for x in range(1, N + 1) if x % 2 == 1)
    evens_list = [x for x in range(2, N + 1) if x % 2 == 0]
    
    # Precompute which evens conflict with which odds
    # But since odds never conflict with each other, we just need to check even-odd and even-even conflicts.
    
    # For each even e, determine which odds it conflicts with.
    # If e conflicts with any odd, it cannot be added (since we want to include all odds).
    # Wait, we don't HAVE to include all odds. But it's a good starting point.
    
    # Let's find evens that don't conflict with ANY odd.
    # If an even e conflicts with some odd o, then we can't have both e and o.
    # Since we want to maximize size, and there are 128 odds, it's probably best to keep all odds
    # and only add evens that don't conflict with any odd.
    
    safe_evens = []
    for e in evens_list:
        conflicts_with_odd = False
        for o in odds_set:
            if (e, o) in conflict_set or (o, e) in conflict_set:
                conflicts_with_odd = True
                break
        if not conflicts_with_odd:
            safe_evens.append(e)
    
    # Now find the max independent set among safe_evens in the conflict graph (only even-even conflicts matter)
    # Since safe_evens don't conflict with odds, we can add them freely as long as they don't conflict with each other.
    
    # Build subgraph on safe_evens
    # Find max independent set in this subgraph
    
    # For a small set, we can use brute force or B&B
    # Let's use a simple B&B
    
    def max_independent_subset(vertices, conflict_pairs):
        """Find max independent set in the subgraph induced by vertices."""
        if not vertices:
            return set()
        
        # Convert to list
        verts = list(vertices)
        n = len(verts)
        
        # If n is small, use brute force
        if n <= 20:
            best = set()
            for mask in range(1 << n):
                s = set()
                for i in range(n):
                    if mask & (1 << i):
                        s.add(verts[i])
                # Check if independent
                valid = True
                verts_list = list(s)
                for i in range(len(verts_list)):
                    for j in range(i + 1, len(verts_list)):
                        a, b = verts_list[i], verts_list[j]
                        if (a, b) in conflict_pairs or (b, a) in conflict_pairs:
                            valid = False
                            break
                    if not valid:
                        break
                if valid and len(s) > len(best):
                    best = s
            return best
        
        # For larger sets, use greedy + local search
        # Greedy: order by degree (ascending), add if possible
        degrees = {}
        for v in verts:
            deg = 0
            for u in verts:
                if u != v and ((v, u) in conflict_pairs or (u, v) in conflict_pairs):
                    deg += 1
            degrees[v] = deg
        
        ordered = sorted(verts, key=lambda x: degrees[x])
        s = set()
        for v in ordered:
            if all((v, u) not in conflict_pairs and (u, v) not in conflict_pairs for u in s):
                s.add(v)
        
        # Local search
        improved = True
        while improved:
            improved = False
            for rem in list(s):
                new_s = s - {rem}
                for c in verts:
                    if c in new_s:
                        continue
                    if all((c, u) not in conflict_pairs and (u, c) not in conflict_pairs for u in new_s):
                        if len(new_s | {c}) > len(s):
                            s = new_s | {c}
                            improved = True
                            break
                if improved:
                    break
        
        return s
    
    # Conflict pairs among safe_evens
    even_conflicts = set()
    for i in range(len(safe_evens)):
        for j in range(i + 1, len(safe_evens)):
            a, b = safe_evens[i], safe_evens[j]
            if (a, b) in conflict_set or (b, a) in conflict_set:
                even_conflicts.add((a, b))
    
    if safe_evens:
        safe_evens_set = set(safe_evens)
        max_safe = max_independent_subset(safe_evens_set, even_conflicts)
        candidate = odds_set | max_safe
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_set = candidate
    
    # Also try: don't include all odds, but include more evens
    # This is harder, but let's try a heuristic:
    # For each even, compute the number of odds it conflicts with.
    # If it conflicts with few odds, it might be worth dropping those odds to include the even.
    
    # This is a complex optimization. Given time constraints, let's stick with the above
    # and do a final verification.
    
    # Final verification
    best_list = sorted(best_set)
    valid = True
    for i in range(len(best_list)):
        for j in range(i + 1, len(best_list)):
            a, b = best_list[i], best_list[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break
    
    if not valid or len(best_list) == 0:
        return list(range(1, N + 1, 2))
    
    return best_list
# EVOLVE-BLOCK-END
