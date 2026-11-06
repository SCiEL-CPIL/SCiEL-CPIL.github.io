import time
import random
import itertools

def construct():
    start_time = time.time()
    max_time = 110  # Safety margin for 120s limit
    
    # Precompute C6 cycles for a maximum n=20
    # A C6 is a set of 6 distinct vertices {v0, v1, v2, v3, v4, v5}
    # The edges in the cycle are (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
    # Note: The cycle is not induced, so other edges between these vertices can be any color.
    # We just need to check if the 6 cycle edges are all the same color.
    
    max_n = 20
    cycles = []
    for combo in itertools.combinations(range(max_n), 6):
        # 5 distinct cycles for 6 vertices (up to rotation/reflection)
        # Actually, for a set of 6 vertices, there are (6-1)! / 2 = 60/2 = 30 cycles?
        # No, the number of cycles in K6 is (6-1)!/2 = 30.
        # Wait, (n-1)!/2 is the number of Hamiltonian cycles in Kn.
        # For n=6, (6-1)!/2 = 60/2 = 30.
        # Let's generate them explicitly to be safe.
        # Fix v0. Permute the remaining 5.
        # v0, v1, v2, v3, v4, v5 is a cycle.
        # v0, v5, v4, v3, v2, v1 is the same cycle reversed.
        # So we can just iterate permutations of the last 5, but divide by 2?
        # Or just generate all 30.
        
        # Let's use a simpler method:
        # For a set of 6 vertices, the cycles are defined by the order.
        # We can just store the 6 edges for each of the 30 cycles.
        # However, generating 30 cycles per combination of 6 vertices from 20
        # is 38760 * 30 = 1,162,800 cycles.
        # Each cycle has 6 edges.
        # Total edge references: ~7 million.
        # This might be too slow to precompute or too much memory.
        
        # Alternative: Count monochromatic C6s on the fly or use a faster check.
        # But checking all 1.16M cycles for every move is too slow.
        
        # Let's reconsider the cycle count.
        # The task says "No colour class may contain a six-cycle".
        # This means we are looking for a $C_6$ in the subgraph induced by edges of color $c$.
        
        # Let's optimize the cycle generation.
        # For a given set of 6 vertices, there are 30 cycles.
        # We can represent a cycle by the sorted tuple of its edges.
        
        pass # We will generate cycles lazily or use a different strategy if this is too heavy.
        
    # Actually, 1.16M cycles is a lot.
    # Let's try a different heuristic: Local search on edge colors.
    # We can estimate the number of C6s by checking specific patterns.
    
    # Let's stick to the Simulated Annealing but with a smarter evaluation.
    # Instead of precomputing all cycles, we can use a random sampling of cycles for evaluation?
    # No, we need exact validity.
    
    # Let's try to construct a valid coloring for n=10, 11, etc.
    # The seed gives n=9.
    # Let's try to improve n.
    
    def edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * n - (i + 1) // 2 + (j - i - 1)
        
    def try_solve(n, time_budget):
        # Initialize random coloring
        colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        
        # Precompute cycles for this specific n
        # Number of cycles in Kn is (n-1)!/2
        # For n=10, 9!/2 = 181440.
        # For n=11, 10!/2 = 1814400.
        # For n=12, 11!/2 = 19958400.
        # Precomputing cycles for n>=12 is likely too slow/memory heavy.
        # We will limit our search to n <= 11 with full cycle precomputation.
        # For n > 11, we might use a heuristic check or just return the best found.
        
        if n > 11:
            # Fallback: return a known valid coloring for n=9 or 10 if we can find one quickly
            # Or just try a few random restarts and check a subset of cycles?
            # Actually, let's just try n=10 and 11 with SA.
            # If n > 11, we assume it's hard and return a smaller n if found.
            pass

        cycles = []
        # Generate all C6s in Kn
        # We iterate over all combinations of 6 vertices
        for combo in itertools.combinations(range(n), 6):
            # Generate 30 cycles for these 6 vertices
            # Fix vertex 0 of the combo. Permute the other 5.
            # Each permutation (v1, v2, v3, v4, v5) defines a cycle (0, v1, v2, v3, v4, v5, 0)
            # But (v1, v2, v3, v4, v5) and (v5, v4, v3, v2, v1) are the same cycle.
            # So we only need to consider half of the permutations?
            # Number of permutations of 5 items is 120. Half is 60.
            # But wait, the cycle (0, 1, 2, 3, 4, 5) is the same as (0, 5, 4, 3, 2, 1).
            # So yes, 60 unique cycles per set of 6 vertices?
            # No, the formula (n-1)!/2 for Hamiltonian cycles in Kn is correct.
            # For n=6, (6-1)!/2 = 30.
            # So for each set of 6 vertices, there are 30 cycles.
            
            # Let's generate the 30 cycles.
            # We can fix the first vertex as the smallest in the combo to avoid symmetry?
            # No, the vertices are distinct.
            # Let's just generate all permutations of the last 5 vertices and take the first half?
            # It's easier to just generate all 120 and deduplicate by edge set?
            # Or just accept the 30 unique cycles.
            
            # Efficient way:
            # Fix v0.
            # Permute v1..v5.
            # Cycle is (v0, v1, v2, v3, v4, v5, v0).
            # To get 30 unique cycles, we can fix v1 to be the smallest of the remaining 5?
            # No, that's not quite right.
            # Let's just generate all 120 and use a set of frozensets of edges to deduplicate.
            # This might be slow for precomputation.
            
            # Alternative: Just store the 6 edges for each of the 30 cycles.
            # How to get 30 unique cycles from 6 vertices?
            # The number of ways to arrange 6 items in a circle is (6-1)! = 120.
            # Each cycle is counted twice (clockwise/counter-clockwise).
            # So 60?
            # Wait, (n-1)!/2 is the number of *undirected* cycles.
            # So 30 is correct.
            
            # Let's just iterate over all permutations of the 5 remaining vertices.
            # If we do this for all 6 vertices, we will generate 120 cycles per combo.
            # Many will be duplicates.
            # To be safe and simple, let's just generate all 120 and add them to a set of tuples of edges.
            # This will take some time but ensures correctness.
            
            pass # This precomputation is likely too slow for n=11 in Python.
            
        # Given the time constraints, a full exact search for C6 in K11+ is difficult in Python.
        # Let's return a known good result.
        # The seed provides n=9.
        # We can try to find n=10 or 11 with a simpler heuristic.
        
        # Let's try a random restart local search for n=10.
        # We will check for C6s by sampling? No, we need exact.
        
        # Let's just return the seed result if we can't find better quickly.
        # But we can try to improve the seed.
        
        # Let's try n=10 with a simple heuristic:
        # Random coloring, then minimize conflicts.
        # Conflict: A C6 is a conflict.
        # We can't easily count C6s without precomputation.
        
        # Let's use a different approach:
        # We know that a 4-coloring of K_n with no monochromatic C6 exists for n=17.
        # We can try to construct one using a known algebraic structure or random search with a specific pattern.
        
        # For now, let's return the seed result to be safe, but try to see if we can get n=10.
        
        return None

    # Let's try to find a valid coloring for n=10.
    # If we fail, we return n=9.
    
    # Actually, the prompt says "Maximize n".
    # The seed is n=9.
    # If we can't find n=10, we return n=9.
    
    # Let's try a simple random search for n=10.
    # We will check for C6s by iterating over all 6-vertex subsets and checking if any of the 30 cycles are monochromatic.
    # Number of 6-subsets in K10 is C(10,6) = 210.
    # 210 * 30 = 6300 cycles to check.
    # This is very fast!
    # So we CAN do exact search for n=10.
    
    # For n=11: C(11,6) = 462. 462 * 30 = 13860 cycles. Still fast.
    # For n=12: C(12,6) = 924. 924 * 30 = 27720 cycles. Still fast.
    # For n=13: C(13,6) = 1716. 1716 * 30 = 51480 cycles. Fast.
    # For n=14: C(14,6) = 3003. 3003 * 30 = 90090 cycles. Fast.
    # For n=15: C(15,6) = 5005. 5005 * 30 = 150150 cycles. Fast.
    # For n=16: C(16,6) = 8008. 8008 * 30 = 240240 cycles. Fast.
    # For n=17: C(17,6) = 12376. 12376 * 30 = 371280 cycles. Fast.
    # For n=18: C(18,6) = 18564. 18564 * 30 = 556920 cycles. Fast.
    # For n=19: C(19,6) = 27132. 27132 * 30 = 813960 cycles. Fast.
    # For n=20: C(20,6) = 38760. 38760 * 30 = 1162800 cycles. Still manageable.
    
    # So we CAN precompute cycles for n=20!
    # Let's do it.
    
    def precompute_cycles(n):
        cycles = []
        for combo in itertools.combinations(range(n), 6):
            # Generate 30 unique cycles
            # Fix v0 = combo[0]
            v0 = combo[0]
            rest = combo[1:]
            # Permute rest
            seen = set()
            for perm in itertools.permutations(rest):
                # Cycle: v0, perm[0], perm[1], perm[2], perm[3], perm[4], v0
                # To avoid duplicates, we can require perm[0] < perm[4]?
                # Or just add the edge set to a set.
                edges = []
                edges.append((v0, perm[0]))
                edges.append((perm[0], perm[1]))
                edges.append((perm[1], perm[2]))
                edges.append((perm[2], perm[3]))
                edges.append((perm[3], perm[4]))
                edges.append((perm[4], v0))
                
                # Normalize edges
                norm_edges = tuple(sorted((min(e[0], e[1]), max(e[0], e[1])) for e in edges))
                if norm_edges not in seen:
                    seen.add(norm_edges)
                    cycles.append(norm_edges)
        return cycles

    # Precompute for n=20
    # This might take a few seconds.
    time_limit = 10
    cycles_20 = precompute_cycles(20)
    
    # Now, use Simulated Annealing to find a valid coloring for n=20, 19, etc.
    # We start with n=20.
    # If we find a valid coloring, we return it.
    # If we don't, we try n=19, etc.
    
    def count_c6s(colors, n, cycles):
        count = 0
        for cycle in cycles:
            # Check if all edges in cycle have the same color
            c0 = colors[edge_index(cycle[0][0], cycle[0][1], n)]
            # Actually, cycle edges are stored as (u, v).
            # We need to map (u, v) to index.
            # Let's store the indices in the cycle tuple to speed this up.
            pass
        return count

    # Let's optimize: store edge indices in the cycle.
    def precompute_cycles_optimized(n):
        cycles = []
        for combo in itertools.combinations(range(n), 6):
            v0 = combo[0]
            rest = combo[1:]
            seen = set()
            for perm in itertools.permutations(rest):
                edges = []
                edges.append((v0, perm[0]))
                edges.append((perm[0], perm[1]))
                edges.append((perm[1], perm[2]))
                edges.append((perm[2], perm[3]))
                edges.append((perm[3], perm[4]))
                edges.append((perm[4], v0))
                
                norm_edges = tuple(sorted((min(e[0], e[1]), max(e[0], e[1])) for e in edges))
                if norm_edges not in seen:
                    seen.add(norm_edges)
                    # Store indices
                    idxs = tuple(edge_index(e[0], e[1], n) for e in norm_edges)
                    cycles.append(idxs)
        return cycles

    # Precompute for n=20
    cycles_20 = precompute_cycles_optimized(20)
    
    # Simulated Annealing
    def solve(n, time_budget):
        # Use cycles for n=20, but filter to those involving only vertices < n?
        # No, cycles are specific to the vertex set.
        # We need cycles for the specific n.
        # Let's precompute cycles for each n we try.
        
        cycles_n = precompute_cycles_optimized(n)
        
        colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        
        # Initial count
        def get_count(colors):
            count = 0
            for cycle in cycles_n:
                c0 = colors[cycle[0]]
                # Check if all are c0
                # Optimization: break early if mismatch
                is_mono = True
                for idx in cycle[1:]:
                    if colors[idx] != c0:
                        is_mono = False
                        break
                if is_mono:
                    count += 1
            return count
            
        current_count = get_count(colors)
        
        temp = 1.0
        cool_rate = 0.999
        min_temp = 0.01
        
        start = time.time()
        
        while time.time() - start < time_budget and temp > min_temp:
            # Pick random edge
            i = random.randint(0, len(colors) - 1)
            old_color = colors[i]
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
                
            colors[i] = new_color
            new_count = get_count(colors)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < pow(2.71828, -delta / temp):
                current_count = new_count
                if current_count == 0:
                    return colors
            else:
                colors[i] = old_color
                
            temp *= cool_rate
            
        return None

    # Try n=20 down to 10
    for n in range(20, 9, -1):
        if time.time() - start_time > max_time:
            break
            
        # Budget for this n
        budget = min(10, max_time - (time.time() - start_time))
        if budget <= 0:
            break
            
        result = solve(n, budget)
        if result:
            return {'n': n, 'colors': result}
            
    # If no better found, return seed
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

