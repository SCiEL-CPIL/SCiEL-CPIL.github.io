import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Helper to get edge index for (i, j) in lex order list
    def get_edge_idx(i, j, n):
        if i > j:
            i, j = j, i
        # Number of edges (u, v) with u < i
        # u=0: n-1 edges
        # u=1: n-2 edges
        # ...
        # u=i-1: n-i edges
        # Total edges before i: sum_{k=0}^{i-1} (n-1-k)
        # = i*n - i*(i+1)//2
        # Within row i, j is the (j-i-1)-th edge (0-indexed)
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def generate_c6s(n):
        """Generate all 6-cycles in K_n as lists of edge indices."""
        cycles = []
        # Choose 6 distinct vertices
        for verts in itertools.combinations(range(n), 6):
            # There are 60 distinct cycles on 6 vertices
            # Fix one vertex to break rotational symmetry? No, just generate all permutations and deduplicate or use known count.
            # Easier: generate all permutations of the 6 vertices, 
            # but a cycle (v0..v5) is same as (v5..v0) and reverse.
            # Number of distinct cycles is (6-1)! / 2 = 60.
            # We can generate them by fixing v0 and permuting the rest, then dividing by 2 for reversal?
            # Actually, just iterate through all permutations of the 6 vertices.
            # A permutation (p0, p1, p2, p3, p4, p5) defines a cycle p0-p1-p2-p3-p4-p5-p0.
            # To avoid duplicates:
            # 1. Fix p0 to be the smallest vertex in the set? No, the set is fixed.
            # 2. Just generate all 720 permutations. 
            # 3. A cycle is represented by its edges. 
            # 4. We can use a set of frozensets of edges to deduplicate.
            # But 720 * 12376 is ~9M iterations. Might be slow in Python.
            
            # Optimized:
            # Fix v0 = verts[0].
            # Permute the other 5: 5! = 120.
            # Each cycle appears twice (once forward, once backward) in these 120?
            # Yes. So we generate 120 permutations, take the first 60?
            # Or just generate all 120, create edge tuple, and add to a local set.
            
            local_cycles = set()
            v0 = verts[0]
            rest = verts[1:]
            for perm in itertools.permutations(rest):
                path = (v0,) + perm
                edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    idx = get_edge_idx(u, v, n)
                    edges.append(idx)
                # Sort edges to create a canonical representation for deduplication
                local_cycles.add(tuple(sorted(edges)))
            
            cycles.extend(local_cycles)
        return cycles

    def check_coloring(n, colors):
        """Returns True if no color class has a C6."""
        # Precompute cycles for this n? 
        # It's better to precompute outside if we retry, but n changes.
        # For now, assume we pass cycles.
        pass

    # We will try n from 17 down to 9
    # To be efficient, we should precompute cycles for each n we try?
    # Or just try 17 first.
    
    best_n = 9
    best_colors = [min(j - i, 17 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)] # Seed solution for n=9? 
    # Wait, seed uses n=9. 
    # Let's use the seed's logic for fallback.
    def seed_solution(n):
        if n == 9:
            return [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
        return None

    current_best = seed_solution(9)
    current_n = 9

    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
        
        # Time slice for this n
        n_deadline = min(deadline, time.time() + 10.0)
        
        # Generate cycles
        # This might take a second for n=17
        if time.time() > n_deadline: continue
        cycles = generate_c6s(n)
        
        if not cycles:
            # n < 6, trivial
            colors = [0] * (n*(n-1)//2)
            current_n = n
            current_best = colors
            continue

        # Local Search
        # Initialize random coloring
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Count bad cycles
        def count_bad():
            count = 0
            for cyc in cycles:
                c0 = colors[cyc[0]]
                if c0 == colors[cyc[1]] == colors[cyc[2]] == colors[cyc[3]] == colors[cyc[4]] == colors[cyc[5]]:
                    count += 1
            return count

        bad_count = count_bad()
        
        while time.time() < n_deadline:
            if bad_count == 0:
                current_n = n
                current_best = colors[:]
                break
            
            # Pick a random bad cycle
            # To be faster, we could maintain a list of bad cycles, but updating it is hard.
            # Just pick random cycles until we find a bad one.
            # Or pick a random edge and check if it's in a bad cycle?
            # Let's pick a random cycle.
            # Optimization: If bad_count is small, this is fast. If large, it's slow.
            # Better: Pick a random edge, find a bad cycle containing it?
            # Simple approach: Pick random cycle.
            
            found_bad = False
            # Try a few random cycles
            for _ in range(10):
                cyc = random.choice(cycles)
                c0 = colors[cyc[0]]
                if c0 == colors[cyc[1]] == colors[cyc[2]] == colors[cyc[3]] == colors[cyc[4]] == colors[cyc[5]]:
                    found_bad = True
                    break
            
            if not found_bad:
                # If we can't find a bad cycle quickly, maybe it's 0?
                # Re-check
                bad_count = count_bad()
                if bad_count == 0:
                    current_n = n
                    current_best = colors[:]
                    break
                continue

            # Flip a random edge in this cycle to a different color
            edge_idx_in_cycle = random.randint(0, 5)
            e = cyc[edge_idx_in_cycle]
            old_c = colors[e]
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            # Calculate delta
            # This requires knowing which cycles contain 'e' and are affected.
            # Too complex for simple implementation.
            # Just flip and re-count?
            # Re-counting is 742k ops. ~0.1s.
            # 10s / 0.1s = 100 iterations. Too few.
            
            # We need a faster delta calculation.
            # Precompute: for each edge, which cycles contain it?
            # edge_to_cycles = {edge_idx: [cycle_indices]}
            
            # Let's build this map.
            # This is heavy but done once per n.
            
            # Re-implement loop with precomputed map
            break # Break out of inner while to rebuild logic
            
        # Re-do the search with precomputed map
        if time.time() < n_deadline:
            # Build edge to cycles map
            edge_to_cycles = {i: [] for i in range(num_edges)}
            for i, cyc in enumerate(cycles):
                for e in cyc:
                    edge_to_cycles[e].append(i)
            
            # Current bad count
            bad_count = count_bad()
            
            while time.time() < n_deadline:
                if bad_count == 0:
                    current_n = n
                    current_best = colors[:]
                    break
                
                # Pick a random bad cycle
                # We don't have a list of bad cycles.
                # Pick a random edge, then a random cycle containing it?
                # If the graph is dense with bad cycles, this works.
                
                # Better: Pick a random edge.
                e = random.randint(0, num_edges - 1)
                if not edge_to_cycles[e]:
                    continue
                    
                # Pick a random cycle containing e
                cyc_idx = random.choice(edge_to_cycles[e])
                cyc = cycles[cyc_idx]
                
                # Check if it's bad
                c0 = colors[cyc[0]]
                is_bad = (c0 == colors[cyc[1]] == colors[cyc[2]] == colors[cyc[3]] == colors[cyc[4]] == colors[cyc[5]])
                
                if not is_bad:
                    continue
                
                # Flip e
                old_c = colors[e]
                new_c = random.choice([c for c in range(4) if c != old_c])
                
                # Calculate new bad count
                # Only cycles containing e are affected.
                delta = 0
                for ci in edge_to_cycles[e]:
                    c = cycles[ci]
                    # Check if c was bad
                    c0 = colors[c[0]]
                    was_bad = (c0 == colors[c[1]] == colors[c[2]] == colors[c[3]] == colors[c[4]] == colors[c[5]])
                    
                    # Apply flip temporarily
                    colors[e] = new_c
                    
                    # Check if c is bad now
                    c0_new = colors[c[0]]
                    is_bad_now = (c0_new == colors[c[1]] == colors[c[2]] == colors[c[3]] == colors[c[4]] == colors[c[5]])
                    
                    if was_bad and not is_bad_now:
                        delta -= 1
                    elif not was_bad and is_bad_now:
                        delta += 1
                    # If both bad or both not bad, delta 0
                    # Note: If was_bad, it was monochromatic old_c.
                    # If new_c == old_c, no change. But we chose new_c != old_c.
                    # So if was_bad, it becomes NOT bad (unless other edges match new_c? No, it was monochromatic old_c, so all other edges are old_c. Changing e to new_c makes it non-monochromatic).
                    # So if was_bad, is_bad_now is always False.
                    # If not was_bad, it might become bad if all other 5 edges are new_c.
                    
                    # Revert flip for next check?
                    # We are modifying 'colors' in place.
                    # We need to be careful.
                    # Let's just calculate the change based on the state BEFORE flip.
                    
                # This logic is getting messy. 
                # Simpler: Just flip and re-count if bad_count is small?
                # Or, accept the flip if it doesn't increase bad count?
                # We don't know the change without checking.
                
                # Let's just do a simple hill climb:
                # Flip e.
                # Count bad cycles containing e.
                # If new_count < old_count, keep. Else revert.
                
                colors[e] = new_c
                new_bad_count = count_bad() # Expensive!
                
                if new_bad_count < bad_count:
                    bad_count = new_bad_count
                else:
                    colors[e] = old_c
                    # bad_count unchanged

    return {'n': current_n, 'colors': current_best}
