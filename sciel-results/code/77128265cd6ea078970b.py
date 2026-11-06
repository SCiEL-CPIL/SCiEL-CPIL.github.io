import time
import random
import math
import itertools

def construct():
    deadline = time.time() + 115  # Safety margin
    
    def generate_c6s(n):
        """Generate all 6-cycles in K_n as lists of edge indices."""
        # Edge index mapping
        def edge_idx(i, j):
            if i > j: i, j = j, i
            return i * (i - 1) // 2 + (j - i - 1)
        
        cycles = []
        vertices = list(range(n))
        # A 6-cycle is defined by 6 distinct vertices.
        # There are 5 distinct cycles on 6 vertices.
        # We can generate all combinations of 6 vertices, and for each,
        # generate the 5 cycles.
        
        # To avoid duplicates, we can enforce an ordering on vertices?
        # Actually, the 5 cycles on vertices {v0..v5} are distinct sets of edges.
        # We just need to ensure we don't add the same set of edges twice.
        # Using a set of frozensets of edge indices is safe.
        
        seen = set()
        for v in itertools.combinations(vertices, 6):
            # v is a tuple of 6 vertices
            # Generate the 5 Hamiltonian cycles on these vertices
            # Fix v[0] to break rotational symmetry? No, we just need the edge sets.
            # The 5 cycles correspond to the 5 ways to connect 6 points in a cycle.
            # One way: fix the first vertex, permute the other 5, and divide by 2 (reversal).
            # But it's easier to just list the 5 specific permutations relative to a sorted order.
            
            # Let's just iterate over all permutations of the 6 vertices? 720 is small.
            # But we only need unique edge sets.
            # A cycle is (v0, v1, v2, v3, v4, v5, v0).
            # Edges: (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0).
            
            # We can generate all permutations of the 6 vertices,
            # compute the edge set, and add to 'seen'.
            # 720 * 18564 combinations is too slow?
            # 18564 * 720 ~ 13 million. Might be too slow in Python.
            
            # Better: There are exactly 5 distinct cycles on 6 labeled vertices.
            # Let's hardcode the 5 patterns relative to sorted vertices v0..v5.
            # Patterns (indices 0..5):
            # 1: 0-1-2-3-4-5-0
            # 2: 0-1-2-4-3-5-0
            # 3: 0-1-3-2-4-5-0
            # 4: 0-1-3-4-2-5-0
            # 5: 0-1-4-2-3-5-0
            # Wait, are these all distinct?
            # Yes, these are the 5 non-isomorphic Hamiltonian cycles on K6?
            # No, K6 has 5 distinct Hamiltonian cycles up to isomorphism?
            # Actually, the number of distinct Hamiltonian cycles in K_n is (n-1)! / 2.
            # For n=6, it is 5! / 2 = 60.
            # So there are 60 distinct edge sets for each set of 6 vertices.
            # My previous count of 5 was wrong.
            # Total cycles in K_n is \binom{n}{6} * 60.
            # For n=18, 18564 * 60 = 1,113,840 cycles.
            # This is still manageable.
            
            # So for each combination of 6 vertices, we generate 60 cycles.
            # 18564 * 60 = 1.1M cycles.
            # Storing 1.1M lists of 6 ints is ~ 1.1M * 6 * 8 bytes ~ 50MB. Acceptable.
            
            pass # Logic below
            
        # Efficient generation:
        # For each 6-combo, generate all 60 permutations that represent unique cycles.
        # To get 60 unique cycles from 6 vertices:
        # Fix v0. Permute v1..v5. 5! = 120 permutations.
        # Each cycle is counted twice (forward/backward). So 60 unique cycles.
        # So for each 6-combo, we can just iterate over permutations of the last 5 vertices.
        
        for v in itertools.combinations(vertices, 6):
            # v[0] is fixed as the start
            rest = v[1:]
            for p in itertools.permutations(rest):
                # Cycle: v[0] - p[0] - p[1] - p[2] - p[3] - p[4] - v[0]
                # This generates 120 cycles, but each is generated twice (reverse).
                # To avoid duplicates, we can enforce p[0] < p[4]? No, that's for undirected graphs.
                # The edge set of (A,B,C,D,E,F) is the same as (F,E,D,C,B,A).
                # So we can just use a set of frozensets of edge indices.
                
                edges = [
                    edge_idx(v[0], p[0]),
                    edge_idx(p[0], p[1]),
                    edge_idx(p[1], p[2]),
                    edge_idx(p[2], p[3]),
                    edge_idx(p[3], p[4]),
                    edge_idx(p[4], v[0])
                ]
                # Check if this specific permutation is a "canonical" one to avoid duplicates?
                # Easier: just add to a global set.
                # But 1.1M * 120 operations is 130M ops. Too slow.
                
                # Optimization: Only generate 60 unique cycles per combo.
                # Condition: p[0] < p[4] ensures we don't double count?
                # Let's check:
                # Cycle C: v0-p0-p1-p2-p3-p4-v0.
                # Reverse: v0-p4-p3-p2-p1-p0-v0.
                # In the reverse, the "p" part is (p4, p3, p2, p1, p0).
                # If we require p[0] < p[4], then for the reverse, the new p[0] is old p[4] and new p[4] is old p[0].
                # So exactly one of the pair will satisfy p[0] < p[4].
                # So we just check if p[0] < p[4].
                
                if p[0] < p[4]:
                    edges = [
                        edge_idx(v[0], p[0]),
                        edge_idx(p[0], p[1]),
                        edge_idx(p[1], p[2]),
                        edge_idx(p[2], p[3]),
                        edge_idx(p[3], p[4]),
                        edge_idx(p[4], v[0])
                    ]
                    cycles.append(tuple(edges))
        
        return cycles

    def solve(n):
        """Try to find a valid coloring for K_n."""
        num_edges = n * (n - 1) // 2
        if num_edges > 190: # n=20 has 190 edges
            return None
            
        # Precompute cycles
        cycles = generate_c6s(n)
        
        # Map edge index to list of (cycle_index, other_edges)
        # other_edges is a tuple of 5 edge indices
        edge_to_cycles = [[] for _ in range(num_edges)]
        for c_idx, cycle in enumerate(cycles):
            for e in cycle:
                others = [x for x in cycle if x != e]
                edge_to_cycles[e].append((c_idx, tuple(others)))
        
        # Initialize random coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Calculate initial cost
        def calc_cost(colors):
            cost = 0
            for cycle in cycles:
                c0 = colors[cycle[0]]
                if all(colors[e] == c0 for e in cycle[1:]):
                    cost += 1
            return cost
            
        current_cost = calc_cost(colors)
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        best_colors = colors[:]
        best_cost = current_cost
        
        start_time = time.time()
        steps = 0
        
        while time.time() < deadline - 5: # Leave 5s for returning/verification
            steps += 1
            
            # Pick random edge
            e = random.randint(0, num_edges - 1)
            old_color = colors[e]
            
            # Try other 3 colors
            best_delta = float('inf')
            best_new_color = -1
            
            for new_color in range(4):
                if new_color == old_color:
                    continue
                    
                # Calculate delta cost
                delta = 0
                for c_idx, others in edge_to_cycles[e]:
                    # Check if cycle was monochromatic with old_color
                    # It was monochromatic if all others had old_color AND e had old_color (which is true)
                    # So we just check if all others have old_color
                    is_old_mono = all(colors[x] == old_color for x in others)
                    if is_old_mono:
                        delta -= 1 # We break a monochromatic cycle
                    
                    # Check if cycle becomes monochromatic with new_color
                    # It becomes monochromatic if all others have new_color AND e has new_color (which is true)
                    is_new_mono = all(colors[x] == new_color for x in others)
                    if is_new_mono:
                        delta += 1 # We create a monochromatic cycle
                        
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
                    
            # Apply best move
            if best_new_color != -1:
                colors[e] = best_new_color
                current_cost += best_delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    
                if best_cost == 0:
                    return best_colors
                    
            # Cooling
            T *= cooling
            
        return best_colors if best_cost == 0 else None

    # Try n from 18 down to 17, then 16...
    # We want to maximize n.
    # Start with 18. If it fails, try 17.
    
    for n in [18, 17, 16, 15]:
        if time.time() > deadline - 10:
            break
            
        result = solve(n)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback: return a small valid coloring
    # n=9 is always possible?
    # Just return n=1 with empty colors? No, n>=1.
    # Let's return a known valid small one.
    # Or just return the last attempt's best even if not 0? No, must be valid.
    # If we fail to find valid, we should return a valid one.
    # A random coloring of K_5 is likely valid (no C6 possible in K5).
    # So n=5 is always valid.
    
    # Let's try to find a valid one for n=9 quickly if 17/18 failed.
    # But the loop above tries 18, 17, 16, 15.
    # If all fail, we return n=5.
    
    colors = [random.randint(0, 3) for _ in range(10)] # K5 has 10 edges
    return {'n': 5, 'colors': colors}
