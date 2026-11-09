import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        if time.time() > deadline - 5:
            break
        
        # Precomputation for K_n
        num_edges = n * (n - 1) // 2
        edge_indices = []
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                edge_indices.append((i, j))
                idx += 1
        
        # Generate all C6 cycles
        # A C6 is defined by 6 distinct vertices.
        # To avoid duplicates, we fix the smallest vertex as the start,
        # and ensure the second smallest is less than the largest?
        # Simpler: Generate all combinations of 6 vertices.
        # For each combination, generate all distinct cycles.
        # Number of distinct cycles on 6 labeled vertices is (6-1)!/2 = 60.
        
        cycles = []
        cycle_edges = []
        
        import itertools
        
        # Optimization: Only generate cycles if n is small enough or time permits
        # For n=17, 12376 combinations * 60 cycles = 742,560 cycles.
        # This is a lot to precompute in Python.
        
        # Alternative: Generate cycles on the fly? No, need edge_to_cycles.
        
        # Let's check time for precomputation.
        # 12376 * 60 = 742,560 cycles.
        # Each cycle has 6 edges.
        # Total operations ~ 4.5M. Should take < 1 sec.
        
        for vertices in itertools.combinations(range(n), 6):
            # Generate all 60 distinct cycles on these 6 vertices
            # Fix v0 = vertices[0]
            # Permute the other 5.
            # Cycle is v0, v1, v2, v3, v4, v5, v0
            # To avoid duplicates:
            # 1. v1 < v5 (to handle reversal)
            # 2. This generates 30 cycles per set of 6 vertices?
            # Total cycles in K6 is 60.
            # If we fix v0, we have 5! = 120 permutations.
            # Reversal symmetry: (v1..v5) is same as (v5..v1).
            # So 120 / 2 = 60 cycles.
            # Condition: v1 < v5 ensures we pick one from each pair.
            
            v0 = vertices[0]
            others = vertices[1:]
            for perm in itertools.permutations(others):
                v1, v2, v3, v4, v5 = perm
                if v1 > v5:
                    continue
                
                # Build cycle edges
                cycle_verts = [v0, v1, v2, v3, v4, v5]
                c_edges = []
                for k in range(6):
                    u = cycle_verts[k]
                    v = cycle_verts[(k + 1) % 6]
                    if u > v:
                        u, v = v, u
                    c_edges.append(edge_map[(u, v)])
                
                cycles.append(cycle_verts) # Not strictly needed, just for debug
                cycle_edges.append(c_edges)
        
        num_cycles = len(cycles)
        
        # Build edge_to_cycles
        edge_to_cycles = [[] for _ in range(num_edges)]
        for c_idx, c_edges in enumerate(cycle_edges):
            for e_idx in c_edges:
                edge_to_cycles[e_idx].append(c_idx)
        
        # Initialize coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Initialize cycle counts
        # cycle_counts[c_idx][k]
        cycle_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
        for c_idx, c_edges in enumerate(cycle_edges):
            for e_idx in c_edges:
                cycle_counts[c_idx][colors[e_idx]] += 1
        
        # Calculate initial mono_count
        mono_count = 0
        for c_idx in range(num_cycles):
            counts = cycle_counts[c_idx]
            if any(c == 6 for c in counts):
                mono_count += 1
        
        # Search
        # We use a simple local search with random restarts implicitly by trying different n
        # and within n, we just run until timeout for this n.
        
        # To improve success rate, we can add a "shake" if stuck.
        # But simple hill climbing with random restarts (by re-initializing) is better.
        # However, re-initialization takes time.
        # Let's just run the search for this n until deadline - 5s.
        
        while time.time() < deadline - 5:
            if mono_count == 0:
                return {'n': n, 'colors': colors}
            
            # Pick random edge
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            
            # Pick new color
            new_color = random.choice([c for c in range(4) if c != old_color])
            
            # Calculate delta
            delta = 0
            for c_idx in edge_to_cycles[e_idx]:
                counts = cycle_counts[c_idx]
                if counts[old_color] == 6:
                    delta -= 1
                if counts[new_color] == 5:
                    delta += 1
            
            # Acceptance criterion
            # Always accept if delta < 0
            # Accept if delta == 0 with some prob
            # Accept if delta > 0 with very low prob (annealing)
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.5:
                    accept = True
            else:
                if random.random() < 0.01: # 1% chance to escape local minima
                    accept = True
            
            if accept:
                colors[e_idx] = new_color
                mono_count += delta
                for c_idx in edge_to_cycles[e_idx]:
                    cycle_counts[c_idx][old_color] -= 1
                    cycle_counts[c_idx][new_color] += 1
    
    # If nothing found, return the seed solution (n=9)
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
