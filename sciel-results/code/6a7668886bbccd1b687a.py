import random
import time
import itertools

def construct():
    # Target n=17
    n = 17
    # Edge index mapping
    def edge_index(i, j):
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)
        
    num_edges = n * (n - 1) // 2
    
    # Precompute all 6-cycles in K_n
    # A 6-cycle is a sequence of 6 distinct vertices v0..v5
    # Edges: (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
    # To avoid duplicates, we can fix v0=0 and v1 < v5?
    # Actually, for the check, we just need the list of edge indices.
    # There are ~4.4M cycles. Generating them might take time.
    # Let's generate them once.
    
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Generate cycles
    cycles = []
    vertices = list(range(n))
    # Fix v0=0 to reduce symmetry? No, we need all cycles.
    # Number of cycles is (n-1)! / (2 * 6) ? No.
    # Let's just iterate over combinations of 6 vertices.
    # C(17,6) = 12376 combinations.
    # For each combination, there are (6-1)! / 2 = 60 cycles.
    # Total 12376 * 60 = 742,560 unique cycles.
    # This is much smaller than 4.4M (which was closed walks).
    # 742k cycles is very manageable.
    
    for comb in itertools.combinations(vertices, 6):
        v0, v1, v2, v3, v4, v5 = comb
        # Generate all permutations of v1..v5
        for perm in itertools.permutations((v1, v2, v3, v4, v5)):
            # The cycle is v0, perm[0], perm[1], perm[2], perm[3], perm[4], v0
            # To avoid counting the same cycle multiple times (direction/reversal),
            # we can enforce a canonical form, but for checking monochromaticity,
            # we just need to check if the 6 edges exist in the same color.
            # However, iterating all permutations is 120 per comb.
            # 12376 * 120 = 1.4M checks.
            # Let's just generate the edge indices for the cycle.
            edges = [
                edge_index(v0, perm[0]),
                edge_index(perm[0], perm[1]),
                edge_index(perm[1], perm[2]),
                edge_index(perm[2], perm[3]),
                edge_index(perm[3], perm[4]),
                edge_index(perm[4], v0)
            ]
            # To avoid duplicates in the list of cycles to check, 
            # we can sort the edges or use a set.
            # But 1.4M entries in a list is fine.
            cycles.append(tuple(edges))
            
    # Remove duplicates to save time
    # Note: A cycle is a set of 6 edges.
    # Different permutations yield the same set of edges.
    # We can use a set of frozensets? Or just a set of tuples of sorted edges.
    unique_cycles = set()
    for cyc in cycles:
        unique_cycles.add(tuple(sorted(cyc)))
    cycles = list(unique_cycles)
    
    # Now we have the list of unique 6-cycles (sets of 6 edge indices).
    # Size is approx 742k.
    
    def count_c6(colors):
        count = 0
        for cyc in cycles:
            c0 = colors[cyc[0]]
            if c0 == colors[cyc[1]] == colors[cyc[2]] == colors[cyc[3]] == colors[cyc[4]] == colors[cyc[5]]:
                count += 1
        return count

    # Simulated Annealing
    # Init
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    current_energy = count_c6(colors)
    
    best_colors = colors[:]
    best_energy = current_energy
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    while T > min_T and time.time() < deadline:
        # Pick random edge
        e = random.randint(0, num_edges - 1)
        old_color = colors[e]
        new_color = random.choice([c for c in range(4) if c != old_color])
        
        # Calculate delta
        # This requires checking all cycles containing edge e.
        # Precomputing "cycles containing edge e" is expensive but doable.
        # Or, just recompute count_c6? 742k checks * 6 ops = 4.4M ops.
        # 4.4M ops in Python is ~0.05s.
        # If we do 1000 steps, that's 50s. Too slow.
        
        # We need a faster delta calculation.
        # Let's precompute for each edge, the list of cycles it belongs to.
        edge_to_cycles = {i: [] for i in range(num_edges)}
        for i, cyc in enumerate(cycles):
            for e in cyc:
                edge_to_cycles[e].append(i)
                
        # Now, to calculate delta for changing edge e from old_color to new_color:
        # For each cycle in edge_to_cycles[e]:
        #   Check if the other 5 edges are all new_color. If so, +1 (new C6).
        #   Check if the other 5 edges are all old_color. If so, -1 (old C6 removed).
        
        delta = 0
        for idx in edge_to_cycles[e]:
            cyc = cycles[idx]
            # Get the other 5 edges
            other_edges = [ce for ce in cyc if ce != e]
            
            # Check if all other edges are new_color
            if all(colors[oe] == new_color for oe in other_edges):
                delta += 1
            
            # Check if all other edges are old_color
            if all(colors[oe] == old_color for oe in other_edges):
                delta -= 1
        
        new_energy = current_energy + delta
        
        if new_energy < current_energy or random.random() < pow(2.718, -new_energy / T) if new_energy > current_energy else True:
            colors[e] = new_color
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_colors = colors[:]
                
        T *= cooling
        
        if best_energy == 0:
            break

    # Verify best
    if count_c6(best_colors) == 0:
        return {'n': n, 'colors': best_colors}
    
    # If not found, try n=16
    n = 16
    num_edges = n * (n - 1) // 2
    def edge_index16(i, j):
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)
        
    # Regenerate cycles for n=16
    cycles16 = []
    vertices16 = list(range(n))
    for comb in itertools.combinations(vertices16, 6):
        v0, v1, v2, v3, v4, v5 = comb
        for perm in itertools.permutations((v1, v2, v3, v4, v5)):
            edges = [
                edge_index16(v0, perm[0]),
                edge_index16(perm[0], perm[1]),
                edge_index16(perm[1], perm[2]),
                edge_index16(perm[2], perm[3]),
                edge_index16(perm[3], perm[4]),
                edge_index16(perm[4], v0)
            ]
            cycles16.append(tuple(sorted(edges)))
    unique_cycles16 = set(cycles16)
    cycles16 = list(unique_cycles16)
    
    def count_c6_16(colors):
        count = 0
        for cyc in cycles16:
            c0 = colors[cyc[0]]
            if c0 == colors[cyc[1]] == colors[cyc[2]] == colors[cyc[3]] == colors[cyc[4]] == colors[cyc[5]]:
                count += 1
        return count

    colors16 = [random.randint(0, 3) for _ in range(num_edges)]
    current_energy16 = count_c6_16(colors16)
    best_colors16 = colors16[:]
    best_energy16 = current_energy16
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute edge to cycles for 16
    edge_to_cycles16 = {i: [] for i in range(num_edges)}
    for i, cyc in enumerate(cycles16):
        for e in cyc:
            edge_to_cycles16[e].append(i)

    while T > min_T and time.time() < deadline:
        e = random.randint(0, num_edges - 1)
        old_color = colors16[e]
        new_color = random.choice([c for c in range(4) if c != old_color])
        
        delta = 0
        for idx in edge_to_cycles16[e]:
            cyc = cycles16[idx]
            other_edges = [ce for ce in cyc if ce != e]
            
            if all(colors16[oe] == new_color for oe in other_edges):
                delta += 1
            if all(colors16[oe] == old_color for oe in other_edges):
                delta -= 1
        
        new_energy16 = current_energy16 + delta
        
        if new_energy16 < current_energy16 or random.random() < pow(2.718, -new_energy16 / T) if new_energy16 > current_energy16 else True:
            colors16[e] = new_color
            current_energy16 = new_energy16
            
            if current_energy16 < best_energy16:
                best_energy16 = current_energy16
                best_colors16 = colors16[:]
                
        T *= cooling
        
        if best_energy16 == 0:
            break

    if count_c6_16(best_colors16) == 0:
        return {'n': n, 'colors': best_colors16}
        
    # Fallback to seed
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
