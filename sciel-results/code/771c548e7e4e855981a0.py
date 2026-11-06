import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Precompute all 6-cycles (as tuples of 6 distinct vertices) for a given n
    # A 6-cycle is a sequence of 6 distinct vertices v0,v1,...,v5 where edges
    # (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v5),(v5,v0) exist.
    # We enumerate all cyclic orderings of 6-subsets, modulo rotation and reversal.
    # Number of distinct cycles per 6-subset: (6-1)!/2 = 60
    # Total 6-cycles in K_n: C(n,6) * 60
    
    # For n=17: C(17,6)*60 = 12376*60 = 742560
    # For n=18: C(18,6)*60 = 18564*60 = 1113840
    
    def generate_c6s(n):
        """Generate all 6-cycles as tuples of 6 edges (i,j) with i<j."""
        cycles = []
        verts = range(n)
        for combo in itertools.combinations(verts, 6):
            # All cyclic permutations of 6 vertices, modulo rotation and reversal
            # Fix combo[0] as start to break rotation symmetry
            # Generate permutations of remaining 5, then break reversal
            rest = list(combo[1:])
            for perm in itertools.permutations(rest):
                cycle = (combo[0],) + perm
                # Break reversal: require cycle[1] < cycle[5]
                if cycle[1] >= cycle[5]:
                    continue
                # Convert to edge list
                edges = []
                for k in range(6):
                    a, b = cycle[k], cycle[(k+1) % 6]
                    if a > b:
                        a, b = b, a
                    edges.append((a, b))
                cycles.append(tuple(edges))
        return cycles
    
    # Map edge (i,j) to index in lexicographic order
    def edge_to_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
    
    def count_c6s_fast(colors, n, c6_list):
        """Count number of monochromatic 6-cycles."""
        # For each color, check each 6-cycle
        count = 0
        for c in range(4):
            for edges in c6_list:
                # Check if all edges in this cycle have color c
                all_same = True
                for (i, j) in edges:
                    idx = edge_to_idx(i, j, n)
                    if colors[idx] != c:
                        all_same = False
                        break
                if all_same:
                    count += 1
        return count
    
    def has_c6(colors, n, c6_list):
        """Check if any monochromatic 6-cycle exists."""
        for c in range(4):
            for edges in c6_list:
                all_same = True
                for (i, j) in edges:
                    idx = edge_to_idx(i, j, n)
                    if colors[idx] != c:
                        all_same = False
                        break
                if all_same:
                    return True
        return False
    
    # Try to find valid coloring
    # Strategy: 
    # 1. Start with n=17, try to find valid coloring
    # 2. If found, try n=18
    # 3. Use simulated annealing / genetic algorithm
    
    def solve_n(n, time_budget):
        """Try to find valid 4-coloring of K_n with no monochromatic C6."""
        num_edges = n * (n - 1) // 2
        c6_list = generate_c6s(n)
        
        # Precompute for each color c and each 6-cycle, the list of edge indices
        # This allows faster checking
        c6_edge_indices = []
        for edges in c6_list:
            c6_edge_indices.append(tuple(edge_to_idx(i, j, n) for (i, j) in edges))
        
        def count_c6s_v2(colors):
            count = 0
            for c in range(4):
                for edge_idxs in c6_edge_indices:
                    all_same = True
                    for idx in edge_idxs:
                        if colors[idx] != c:
                            all_same = False
                            break
                    if all_same:
                        count += 1
            return count
        
        def has_c6_v2(colors):
            for c in range(4):
                for edge_idxs in c6_edge_indices:
                    all_same = True
                    for idx in edge_idxs:
                        if colors[idx] != c:
                            all_same = False
                            break
                    if all_same:
                        return True
            return False
        
        # Simulated annealing
        best_n_found = None
        best_colors_found = None
        
        for n_try in [n, n - 1, n - 2, n - 3]:
            if n_try < 10:
                break
            if time.time() > deadline:
                break
            
            num_edges_try = n_try * (n_try - 1) // 2
            c6_list_try = generate_c6s(n_try)
            c6_edge_indices_try = []
            for edges in c6_list_try:
                c6_edge_indices_try.append(tuple(edge_to_idx(i, j, n_try) for (i, j) in edges))
            
            def count_c6s_try(colors):
                count = 0
                for c in range(4):
                    for edge_idxs in c6_edge_indices_try:
                        all_same = True
                        for idx in edge_idxs:
                            if colors[idx] != c:
                                all_same = False
                                break
                        if all_same:
                            count += 1
                return count
            
            def has_c6_try(colors):
                for c in range(4):
                    for edge_idxs in c6_edge_indices_try:
                        all_same = True
                        for idx in edge_idxs:
                            if colors[idx] != c:
                                all_same = False
                                break
                        if all_same:
                            return True
                return False
            
            # Try multiple random restarts
            for restart in range(20):
                if time.time() > deadline:
                    break
                
                # Initialize
                colors = [random.randint(0, 3) for _ in range(num_edges_try)]
                current_cost = count_c6s_try(colors)
                
                if current_cost == 0:
                    best_n_found = n_try
                    best_colors_found = colors[:]
                    break
                
                # Simulated annealing
                T = 1.0
                T_min = 0.01
                cooling = 0.995
                
                while T > T_min and time.time() < deadline:
                    # Pick a random edge to flip
                    idx = random.randint(0, num_edges_try - 1)
                    old_c = colors[idx]
                    new_c = random.randint(0, 3)
                    if new_c == old_c:
                        continue
                    colors[idx] = new_c
                    
                    new_cost = count_c6s_try(colors)
                    delta = new_cost - current_cost
                    
                    if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                        current_cost = new_cost
                        if current_cost == 0:
                            best_n_found = n_try
                            best_colors_found = colors[:]
                            break
                    else:
                        colors[idx] = old_c
                    
                    T *= cooling
                
                if best_n_found is not None:
                    break
            
            if best_n_found is not None:
                break
        
        return best_n_found, best_colors_found
    
    # Start with n=18, then 17
    for n_attempt in [18, 17]:
        n_found, colors_found = solve_n(n_attempt, 110)
        if n_found is not None:
            return {'n': n_found, 'colors': colors_found}
    
    # Fallback: return a known valid coloring for smaller n
    # For n=9, use a simple construction
    n = 9
    colors = [0] * (n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Use a pattern that avoids C6
            colors[idx] = (i + j) % 4
            idx += 1
    
    # Verify and if invalid, try random search for small n
    if time.time() < deadline - 5:
        # Quick verification
        c6_list = generate_c6s(n)
        c6_edge_indices = []
        for edges in c6_list:
            c6_edge_indices.append(tuple(edge_to_idx(i, j, n) for (i, j) in edges))
        
        def has_c6_fallback(colors):
            for c in range(4):
                for edge_idxs in c6_edge_indices:
                    all_same = True
                    for idx in edge_idxs:
                        if colors[idx] != c:
                            all_same = False
                            break
                    if all_same:
                        return True
            return False
        
        if not has_c6_fallback(colors):
            return {'n': n, 'colors': colors}
    
    # Last resort: return minimal valid coloring
    n = 5
    colors = [0] * (n * (n - 1) // 2)
    return {'n': n, 'colors': colors}
