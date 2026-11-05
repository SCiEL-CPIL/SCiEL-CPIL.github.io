import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    
    # List of all edges (i, j)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # List of all K4s (sets of 4 vertices)
    k4s = list(combinations(range(n), 4))
    num_k4s = len(k4s)  # 1820
    
    # For each K4, find the indices of its 6 edges
    k4_edge_indices = []
    for k4 in k4s:
        vs = list(k4)
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(vs[i], vs[j])])
        k4_edge_indices.append(tuple(e_indices))
        
    # For each edge, find which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)

    def compute_deficit(colors):
        """Compute total deficit and per-K4 distinct counts."""
        k4_distinct = [0] * num_k4s
        total_def = 0
        for k4_idx, e_indices in enumerate(k4_edge_indices):
            c0 = colors[e_indices[0]]
            c1 = colors[e_indices[1]]
            c2 = colors[e_indices[2]]
            c3 = colors[e_indices[3]]
            c4 = colors[e_indices[4]]
            c5 = colors[e_indices[5]]
            # Fast distinct count for 6 elements
            if c0 == c1 == c2 == c3 == c4 == c5:
                d = 1
            elif c0 == c1 == c2 == c3 == c4:
                d = 2
            elif c0 == c1 == c2 == c3:
                if c4 == c5:
                    d = 2
                else:
                    d = 3
            elif c0 == c1 == c2:
                if c3 == c4 == c5:
                    d = 2
                elif c3 == c4:
                    d = 3
                elif c3 == c5:
                    d = 3
                elif c4 == c5:
                    d = 3
                else:
                    d = 4
            else:
                # More complex case, use set
                d = len({c0, c1, c2, c3, c4, c5})
            k4_distinct[k4_idx] = d
            if d < 5:
                total_def += 5 - d
        return total_def, k4_distinct

    def local_search_smooth(max_colors, time_limit, initial_colors=None):
        """
        Local search with fast distinct counting.
        """
        if initial_colors is not None:
            colors = initial_colors[:]
        else:
            colors = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
        
        total_def, k4_distinct = compute_deficit(colors)
        best_deficit = total_def
        best_colors = colors[:]
        
        # Simulated Annealing parameters
        temp = 15.0
        min_temp = 0.01
        cool_rate = 0.9997
        
        steps_since_improve = 0
        max_steps_no_improve = 8000
        
        # Precompute edge_to_k4s as tuples for speed
        edge_to_k4s_tuple = [tuple(lst) for lst in edge_to_k4s]
        
        # Main Loop
        while time.time() < start_time + time_limit:
            if total_def == 0:
                return colors, 0
            
            if steps_since_improve > max_steps_no_improve:
                # Restart with new random init
                colors = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
                total_def, k4_distinct = compute_deficit(colors)
                temp = 15.0
                steps_since_improve = 0
                if total_def < best_deficit:
                    best_deficit = total_def
                    best_colors = colors[:]
                continue
            
            # Pick a random edge, but bias toward edges in deficient K4s
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            
            # Pick a new color different from old
            new_color = random.randint(0, max_colors - 1)
            if new_color == old_color:
                continue
            
            # Calculate Delta Deficit
            delta_deficit = 0
            affected_k4s = edge_to_k4s_tuple[e_idx]
            
            # Store old states for revert if needed
            old_states = []
            
            for k4_idx in affected_k4s:
                e_indices = k4_edge_indices[k4_idx]
                old_d = k4_distinct[k4_idx]
                old_def = 5 - old_d if old_d < 5 else 0
                old_states.append((k4_idx, old_d, old_def))
                
                # Get the 6 edge colors, update one
                c0 = colors[e_indices[0]]
                c1 = colors[e_indices[1]]
                c2 = colors[e_indices[2]]
                c3 = colors[e_indices[3]]
                c4 = colors[e_indices[4]]
                c5 = colors[e_indices[5]]
                
                # Find which position in this K4 corresponds to e_idx
                # and update the color
                new_c0, new_c1, new_c2, new_c3, new_c4, new_c5 = c0, c1, c2, c3, c4, c5
                for pos, e_i in enumerate(e_indices):
                    if e_i == e_idx:
                        if pos == 0: new_c0 = new_color
                        elif pos == 1: new_c1 = new_color
                        elif pos == 2: new_c2 = new_color
                        elif pos == 3: new_c3 = new_color
                        elif pos == 4: new_c4 = new_color
                        elif pos == 5: new_c5 = new_color
                        break
                
                # Count distinct in the new 6 colors
                if new_c0 == new_c1 == new_c2 == new_c3 == new_c4 == new_c5:
                    new_d = 1
                elif new_c0 == new_c1 == new_c2 == new_c3 == new_c4:
                    new_d = 2
                elif new_c0 == new_c1 == new_c2 == new_c3:
                    if new_c4 == new_c5:
                        new_d = 2
                    else:
                        new_d = 3
                elif new_c0 == new_c1 == new_c2:
                    if new_c3 == new_c4 == new_c5:
                        new_d = 2
                    elif new_c3 == new_c4:
                        new_d = 3
                    elif new_c3 == new_c5:
                        new_d = 3
                    elif new_c4 == new_c5:
                        new_d = 3
                    else:
                        new_d = 4
                else:
                    new_d = len({new_c0, new_c1, new_c2, new_c3, new_c4, new_c5})
                
                new_def = 5 - new_d if new_d < 5 else 0
                k4_distinct[k4_idx] = new_d
                delta_deficit += new_def - old_def
            
            # Acceptance Criteria
            accept = False
            if delta_deficit <= 0:
                accept = True
            else:
                if delta_deficit < 100 * temp:
                    if random.random() < math.exp(-delta_deficit / temp):
                        accept = True
            
            if accept:
                colors[e_idx] = new_color
                total_def += delta_deficit
                if total_def < best_deficit:
                    best_deficit = total_def
                    best_colors = colors[:]
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1
            else:
                # Revert distinct counts
                for k4_idx, old_d, old_def in old_states:
                    k4_distinct[k4_idx] = old_d
                steps_since_improve += 1
            
            # Cool down
            temp *= cool_rate
            if temp < min_temp:
                temp = min_temp
                
        return best_colors, best_deficit

    def try_structured_construction(max_colors, time_limit):
        """
        Try structured constructions based on group theory and combinatorial designs.
        """
        end_time = start_time + time_limit
        best_sol = None
        best_def = float('inf')
        
        # Construction 1: Z_16 based with various edge color functions
        # color(i,j) = f(i, j) mod max_colors where f is a polynomial
        
        # Try various polynomial forms over Z_16
        forms = [
            # Linear combinations
            lambda i, j, m: (i + j) % m,
            lambda i, j, m: (i + 2*j) % m,
            lambda i, j, m: (2*i + j) % m,
            lambda i, j, m: (i + 3*j) % m,
            lambda i, j, m: (3*i + j) % m,
            lambda i, j, m: (i + 5*j) % m,
            lambda i, j, m: (5*i + j) % m,
            lambda i, j, m: (i + 7*j) % m,
            lambda i, j, m: (7*i + j) % m,
            lambda i, j, m: (i + 9*j) % m,
            lambda i, j, m: (9*i + j) % m,
            lambda i, j, m: (i + 11*j) % m,
            lambda i, j, m: (11*i + j) % m,
            lambda i, j, m: (i + 13*j) % m,
            lambda i, j, m: (13*i + j) % m,
            lambda i, j, m: (i + 15*j) % m,
            lambda i, j, m: (15*i + j) % m,
            # Differences
            lambda i, j, m: (i - j) % m,
            lambda i, j, m: (2*i - j) % m,
            lambda i, j, m: (i - 2*j) % m,
            lambda i, j, m: (3*i - j) % m,
            lambda i, j, m: (i - 3*j) % m,
            lambda i, j, m: (5*i - j) % m,
            lambda i, j, m: (i - 5*j) % m,
            lambda i, j, m: (7*i - j) % m,
            lambda i, j, m: (i - 7*j) % m,
            lambda i, j, m: (9*i - j) % m,
            lambda i, j, m: (i - 9*j) % m,
            # Products
            lambda i, j, m: (i * j) % m,
            lambda i, j, m: (i * j + i + j) % m,
            lambda i, j, m: (i * j - i - j) % m,
            lambda i, j, m: (i * j + i) % m,
            lambda i, j, m: (i * j + j) % m,
            lambda i, j, m: (i * j - i) % m,
            lambda i, j, m: (i * j - j) % m,
            lambda i, j, m: (2*i * j) % m,
            lambda i, j, m: (3*i * j) % m,
            lambda i, j, m: (5*i * j) % m,
            # Quadratic
            lambda i, j, m: (i*i + j*j) % m,
            lambda i, j, m: (i*i + i*j + j*j) % m,
            lambda i, j, m: (i*i - j*j) % m,
            lambda i, j, m: (i*i + 2*i*j + j*j) % m,
            lambda i, j, m: (i*i + 3*i*j + j*j) % m,
            lambda i, j, m: (i*i + 5*i*j + j*j) % m,
            # Cubic
            lambda i, j, m: (i*i*i + j*j*j) % m,
            lambda i, j, m: (i*i*j + i*j*j) % m,
            lambda i, j, m: (i*i*i + i*j*j) % m,
            lambda i, j, m: (i*i*j + j*j*j) % m,
            # XOR-based
            lambda i, j, m: (i ^ j) % m,
            lambda i, j, m: ((i ^ j) + i + j) % m,
            lambda i, j, m: ((i ^ j) * 3 + i + j) % m,
            lambda i, j, m: ((i ^ j) + i * j) % m,
            lambda i, j, m: ((i ^ j) ^ (i + j)) % m,
        ]
        
        for func in forms:
            if time.time() > end_time:
                break
            
            # Generate coloring
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(func(i, j, max_colors))
            
            # Check validity
            deficit, _ = compute_deficit(colors)
            
            if deficit < best_def:
                best_def = deficit
                best_sol = colors[:]
            
            if deficit == 0:
                return colors, 0
            
            # If close, try local search refinement
            if deficit <= 10 and time.time() < end_time - 3:
                refined, ref_def = local_search_smooth(max_colors, min(3.0, end_time - time.time() - 1.0), initial_colors=colors)
                if ref_def < best_def:
                    best_def = ref_def
                    best_sol = refined[:]
                if ref_def == 0:
                    return refined, 0
        
        # Try with vertex relabeling: permute vertices
        if best_sol is not None and time.time() < end_time - 5:
            # Try random permutations of the best solution
            for perm_attempt in range(100):
                if time.time() > end_time - 2:
                    break
                perm = list(range(n))
                random.shuffle(perm)
                
                # Apply permutation to edges
                new_colors = [0] * num_edges
                for e_idx, (i, j) in enumerate(edges):
                    ni = perm[i]
                    nj = perm[j]
                    if ni > nj:
                        ni, nj = nj, ni
                    new_colors[e_idx] = best_sol[edge_index[(ni, nj)]]
                
                deficit, _ = compute_deficit(new_colors)
                if deficit < best_def:
                    best_def = deficit
                    best_sol = new_colors[:]
                if deficit == 0:
                    return new_colors, 0
        
        if best_sol is not None:
            return best_sol, best_def
        return None, float('inf')

    # Strategy:
    # 1. Try structured construction for C=13
    # 2. Local search for C=13
    # 3. Try C=14
    # 4. Try C=15
    
    # Try C=13 with structured construction first
    if time.time() < deadline - 30:
        sol, deficit = try_structured_construction(13, min(20.0, deadline - time.time() - 30))
        if deficit == 0:
            return sol
    
    # Try C=13 with local search
    if time.time() < deadline - 20:
        sol, deficit = local_search_smooth(13, min(20.0, deadline - time.time() - 20))
        if deficit == 0:
            return sol
    
    # Try C=14 with structured construction
    if time.time() < deadline - 12:
        sol, deficit = try_structured_construction(14, min(10.0, deadline - time.time() - 12))
        if deficit == 0:
            return sol
    
    # Try C=14 with local search
    if time.time() < deadline - 6:
        sol, deficit = local_search_smooth(14, min(8.0, deadline - time.time() - 6))
        if deficit == 0:
            return sol
    
    # Try C=15
    remaining = deadline - time.time()
    if remaining > 3.0:
        sol, deficit = local_search_smooth(15, remaining - 1.0)
        if deficit == 0:
            return sol
    
    # Fallback: Return a valid 15-color solution if we can find one quickly
    # Use a known construction: color(i,j) = (i + j) % 15 doesn't work, but let's try
    # Actually, let's just return the best we found or a simple valid one
    
    # Try a few more quick local searches with C=15
    for _ in range(3):
        if time.time() > deadline - 1.0:
            break
        sol, deficit = local_search_smooth(15, 1.0)
        if deficit == 0:
            return sol
    
    # Fallback: Return seed solution (17 colors)
    seed_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    return seed_colors
