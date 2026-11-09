# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Fallback: known n=9 solution (cyclic coloring)
    n0 = 9
    best_n = n0
    best_colors = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    
    # Try to find solutions for n from 17 down to 10
    for target_n in range(17, 9, -1):
        if time.time() > deadline - 20:
            break
        
        solution = _solve_ilp(target_n, deadline)
        if solution is not None:
            best_n = target_n
            best_colors = solution
            break
    
    return {'n': best_n, 'colors': best_colors}


def _solve_ilp(n, deadline):
    """
    Solve 4-coloring of K_n with no monochromatic C6 using ILP.
    
    Variables: x[e][c] = 1 if edge e has color c, for e in edges, c in {0,1,2,3}
    Constraints:
      1. For each edge e: sum_c x[e][c] = 1
      2. For each potential monochromatic C6 (v0,v1,...,v5) and each color c:
         sum of (1 - x[e_i][c]) >= 1  for the 6 edges e_i in the cycle
         i.e., at least one edge is NOT color c.
      3. Symmetry breaking: fix colors of first few edges.
    
    Objective: minimize 0 (feasibility problem)
    """
    # Build edge list
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(edges)}
    
    # Variables: x[idx][c] for idx in range(num_edges), c in range(4)
    # Total variables: num_edges * 4
    num_vars = num_edges * 4
    
    def var_idx(e_idx, c):
        return e_idx * 4 + c
    
    # Objective: minimize 0 (feasibility)
    c_obj = np.zeros(num_vars)
    
    # Build constraint matrix A and bounds for A_ub @ x <= b_ub
    # We'll use LinearConstraint with A @ x between lb and ub
    
    # Constraint 1: For each edge, exactly one color
    # sum_c x[e][c] = 1  =>  1 <= sum_c x[e][c] <= 1
    # This gives num_edges constraints
    
    # Constraint 2: For each C6 and each color, at least one edge is not that color
    # sum of (1 - x[e_i][c]) >= 1
    # => -sum x[e_i][c] >= 1 - 6 = -5
    # => sum x[e_i][c] <= 5
    # This gives num_c6s * 4 constraints
    
    # First, enumerate all C6s (6-cycles) in K_n
    # A C6 is a set of 6 vertices {v0,...,v5} and a cyclic ordering.
    # To avoid duplicates, we fix v0 < v1 < ... < v5 and consider all cyclic orderings.
    # But there are many. Let's be smarter: enumerate all 6-subsets and all Hamiltonian cycles.
    
    # For n up to 17, number of 6-subsets is C(n,6) which for n=17 is 12376.
    # For each 6-subset, number of distinct C6s (up to rotation/reflection) is (6-1)!/2 = 60.
    # So total C6s ~ 12376 * 60 = ~742,560 for n=17. That's a lot.
    # For n=16: C(16,6)=8008, total ~480,480.
    # For n=15: C(15,6)=5005, total ~300,300.
    
    # This might be too many constraints. Let's try a different approach:
    # Instead of enumerating all C6s, we can use a weaker but more tractable formulation.
    # Or we can limit the number of C6s we include.
    
    # Alternative: use a SAT-like approach with fewer constraints.
    # Let's try with n <= 14 first where the number of C6s is manageable.
    
    if n > 14:
        # For larger n, use a heuristic approach with ILP on a subset
        # Or fall back to local search
        return _solve_local_search(n, deadline)
    
    # Enumerate all C6s
    c6_list = []
    for subset in combinations(range(n), 6):
        # For each 6-subset, enumerate all distinct C6s
        # Fix the first vertex as the smallest, then permute the rest
        # A C6 is determined by the cyclic order of 6 vertices.
        # Two C6s are the same if they have the same set of edges.
        # Number of distinct C6s on 6 labeled vertices: (6-1)!/2 = 60
        
        verts = list(subset)
        # Generate all permutations of verts[1:] (5! = 120), take unique cycles
        seen_cycles = set()
        for perm in _permutations(verts[1:]):
            cycle = [verts[0]] + list(perm)
            # Normalize: rotate so smallest element is first, then take the shorter direction
            min_idx = cycle.index(min(cycle))
            rot1 = cycle[min_idx:] + cycle[:min_idx]
            rot2 = rot1[:1] + rot1[1:][::-1]
            norm = tuple(rot1) if rot1 < rot2 else tuple(rot2)
            if norm not in seen_cycles:
                seen_cycles.add(norm)
                c6_list.append(cycle)
    
    # Now build the ILP
    # Constraints:
    # 1. For each edge e: sum_c x[e][c] = 1
    # 2. For each C6 (v0,...,v5) and each color c: sum_{i=0}^{5} x[edge(v_i, v_{i+1})][c] <= 5
    
    # Build A matrix for LinearConstraint
    # We'll use sparse representation conceptually, but scipy.optimize.milp uses dense or sparse
    
    # Number of constraints: num_edges + len(c6_list) * 4
    num_constraints = num_edges + len(c6_list) * 4
    
    # Build A as a list of rows (we'll convert to sparse)
    from scipy.sparse import lil_matrix
    
    A = lil_matrix((num_constraints, num_vars), dtype=np.float64)
    lb = np.zeros(num_constraints)
    ub = np.zeros(num_constraints)
    
    # Constraint 1: edge sum = 1
    for e_idx in range(num_edges):
        row = e_idx
        for c in range(4):
            A[row, var_idx(e_idx, c)] = 1
        lb[row] = 1
        ub[row] = 1
    
    # Constraint 2: C6 constraints
    constraint_row = num_edges
    for cycle in c6_list:
        # Get the 6 edges in the cycle
        cycle_edges = []
        for i in range(6):
            u = cycle[i]
            v = cycle[(i + 1) % 6]
            if u > v:
                u, v = v, u
            e_idx = edge_to_idx[(u, v)]
            cycle_edges.append(e_idx)
        
        for c in range(4):
            row = constraint_row
            constraint_row += 1
            for e_idx in cycle_edges:
                A[row, var_idx(e_idx, c)] = 1
            lb[row] = 0
            ub[row] = 5  # sum x[e][c] <= 5
    
    # Symmetry breaking: fix some edges to specific colors
    # Fix edge (0,1) to color 0
    # This reduces symmetry by a factor of 4
    if num_edges > 0:
        # Add constraint: x[edge(0,1)][0] = 1
        # We can do this by fixing the variable
        e01_idx = edge_to_idx[(0, 1)]
        v01_c0 = var_idx(e01_idx, 0)
        # Set lower and upper bound of this variable to 1
        # We'll handle this via variable bounds
    
    # Variable bounds: all x are in {0, 1}
    # We'll use integrality
    integrality = np.ones(num_vars, dtype=np.int8)
    
    # Variable bounds
    var_lb = np.zeros(num_vars)
    var_ub = np.ones(num_vars)
    
    # Apply symmetry breaking: fix x[edge(0,1)][0] = 1
    if num_edges > 0:
        e01_idx = edge_to_idx[(0, 1)]
        v01_c0 = var_idx(e01_idx, 0)
        var_lb[v01_c0] = 1
        var_ub[v01_c0] = 1
    
    # Also fix x[edge(0,2)][1] = 1 to break more symmetry
    if num_edges > 1:
        e02_idx = edge_to_idx[(0, 2)]
        v02_c1 = var_idx(e02_idx, 1)
        var_lb[v02_c1] = 1
        var_ub[v02_c1] = 1
    
    bounds = Bounds(var_lb, var_ub)
    
    # Solve
    try:
        result = milp(
            c=c_obj,
            constraints=LinearConstraint(A.tocsr(), lb, ub),
            integrality=integrality,
            bounds=bounds
        )
    except Exception as e:
        return None
    
    if result.status != 0:  # 0 means optimal
        return None
    
    x = result.x
    if x is None:
        return None
    
    # Extract colors
    colors = []
    for e_idx in range(num_edges):
        for c in range(4):
            if x[var_idx(e_idx, c)] > 0.5:
                colors.append(c)
                break
        else:
            # No color assigned (shouldn't happen)
            return None
    
    return colors


def _permutations(iterable):
    """Generate all permutations of an iterable."""
    if len(iterable) <= 1:
        yield tuple(iterable)
        return
    for i, item in enumerate(iterable):
        rest = list(iterable[:i]) + list(iterable[i+1:])
        for perm in _permutations(rest):
            yield (item,) + perm


def _solve_local_search(n, deadline):
    """Fallback local search for larger n."""
    num_edges = n * (n - 1) // 2
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Build edge list
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(edges)}
    
    # Build adjacency
    adj = [[0] * n for _ in range(4)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        adj[c][i] |= (1 << j)
        adj[c][j] |= (1 << i)
    
    def find_c6():
        """Find a monochromatic C6, return (color, edge_idx) or (None, None)."""
        for c in range(4):
            adj_c = adj[c]
            for u in range(n):
                # Find C6 starting from u
                # Path: u -> v1 -> v2 -> v3 -> v4 -> v5 -> u
                N_u = adj_c[u] & ~((1 << (u + 1)) - 1)
                if N_u == 0:
                    continue
                
                def dfs(current, depth, visited, path):
                    if depth == 5:
                        if (adj_c[current] >> u) & 1:
                            return path + [current]
                        return None
                    candidates = adj_c[current] & ~((1 << (u + 1)) - 1) & ~visited
                    c_bits = candidates
                    while c_bits:
                        c_bit = c_bits & -c_bits
                        cv = c_bit.bit_length() - 1
                        c_bits ^= c_bit
                        result = dfs(cv, depth + 1, visited | c_bit, path + [cv])
                        if result is not None:
                            return result
                    return None
                
                s_bits = N_u
                while s_bits:
                    s_bit = s_bits & -s_bits
                    s = s_bit.bit_length() - 1
                    s_bits ^= s_bit
                    result = dfs(s, 1, s_bit, [s])
                    if result is not None:
                        cycle = [u] + result
                        # Pick a random edge in the cycle
                        k = random.randint(0, 5)
                        i, j = cycle[k], cycle[(k + 1) % 6]
                        if i > j:
                            i, j = j, i
                        return c, edge_to_idx[(i, j)]
        return None, None
    
    max_steps = 30000
    step = 0
    while step < max_steps:
        if time.time() > deadline:
            return None
        
        c, idx = find_c6()
        if c is None:
            return colors
        
        # Flip the edge
        old_c = colors[idx]
        new_c = random.choice([cc for cc in range(4) if cc != old_c])
        colors[idx] = new_c
        
        u, v = edges[idx]
        adj[old_c][u] &= ~(1 << v)
        adj[old_c][v] &= ~(1 << u)
        adj[new_c][u] |= (1 << v)
        adj[new_c][v] |= (1 << u)
        
        step += 1
    
    return None
# EVOLVE-BLOCK-END
