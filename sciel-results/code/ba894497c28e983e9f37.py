"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = None
_TIME_LIMIT = 110.0  # safety margin before 120s kill

def _check_time():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return time.time() >= _DEADLINE

def _get_c6s(n):
    """Precompute all 6-cycles in K_n. A 6-cycle is a set of 6 vertices with 6 edges forming a cycle."""
    c6s = []
    for verts in combinations(range(n), 6):
        # Generate all distinct 6-cycles on these 6 vertices
        # Fix vertex 0 at position 0 to avoid rotation symmetry
        # We need all Hamiltonian cycles on 6 vertices
        # Number of distinct cycles on 6 labeled vertices: (6-1)!/2 = 60
        v0 = verts[0]
        rest = verts[1:]
        # Generate permutations of rest, take first half to avoid reversal symmetry
        from itertools import permutations
        count = 0
        seen = set()
        for perm in permutations(rest):
            cycle = (v0, perm[0], perm[1], perm[2], perm[3], perm[4], v0)
            # Normalize: use frozenset of edges
            edges = frozenset(
                (min(cycle[i], cycle[i+1]), max(cycle[i], cycle[i+1]))
                for i in range(6)
            )
            if edges not in seen:
                seen.add(edges)
                c6s.append(verts + perm)  # store as tuple of 6 vertices in cycle order
                count += 1
                if count >= 1:  # only one unique cycle per 6-set? No, there are 60.
                    pass
    # Actually, for each set of 6 vertices, there are (6-1)!/2 = 60 distinct cycles
    # Let me redo this more carefully
    c6s = []
    for verts in combinations(range(n), 6):
        v0 = verts[0]
        rest = verts[1:]
        seen_edges = set()
        for perm in permutations(rest):
            cycle = (v0, perm[0], perm[1], perm[2], perm[3], perm[4])
            edges = frozenset(
                (min(cycle[i], cycle[(i+1)%6]), max(cycle[i], cycle[(i+1)%6]))
                for i in range(6)
            )
            if edges not in seen_edges:
                seen_edges.add(edges)
                # Store as list of 6 edge indices (pairs)
                edge_list = [(cycle[i], cycle[(i+1)%6]) for i in range(6)]
                c6s.append((verts, edge_list))
    return c6s

def _edge_index(i, j, n):
    """Get the index in the colors list for edge (i,j) where i<j."""
    if i > j:
        i, j = j, i
    # Number of edges before vertex i: i*(n-1) - i*(i-1)//2
    # Simpler: sum of (n-k) for k=0..i-1 minus i*(i+1)/2... let me just use a formula
    # Edges before (i, i+1): for vertices 0..i-1, each has (n - vertex) edges
    # Actually simpler: index = sum_{k=0}^{i-1} (n-k) - i*(i+1)/2 + (j-i-1)
    # Let me just precompute
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _solve_n(n, c6s, time_budget):
    """Try to find a valid 4-coloring of K_n using backtracking with forward checking."""
    num_edges = n * (n - 1) // 2
    colors = [-1] * num_edges
    
    # Build edge index lookup
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i,j)] = idx
            idx += 1
    
    # For each edge, precompute which C6s it belongs to
    edge_to_c6s = [[] for _ in range(num_edges)]
    for c6_idx, (verts, edge_list) in enumerate(c6s):
        for (u, v) in edge_list:
            if u > v:
                u, v = v, u
            ei = edge_idx[(u,v)]
            edge_to_c6s[ei].append(c6_idx)
    
    # For each C6, store its 6 edge indices
    c6_edges = []
    for c6_idx, (verts, edge_list) in enumerate(c6s):
        c6_edge_indices = []
        for (u, v) in edge_list:
            if u > v:
                u, v = v, u
            c6_edge_indices.append(edge_idx[(u,v)])
        c6_edges.append(c6_edge_indices)
    
    # Forward checking: for each C6, track how many edges are colored with each color
    # c6_color_count[c6_idx][color] = number of edges in that C6 colored with that color
    # If any color reaches 6, that C6 is monochromatic -> invalid
    
    # Order edges for backtracking: try to find a good ordering
    # Use a simple heuristic: edges that appear in more C6s first
    edge_c6_count = [len(edge_to_c6s[e]) for e in range(num_edges)]
    edge_order = sorted(range(num_edges), key=lambda e: -edge_c6_count[e])
    
    # Reverse mapping: position in search order -> edge index
    pos_to_edge = edge_order
    edge_to_pos = [0] * num_edges
    for pos, e in enumerate(edge_order):
        edge_to_pos[e] = pos
    
    best_solution = None
    start_time = time.time()
    
    def backtrack(pos):
        nonlocal best_solution
        if time.time() - start_time > time_budget:
            return False
        if pos == num_edges:
            best_solution = colors[:]
            return True
        
        e = pos_to_edge[pos]
        # Try colors 0,1,2,3
        for c in range(4):
            colors[e] = c
            # Check if this creates a monochromatic C6
            valid = True
            for c6_idx in edge_to_c6s[e]:
                c6e = c6_edges[c6_idx]
                # Check if all 6 edges in this C6 are now color c
                all_c = True
                for ei in c6e:
                    if colors[ei] != c:
                        all_c = False
                        break
                if all_c:
                    valid = False
                    break
            
            if valid:
                if backtrack(pos + 1):
                    return True
            
            colors[e] = -1
        
        return False
    
    if backtrack(0):
        return best_solution
    
    return None

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    
    # Start with known good solution
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try to find larger solutions
    # Precompute C6s for n up to 15 (limit due to combinatorial explosion)
    # For n=10: C(10,6)=210 sets, each with 60 cycles = 12600 C6s
    # For n=11: C(11,6)=462 * 60 = 27720
    # For n=12: C(12,6)=924 * 60 = 55440
    
    # Try from high n down
    for n_try in [14, 13, 12, 11, 10]:
        if _check_time():
            break
        
        if n_try <= best_n:
            break
        
        start = time.time()
        c6s = _get_c6s(n_try)
        elapsed = time.time() - start
        
        # Estimate remaining time
        remaining = _TIME_LIMIT - (time.time() - (_DEADLINE - _TIME_LIMIT))
        if remaining < 10:
            break
        
        time_budget = max(5, remaining - 5)
        
        solution = _solve_n(n_try, c6s, time_budget)
        
        if solution is not None:
            best_n = n_try
            best_colors = solution
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
