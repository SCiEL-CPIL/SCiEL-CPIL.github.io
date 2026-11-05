import time
import random
from random import Random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DEADLINE_MARGIN = 10  # seconds before 150s

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def build_adj(edges):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj

def find_cycles_pool(edges, n_vertices, target_cycles=200, max_time=5.0):
    """
    Generate a pool of simple cycles using randomized DFS and greedy extension.
    Returns a list of tuples, each tuple is a cycle (list of vertices).
    """
    if not edges:
        return []
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = build_adj(edge_set)
    nodes = list(range(n_vertices))
    
    cycles = []
    seen_cycles = set()
    rng = Random(42)
    start_time = time.time()
    
    # Try to find many cycles
    attempts = 0
    max_attempts = 1000
    
    while len(cycles) < target_cycles and attempts < max_attempts and (time.time() - start_time < max_time):
        attempts += 1
        # Pick a random start node
        if not nodes:
            break
        start_node = rng.choice(nodes)
        
        # Randomized DFS to find a cycle back to start_node
        visited = {start_node}
        path = [start_node]
        stack = [(start_node, list(adj.get(start_node, [])))]
        
        found = False
        while stack and not found:
            node, neighbors = stack[-1]
            if not neighbors:
                stack.pop()
                if path:
                    path.pop()
                    visited.discard(path[-1] if path else node)
                continue
            
            # Shuffle neighbors for randomization
            rng.shuffle(neighbors)
            
            for nb in neighbors:
                e = (node, nb) if node < nb else (nb, node)
                if e not in edge_set:
                    continue
                
                if nb == start_node and len(path) >= 3:
                    # Found a cycle
                    cycle = path[:]
                    # Normalize cycle for seen set (rotate to start with min vertex)
                    min_idx = cycle.index(min(cycle))
                    norm_cycle = tuple(cycle[min_idx:] + cycle[:min_idx])
                    
                    if norm_cycle not in seen_cycles:
                        seen_cycles.add(norm_cycle)
                        cycles.append(cycle)
                        found = True
                        break
                elif nb not in visited and len(path) < 15: # Limit cycle length to keep ILP manageable
                    visited.add(nb)
                    path.append(nb)
                    stack.append((nb, list(adj.get(nb, []))))
                    break
            
            if found:
                break
            if not neighbors:
                stack.pop()
                if path:
                    path.pop()
                    visited.discard(path[-1] if path else node)
        
        # To find different cycles, we could remove one edge from the found cycle temporarily?
        # Or just let randomization handle it.
        # A better strategy: if we found a cycle, we don't remove edges yet, 
        # but we can bias the search. For now, simple random restarts.
        
        # Optional: 2-opt refinement to make cycles "nicer" or longer? 
        # The task asks for few pieces. Longer cycles are generally better for covering edges 
        # if they don't overlap too much, but the ILP will decide the combination.
        # Actually, having many short cycles gives the ILP more flexibility to pack them.
        # So we want a diverse pool.
        
    return cycles

def decompose_graph_ilp(edges, max_time=40.0):
    """
    Decompose edges by:
    1. Generating a pool of cycles.
    2. Formulating an ILP to select a subset of these cycles that are edge-disjoint
       and cover as many edges as possible (minimizing number of pieces).
       Actually, we want to minimize the total number of pieces.
       Let y_c be 1 if cycle c is selected.
       Let z_e be 1 if edge e is NOT covered by any selected cycle (i.e., becomes a single edge piece).
       Objective: Minimize sum(y_c) + sum(z_e)
       Constraints:
       For each edge e: sum_{c contains e} y_c + z_e >= 1
       For each edge e: sum_{c contains e} y_c <= 1 (Edge-disjointness of cycles)
       y_c, z_e are binary.
       
       Wait, if we select a cycle, it covers its edges.
       If an edge is not covered by any selected cycle, it must be a single piece.
       So z_e = 1 - sum_{c contains e} y_c.
       Since sum_{c contains e} y_c can be 0 or 1 (due to disjointness), z_e is determined.
       Objective: Minimize sum(y_c) + sum_e (1 - sum_{c contains e} y_c)
                 = Minimize sum(y_c) + |E| - sum_e sum_{c contains e} y_c
                 = Minimize sum(y_c) + |E| - sum_c (len(c) * y_c)
                 = Minimize |E| + sum_c (1 - len(c)) * y_c
       
       Since |E| is constant, we minimize sum_c (1 - len(c)) * y_c.
       Note that 1 - len(c) is negative for len(c) >= 2.
       So we want to select cycles with large len(c) to reduce the cost.
       This effectively maximizes sum_c (len(c) - 1) * y_c.
       Which is equivalent to maximizing the total number of edges covered by cycles minus the number of cycles.
       This makes sense: a cycle of length L covers L edges but counts as 1 piece.
       Savings = L - 1.
       
       So the objective is: Maximize sum_c (len(c) - 1) * y_c
       Subject to:
       sum_{c contains e} y_c <= 1 for all edges e.
       y_c binary.
    """
    start = time.time()
    edge_set = set(tuple(sorted(e)) for e in edges)
    if not edge_set:
        return []
    
    n_edges = len(edge_set)
    n_vertices = max([v for e in edge_set for v in e]) + 1 if edge_set else 0
    
    # Generate cycle pool
    # Allocate time for cycle generation
    gen_time = min(10.0, max_time * 0.3)
    cycles = find_cycles_pool(edge_set, n_vertices, target_cycles=300, max_time=gen_time)
    
    if not cycles:
        # Fallback: all single edges
        return [list(e) for e in edge_set]
    
    # Build ILP
    # Variables: y_c for each cycle in pool
    num_cycles = len(cycles)
    
    # Objective coefficients: c_i = -(len(cycle_i) - 1)  (Minimizing negative savings)
    c = np.array([-(len(cyc) - 1) for cyc in cycles])
    
    # Constraints: For each edge, sum of y_c for cycles containing it <= 1
    # We need to build the constraint matrix A (n_edges x num_cycles)
    # A is sparse, but for small problems (n_edges < 500, num_cycles < 300), dense is fine.
    edge_to_idx = {e: i for i, e in enumerate(sorted(edge_set))}
    A = np.zeros((n_edges, num_cycles))
    
    for j, cyc in enumerate(cycles):
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            e = (u, v) if u < v else (v, u)
            if e in edge_to_idx:
                A[edge_to_idx[e], j] = 1
            # else: this shouldn't happen if cycles are valid subgraphs
    
    # Bounds: y_c in [0, 1]
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(num_cycles)
    
    # Constraints: A @ y <= 1
    cons = LinearConstraint(A, lb=None, ub=1)
    
    # Solve ILP
    remaining_time = max_time - (time.time() - start) - 2.0 # Leave 2s for post-processing
    if remaining_time < 1.0:
        return [list(e) for e in edge_set]
        
    try:
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds,
                   options={"time_limit": remaining_time, "mip_rel_gap": 0.0})
        
        if res.status not in [0, 1]:
            # ILP failed or timed out without solution
            return [list(e) for e in edge_set]
            
        y = res.x
        selected_cycles = []
        covered_edges = set()
        
        for j in range(num_cycles):
            if y[j] > 0.5:
                cyc = cycles[j]
                # Verify edges are actually in graph and not already covered (should be guaranteed by ILP)
                valid = True
                for i in range(len(cyc)):
                    u = cyc[i]
                    v = cyc[(i + 1) % len(cyc)]
                    e = (u, v) if u < v else (v, u)
                    if e in covered_edges or e not in edge_set:
                        valid = False
                        break
                if valid:
                    selected_cycles.append(cyc)
                    for i in range(len(cyc)):
                        u = cyc[i]
                        v = cyc[(i + 1) % len(cyc)]
                        e = (u, v) if u < v else (v, u)
                        covered_edges.add(e)
                        
    except Exception:
        return [list(e) for e in edge_set]
    
    # Construct final pieces
    pieces = []
    for cyc in selected_cycles:
        pieces.append(cyc[:])
        
    # Add remaining edges as single pieces
    for e in edge_set:
        if e not in covered_edges:
            pieces.append([e[0], e[1]])
            
    return pieces

def construct():
    start = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    per_graph_time = max(1.0, (DEADLINE_MARGIN - 5) / 3)
    
    for i, spec in enumerate(specs):
        edges = instance(*spec)
        cur_time = time.time()
        remaining_time = min(per_graph_time, DEADLINE_MARGIN - (cur_time - start))
        if remaining_time < 1.0:
            # No time left, return all edges as single pieces
            results.append([list(e) for e in edges])
            continue
        
        pieces = decompose_graph_ilp(edges, max_time=remaining_time)
        results.append(pieces)
        
        # Check deadline
        cur_time = time.time()
        if cur_time - start > DEADLINE_MARGIN - 3:
            # Return remaining as individual edges
            for remaining_spec in specs[i + 1:]:
                e = instance(*remaining_spec)
                results.append([list(e) for e in e])
            break
    
    # Ensure we have exactly 3 results
    while len(results) < 3:
        results.append([])
    
    return results[:3]
