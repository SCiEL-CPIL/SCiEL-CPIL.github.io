"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _canonical_cycle(cycle):
    """
    Returns a canonical representation of a cycle to check for uniqueness.
    A cycle is a list of vertices [v0, v1, ..., vk-1].
    We represent it by the frozenset of its edges.
    """
    edges = set()
    k = len(cycle)
    for i in range(k):
        u = cycle[i]
        v = cycle[(i + 1) % k]
        edges.add((min(u, v), max(u, v)))
    return frozenset(edges)

def _find_cycles_greedy_and_enumerate(adj, n, max_len=8, max_candidates=2000, deadline=None):
    """
    Finds cycles in the graph.
    1. Enumerates all cycles of length 3 and 4.
    2. Uses randomized DFS to find other cycles up to max_len.
    Returns a list of unique cycles (as lists of vertices).
    """
    cycles = []
    cycle_edge_sets = set()
    
    # Helper to add a cycle if valid and new
    def add_cycle(cyc):
        if len(cyc) < 3:
            return
        # Check if it's a simple cycle (vertices distinct except start=end)
        # Our DFS ensures this by tracking visited nodes in the path
        # But we need to ensure it's not a duplicate
        edge_set = _canonical_cycle(cyc)
        if edge_set not in cycle_edge_sets:
            cycle_edge_sets.add(edge_set)
            cycles.append(cyc)
    
    # 1. Enumerate all triangles (length 3)
    # For each edge (u,v), find common neighbors w
    # This is faster than DFS for triangles
    neighbors = {u: set(adj[u]) for u in adj}
    for u in adj:
        for v in adj[u]:
            if u < v:
                common = neighbors[u] & neighbors[v]
                for w in common:
                    # Cycle u-v-w-u
                    # Canonical form to avoid duplicates? 
                    # We just add it, uniqueness is handled by edge_set
                    # But we must ensure u,v,w are distinct. They are.
                    # Order doesn't matter for edge set.
                    add_cycle([u, v, w])
    
    # 2. Enumerate all 4-cycles
    # For each pair of edges (u,v) and (w,x) that share a vertex?
    # Easier: For each vertex u, look at pairs of neighbors (v, w)
    # If v and w have a common neighbor z (z != u), then u-v-z-w-u is a 4-cycle.
    # To avoid duplicates, we can just generate and let edge_set filter.
    for u in adj:
        nbs = list(neighbors[u])
        for i in range(len(nbs)):
            for j in range(i + 1, len(nbs)):
                v = nbs[i]
                w = nbs[j]
                # Find common neighbors of v and w, excluding u
                common_vw = neighbors[v] & neighbors[w]
                common_vw.discard(u)
                for z in common_vw:
                    # Cycle u-v-z-w-u
                    add_cycle([u, v, z, w])

    if len(cycles) >= max_candidates or (deadline and time.time() > deadline):
        return cycles

    # 3. Randomized DFS for longer cycles (5 to max_len)
    rng = random.Random(123)
    vertices = list(adj.keys())
    
    # Shuffle vertices for randomization
    for start in vertices:
        if len(cycles) >= max_candidates or (deadline and time.time() > deadline):
            break
            
        # Try multiple random paths from start
        for _ in range(10):
            if len(cycles) >= max_candidates or (deadline and time.time() > deadline):
                break
                
            path = [start]
            visited = {start}
            # Shuffle neighbors for randomization
            adj_copy = {v: list(adj[v]) for v in adj}
            for v in adj_copy:
                rng.shuffle(adj_copy[v])
                
            # DFS
            stack = [(start, [start], {start})]
            found_cycle = False
            
            while stack and not found_cycle:
                if len(cycles) >= max_candidates or (deadline and time.time() > deadline):
                    break
                    
                node, p, vis = stack.pop()
                
                for nb in adj_copy.get(node, []):
                    if nb == start and len(p) >= 3:
                        # Found a cycle
                        # Check if it's a new one
                        add_cycle(p)
                        found_cycle = True
                        break
                    elif nb not in vis:
                        # Only expand if path length < max_len
                        if len(p) < max_len:
                            stack.append((nb, p + [nb], vis | {nb}))

    return cycles

def _decompose_lp(edges, n, seed, time_limit=40.0):
    """
    Decompose edges using ILP-based maximum cycle packing.
    """
    if not edges:
        return []
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find candidate cycles
    # We prioritize finding many cycles.
    # The ILP will select the best combination.
    candidates = _find_cycles_greedy_and_enumerate(adj, n, max_len=8, max_candidates=1000, deadline=deadline)
    
    if not candidates:
        return [[u, v] for u, v in edges]
    
    edge_list = list(edges)
    edge_idx = {e: i for i, e in enumerate(edge_list)}
    m = len(edge_list)
    k = len(candidates)
    
    # Objective: Maximize sum(|c| - 1) for selected cycles
    # This is equivalent to minimizing pieces.
    # c_j = |cycle_j| - 1
    c_obj = np.array([len(cycle) - 1 for cycle in candidates], dtype=np.float64)
    
    # Constraints: For each edge, sum of cycles using it <= 1
    # A[i][j] = 1 if cycle j uses edge i
    # We need to build A.
    # A is m x k.
    # To save memory and time, we can build it as a sparse matrix or just dense if m,k are small.
    # m ~ 150, k ~ 1000. Dense is fine.
    A = np.zeros((m, k), dtype=np.float64)
    
    for j, cycle in enumerate(candidates):
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            if e in edge_idx:
                A[edge_idx[e], j] = 1.0
    
    # Variables x_j in {0, 1}
    integrality = np.ones(k, dtype=int)
    
    # Bounds: 0 <= x_j <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Constraints: A @ x <= 1
    constraints = LinearConstraint(A, ub=np.ones(m))
    
    # Minimize -c_obj @ x
    res = milp(
        c=-c_obj,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds
    )
    
    if res.status == 0:
        # Extract selected cycles
        x = res.x
        selected_cycles = []
        for j in range(k):
            if x[j] > 0.5:
                selected_cycles.append(candidates[j])
    else:
        # Fallback to greedy if ILP fails
        selected_cycles = []
        rem_edges = set(edge_list)
        adj_rem = defaultdict(set)
        for u, v in rem_edges:
            adj_rem[u].add(v)
            adj_rem[v].add(u)
            
        # Simple greedy: pick longest available cycle
        # Reuse candidates?
        # Just pick from candidates that are still valid
        valid_candidates = []
        for cyc in candidates:
            cyc_edges = set()
            valid = True
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i+1) % len(cyc)]
                e = (min(u,v), max(u,v))
                if e not in rem_edges:
                    valid = False
                    break
                cyc_edges.add(e)
            if valid:
                valid_candidates.append((len(cyc), cyc))
        
        valid_candidates.sort(reverse=True)
        for _, cyc in valid_candidates:
            cyc_edges = set()
            valid = True
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i+1) % len(cyc)]
                e = (min(u,v), max(u,v))
                if e not in rem_edges:
                    valid = False
                    break
                cyc_edges.add(e)
            if valid:
                selected_cycles.append(cyc)
                for e in cyc_edges:
                    rem_edges.discard(e)
                    u, v = e
                    adj_rem[u].discard(v)
                    adj_rem[v].discard(u)
                if time.time() > deadline:
                    break

    # Remove edges used by selected cycles
    used_edges = set()
    for cycle in selected_cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            used_edges.add(e)
    
    remaining = [e for e in edge_list if e not in used_edges]
    
    pieces = list(selected_cycles)
    
    # Add remaining edges as single pieces
    for e in remaining:
        pieces.append([e[0], e[1]])
        
    return pieces

def _decompose(edges, n, seed, time_limit=40.0):
    """
    Main decomposition function.
    """
    if not edges:
        return []
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Try ILP approach
    pieces = _decompose_lp(edges, n, seed, time_limit=time_limit)
    
    # Optional: Local search improvement?
    # Given the ILP optimality on the candidate set, and the candidate set being quite comprehensive (all 3, 4-cycles + random longer),
    # further improvement is likely marginal.
    
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    total_deadline = time.time() + 140.0
    
    for idx, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        
        remaining_time = total_deadline - time.time()
        graphs_left = len(specs) - idx
        time_for_this = max(10.0, remaining_time / graphs_left - 1.0)
        
        pieces = _decompose(edges, n, seed, time_limit=time_for_this)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
