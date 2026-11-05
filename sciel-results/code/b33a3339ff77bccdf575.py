import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_random_cycles(adj, n, edge_set, num_cycles=200, time_limit=5.0):
    """
    Generate a pool of random cycles.
    """
    cycles = []
    seen_cycles = set()
    start_time = time.time()
    rng = random.Random(42)
    
    # Helper to find a cycle from a start edge
    def _extend_to_cycle(start_edge):
        u, v = start_edge
        # Path from v to u
        path = [v]
        visited = {v}
        current = v
        
        # Randomized DFS
        while True:
            neighbors = list(adj[current])
            # Filter neighbors: must be in edge_set with current
            # Actually, we just need to ensure the edge exists in the original graph
            # and hasn't been used in this specific path? 
            # For cycle generation, we just need a simple cycle in the graph.
            # We don't care about edge-disjointness during generation, only during selection.
            
            if not neighbors:
                return None
            
            # Prefer neighbors that are not visited, or if all visited, check for u
            unvisited = [x for x in neighbors if x not in visited]
            
            if u in neighbors and len(path) >= 2:
                # Can close cycle
                return path + [u]
            
            if not unvisited:
                return None
            
            # Pick random unvisited
            next_v = rng.choice(unvisited)
            path.append(next_v)
            visited.add(next_v)
            current = next_v
            
            # Safety break
            if len(path) > n:
                return None

    # Iterate through edges to start cycles
    edge_list = list(edge_set)
    rng.shuffle(edge_list)
    
    for _ in range(num_cycles):
        if time.time() - start_time > time_limit:
            break
        if not edge_list:
            break
        e = rng.choice(edge_list)
        cyc = _extend_to_cycle(e)
        if cyc:
            # Canonical form for deduplication
            # Rotate to min vertex, then reverse if second vertex is larger
            min_idx = cyc.index(min(cyc))
            rotated = cyc[min_idx:] + cyc[:min_idx]
            if rotated[1] > rotated[-1]:
                rotated.reverse()
            key = tuple(rotated)
            if key not in seen_cycles:
                seen_cycles.add(key)
                cycles.append(cyc)
                
    return cycles

def _solve_one(edges, n, deadline):
    if not edges:
        return []
        
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = _build_adj(edge_set, n)
    
    # Generate pool of cycles
    pool = _find_random_cycles(adj, n, edge_set, num_cycles=500, time_limit=5.0)
    
    if not pool:
        return [list(e) for e in edge_set]
        
    # Precompute edge sets for each cycle
    cycle_edges = []
    for cyc in pool:
        es = set()
        for i in range(len(cyc)):
            u, v = cyc[i], cyc[(i+1)%len(cyc)]
            es.add(tuple(sorted((u, v))))
        cycle_edges.append(es)
        
    # Simulated Annealing on the selection of cycles
    # State: list of indices of selected cycles
    # Objective: Minimize pieces = len(selected) + (len(edge_set) - sum(len(es) for es in selected))
    #           = len(selected) + len(edge_set) - sum(len(es))
    #           Minimize: len(selected) - sum(len(es))
    
    rng = random.Random(123)
    best_selection = []
    best_score = float('inf')
    current_selection = []
    current_used_edges = set()
    current_score = len(edge_set) # Initially no cycles, pieces = all edges
    
    # Initial greedy solution
    # Sort cycles by length descending
    indices_by_len = sorted(range(len(pool)), key=lambda i: len(pool[i]), reverse=True)
    for i in indices_by_len:
        if cycle_edges[i].isdisjoint(current_used_edges):
            current_selection.append(i)
            current_used_edges.update(cycle_edges[i])
            current_score = len(current_selection) + len(edge_set) - sum(len(cycle_edges[j]) for j in current_selection)
            
    best_selection = current_selection[:]
    best_score = current_score
    
    start_time = time.time()
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline - 1.0:
        if T < 0.01:
            break
            
        # Generate a move
        move_type = rng.random()
        new_selection = current_selection[:]
        new_used_edges = set(current_used_edges)
        
        if move_type < 0.3:
            # Add a random cycle
            if not pool: continue
            i = rng.randint(0, len(pool)-1)
            if i in new_selection: continue
            if not cycle_edges[i].isdisjoint(new_used_edges): continue
            new_selection.append(i)
            new_used_edges.update(cycle_edges[i])
            
        elif move_type < 0.6:
            # Remove a random cycle
            if not new_selection: continue
            i = rng.randint(0, len(new_selection)-1)
            to_remove = new_selection[i]
            new_selection.pop(i)
            # Remove edges
            new_used_edges.difference_update(cycle_edges[to_remove])
            
        else:
            # Swap: Remove one, try to add another
            if not new_selection: continue
            i = rng.randint(0, len(new_selection)-1)
            to_remove = new_selection[i]
            new_selection.pop(i)
            new_used_edges.difference_update(cycle_edges[to_remove])
            
            # Try to add a random one
            if pool:
                j = rng.randint(0, len(pool)-1)
                if j not in new_selection and cycle_edges[j].isdisjoint(new_used_edges):
                    new_selection.append(j)
                    new_used_edges.update(cycle_edges[j])
                    
        if not new_selection:
            new_score = len(edge_set)
        else:
            new_score = len(new_selection) + len(edge_set) - sum(len(cycle_edges[j]) for j in new_selection)
            
        # Acceptance
        delta = new_score - current_score
        if delta < 0 or rng.random() < pow(2.718, -delta / T):
            current_selection = new_selection
            current_used_edges = new_used_edges
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_selection = current_selection[:]
                
        T *= cooling
        
    # Construct result
    pieces = []
    for i in best_selection:
        pieces.append(pool[i])
        
    # Add remaining edges
    used_edges = set()
    for i in best_selection:
        used_edges.update(cycle_edges[i])
        
    for e in edge_set:
        if e not in used_edges:
            pieces.append(list(e))
            
    return pieces

def construct():
    start_time = time.time()
    total_budget = 145.0
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)
        
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0
            
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
            
    return results
