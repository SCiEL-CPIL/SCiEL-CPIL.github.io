"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from scipy.optimize import linprog
import numpy as np

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycles_in_graph(adj, n, max_cycles=200, deadline=None):
    """Find many distinct cycles in the graph using randomized DFS."""
    cycles = []
    vertices = [v for v in range(n) if adj.get(v)]
    if not vertices:
        return cycles
    
    rng = random.Random(42)
    found_edges = set()
    
    for start in vertices:
        if deadline is not None and time.time() > deadline:
            break
        if len(cycles) >= max_cycles:
            break
        
        # Multiple randomized DFS attempts from this start
        for attempt in range(3):
            if deadline is not None and time.time() > deadline:
                break
            if len(cycles) >= max_cycles:
                break
            
            # Randomize neighbor order
            adj_copy = {v: list(adj[v]) for v in adj if adj[v]}
            for v in adj_copy:
                rng.shuffle(adj_copy[v])
            
            # DFS to find a cycle back to start
            stack = [(start, [start])]
            visited = {start}
            found = False
            
            while stack and not found:
                if deadline is not None and time.time() > deadline:
                    break
                node, path = stack.pop()
                for nb in adj_copy.get(node, []):
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        cycle = path[:]
                        cycle_edges = frozenset(
                            (min(cycle[i], cycle[(i+1) % len(cycle)]), 
                             max(cycle[i], cycle[(i+1) % len(cycle)]))
                            for i in range(len(cycle))
                        )
                        if cycle_edges not in found_edges:
                            cycles.append(cycle)
                            found_edges.add(cycle_edges)
                            found = True
                            break
                    elif nb not in visited:
                        visited.add(nb)
                        stack.append((nb, path + [nb]))
    
    return cycles

def _decompose_lp(edges, n, seed, time_limit=40.0):
    """Decompose edges using LP-based maximum cycle packing."""
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
    cycles = _find_cycles_in_graph(adj, n, max_cycles=300, deadline=deadline)
    
    if not cycles:
        return [[u, v] for u, v in edges]
    
    edge_list = list(edges)
    edge_idx = {e: i for i, e in enumerate(edge_list)}
    m = len(edge_list)
    k = len(cycles)
    
    # Build constraint matrix: each cycle uses its edges
    # A[i][j] = 1 if cycle j uses edge i
    A = np.zeros((m, k), dtype=np.float64)
    for j, cycle in enumerate(cycles):
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            if e in edge_idx:
                A[edge_idx[e], j] = 1.0
    
    # Objective: maximize number of cycles = minimize -sum(x_j)
    c = np.full(k, -1.0, dtype=np.float64)
    
    # Constraints: sum of cycles using each edge <= 1
    bounds = [(0, 1)] * k
    A_ub = A
    b_ub = np.ones(m, dtype=np.float64)
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.status == 0:
        # Extract selected cycles (round to nearest integer)
        x = np.round(result.x)
        selected_cycles = []
        for j in range(k):
            if x[j] > 0.5:
                selected_cycles.append(cycles[j])
    else:
        selected_cycles = []
    
    # Remove edges used by selected cycles
    used_edges = set()
    for cycle in selected_cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            used_edges.add(e)
    
    remaining = [e for e in edge_list if e not in used_edges]
    
    # For remaining edges, try to find more cycles greedily
    pieces = [cycle for cycle in selected_cycles]
    
    if remaining:
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Greedy cycle finding on remaining
        rem_set = set(remaining)
        while rem_set:
            if time.time() > deadline:
                break
            
            # Find longest cycle in remaining
            best_cycle = []
            for start in list(rem_adj.keys()):
                if not rem_adj.get(start):
                    continue
                # Greedy path extension
                path = [start]
                visited = {start}
                while True:
                    u = path[-1]
                    candidates = [v for v in rem_adj[u] if v not in visited]
                    if not candidates:
                        break
                    # Pick neighbor with lowest degree (Warnsdorff-like)
                    candidates.sort(key=lambda v: len(rem_adj[v]))
                    next_v = candidates[0]
                    path.append(next_v)
                    visited.add(next_v)
                
                if len(path) >= 3 and path[-1] in rem_adj.get(start, set()):
                    if len(path) > len(best_cycle):
                        best_cycle = path
                
                # Check for cycles in path
                for i in range(len(path) - 2):
                    for j in range(i + 2, len(path)):
                        if path[i] in rem_adj.get(path[j], set()):
                            cycle = path[i:j+1]
                            if len(cycle) > len(best_cycle):
                                best_cycle = cycle
                            break
            
            if best_cycle:
                # Remove cycle edges
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    e = (min(u, v), max(u, v))
                    if e in rem_set:
                        rem_set.discard(e)
                        rem_adj[u].discard(v)
                        rem_adj[v].discard(u)
                pieces.append(best_cycle)
            else:
                # No cycle, take single edge
                e = next(iter(rem_set))
                rem_set.discard(e)
                u, v = e
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)
                pieces.append([u, v])
    
    return pieces

def _decompose(edges, n, seed, time_limit=40.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Try LP-based approach
    pieces_lp = _decompose_lp(edges, n, seed, time_limit=min(20.0, time_limit * 0.6))
    count_lp = len(pieces_lp)
    
    # Try greedy multi-start approach
    best_pieces = pieces_lp
    best_count = count_lp
    
    def greedy_decompose(remaining_edges, n, seed_offset=0):
        remaining = set(remaining_edges)
        pieces = []
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_counter = 0
        while remaining:
            if time.time() > deadline:
                for e in remaining:
                    pieces.append([e[0], e[1]])
                return pieces
            
            # Find a long cycle
            best_cycle = []
            vertices = [v for v in adj if adj[v]]
            for start in vertices[:6]:
                path = [start]
                visited = {start}
                while True:
                    u = path[-1]
                    candidates = [(len(adj[v]), v) for v in adj[u] if v not in visited]
                    if not candidates:
                        break
                    candidates.sort(reverse=True)
                    next_v = candidates[0][1]
                    path.append(next_v)
                    visited.add(next_v)
                
                if len(path) >= 3 and start in adj.get(path[-1], set()):
                    if len(path) > len(best_cycle):
                        best_cycle = path
                
                for i in range(len(path) - 2):
                    for j in range(i + 2, len(path)):
                        if path[i] in adj.get(path[j], set()):
                            cycle = path[i:j+1]
                            if len(cycle) > len(best_cycle):
                                best_cycle = cycle
                            break
            
            if best_cycle:
                cycle_edges = set()
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    e = (min(u, v), max(u, v))
                    cycle_edges.add(e)
                
                for e in cycle_edges:
                    if e in remaining:
                        remaining.discard(e)
                        u, v = e
                        adj[u].discard(v)
                        adj[v].discard(u)
                
                pieces.append(best_cycle)
                cycle_counter += 1
            else:
                e = min(remaining, key=lambda e: (len(adj.get(e[0], set())) + len(adj.get(e[1], set())), e))
                remaining.discard(e)
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append([u, v])
        
        return pieces
    
    # A few greedy restarts
    num_restarts = min(20, max(5, int((deadline - time.time()) / 2.0)))
    for restart in range(num_restarts):
        if time.time() > deadline - 1.0:
            break
        pieces = greedy_decompose(edges, n, restart)
        if len(pieces) < best_count:
            best_count = len(pieces)
            best_pieces = pieces
    
    if best_pieces is None:
        best_pieces = [[u, v] for u, v in edges]
    
    return best_pieces

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
