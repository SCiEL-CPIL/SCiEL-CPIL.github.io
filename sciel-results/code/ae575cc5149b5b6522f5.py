"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from collections import defaultdict, deque
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 145

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    def find_cycles_bfs(adj, start, max_len, visited_edges, limit):
        """Find a simple cycle starting from 'start' using BFS-like DFS."""
        # This is a heuristic to find *a* cycle, not necessarily the longest
        # But for ILP, having a good pool is better than one long one?
        # Actually, ILP needs a pool of *all* potential cycles or a representative set.
        # Generating ALL cycles is exponential.
        # Strategy: Generate a pool of random cycles.
        
        path = [start]
        visited = {start}
        
        def dfs(u):
            if time.time() > limit:
                return None
            if len(path) >= max_len:
                return None
            
            neighbors = list(adj.get(u, set()))
            random.shuffle(neighbors)
            
            for v in neighbors:
                if v == start and len(path) >= 3:
                    return path[:]
                if v not in visited:
                    visited.add(v)
                    path.append(v)
                    res = dfs(v)
                    if res:
                        return res
                    path.pop()
                    visited.discard(v)
            return None
            
        return dfs(start)

    def generate_cycle_pool(adj, n, max_cycles, time_limit):
        """Generate a diverse set of cycles for the ILP."""
        cycles = []
        seen_cycles = set()
        vertices = list(adj.keys())
        
        # 1. Find some long cycles
        for _ in range(50):
            if time.time() > time_limit:
                break
            if not vertices:
                break
            start = random.choice(vertices)
            # Try to find a cycle of random length between 3 and 15
            max_len = random.randint(3, min(15, n))
            cyc = find_cycles_bfs(adj, start, max_len, set(), time.time() + 0.5)
            if cyc:
                # Normalize cycle for deduplication (start at min vertex, direction doesn't matter for edges but does for vertex list)
                # Actually, we just need the set of edges.
                edge_set = frozenset(edge_key(cyc[i], cyc[(i+1)%len(cyc)]) for i in range(len(cyc)))
                if edge_set not in seen_cycles:
                    seen_cycles.add(edge_set)
                    cycles.append(cyc)
                    
        # 2. Find 3-cycles (triangles) explicitly - they are common
        # Simple triangle finding
        for u in vertices:
            if time.time() > time_limit:
                break
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            cyc = [u, v, w]
                            edge_set = frozenset(edge_key(cyc[i], cyc[(i+1)%len(cyc)]) for i in range(len(cyc)))
                            if edge_set not in seen_cycles:
                                seen_cycles.add(edge_set)
                                cycles.append(cyc)
                                
        return cycles

    def solve_ilp(edges, n, time_limit):
        """
        Solve the ILP for a single graph.
        Variables:
        x_c for each cycle c in pool
        x_e for each edge e in graph
        Minimize sum(x)
        Subject to: for each edge e, sum(x_c for c using e) + x_e == 1
        """
        edge_list = [edge_key(u, v) for u, v in edges]
        edge_index = {e: i for i, e in enumerate(edge_list)}
        m = len(edge_list)
        
        # Generate pool
        adj = defaultdict(set)
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            
        pool = generate_cycle_pool(adj, n, 200, time.time() + 2.0)
        
        if not pool:
            return [[u, v] for u, v in edges]

        # Build ILP matrix
        # Rows: m (edges)
        # Cols: len(pool) + m (cycles + edges)
        
        num_cycles = len(pool)
        num_vars = num_cycles + m
        
        # Objective: minimize sum of all vars
        c = np.ones(num_vars)
        
        # Constraints: A @ x == 1
        # A is m x num_vars
        A = np.zeros((m, num_vars))
        
        # Fill cycle columns
        for i, cyc in enumerate(pool):
            for j in range(len(cyc)):
                u = cyc[j]
                v = cyc[(j+1) % len(cyc)]
                ek = edge_key(u, v)
                if ek in edge_index:
                    A[edge_index[ek], i] = 1
                    # Note: A cycle uses each edge exactly once, so 1.0
                    # Wait, if a cycle is invalid (reuses edge), it shouldn't be in pool.
                    # Our generator ensures simple cycles.
        
        # Fill edge columns (identity part)
        for i in range(m):
            A[i, num_cycles + i] = 1.0
            
        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: all binary
        integrality = np.ones(num_vars)
        
        # Constraints: A @ x == 1
        constraints = LinearConstraint(A, lb=np.ones(m), ub=np.ones(m))
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.status == 0 and res.x is not None:
                pieces = []
                # Extract cycles
                for i in range(num_cycles):
                    if res.x[i] > 0.5:
                        pieces.append(pool[i])
                # Extract edges
                for i in range(m):
                    if res.x[num_cycles + i] > 0.5:
                        u, v = edge_list[i]
                        pieces.append([u, v])
                return pieces
            else:
                # Fallback to greedy if ILP fails
                return greedy_decompose(edges, time_limit)
        except Exception:
            return greedy_decompose(edges, time_limit)

    def greedy_decompose(edges, time_limit):
        """Fallback greedy decomposition."""
        remaining = set(edge_key(u, v) for u, v in edges)
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
            
        pieces = []
        
        def commit_cycle(cyc):
            for i in range(len(cyc)):
                u, v = cyc[i], cyc[(i+1)%len(cyc)]
                ek = edge_key(u, v)
                remaining.discard(ek)
                adj[u].discard(v)
                adj[v].discard(u)
            pieces.append(cyc)
            
        def commit_edge(u, v):
            ek = edge_key(u, v)
            remaining.discard(ek)
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])
            
        while remaining:
            if time.time() > time_limit:
                break
            # Find any cycle
            found = False
            for u in list(adj.keys()):
                if time.time() > time_limit:
                    break
                if len(adj[u]) < 2:
                    continue
                # Try to find a cycle starting at u
                cyc = find_cycles_bfs(adj, u, 10, set(), time.time() + 0.1)
                if cyc:
                    commit_cycle(cyc)
                    found = True
                    break
            if not found:
                # No cycles, take an edge
                u, v = list(remaining)[0]
                commit_edge(u, v)
                
        for u, v in list(remaining):
            commit_edge(u, v)
            
        return pieces

    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    
    for spec in specs:
        if time.time() > deadline - 15:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        
        time_per_instance = min(40, deadline - time.time() - 15)
        if time_per_instance < 5:
            time_per_instance = 5
            
        # Try ILP first
        pieces = solve_ilp(edges, n, time.time() + time_per_instance)
        
        # If ILP was too slow or failed, greedy fallback is already handled in solve_ilp
        # But we can also try greedy if ILP result is bad?
        # ILP is exact for the pool, so it should be good.
        
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END
