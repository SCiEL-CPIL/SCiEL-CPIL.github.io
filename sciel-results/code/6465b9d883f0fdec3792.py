"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    import itertools
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_all_cycles(n, edges_set, max_len=8):
        """Find all simple cycles up to max_len in the graph."""
        cycles = []
        adj = {i: set() for i in range(n)}
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Use DFS to find cycles, canonicalize by starting at smallest vertex
        for start in range(n):
            # DFS from start, only visit vertices > start to avoid duplicates
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                for nb in sorted(adj[node]):
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        cycles.append(tuple(path))
                    elif nb > start and nb not in path and len(path) < max_len:
                        stack.append((nb, path + [nb]))
        return cycles
    
    def decompose(n, edges_list):
        edges_set = {tuple(sorted(e)) for e in edges_list}
        m = len(edges_list)
        if m == 0:
            return []
        
        # Find all cycles
        cycles = find_all_cycles(n, edges_set, max_len=10)
        
        # If no cycles found, return all edges as single pieces
        if not cycles:
            return [[u, v] for u, v in edges_list]
        
        # Build edge index mapping
        edge_idx = {e: i for i, e in enumerate(sorted(edges_set))}
        sorted_edges = sorted(edges_set)
        
        # For each cycle, get the list of edge indices it uses
        cycle_edges = []
        cycle_weights = []
        for cyc in cycles:
            cyc_edge_list = []
            for i in range(len(cyc)):
                u, v = cyc[i], cyc[(i+1) % len(cyc)]
                e = tuple(sorted((u, v)))
                if e in edge_idx:
                    cyc_edge_list.append(edge_idx[e])
                else:
                    cyc_edge_list = None
                    break
            if cyc_edge_list is not None and len(cyc_edge_list) == len(cyc):
                # Check all edges are distinct in the cycle
                if len(set(cyc_edge_list)) == len(cyc_edge_list):
                    cycle_edges.append(cyc_edge_list)
                    cycle_weights.append(len(cyc) - 1)  # savings
        
        ncycles = len(cycle_edges)
        if ncycles == 0:
            return [[u, v] for u, v in edges_list]
        
        # ILP: maximize sum(weights[i] * x[i]) subject to each edge used at most once
        # x[i] in {0, 1}
        
        # c: objective coefficients (we want to maximize, so minimize negative)
        c = np.array([-w for w in cycle_weights])
        
        # A: constraint matrix, one row per edge
        A = np.zeros((m, ncycles))
        for i, cyc_e in enumerate(cycle_edges):
            for eidx in cyc_e:
                A[eidx, i] = 1
        
        # Constraints: A @ x <= 1 for each edge
        constraints = LinearConstraint(A, 0, 1)
        integrality = np.ones(ncycles)
        bounds = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status != 0:
            # Fallback: greedy
            return [[u, v] for u, v in edges_list]
        
        x = res.x
        selected = [i for i in range(ncycles) if x[i] > 0.5]
        
        # Collect used edges
        used_edges = set()
        pieces = []
        for i in selected:
            cyc = cycles[i]
            piece = list(cyc)
            pieces.append(piece)
            for j in range(len(cyc)):
                u, v = cyc[j], cyc[(j+1) % len(cyc)]
                used_edges.add(tuple(sorted((u, v))))
        
        # Add remaining edges as single pieces
        for e in sorted_edges:
            if e not in used_edges:
                pieces.append([e[0], e[1]])
        
        return pieces
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(n, edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
