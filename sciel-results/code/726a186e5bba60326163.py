"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def solve_path_cover(edges, time_limit=30.0):
        """
        Solves the path cover problem using a constructive heuristic based on
        repeatedly finding long paths in the residual graph.
        We use a strategy of picking a start node and extending the path greedily,
        but with a twist: we prioritize nodes that are "dead ends" in the current
        residual graph if possible, or use a Warnsdorff-like rule to avoid trapping
        ourselves.
        
        Actually, a better approach for minimizing paths is to find a path cover
        that is close to a Hamiltonian path if the graph is dense enough, or
        otherwise a set of long paths.
        
        We will use a randomized greedy algorithm with restarts.
        """
        if not edges:
            return []
        
        edge_set = set(edges)
        m = len(edge_set)
        
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        best = None
        best_count = float('inf')
        t_start = time.time()
        
        max_restarts = 10000
        
        for restart in range(max_restarts):
            if time.time() - t_start > time_limit:
                break
            if m == 0:
                break
                
            # Randomize the order of consideration to escape local optima
            # We will build paths one by one.
            # Strategy: 
            # 1. Pick a random starting vertex from the remaining graph.
            # 2. Extend the path greedily. To make it "long", we prefer neighbors
            #    with few remaining options (Warnsdorff) to leave "open" nodes for later?
            #    Or do we prefer neighbors with many options to keep the path going?
            #    For finding a single long path, Warnsdorff (fewest onward moves) is good.
            #    For covering all edges with few paths, we want to "use up" edges efficiently.
            
            # Let's try a different heuristic: 
            # Always start from a vertex with the highest degree in the residual graph?
            # Or random?
            
            cur_edges = set(edge_set)
            paths = []
            
            while cur_edges:
                # Compute degrees in current residual graph
                deg = defaultdict(int)
                for u, v in cur_edges:
                    deg[u] += 1
                    deg[v] += 1
                
                if not deg:
                    break
                    
                # Choose start: 
                # Heuristic: Pick a random vertex with the maximum degree? 
                # Or just random?
                # Let's try: Random vertex, but weighted by degree?
                # Or: Pick a random vertex, but if there are odd-degree vertices, prefer them?
                # In a path cover, the number of paths is related to the number of odd-degree
                # vertices and the structure.
                
                # Let's just pick a random vertex from those with max degree, to keep paths long.
                max_d = max(deg.values())
                candidates = [v for v, d in deg.items() if d == max_d]
                start = random.choice(candidates)
                
                # Build path
                path = [start]
                visited = {start}
                last = start
                removed = set()
                
                while True:
                    # Find available neighbors of last in cur_edges not visited
                    nbrs = []
                    for nb in adj[last]:
                        if nb in visited:
                            continue
                        e = tuple(sorted((last, nb)))
                        if e in cur_edges and e not in removed:
                            # Count onward moves for nb
                            onward = 0
                            for nn in adj[nb]:
                                if nn in visited or nn == last:
                                    continue
                                e2 = tuple(sorted((nb, nn)))
                                if e2 in cur_edges and e2 not in removed:
                                    onward += 1
                            nbrs.append((onward, nb))
                    
                    if not nbrs:
                        break
                    
                    # Warnsdorff: pick neighbor with fewest onward moves
                    # This tends to create long paths that don't trap themselves.
                    min_onward = min(x[0] for x in nbrs)
                    best_nbs = [nb for o, nb in nbrs if o == min_onward]
                    nxt = random.choice(best_nbs)
                    
                    e = tuple(sorted((last, nxt)))
                    cur_edges.discard(e)
                    removed.add(e)
                    path.append(nxt)
                    visited.add(nxt)
                    last = nxt
                
                if len(path) >= 2:
                    paths.append(path)
                else:
                    # Should not happen if edges exist and start has degree > 0
                    # Fallback: add remaining edges as individual paths
                    for e in cur_edges:
                        paths.append(list(e))
                    cur_edges = set()
                    break
            
            cnt = len(paths)
            if cnt < best_count:
                best_count = cnt
                best = [p[:] for p in paths]
                
                # Early stop if we hit a very good number?
                # Gallai's conjecture: floor(n/2).
                # We don't know n easily here without passing it, but we can estimate.
                # Or just run until time limit.
        
        if best is None:
            # Fallback: trivial decomposition
            return [[u, v] for u, v in edge_set]
            
        return best

    def validate_and_fix(paths, edges):
        """
        Ensure paths are valid:
        1. Simple paths
        2. Edges in paths are in graph
        3. Edge-disjoint
        4. Cover all edges
        """
        edge_set = set(edges)
        used_edges = set()
        valid_paths = []
        for p in paths:
            # Check simple
            if len(p) != len(set(p)):
                # Path is not simple. Break it?
                pass
            
            ok = True
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                if e not in edge_set:
                    ok = False
                    break
                if e in used_edges:
                    ok = False
                    break
                used_edges.add(e)
            
            if ok:
                valid_paths.append(p)
            else:
                # If a path is invalid, add its edges individually?
                for i in range(len(p) - 1):
                    e = tuple(sorted((p[i], p[i+1])))
                    if e in edge_set and e not in used_edges:
                        used_edges.add(e)
                        valid_paths.append([p[i], p[i+1]])
        
        # Check coverage
        if used_edges != edge_set:
            # Missing edges
            for e in edge_set - used_edges:
                valid_paths.append(list(e))
        
        return valid_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for i, spec in enumerate(specs):
        g_edges = instance(*spec)
        time_left = deadline - time.time()
        
        if time_left < 2:
            # Emergency fallback
            paths = [[u, v] for u, v in g_edges]
        else:
            # Allocate time
            # We have ~130s total. 3 graphs. ~40s each.
            # But greedy is fast. Let's give 10s per graph initially, and if we finish fast,
            # we can't easily redistribute in this loop structure without knowing future.
            # Let's just use a fixed 10s per graph, which should be plenty for n=20-30.
            # If we want to be safer, 20s per graph.
            paths = solve_path_cover(g_edges, time_limit=min(20.0, time_left / 3.0))
            
            # Final validation
            paths = validate_and_fix(paths, g_edges)
            
        results.append(paths)
        
    return results

# EVOLVE-BLOCK-END
