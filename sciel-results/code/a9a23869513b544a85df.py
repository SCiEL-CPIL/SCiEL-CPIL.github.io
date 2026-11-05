import time
import random
from collections import defaultdict, deque

def construct():
    """
    Decomposes the edges of three fixed instance graphs into the minimum number of
    edge-disjoint simple paths.
    
    Returns:
        A list of three lists of paths. Each path is a list of vertices.
    """
    # Deadline guard: 150s limit, leave 10s safety margin
    start_time = time.time()
    deadline = start_time + 140.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_eulerian_circuit(adj, start):
        """
        Iterative Hierholzer's algorithm to find an Eulerian circuit.
        adj: dict of list of (neighbor, edge_id)
        Returns list of vertices in the circuit.
        """
        # We need to track used edges. 
        # Since we modify adj in place, we'll use a copy or a separate used set.
        # For efficiency, we'll pop from the adjacency lists.
        
        # Make a deep copy of the adjacency list structure to allow popping
        # Actually, we can just use the provided adj and pop from it.
        # But we need to be careful if we want to reuse the structure. 
        # Here we assume adj is a fresh structure for this call.
        
        stack = [start]
        path = []
        
        while stack:
            v = stack[-1]
            if adj[v]:
                # Pick an edge
                # Random choice? Hierholzer works with any order.
                # But for deterministic behavior in retries, maybe fixed or random.
                # Let's just pop the last one (LIFO) for speed.
                u, edge_id = adj[v].pop()
                stack.append(u)
            else:
                path.append(stack.pop())
                
        path.reverse()
        return path

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # 1. Compute degrees and identify odd vertices
        deg = defaultdict(int)
        adj = defaultdict(list)
        
        for idx, (u, v) in enumerate(edges):
            deg[u] += 1
            deg[v] += 1
            adj[u].append((v, idx))
            adj[v].append((u, idx))
            
        odd_vertices = [v for v in range(n) if deg[v] % 2 != 0]
        k = len(odd_vertices)
        
        if k == 0:
            # Eulerian graph: 1 path (circuit)
            # Find Eulerian circuit
            # Start from any vertex with non-zero degree
            start = next(iter(adj), 0)
            circuit = find_eulerian_circuit(adj, start)
            # The circuit is a cycle. We can return it as a single path.
            # Note: The path should be simple? 
            # An Eulerian circuit in a simple graph is not necessarily a simple path (it repeats vertices).
            # The task requires "simple paths". 
            # Wait, "Paths must be simple". 
            # A simple path has no repeated vertices.
            # An Eulerian circuit is a closed walk. If it repeats vertices, it's not a simple path.
            # However, the decomposition covers edges.
            # If the graph is Eulerian, can we decompose it into simple paths?
            # Yes, trivially, but minimizing count is the goal.
            # If we use 1 path, it must be simple.
            # But an Eulerian circuit is not simple if the graph has cycles that aren't just the whole graph.
            # Actually, if the graph is Eulerian, the minimum number of *simple* paths to cover edges is NOT necessarily 1.
            # Example: A figure-8 graph (two cycles sharing a vertex). 
            # Degrees: center vertex 4, others 2. All even.
            # Eulerian circuit exists. But it visits the center twice.
            # So it's not a simple path.
            # Can we cover it with 1 simple path? No, because the center is a cut-vertex for the edges?
            # Actually, a path cover of edges by simple paths.
            # If we have a figure-8, we can take cycle 1 and cycle 2 as two paths.
            # Can we do better? No, because any path entering the center must leave.
            # If we try to make one path, we enter center, go around cycle 1, return to center, go around cycle 2, return to center.
            # This path visits center 3 times. Not simple.
            # So the "Eulerian circuit -> 1 path" logic is flawed for "simple paths" requirement.
            
            # RE-READING TASK: "Paths must be simple".
            # This changes everything.
            # The lower bound based on odd vertices applies to *trails* (edge-disjoint walks), not *simple paths*.
            # For simple paths, the problem is harder.
            
            # However, for random graphs with n=20-30, p=0.25-0.3, the graphs are sparse.
            # The "odd vertex pairing" method produces trails. 
            # If a trail is not simple, it contains a cycle.
            # We can decompose a non-simple trail into simple paths?
            # No, we want to MINIMIZE the number of paths.
            
            # Let's look at the previous program's score: 11.36.
            # Theoretical lower bound for trails is small.
            # The previous program used greedy + merge.
            
            # If the "simple path" constraint is strict, then an Eulerian circuit is only valid if it is simple (i.e., the graph is a single cycle).
            # Otherwise, we might need more paths.
            
            # Let's stick to the previous approach's spirit but optimize it.
            # The previous approach found ~11 paths.
            # The lower bound for simple path cover is higher than for trail cover.
            
            # Let's try a different strategy:
            # 1. Find a maximal set of edge-disjoint simple paths greedily.
            # 2. Use Local Search (2-opt style) to reduce count.
            
            # Actually, the previous code's `merge_paths` is a form of local search.
            # The main weakness was the initial greedy decomposition.
            
            # Let's try to improve the greedy decomposition to produce longer simple paths.
            # Strategy: 
            # - Always extend the path in the direction that leads to a vertex with lower degree (to avoid trapping high-degree vertices).
            # - Or, prioritize edges that are "bridges" or part of biconnected components?
            
            # Let's refine the `greedy_decompose` from the original code.
            # The original code used `random` and `low_deg` strategies.
            # Let's add a strategy that prefers extending towards the "center" of the remaining graph or avoids creating isolated vertices.
            
            # Also, the merge phase can be improved.
            # Instead of just checking endpoints, we can try to "splice" paths.
            
            # Let's rewrite the solver with a focus on:
            # 1. Better Greedy: Use a heuristic that tries to keep the path "open" (endpoints have high degree in remaining graph).
            # 2. Better Merge: Iteratively try to merge any two paths if their endpoints are connected and the result is simple.
            # 3. Simulated Annealing / Random Restarts: Since n is small, we can afford many restarts.
            
            pass # Placeholder for logic below

        # We will use the improved greedy + merge approach.
        
        def greedy_decompose(edges_list, rng, strategy='low_deg'):
            remaining = set(edges_list)
            paths = []
            
            # Pre-compute adjacency for speed, but update dynamically.
            # Since n is small, linear scan of remaining edges is okay, but let's use a dict of sets for neighbors.
            # adj[u] = set of (v, edge_tuple)
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add((v, (u, v)))
                adj[v].add((u, (u, v)))
                
            def get_neighbors(u):
                return [v for v, _ in adj[u]]
                
            def remove_edge(u, v):
                e = tuple(sorted((u, v)))
                remaining.remove(e)
                adj[u].remove((v, e))
                adj[v].remove((u, e))
                
            def extend_path(path):
                changed = True
                while changed:
                    changed = False
                    # Try extending back
                    if path:
                        back = path[-1]
                        neighbors = get_neighbors(back)
                        valid = [n for n in neighbors if n not in path]
                        if valid:
                            if strategy == 'low_deg':
                                # Choose neighbor with lowest degree in remaining graph
                                # This tends to keep high-degree vertices for later
                                degs = {n: len(get_neighbors(n)) for n in valid}
                                min_d = min(degs.values())
                                cands = [n for n in valid if degs[n] == min_d]
                                next_node = rng.choice(cands)
                            else:
                                next_node = rng.choice(valid)
                            path.append(next_node)
                            remove_edge(back, next_node)
                            changed = True
                            
                    # Try extending front
                    if path:
                        front = path[0]
                        neighbors = get_neighbors(front)
                        valid = [n for n in neighbors if n not in path]
                        if valid:
                            if strategy == 'low_deg':
                                degs = {n: len(get_neighbors(n)) for n in valid}
                                min_d = min(degs.values())
                                cands = [n for n in valid if degs[n] == min_d]
                                next_node = rng.choice(cands)
                            else:
                                next_node = rng.choice(valid)
                            path.insert(0, next_node)
                            remove_edge(front, next_node)
                            changed = True

            while remaining:
                if time.time() > deadline:
                    break
                # Pick start edge
                # Heuristic: Pick edge connecting two vertices with high degree?
                # Or just random?
                # Let's try picking an edge incident to the vertex with highest degree.
                if not remaining:
                    break
                # Calculate degrees in remaining
                rdeg = defaultdict(int)
                for u, v in remaining:
                    rdeg[u] += 1
                    rdeg[v] += 1
                max_d = max(rdeg.values())
                cands = [v for v, d in rdeg.items() if d == max_d]
                start_u = rng.choice(cands)
                
                neighbors = get_neighbors(start_u)
                if not neighbors:
                    continue
                start_v = rng.choice(neighbors)
                
                path = [start_u, start_v]
                remove_edge(start_u, start_v)
                extend_path(path)
                paths.append(path)
                
            return paths

        def merge_paths(paths, edges_list, rng):
            current_paths = [list(p) for p in paths]
            if len(current_paths) <= 1:
                return current_paths
                
            edge_set = set(edges_list)
            
            improved = True
            while improved:
                improved = False
                if time.time() > deadline:
                    break
                    
                n_paths = len(current_paths)
                found = False
                for i in range(n_paths):
                    if found: break
                    for j in range(n_paths):
                        if i == j: continue
                        if found: break
                        
                        p_i = current_paths[i]
                        p_j = current_paths[j]
                        
                        # 4 merges
                        # 1. end(i) - start(j)
                        if (p_i[-1], p_j[0]) in edge_set or (p_j[0], p_i[-1]) in edge_set:
                            new_path = p_i + p_j[1:]
                            if len(set(new_path)) == len(new_path):
                                current_paths[i] = new_path
                                del current_paths[j]
                                improved = True
                                found = True
                                break
                        # 2. end(i) - end(j)
                        if (p_i[-1], p_j[-1]) in edge_set or (p_j[-1], p_i[-1]) in edge_set:
                            new_path = p_i + p_j[::-1][1:]
                            if len(set(new_path)) == len(new_path):
                                current_paths[i] = new_path
                                del current_paths[j]
                                improved = True
                                found = True
                                break
                        # 3. start(i) - start(j)
                        if (p_i[0], p_j[0]) in edge_set or (p_j[0], p_i[0]) in edge_set:
                            new_path = p_i[::-1] + p_j[1:]
                            if len(set(new_path)) == len(new_path):
                                current_paths[i] = new_path
                                del current_paths[j]
                                improved = True
                                found = True
                                break
                        # 4. start(i) - end(j)
                        if (p_i[0], p_j[-1]) in edge_set or (p_j[-1], p_i[0]) in edge_set:
                            new_path = p_i[::-1] + p_j[::-1][1:]
                            if len(set(new_path)) == len(new_path):
                                current_paths[i] = new_path
                                del current_paths[j]
                                improved = True
                                found = True
                                break
            return current_paths

        def validate_paths(paths, edges_list):
            if not paths:
                return len(edges_list) == 0
            used_edges = set()
            edge_set = set(edges_list)
            for path in paths:
                if len(path) < 1: return False
                if len(set(path)) != len(path): return False
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set: return False
                    if e in used_edges: return False
                    used_edges.add(e)
            return len(used_edges) == len(edge_set)

        best_paths = None
        best_count = float('inf')
        
        # Run multiple iterations
        iterations = 2000
        for i in range(iterations):
            if time.time() > deadline:
                break
            rng = random.Random(seed * 10000 + i)
            strategy = 'low_deg' if i % 3 != 0 else 'random'
            
            paths = greedy_decompose(edges, rng, strategy)
            
            if not validate_paths(paths, edges):
                continue
                
            merged_paths = merge_paths(paths, edges, rng)
            
            if validate_paths(merged_paths, edges):
                if len(merged_paths) < best_count:
                    best_count = len(merged_paths)
                    best_paths = [list(p) for p in merged_paths]
                    
        if best_paths is None:
            # Fallback: single edges
            return [[list(e)] for e in edges]
            
        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        paths = solve_instance(n, p, seed)
        results.append(paths)
        
        if time.time() > deadline:
            break
            
    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[list(e)] for e in edges])

    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
