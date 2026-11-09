import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Copied from verifier
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    def solve_instance(n, p, seed):
        edges_list = instance(n, p, seed)
        if not edges_list:
            return []
        
        # Build adjacency list for the full graph to help with heuristics
        # But we need to track unused edges dynamically.
        
        def greedy_path_cover(edge_set, rng):
            """
            Greedy path cover using randomization for tie-breaking.
            Returns list of paths.
            """
            unused = set(edge_set)
            paths = []
            
            # Pre-compute degrees for initial sorting? 
            # Degrees change as edges are removed.
            
            while unused:
                # Find a good starting vertex.
                # Heuristic: Vertex with highest degree in the remaining graph.
                # To add randomness, we can pick randomly from top K degrees.
                deg = {}
                for u, v in unused:
                    deg[u] = deg.get(u, 0) + 1
                    deg[v] = deg.get(v, 0) + 1
                
                if not deg:
                    break
                
                # Sort vertices by degree descending
                sorted_verts = sorted(deg.items(), key=lambda x: x[1], reverse=True)
                max_deg = sorted_verts[0][1]
                
                # Pick a random vertex from those with high degree (e.g., top 3 or max)
                # This introduces variation.
                candidates = [v for v, d in sorted_verts if d == max_deg]
                start = rng.choice(candidates)
                
                # Greedy DFS from start
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    # Find neighbors of current in unused edges that are not visited
                    next_candidates = []
                    for u, v in unused:
                        if u == current and v not in visited:
                            next_candidates.append(v)
                        elif v == current and u not in visited:
                            next_candidates.append(u)
                    
                    if not next_candidates:
                        break
                    
                    # Heuristic: Choose neighbor with highest degree in remaining graph
                    # If tie, choose randomly.
                    # We need degrees of these candidates in the *remaining* graph.
                    # We already have `deg` from the start of the iteration, but it's stale.
                    # Recalculating degrees for just the candidates is fast enough for small N.
                    # Or just use the stale `deg` as a proxy, or just pick randomly from candidates.
                    # Let's use stale `deg` for speed, it's a good proxy.
                    
                    # Update degrees for candidates based on current `deg` (stale is okay for heuristic)
                    # Actually, let's just pick the one with max stale degree.
                    best_next = None
                    best_deg = -1
                    # To allow randomness, we can collect all with max deg and pick one.
                    max_cand_deg = -1
                    for c in next_candidates:
                        if deg.get(c, 0) > max_cand_deg:
                            max_cand_deg = deg.get(c, 0)
                    
                    final_candidates = [c for c in next_candidates if deg.get(c, 0) == max_cand_deg]
                    next_node = rng.choice(final_candidates)
                    
                    # Remove edge (current, next_node) from unused
                    # Find the exact tuple
                    if (current, next_node) in unused:
                        unused.remove((current, next_node))
                    elif (next_node, current) in unused:
                        unused.remove((next_node, current))
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                    # Update stale degrees? Not strictly necessary for the heuristic to work reasonably well.
                    # But for accuracy of subsequent steps in this path, it might help.
                    # Given N is small, we can just leave it.
                
                paths.append(path)
            
            return paths

        # Run multiple randomized trials
        best_paths = None
        best_count = float('inf')
        
        # Number of trials depends on time. 
        # Each trial is very fast. Let's do 1000 trials or until time runs out.
        rng = random.Random(seed) # Fixed seed for reproducibility of the "randomization" strategy itself?
        # Actually, using a different seed for the solver logic might help explore different solutions.
        # But the task asks for a deterministic construct() ideally? 
        # The verifier likely runs construct() once. 
        # We can use a fixed seed for the internal randomization to ensure reproducibility if needed, 
        # or just use time-based? 
        # Standard practice in these tasks: deterministic output is preferred if possible, 
        # but randomized search is common. 
        # Let's use a fixed seed for the internal RNG to make the result reproducible for the same input.
        
        solver_rng = random.Random(12345)
        
        trials = 0
        while time.time() < deadline - 5.0: # Leave 5s for other instances
            paths = greedy_path_cover(set(edges_list), solver_rng)
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
            trials += 1
            
            # If we hit a very good bound, we might stop early?
            # Gallai bound for n=30, p=0.25 is ~15. 
            # If we get close to 15, further improvement is hard.
            # But let's just run the full budget.
            
        if best_paths is None:
            # Fallback: single edge paths
            return [[u, v] for u, v in edges_list]
            
        return best_paths

    results = []
    for spec in specs:
        n, p, seed = spec
        if time.time() > deadline - 10.0:
            # Not enough time for next instance, use fallback
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
        else:
            paths = solve_instance(n, p, seed)
            results.append(paths)
            
    return results
