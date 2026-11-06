import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 40
    # Initialize adjacency matrix
    adj = [[False] * n for _ in range(n)]
    
    # Start with a random sparse graph
    random.seed(42)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i][j] = True
                adj[j][i] = True

    def find_triangle():
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    # Check common neighbors
                    # Optimization: iterate over neighbors of i that are > j
                    # But simple triple loop is fine for n=40
                    for k in range(j + 1, n):
                        if adj[i][k] and adj[j][k]:
                            return (i, j, k)
        return None

    def find_independent_set_10():
        # Greedy search for independent set of size 10
        # We can use a simple backtracking or greedy heuristic.
        # Given n=40, a greedy approach that picks a vertex with min degree in the current subgraph
        # or just random sampling might work, but let's use a deterministic greedy one first.
        
        # A more robust way: try to find an independent set using a simple recursive search with pruning
        # or just check if the complement graph has a clique of size 10.
        
        # Let's use a simple greedy independent set finder.
        # It might not find the max, but if it finds size 10, we have a violation.
        # If it doesn't find size 10, there might still be one.
        # For safety, let's use a slightly smarter heuristic: 
        # Repeatedly pick a vertex with minimum degree in the remaining graph.
        
        # Actually, for n=40, we can just use a standard algorithm to find max independent set size?
        # No, that's NP-hard. But we just need to find *one* of size 10.
        
        # Let's try a randomized greedy approach multiple times?
        # Or just a deterministic greedy:
        available = list(range(n))
        indep = []
        # Sort by degree?
        # Let's just pick vertices one by one.
        # To be effective, we should pick vertices that are "easy" to keep.
        
        # A better heuristic for finding a large independent set:
        # 1. Compute degrees.
        # 2. Pick vertex with min degree.
        # 3. Remove it and its neighbors.
        # 4. Repeat.
        
        # This greedy might fail to find a set of size 10 even if one exists.
        # However, if the graph is close to valid, the independent sets are likely to be found.
        # Let's implement a backtracking search for independent set of size 10.
        # Since n=40, and we stop at 10, it should be fast if the graph is dense enough.
        # If the graph is too sparse, we'll find it quickly.
        
        def backtrack(start_idx, current_set):
            if len(current_set) == 10:
                return current_set
            remaining_needed = 10 - len(current_set)
            if len(available) - start_idx < remaining_needed:
                return None
            
            # Heuristic: try vertices in order of increasing degree in the remaining subgraph?
            # Or just order?
            # Let's stick to simple index order for determinism in this step, 
            # but we can randomize the order of vertices initially? No, adj is fixed.
            
            for i in range(start_idx, len(available)):
                v = available[i]
                # Check if v is independent from all in current_set
                is_indep = True
                for u in current_set:
                    if adj[u][v]:
                        is_indep = False
                        break
                if not is_indep:
                    continue
                
                new_set = current_set + [v]
                # Pruning: if v is connected to many remaining vertices, it's a bad choice?
                # Not necessarily.
                
                # Recurse
                # We need to filter available vertices for the next step
                # Those not connected to v
                new_available = [u for u in available[i+1:] if not adj[v][u]]
                
                res = backtrack(0, new_set)
                if res:
                    return res
                
            return None

        # To speed up, we can order available vertices by degree?
        # Let's just use the list as is.
        return backtrack(0, [])

    # Main search loop
    while time.time() < deadline:
        # Check for triangles
        tri = find_triangle()
        if tri:
            i, j, k = tri
            # Remove a random edge from the triangle
            # To be deterministic-ish but varied, pick first one
            # Or randomize to escape local minima?
            # Let's pick randomly
            r = random.random()
            if r < 1/3:
                adj[i][j] = False
                adj[j][i] = False
            elif r < 2/3:
                adj[i][k] = False
                adj[k][i] = False
            else:
                adj[j][k] = False
                adj[k][j] = False
            continue
        
        # Check for independent set of size 10
        indep_set = find_independent_set_10()
        if indep_set:
            # Add edges to break the independent set
            # We need to add enough edges so that no 10 vertices are independent.
            # Adding edges between pairs in the set helps.
            # Let's add edges between all pairs in the set? That makes it a clique, definitely breaking independence.
            # But adding too many edges might create triangles.
            # Better: add edges strategically.
            # Let's add edges between a random pair in the set.
            # Or add edges to make the set smaller.
            # If we add an edge (u,v), any independent set containing both u and v is broken.
            # To break this specific set of 10, we just need to ensure that in the new graph,
            # these 10 vertices are no longer independent. Adding ONE edge between two of them is sufficient
            # to break THIS specific set. But other sets might form.
            
            # Let's add edges between all pairs in the set? No, that's 45 edges, likely creates triangles.
            # Let's add edges to a random vertex in the set to all others?
            # Or just add one random edge.
            
            # Strategy: Add an edge between two random vertices in the set.
            u, v = random.sample(indep_set, 2)
            if not adj[u][v]:
                adj[u][v] = True
                adj[v][u] = True
            continue
        
        # If no triangle and no independent set of size 10, we are done!
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    # If deadline reached, return best effort (should not happen if loop breaks on success)
    # But if we are stuck, return what we have?
    # The problem says "Maximize n". If we fail to find a valid graph on 40, we should try 39?
    # The loop above only works on 40.
    # If it times out, it's invalid.
    # To be safe, if we don't find a solution on 40 in time, we should probably return a known valid smaller graph?
    # But the goal is to maximize n.
    # If we can't find 40, returning 39 is better than invalid 40.
    # However, the search is very fast usually.
    
    # Fallback: return the seed graph if we fail to improve?
    # No, let's just return a valid graph on 39 if we can't find 40?
    # Actually, the loop condition `while time.time() < deadline` will exit.
    # If we exit without returning, we need a fallback.
    
    # Let's re-structure: try 40. If fail, try 39?
    # Given the speed, 40 should be found.
    
    # Fallback to seed just in case of unexpected timeout
    n_seed = 26
    return {'n': n_seed, 'edges': [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]}
# EVOLVE-BLOCK-END
