import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 20
    # Adjacency matrix
    adj = [[0] * n for _ in range(n)]
    edges = set()
    
    def is_valid_add(u, v):
        # Check if adding (u, v) creates a K3,3
        # We need to check if there exist a1, a2 in N(u)\{v} and b1, b2 in N(v)\{u}
        # such that a1, a2 are connected to b1, b2.
        # This is equivalent to: 
        # There do not exist a1, a2 in N(u)\{v} such that 
        # |N(a1) ∩ N(a2) ∩ (N(v)\{u})| >= 2
        
        nu = [x for x in range(n) if adj[u][x] and x != v]
        nv = [y for y in range(n) if adj[v][y] and y != u]
        
        if not nu or not nv:
            return True
            
        # For each pair in nu, check common neighbors in nv
        # We can precompute common neighbors for pairs in nu?
        # Or just iterate.
        for i in range(len(nu)):
            a1 = nu[i]
            # Find neighbors of a1 in nv
            n_a1_in_nv = [y for y in nv if adj[a1][y]]
            if len(n_a1_in_nv) < 2:
                continue
            # If a1 has >= 2 neighbors in nv, check if any other a2 in nu also has >= 2 neighbors in nv
            # and shares at least one with a1? No, they need to share TWO.
            # Actually, if a1 has >= 2 neighbors in nv, let's say b1, b2.
            # Then we need another a2 in nu (a2 != a1) that is also connected to b1 and b2.
            # So we can check for each pair of neighbors (b1, b2) of a1 in nv,
            # if there is another a2 in nu connected to both.
            
            # Optimization: 
            # For each a1 in nu, let S1 = N(a1) ∩ nv.
            # If |S1| >= 2, then for each pair (b1, b2) in S1, 
            # we need to ensure no other a2 in nu is connected to both b1 and b2.
            
            # Simpler: Iterate pairs of a1, a2 in nu.
            # For each pair, compute common neighbors in nv.
            # If count >= 2, invalid.
            
        # Let's just do the pair iteration, it's small.
        for i in range(len(nu)):
            a1 = nu[i]
            for j in range(i+1, len(nu)):
                a2 = nu[j]
                # Count common neighbors of a1 and a2 in nv
                common_count = 0
                for y in nv:
                    if adj[a1][y] and adj[a2][y]:
                        common_count += 1
                        if common_count >= 2:
                            return False
        return True

    def add_edge(u, v):
        adj[u][v] = 1
        adj[v][u] = 1
        edges.add((u, v) if u < v else (v, u))

    def remove_edge(u, v):
        adj[u][v] = 0
        adj[v][u] = 0
        edges.discard((u, v) if u < v else (v, u))

    # Initialize with empty graph or random sparse graph
    # Random start might be better to avoid local minima
    random.seed(42)
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.2: # Sparse start
                add_edge(u, v)

    best_edges = len(edges)
    
    # Local Search
    while time.time() < deadline:
        # Try to add an edge
        added = False
        # Shuffle non-edges to randomize
        non_edges = [(u, v) for u in range(n) for v in range(u+1, n) if not adj[u][v]]
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline:
                break
            if is_valid_add(u, v):
                add_edge(u, v)
                added = True
                best_edges = len(edges)
                # Restart loop to re-shuffle? Or just continue?
                # Continuing is faster, but might get stuck.
                # Let's just continue adding if possible.
        
        if not added:
            # Stuck. Try to remove a random edge to escape local minimum
            if edges:
                edge_list = list(edges)
                random.shuffle(edge_list)
                for u, v in edge_list:
                    if time.time() > deadline:
                        break
                    # Check if removing this edge allows adding others?
                    # Just remove and continue search.
                    remove_edge(u, v)
                    break # Remove one and restart search loop
            else:
                # Empty graph, shouldn't happen if we can add edges
                break

    # Convert set to list of lists
    return [[u, v] for u, v in edges]
