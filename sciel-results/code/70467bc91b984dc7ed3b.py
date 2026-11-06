import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # 5 seconds buffer before 90s kill
    
    N = 256
    # Squares up to 2*256 = 512
    squares = set()
    for k in range(1, 23):
        squares.add(k * k)
    
    # 1. Filter vertices that have self-conflicts (2*v is a square)
    candidates = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            candidates.append(v)
    
    # Map original values to indices in candidates list for efficient adjacency
    # Actually, we can just use the values directly in the adjacency sets/dicts
    # But lists of indices are faster for bit operations or simple loops.
    
    n_cand = len(candidates)
    # adjacency[i] is a set of indices j such that candidates[i] + candidates[j] is NOT a square
    # We want Maximum Clique in this graph.
    
    # Precompute adjacency
    adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        vi = candidates[i]
        for j in range(i + 1, n_cand):
            vj = candidates[j]
            if (vi + vj) not in squares:
                adj[i].add(j)
                adj[j].add(i)
    
    # Tomita's Algorithm for Maximum Clique
    # best_clique stores the indices into 'candidates'
    best_clique = []
    
    def expand(clique, candidates_list):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
        
        if not candidates_list:
            if len(clique) > len(best_clique):
                best_clique = clique[:]
            return
        
        # Greedy Coloring of the subgraph induced by candidates_list
        # We color the vertices in candidates_list
        # The color of a vertex is an upper bound on the clique size containing it
        # We sort candidates by color to prune
        
        # Simple greedy coloring: 
        # colors[v] = 1 + max(colors[u] for u in neighbors in candidates_list processed so far)
        # Actually, standard Tomita uses a specific ordering.
        # Let's use a simpler bound: 
        # If we can color the graph with k colors, max clique <= k.
        
        # To speed up, we pick a vertex v, branch on including v or not.
        # Upper bound: len(candidates_list) is weak.
        # Better: Greedy coloring bound.
        
        # Implementation of greedy coloring for bound
        # We maintain a list of (vertex, color)
        # We process vertices in candidates_list.
        
        # To make it fast, we can just use a simple heuristic bound:
        # The size of the maximum clique is at most the size of the maximum independent set in the complement? No.
        
        # Let's use the standard coloring bound.
        # We color the subgraph G[candidates_list].
        # If the chromatic number is k, then omega(G[candidates_list]) <= k.
        
        # Efficient coloring for small subgraphs:
        # We can just iterate and assign colors.
        
        # However, implementing full Tomita with dynamic coloring is complex in Python within time limits if not careful.
        # Given N=245, a recursive search with simple pruning might be too slow.
        # But the graph is dense?
        # Probability that a+b is a square is low.
        # Number of squares ~ 22.
        # For a fixed v, number of u such that v+u is square is small.
        # So the "conflict" graph is sparse.
        # The "good" graph (non-conflict) is DENSE.
        # Max Clique in a Dense Graph is hard.
        
        # Wait, if the "good" graph is dense, the Max Clique is likely very large.
        # If the conflict graph is sparse, the Max Independent Set is large.
        
        # Let's check density.
        # For v=1, conflicts with 3 (1+3=4), 8 (1+8=9), 15, 24, 35, 48, 63, 80, 99, 120, 143, 168, 195, 224, 255.
        # ~15 conflicts per vertex on average?
        # So degree in conflict graph ~ 15.
        # Degree in good graph ~ 245 - 15 = 230.
        # Dense graph.
        
        # Finding Max Clique in a dense graph of 245 nodes is very hard for exact algorithms.
        # However, we can use a heuristic to find a very large clique, which is likely optimal or near-optimal.
        
        # Heuristic: Greedy construction + Local Search (Simulated Annealing or Hill Climbing).
        # Start with a random large independent set in the conflict graph.
        # Or start with a random large clique in the good graph.
        
        # Let's try a greedy approach to build a large clique in the good graph.
        # Then try to improve it by swapping.
        
        # Given the time constraint (90s), an exact solver might timeout.
        # A good heuristic is safer.
        
        # Heuristic Strategy:
        # 1. Generate a random permutation of candidates.
        # 2. Greedily select vertices that are compatible with all previously selected vertices.
        # 3. This gives a maximal clique.
        # 4. Try to add vertices? No, maximal means can't add.
        # 5. Try to replace a vertex to allow adding more?
        
        # Better Heuristic:
        # We want Max Independent Set in Conflict Graph.
        # Conflict Graph is Sparse.
        # Greedy MIS:
        # Pick vertex with min degree, add to MIS, remove it and its neighbors.
        # This is a standard approximation.
        
        # Let's implement Greedy MIS with multiple restarts and local search.
        
        def greedy_mis(vertices, adj_conflict):
            # vertices: list of indices
            # adj_conflict: list of sets
            mis = []
            remaining = set(vertices)
            while remaining:
                # Pick vertex with min degree in remaining subgraph
                min_deg = float('inf')
                chosen = None
                for v in remaining:
                    deg = len(adj_conflict[v] & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        chosen = v
                mis.append(chosen)
                # Remove chosen and its neighbors
                to_remove = {chosen} | (adj_conflict[chosen] & remaining)
                remaining -= to_remove
            return mis
        
        # Build Conflict Graph Adjacency
        adj_conf = [set() for _ in range(n_cand)]
        for i in range(n_cand):
            vi = candidates[i]
            for j in range(i + 1, n_cand):
                vj = candidates[j]
                if (vi + vj) in squares:
                    adj_conf[i].add(j)
                    adj_conf[j].add(i)
        
        best_size = 0
        best_set = set()
        
        # Run greedy MIS multiple times with different orderings?
        # The min-degree choice is deterministic.
        # To add randomness, we can shuffle the vertices before sorting by degree,
        # or pick randomly among vertices with min degree.
        
        import random
        random.seed(42)
        
        for _ in range(200):
            # Shuffle to break ties or change order if we used index order
            # But min-degree is structural.
            # To get different results, we can randomize the tie-breaking.
            
            mis = []
            remaining = set(range(n_cand))
            while remaining:
                # Find min degree
                min_deg = float('inf')
                min_verts = []
                for v in remaining:
                    deg = len(adj_conf[v] & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        min_verts = [v]
                    elif deg == min_deg:
                        min_verts.append(v)
                
                # Pick randomly from min_verts
                chosen = random.choice(min_verts)
                mis.append(chosen)
                
                to_remove = {chosen} | (adj_conf[chosen] & remaining)
                remaining -= to_remove
            
            if len(mis) > best_size:
                best_size = len(mis)
                best_set = set(mis)
            
            if time.time() > deadline:
                break
        
        # Local Search Improvement
        # Try to swap vertices in the MIS to increase size?
        # Or just accept the best greedy result.
        # Given the sparsity, greedy min-degree is usually very good.
        
        # Convert indices back to values
        result = [candidates[i] for i in best_set]
        
        return result

