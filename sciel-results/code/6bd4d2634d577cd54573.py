"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n = 20
    # Initialize with a random graph or a known good structure
    # A good starting point is often a random graph with moderate density
    # or the seed solution. Let's start with the seed solution (30 edges)
    # but random restarts with SA are better.
    
    best_edges = []
    best_score = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # We will run multiple simulated annealing runs
    # The state is the set of edges.
    # Objective: Maximize |E| subject to no K2,3.
    # Penalty method: Score = |E| - lambda * (# of K2,3 violations)
    # If we maintain validity strictly, we only add edges if they don't create a K2,3.
    # But SA with validity constraints is hard because removing an edge might allow adding two others.
    # Better approach: 
    # 1. Start with a valid graph.
    # 2. Move: Add a random edge. If it creates a K2,3, reject or try to fix by removing a conflicting edge.
    # 3. Move: Remove a random edge.
    # 4. Move: Swap: Remove edge e1, add edge e2 if valid.
    
    # A more robust approach for K2,3-free:
    # The condition "no two vertices have 3 common neighbors" is equivalent to:
    # For every pair of vertices (u, v), |N(u) ∩ N(v)| <= 2.
    
    def count_violations(adj_matrix, n):
        # Returns list of (u, v) pairs that violate the condition
        # Or just count. For SA, we need to know if a move is valid.
        # Efficient check: When adding (u, v), we need to check if there are >= 3 common neighbors.
        # Common neighbors of u and v are w such that adj[u][w] and adj[v][w] are True.
        pass

    def is_valid_addition(adj, u, v, n):
        """Check if adding edge (u, v) keeps the graph K2,3-free."""
        # We need to ensure that for any w, z, the new edge doesn't create a K2,3.
        # Actually, adding (u, v) only affects pairs involving u or v.
        # Specifically, it might create a K2,3 with:
        # 1. u and v as the "2" part? No, K2,3 has two vertices connected to 3 others.
        #    If we add (u, v), u and v are now adjacent.
        #    A K2,3 consists of two vertices {x, y} and three vertices {a, b, c} such that x,y are connected to a,b,c.
        #    If we add (u, v), we might create a K2,3 where {u, v} are the "2" vertices.
        #    This happens if u and v have >= 3 common neighbors.
        #    Let common = N(u) ∩ N(v). If |common| >= 3, then {u, v} and any 3 from common form a K2,3.
        # 2. Or {u, v} are part of the "3" vertices?
        #    If u is connected to x, y, z and v is connected to x, y, z, and u-v is added... 
        #    Wait, the definition is "no two vertices have three common neighbors".
        #    This condition is symmetric. It applies to ANY pair of vertices in the graph.
        #    So, adding (u, v) requires:
        #    a) For the pair (u, v): |N(u) ∩ N(v)| <= 2. (u and v are the "two vertices")
        #    b) For any other vertex w: The pair (u, w) must not have 3 common neighbors.
        #       Adding (u, v) adds v to N(u). So v might become a common neighbor of u and w if v is connected to w.
        #       So if v is connected to w, then v is in N(u) ∩ N(w).
        #       We need to check if |N(u) ∩ N(w)| becomes 3.
        #       This means before adding, |N(u) ∩ N(w)| was 2, and v is connected to w.
        #    c) Similarly for pair (v, w).
        
        # Let's implement the check:
        # 1. Check pair (u, v):
        common_uv = 0
        for w in range(n):
            if adj[u][w] and adj[v][w]:
                common_uv += 1
                if common_uv > 2:
                    return False
        
        # 2. Check pairs (u, w) for all w != v
        for w in range(n):
            if w == u or w == v:
                continue
            # Check if adding (u, v) causes (u, w) to have 3 common neighbors.
            # The new common neighbor is v, IF adj[v][w] is True.
            if adj[v][w]:
                # Count common neighbors of u and w
                count = 0
                for x in range(n):
                    if x == u or x == w:
                        continue
                    if adj[u][x] and adj[w][x]:
                        count += 1
                        if count >= 3:
                            return False
                # Note: We must also consider if v was already a common neighbor?
                # No, we are checking the state AFTER addition.
                # In the state after addition, adj[u][v] is True.
                # So v is a neighbor of u.
                # If adj[v][w] is True, v is a neighbor of w.
                # So v is a common neighbor of u and w.
                # The loop above counts x in N(u) ∩ N(w).
                # Does it count v?
                # adj[u][v] is True (we are adding it).
                # adj[w][v] is True (checked above).
                # So v is counted in the loop.
                # We need total count <= 2.
                # So if count >= 3, it's invalid.
        
        # 3. Check pairs (v, w) for all w != u
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u][w]:
                count = 0
                for x in range(n):
                    if x == v or x == w:
                        continue
                    if adj[v][x] and adj[w][x]:
                        count += 1
                        if count >= 3:
                            return False
        
        return True

    def initialize_graph(n, seed):
        random.seed(seed)
        adj = [[False] * n for _ in range(n)]
        edges = []
        # Start with a sparse random graph
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.2:
                    adj[u][v] = True
                    adj[v][u] = True
                    edges.append((u, v))
        return adj, edges

    def count_edges(adj, n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    count += 1
        return count

    def simulated_annealing(n, seed, max_iters):
        adj, edges = initialize_graph(n, seed)
        current_edges = count_edges(adj, n)
        best_edges_local = edges.copy()
        best_score_local = current_edges
        
        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        alpha = (T_end / T_start) ** (1.0 / max_iters)
        T = T_start
        
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        for i in range(max_iters):
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.6:
                # Add a random edge
                u, v = random.choice(all_edges)
                if not adj[u][v]:
                    if is_valid_addition(adj, u, v, n):
                        # Accept
                        adj[u][v] = True
                        adj[v][u] = True
                        edges.append((u, v))
                        current_edges += 1
                        if current_edges > best_score_local:
                            best_score_local = current_edges
                            best_edges_local = edges.copy()
                    else:
                        # Reject (or could try to fix, but simple SA rejects)
                        pass
            elif move_type < 0.9:
                # Remove a random edge
                if edges:
                    u, v = random.choice(edges)
                    if adj[u][v]:
                        adj[u][v] = False
                        adj[v][u] = False
                        edges.remove((u, v))
                        current_edges -= 1
            else:
                # Swap: Remove one, add another
                if edges:
                    u1, v1 = random.choice(edges)
                    u2, v2 = random.choice(all_edges)
                    if not adj[u2][v2] and (u1, v1) != (u2, v2):
                        # Try removing u1-v1
                        adj[u1][v1] = False
                        adj[v1][u1] = False
                        edges.remove((u1, v1))
                        # Try adding u2-v2
                        if is_valid_addition(adj, u2, v2, n):
                            adj[u2][v2] = True
                            adj[v2][u2] = True
                            edges.append((u2, v2))
                            current_edges += 0 # Net change 0? 
                            # Wait, remove 1, add 1. Net 0.
                            # This move doesn't increase edge count directly.
                            # It's useful for rearrangement.
                            # But we want to maximize edges.
                            # So swaps are only useful if they enable future additions.
                        else:
                            # Revert
                            adj[u1][v1] = True
                            adj[v1][u1] = True
                            edges.append((u1, v1))
                            # Try adding u2-v2 without removing u1-v1?
                            # That's just an add move.
                            # So swap is: Remove e1, Add e2.
                            # If e2 is invalid, revert.
            
            T *= alpha
            
            # Check time
            if time.time() > deadline - 5.0:
                break
        
        return best_edges_local, best_score_local

    # Run multiple SA instances
    num_runs = 50
    for run in range(num_runs):
        if time.time() > deadline - 10.0:
            break
        
        # Estimate max iters based on remaining time
        # is_valid_addition is O(n^2) roughly.
        # Let's say 1000 iters takes ~0.1s? 
        # Let's do a quick benchmark in the first run or just set a fixed reasonable number.
        # n=20 is small. O(n^2) is 400 ops. Very fast.
        # We can do many iterations.
        max_iters = 5000
        
        edges_sa, score_sa = simulated_annealing(n, seed=run, max_iters=max_iters)
        
        if score_sa > best_score:
            best_score = score_sa
            best_edges = edges_sa
    
    return best_edges
# EVOLVE-BLOCK-END
