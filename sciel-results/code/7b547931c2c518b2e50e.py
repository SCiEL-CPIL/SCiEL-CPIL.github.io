import random
from itertools import combinations

def construct():
    n = 18
    vertices = list(range(n))
    
    def is_k4_free(adj, u, v):
        # Check if adding (u,v) creates a K4
        # A K4 is formed if there is a triangle in N(u) intersect N(v)
        common = []
        for w in vertices:
            if w == u or w == v:
                continue
            if adj[u][w] and adj[v][w]:
                common.append(w)
        
        # Check if common neighbors contain a triangle
        for i in range(len(common)):
            for j in range(i+1, len(common)):
                w1 = common[i]
                w2 = common[j]
                if adj[w1][w2]:
                    # w1, w2 are connected.
                    # Is there a w3 connected to both w1 and w2 and u and v?
                    # Actually, if w1-w2 is an edge, and both are in N(u) \cap N(v),
                    # then {u, v, w1, w2} is a K4.
                    return False
        return True

    def max_triangle_free_size(adj):
        # Branch and bound or simple check for n=18
        # We want the largest subset with no triangles.
        # We can check sizes from 18 down to 1.
        # To speed up, we can use a simple heuristic: 
        # If we find one of size k, we know max >= k.
        # We want to minimize the max, so we just need to know if it's >= 12.
        # But for optimization, we need the exact value to compare.
        
        best = 0
        # Check all subsets? 2^18 is too many for repeated calls in local search.
        # Use a greedy approximation or a smarter search.
        # For n=18, checking all combinations of size 12 is 18564.
        # If we do this for every step in local search, it might be slow.
        # Let's estimate: 18564 * (cost to check triangle-free).
        # Cost to check triangle-free for size 12: O(12^3) = 1728.
        # 18564 * 1728 ~ 32 million operations per check.
        # If we do 1000 checks, that's 32 billion. Too slow.
        
        # We need a faster way to estimate or find the max triangle-free set.
        # Or, we only need to verify if size >= 12.
        # If we are in local search, we want to reduce the size.
        # We can maintain the current max set S.
        # When we add an edge (u,v), the only sets that can change are those containing both u and v?
        # No, adding an edge makes sets containing that edge NOT triangle-free.
        # So the new max triangle-free set is the max of:
        # 1. Old max sets not containing both u and v.
        # 2. Subsets of the old max sets that contain both u and v, minus one of them?
        
        # Simpler: Just recompute using a fast heuristic.
        # Heuristic: 
        # 1. Start with all vertices.
        # 2. While there is a triangle, remove one vertex.
        # This gives an independent set? No, triangle-free set.
        # This is the greedy algorithm for maximum triangle-free set.
        # It's an approximation.
        
        # Let's use the greedy approximation for speed in the inner loop.
        # And use exact check only at the end or for the best candidate.
        
        # Greedy Triangle-Free Set:
        # 1. Start with V.
        # 2. While there is a triangle in induced subgraph:
        #    Pick a vertex from a triangle and remove it.
        # This is not optimal but fast.
        
        # Better: 
        # We can check if there exists a triangle-free set of size 12.
        # If yes, return 12 (or higher).
        # If no, check 11, etc.
        
        # Let's stick to the exact check but optimize it.
        # For n=18, 2^18 = 262144.
        # We can precompute triangles?
        # Or use bitmasks.
        
        # Let's implement a bitmask-based checker for "is this subset triangle-free".
        # And iterate through subsets in decreasing order of size.
        # To speed up, we can prune.
        
        # Actually, for the local search, we can just return the size of the greedy triangle-free set.
        # It might be smaller than the true max, leading to premature stopping.
        # But if we use it to guide the search (add edges to break the greedy set), it might still work.
        
        # Let's try to compute the exact max size for the current graph, but only if the greedy estimate is high.
        
        # Greedy:
        remaining = set(vertices)
        adj_local = [adj[i] for i in vertices] # Not needed, use global adj
        
        # Find triangles in remaining
        # This is O(n^3).
        
        # Let's just return a lower bound on the max triangle-free set size.
        # If we find a triangle-free set of size S, the true max is >= S.
        # We want to MINIMIZE the true max.
        # So if we find a set of size 12, we know the score is at least 12.
        # If we can't find one of size 12, the score might be 11.
        
        # So, check if there is a triangle-free set of size 12.
        # If yes, return 12.
        # If no, check 11.
        # If no, return 10 (or lower).
        
        # This is fast if we stop early.
        
        for size in [12, 11, 10, 9]:
            found = False
            for combo in combinations(vertices, size):
                # Check if combo is triangle-free
                is_tf = True
                for i in range(size):
                    u = combo[i]
                    for j in range(i+1, size):
                        v = combo[j]
                        if adj[u][v]:
                            for k in range(j+1, size):
                                w = combo[k]
                                if adj[u][w] and adj[v][w]:
                                    is_tf = False
                                    break
                        if not is_tf:
                            break
                    if not is_tf:
                        break
                if is_tf:
                    return size
            if found:
                return size
        return 9 # Or lower

    best_score = 18
    best_edges = []
    
    for trial in range(200):
        random.seed(trial)
        adj = [[0]*n for _ in range(n)]
        
        # Randomly add edges to make it denser but K4-free
        # Start with empty, add edges in random order
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if is_k4_free(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
                
        # Local search
        for step in range(50):
            score = max_triangle_free_size(adj)
            if score < 12:
                break
                
            # Find a triangle-free set of size 'score' (or 12)
            # We need the actual set to try to break it.
            # Let's find one.
            target_set = None
            for combo in combinations(vertices, score):
                is_tf = True
                for i in range(score):
                    u = combo[i]
                    for j in range(i+1, score):
                        v = combo[j]
                        if adj[u][v]:
                            for k in range(j+1, score):
                                w = combo[k]
                                if adj[u][w] and adj[v][w]:
                                    is_tf = False
                                    break
                        if not is_tf:
                            break
                    if not is_tf:
                        break
                if is_tf:
                    target_set = list(combo)
                    break
            
            if not target_set:
                break
                
            # Try to add an edge within target_set or connecting to it
            # to reduce the size.
            improved = False
            # Try all pairs in target_set
            for i in range(len(target_set)):
                for j in range(i+1, len(target_set)):
                    u = target_set[i]
                    v = target_set[j]
                    if not adj[u][v]:
                        if is_k4_free(adj, u, v):
                            # Add edge
                            adj[u][v] = 1
                            adj[v][u] = 1
                            new_score = max_triangle_free_size(adj)
                            if new_score < score:
                                improved = True
                                break
                            else:
                                # Revert
                                adj[u][v] = 0
                                adj[v][u] = 0
                if improved:
                    break
            
            if not improved:
                # Try adding edges from target_set to outside?
                # Or just break.
                break

        final_score = max_triangle_free_size(adj)
        if final_score < best_score:
            best_score = final_score
            best_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        best_edges.append([i, j])
                        
    return best_edges
