import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    # 1. Define Tutte-Coxeter Graph (n=30)
    # It is 3-regular, triangle-free, alpha=9.
    # We need its edge list.
    # Tutte-Coxeter graph can be generated from the icosahedron or specific adjacency.
    # Let's use a known adjacency list for Tutte-Coxeter.
    # Or generate it: It is the point-line graph of the generalized quadrangle GQ(2,2)? No.
    # It is the incidence graph of the 5-point line?
    # Actually, let's just use a hardcoded valid graph if possible, or generate Tutte-Coxeter.
    
    # Tutte-Coxeter graph edges:
    # Vertices 0-29.
    # It is the bipartite graph between two sets of 15 vertices? No, it's not bipartite?
    # Tutte-Coxeter is 3-regular.
    # Let's use a Python library to generate it? No, standard library only.
    # I will hardcode the edges of the Tutte-Coxeter graph.
    
    # Alternatively, I can use the Andrasfai graph A_9 (n=27) which is easier to generate?
    # A_k is Z_{3k} with edges i~j if (j-i) mod 3k in {1, 3, ..., 3k-2}? No.
    # A_9: n=27.
    # Seed was n=26.
    # Let's try to generate a circulant for n=30 or 33.
    
    # Let's implement a generic local search.
    
    def get_edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def is_triangle_free(adj, n):
        # For each vertex, check if its neighbors form a clique? No, check if any two neighbors are connected.
        for i in range(n):
            neighbors = [j for j in range(n) if adj[i] & (1 << j)]
            for u in range(len(neighbors)):
                for v in range(u+1, len(neighbors)):
                    if adj[neighbors[u]] & (1 << neighbors[v]):
                        return False
        return True

    def max_independent_set_size(adj, n):
        # Branch and bound for alpha
        # Returns the size of the largest independent set
        # We can stop if we find size 10
        best = 0
        # Order vertices by degree?
        # Simple recursive search with pruning
        def dfs(remaining_mask, current_size, last_added):
            nonlocal best
            if current_size >= 10:
                return 10 # Early exit, we only care if it's >= 10
            if remaining_mask == 0:
                if current_size > best:
                    best = current_size
                return best
            
            # Pruning: if current_size + popcount(remaining_mask) <= best, return
            if current_size + bin(remaining_mask).count('1') <= best:
                return best
            
            # Pick a vertex
            # Heuristic: pick vertex with highest degree in remaining graph
            # Or just pick the first one
            # To speed up, pick vertex with max degree
            max_deg = -1
            chosen = -1
            temp_mask = remaining_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                deg = bin(adj[v] & remaining_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
                temp_mask ^= lsb
            
            if chosen == -1:
                return best

            # Option 1: Include chosen
            # New remaining: neighbors of chosen removed
            new_mask = remaining_mask & ~(adj[chosen] | (1 << chosen))
            res1 = dfs(new_mask, current_size + 1, chosen)
            if res1 >= 10: return 10
            
            # Option 2: Exclude chosen
            new_mask = remaining_mask & ~(1 << chosen)
            res2 = dfs(new_mask, current_size, -1)
            if res2 >= 10: return 10
            
            return max(res1, res2)

        return dfs((1 << n) - 1, 0, -1)

    def is_valid(adj, n):
        if not is_triangle_free(adj, n):
            return False
        if max_independent_set_size(adj, n) >= 10:
            return False
        return True

    # Initial Best: Andrasfai A_9 (n=27) or Tutte-Coxeter (n=30)
    # Let's use Tutte-Coxeter if I can generate it.
    # Actually, let's start with n=30 random and search.
    # But a good seed is important.
    
    # Known: n=30 is valid.
    # Let's try to find n=31, 32...
    
    best_n = 0
    best_adj = None
    
    # Seed with a random graph for n=30
    # Or better, seed with a known structure.
    # Let's use a circulant graph C_30(S).
    # Try to find S for n=30.
    # Or just run local search from random.
    
    # To ensure we beat 26, we must find n >= 27.
    # Andrasfai A_9 is n=27.
    # Let's generate A_9.
    # A_k: vertices 0..3k-1. Edge i~j if (j-i) mod 3k is in {1, 4, ..., 3k-2}?
    # For k=9, n=27.
    # Differences: 1, 4, 7, 10, 13, 16, 19, 22, 25.
    # This graph is triangle-free.
    # Independence number of A_9 is 9.
    # So n=27 is valid.
    
    def generate_andrasfai(k):
        n = 3 * k
        adj = [0] * n
        diffs = [1, 4, 7, 10, 13, 16, 19, 22, 25]
        # Generalize: diffs = [1, 1+3, 1+6, ...]
        diffs = [1 + 3*i for i in range(k)]
        for i in range(n):
            for d in diffs:
                j = (i + d) % n
                if i != j:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n

    adj_27, n_27 = generate_andrasfai(9)
    if is_valid(adj_27, n_27):
        best_n = n_27
        best_adj = adj_27
    
    # Now try to improve to n=30, 31, ...
    # We will run local search for each n.
    
    def local_search(adj, n, max_time=5.0):
        # Flip edges to reduce violations
        # Violation = triangles + (alpha - 9) * large_penalty
        # We want 0 triangles and alpha <= 9.
        
        def score(adj, n):
            # Count triangles
            tri_count = 0
            for i in range(n):
                neighbors = [j for j in range(n) if adj[i] & (1 << j)]
                for u in range(len(neighbors)):
                    for v in range(u+1, len(neighbors)):
                        if adj[neighbors[u]] & (1 << neighbors[v]):
                            tri_count += 1
            # Estimate alpha or calculate exact
            # Exact alpha is slow, use greedy upper bound for scoring?
            # No, we need to ensure validity.
            # For local search, we can use a fast heuristic for alpha.
            # But for the final check, we use exact.
            # Let's use a fast greedy independent set size as a proxy for alpha in the score.
            # If greedy finds 10, it's likely invalid.
            
            # Greedy Max Independent Set
            remaining = (1 << n) - 1
            count = 0
            while remaining:
                # Pick vertex with min degree in remaining
                min_deg = n
                chosen = -1
                temp = remaining
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    deg = bin(adj[v] & remaining).count('1')
                    if deg < min_deg:
                        min_deg = deg
                        chosen = v
                    temp ^= lsb
                if chosen == -1: break
                count += 1
                remaining &= ~(adj[chosen] | (1 << chosen))
            
            # Score: high penalty for triangles, high penalty for large independent set
            # We want alpha <= 9.
            # If count >= 10, penalty is high.
            return tri_count * 1000 + max(0, count - 9) * 100

        current_score = score(adj, n)
        start = time.time()
        
        while time.time() - start < max_time:
            # Randomly flip an edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j: continue
            
            # Flip
            adj[i] ^= (1 << j)
            adj[j] ^= (1 << i)
            
            new_score = score(adj, n)
            
            if new_score < current_score:
                current_score = new_score
            else:
                # Revert
                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)
                
            # Check for exact validity periodically
            if int(time.time() - start) % 2 == 0:
                if is_valid(adj, n):
                    return adj, True
        
        return adj, False

    # Try n from 30 to 40
    for n in range(30, 41):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Initialize with random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.3:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Run local search
        adj, found = local_search(adj, n, max_time=2.0)
        
        if found:
            best_n = n
            best_adj = adj
            # Don't break, try to go higher?
            # Yes, keep searching for higher n.
        else:
            # Check if the result is valid anyway
            if is_valid(adj, n):
                if n > best_n:
                    best_n = n
                    best_adj = adj

    # If we didn't find anything better than 27, return 27.
    # But we should ensure we return a valid graph.
    if best_n == 0:
        return {'n': 26, 'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]}
        
    edges = get_edges_from_adj(best_adj, best_n)
    return {'n': best_n, 'edges': edges}

construct()
