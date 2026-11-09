import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    # We aim for n=40, then 39, down to 26 (seed)
    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
            
        # Number of attempts
        # n=40 is hardest. Give it more time initially.
        max_attempts = 500 if n == 40 else 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Represent graph using bitmasks for speed
            # adj[i] is an integer where bit j is 1 if (i,j) is an edge
            adj = [0] * n
            
            # 1. Initialize with a random maximal triangle-free graph
            # We add random edges in random order. If adding (u,v) creates no triangle, add it.
            # To make it "sparse" initially so we have room to add more edges later to shrink alpha,
            # we don't necessarily need to make it maximal immediately.
            # However, a random maximal TF graph is a good start.
            
            # Shuffle vertices to randomize
            vertices = list(range(n))
            random.shuffle(vertices)
            
            # Generate candidate edges
            # To speed up, we can just iterate all pairs and randomly decide to add?
            # Or shuffle all pairs.
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(edges)
            
            for u, v in edges:
                # Check triangle: adj[u] & adj[v] == 0
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # 2. Iteratively shrink independence number by adding edges
            # We want alpha < 10.
            # We repeatedly find a large independent set.
            # If size >= 10, we try to add an edge between two vertices in it.
            
            max_shrink_steps = 100
            success = False
            
            for step in range(max_shrink_steps):
                if time.time() > deadline:
                    break
                    
                # Find a large independent set using a greedy heuristic
                # To find a *large* one, we can use a simple greedy:
                # Pick vertex with smallest degree in the current candidate set?
                # Actually, for independent set, we want vertices that are not connected.
                # Greedy:
                # 1. Start with all vertices available.
                # 2. Pick a vertex v from available.
                # 3. Add v to IS.
                # 4. Remove v and all its neighbors from available.
                # To get a large IS, picking vertices with low degree in the *remaining* graph is good.
                
                available = (1 << n) - 1
                is_set = []
                
                while available:
                    # Pick vertex with minimum degree in the induced subgraph of 'available'
                    # This is expensive to compute exactly.
                    # Approximation: Pick random vertex? Or vertex with min global degree?
                    # Let's try: Pick vertex v in available that minimizes popcount(adj[v] & available)
                    
                    # Optimization: If n is small (<=40), we can iterate all bits.
                    # But doing this for every step of IS construction is O(n^2).
                    # n=40, so 1600 ops per IS construction. 100 steps -> 160k ops. Fast.
                    
                    best_v = -1
                    min_deg = n + 1
                    
                    # Iterate through available bits
                    avail = available
                    while avail:
                        lsb = avail & -avail
                        v = lsb.bit_length() - 1
                        avail -= lsb
                        
                        # Count neighbors of v in available
                        deg = (adj[v] & available).bit_count()
                        
                        if deg < min_deg:
                            min_deg = deg
                            best_v = v
                            # Early exit if deg is 0 (isolated in available)
                            if deg == 0:
                                break
                    
                    if best_v == -1:
                        break
                    
                    is_set.append(best_v)
                    # Remove best_v and its neighbors from available
                    mask = 1 << best_v
                    available &= ~(mask | adj[best_v])
                    
                is_size = len(is_set)
                
                if is_size < 10:
                    success = True
                    break
                else:
                    # Try to add an edge between two vertices in is_set
                    # We need (u, v) in is_set such that (adj[u] & adj[v]) == 0
                    # If we add this edge, we break the independence of this specific set.
                    
                    # To maximize chance of reducing alpha, we should pick edges that connect "central" parts?
                    # Or just any edge.
                    # Let's try all pairs in is_set.
                    added_edge = False
                    # Shuffle is_set to randomize choice
                    random.shuffle(is_set)
                    
                    for i in range(len(is_set)):
                        u = is_set[i]
                        for j in range(i + 1, len(is_set)):
                            v = is_set[j]
                            if (adj[u] & adj[v]) == 0:
                                # Safe to add
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                added_edge = True
                                break
                        if added_edge:
                            break
                    
                    if not added_edge:
                        # Cannot add any edge within this IS without creating a triangle.
                        # This IS is "stable" in the sense that its vertices have overlapping neighborhoods.
                        # We might be stuck.
                        # Try to break it by adding edges *to* vertices outside?
                        # No, adding edges to outside doesn't shrink this IS directly unless we remove vertices from it?
                        # Wait, if we add edges, we might create triangles? No, we only add if no triangle.
                        # If we can't add edges *inside* the IS, we can't shrink *this* IS directly.
                        # But adding edges elsewhere might change the structure so a *different* greedy IS is smaller?
                        # Or we are stuck at alpha >= 10.
                        # Break and restart with a new random graph.
                        break
            
            if success:
                # Construct edge list
                edge_list = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edge_list.append([i, j])
                
                return {'n': n, 'edges': edge_list}
        
    # Fallback: Return a known valid graph (And(9) n=26)
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

if __name__ == "__main__":
    result = construct()
    print(f"n={result['n']}, edges={len(result['edges'])}")
