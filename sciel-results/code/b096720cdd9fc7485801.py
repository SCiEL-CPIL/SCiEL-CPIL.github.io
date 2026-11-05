import time
import random

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n = 40
    m = 401
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = None
    best_score = float('inf')
    
    # Run multiple random restarts with different randomization in the greedy choice
    for restart in range(100):
        if time.time() > deadline:
            break
        
        random.seed(restart)
        
        # Start with a sparse base or empty? 
        # Random sampling found ~10-12. Greedy might do better.
        # Let's try a greedy approach: start with no edges, add one that minimizes the max book increase.
        # However, checking all 780 edges for each of 401 additions is 401 * 780 * O(n^2) which is too slow in Python.
        
        # Optimized Greedy:
        # We maintain adjacency sets and a count of common neighbors for each edge (book size).
        # When we add edge (u,v), it increases the book size of all edges (u,w) and (v,w) where w is a common neighbor of u and v? No.
        # Adding edge (u,v) creates new triangles (u,v,w) for all w in N(u) ∩ N(v).
        # This increases the book size of edge (u,w) by 1 (for each w in N(v) that is connected to u? No, (u,v,w) is a triangle, so edges (u,v), (v,w), (w,u) all get +1 to their book size).
        # Specifically:
        # - Edge (u,v) gets +|N(u) ∩ N(v)| to its book size.
        # - For each w in N(u) ∩ N(v), edges (u,w) and (v,w) get +1 to their book size.
        
        # To keep this fast, we can use a randomized greedy heuristic.
        # 1. Generate a random ordering of edges.
        # 2. Iterate through the order. For each edge, estimate the cost.
        # 3. This is still O(m^2) roughly if we recompute.
        
        # Alternative: Simulated Annealing or Local Search on a random graph.
        # The current program does random sampling. Local search might refine it.
        # Start with a random graph, then swap edges to reduce max book.
        
        # Let's try: 
        # 1. Generate random graph.
        # 2. Compute max book.
        # 3. Identify edges with high book.
        # 4. Try to swap one endpoint of a high-book edge with a non-neighbor to reduce book sizes.
        
        # Actually, a simpler and often effective strategy for "minimize max degree/triangle" is to use a balanced random graph and then do local moves.
        
        # Let's implement a local search:
        # 1. Start with random graph.
        # 2. Compute book sizes for all edges.
        # 3. Find edge e with max book size B_max.
        # 4. Try to move an endpoint of e to a random non-neighbor vertex, or swap endpoints of two edges.
        
        # Given time constraints, let's stick to a faster variant of the current approach but with a better initial guess or a quick local improvement step.
        
        # Let's try: Random Graph + Local Search (Swap)
        # 1. Random sample edges.
        # 2. Compute adj sets and book sizes.
        # 3. While time permits:
        #    a. Find edge (u,v) with max book size.
        #    b. Try to replace (u,v) with (u,w) where w is random and not connected to u, or (v,w).
        #    c. If new max book < old max book, accept.
        
        # This is expensive to recompute fully. 
        # Let's use a simpler heuristic: 
        # The Edwards-Khadzhiivanov-Nikiforov bound is ~6.67. Integer min is likely 7 or 8.
        # Random graphs gave 10-12. 
        # A more structured graph might be better. 
        # What if we use a "blow-up" of a smaller graph?
        # Or simply a random regular graph? Random regular graphs have more uniform degrees, which might help.
        
        # Let's try generating a random d-regular graph (or near-regular) with 401 edges.
        # 401 edges / 40 vertices = avg degree 20.05.
        # So we want a 20-regular graph plus one extra edge.
        # Random 20-regular graphs might have lower max book than random 401-edge graphs because degrees are balanced.
        
        # How to generate random 20-regular graph on 40 vertices?
        # Configuration model.
        
        # 1. Create list of 40*20 = 800 stubs.
        # 2. Randomly pair them up.
        # 3. Resolve multi-edges and self-loops by re-pairing.
        # 4. Add one extra edge.
        
        def generate_random_regular(n, d):
            # n vertices, degree d
            # d must be even or n*d must be even
            if n * d % 2 != 0:
                return None
            
            stubs = [i for i in range(n) for _ in range(d)]
            random.shuffle(stubs)
            
            edges = set()
            attempts = 0
            while attempts < 100:
                edges = set()
                valid = True
                for i in range(0, len(stubs), 2):
                    u = stubs[i]
                    v = stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    edge = (min(u, v), max(u, v))
                    if edge in edges:
                        valid = False
                        break
                    edges.add(edge)
                
                if valid:
                    return list(edges)
                
                # Re-shuffle if invalid
                random.shuffle(stubs)
                attempts += 1
            
            return None

        # Target degree 20 (400 edges) + 1 edge = 401 edges.
        # Let's try to generate a 20-regular graph, then add a random non-edge.
        
        base_edges = generate_random_regular(n, 20)
        if base_edges is None:
            # Fallback to random sample if regular graph generation fails
            base_edges = random.sample(all_edges, m)
        else:
            # base_edges has 400 edges. Need 1 more.
            existing = set(base_edges)
            non_edges = [e for e in all_edges if e not in existing]
            if non_edges:
                extra_edge = random.choice(non_edges)
                base_edges.append(extra_edge)
            else:
                # Should not happen
                base_edges = random.sample(all_edges, m)
        
        # Now we have a candidate graph. Let's evaluate it.
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        max_book = 0
        for u, v in base_edges:
            common = len(adj[u] & adj[v])
            if common > max_book:
                max_book = common
        
        if max_book < best_score:
            best_score = max_book
            best_edges = base_edges
            
            # Optional: Quick local search to improve
            # If we have time, try swapping edges to reduce max_book further
            # But for now, just collect the best regular-ish graph.
            
            # Let's add a few local search steps if we found a good one
            if time.time() < deadline - 20:
                improved = True
                current_edges = list(best_edges)
                current_adj = [set(s) for s in adj]
                current_max = max_book
                
                # Simple local search: try to swap endpoints of the worst edge
                for _ in range(100): # Max 100 swaps per restart
                    if time.time() > deadline:
                        break
                    
                    # Find worst edge
                    worst_edge = None
                    worst_book = -1
                    for u, v in current_edges:
                        common = len(current_adj[u] & current_adj[v])
                        if common > worst_book:
                            worst_book = common
                            worst_edge = (u, v)
                    
                    if worst_edge is None:
                        break
                        
                    u, v = worst_edge
                    # Try to move v to a random non-neighbor of u
                    # Or swap v with a random vertex w not connected to u, and adjust v's connections?
                    # Simpler: Remove (u,v), add (u,w) where w is random.
                    # This changes degrees.
                    
                    # Let's try a 2-opt style move:
                    # Pick another edge (x,y). Swap to (u,x) and (v,y) or (u,y) and (v,x).
                    # This preserves degrees.
                    
                    other_edges = [e for e in current_edges if e != (u,v) and e != (v,u)]
                    if not other_edges:
                        break
                    
                    x, y = random.choice(other_edges)
                    
                    # Option 1: (u,v) -> (u,x), (x,y) -> (v,y)
                    # Check validity: u!=x, v!=y, no self loops, no duplicate edges
                    if u != x and v != y and (u, x) not in current_edges and (v, y) not in current_edges and (u,x) != (x,u) and (v,y) != (y,v):
                         # Actually need to check if (u,x) exists in the set properly
                         pass # This logic is getting complex and slow in Python loops.
                    
                    # Given the 150s limit and Python speed, simple random restarts of regular graphs might be safer.
                    # The regularity constraint alone might drop the score from 10-12 to 8-9.
        
    if best_edges is None:
        # Fallback
        best_edges = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return [[u, v] for u, v in best_edges]
