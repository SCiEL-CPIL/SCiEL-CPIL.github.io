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
    
    def generate_random_regular(n, d):
        """Generate a random d-regular graph using the configuration model."""
        if n * d % 2 != 0:
            return None
        
        # Use a more robust method: generate stubs, shuffle, pair, and resolve conflicts
        # We want a simple graph (no loops, no multi-edges).
        
        # Configuration model approach
        stubs = [i for i in range(n) for _ in range(d)]
        
        for _ in range(1000): # Try up to 1000 times to get a simple graph
            random.shuffle(stubs)
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
        
        return None

    def compute_books(adj, edges):
        """Compute book size for each edge."""
        books = {}
        max_book = 0
        for u, v in edges:
            common = len(adj[u] & adj[v])
            books[(u, v)] = common
            books[(v, u)] = common
            if common > max_book:
                max_book = common
        return books, max_book

    # Run multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        random.seed(restart * 100 + 7)
        
        # 1. Generate a random 20-regular graph (400 edges)
        base_edges = generate_random_regular(n, 20)
        if base_edges is None:
            # Fallback to random sample if regular graph generation fails repeatedly
            base_edges = random.sample(all_edges, m)
        else:
            # base_edges has 400 edges. Need 1 more.
            existing = set(base_edges)
            non_edges = [e for e in all_edges if e not in existing]
            if non_edges:
                # Choose the extra edge to minimize the immediate book increase
                # We want to pick (u, v) such that |N(u) & N(v)| is minimized
                # Approximate: pick a random non-edge, or try a few
                best_extra = None
                min_common = float('inf')
                # Sample a few non-edges to find a good one
                sample_size = min(20, len(non_edges))
                candidates = random.sample(non_edges, sample_size)
                
                # Build adj for the base graph to check common neighbors
                adj_temp = [set() for _ in range(n)]
                for u, v in base_edges:
                    adj_temp[u].add(v)
                    adj_temp[v].add(u)
                    
                for u, v in candidates:
                    common = len(adj_temp[u] & adj_temp[v])
                    if common < min_common:
                        min_common = common
                        best_extra = (u, v)
                
                if best_extra:
                    base_edges.append(best_extra)
                else:
                    base_edges.append(random.choice(non_edges))
            else:
                base_edges = random.sample(all_edges, m)
        
        # 2. Evaluate the initial graph
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        books, max_book = compute_books(adj, base_edges)
        
        if max_book < best_score:
            best_score = max_book
            best_edges = list(base_edges)
            
            # 3. Local Search to improve
            # We want to reduce the max book.
            # Move: Swap an endpoint of a high-book edge with a vertex that has low common neighbors.
            # Or: 2-opt swap: (u,v) and (x,y) -> (u,x) and (v,y).
            
            current_edges = set(base_edges)
            current_adj = [set(s) for s in adj]
            current_books, current_max = compute_books(current_adj, current_edges)
            
            # Precompute degrees for quick checks
            deg = [len(s) for s in current_adj]
            
            # Local search loop
            # Time limit for local search: 1.5s per restart
            ls_deadline = time.time() + 1.5
            no_improve_count = 0
            max_no_improve = 50
            
            while time.time() < ls_deadline and no_improve_count < max_no_improve:
                # Find the edge with the maximum book size
                worst_edge = None
                worst_book_val = -1
                for e in current_edges:
                    b = current_books[e]
                    if b > worst_book_val:
                        worst_book_val = b
                        worst_edge = e
                
                if worst_edge is None:
                    break
                
                u, v = worst_edge
                # Try to swap v with a random non-neighbor w of u
                # This changes edge (u, v) to (u, w).
                # We must ensure (u, w) is not already an edge.
                # Also, we need to update the books for all edges incident to u and v.
                
                # To be efficient, we only try a few random swaps per iteration.
                found_swap = False
                for _ in range(10):
                    if time.time() > ls_deadline:
                        break
                    # Pick a random vertex w
                    w = random.randint(0, n - 1)
                    if w == u or w == v:
                        continue
                    if w in current_adj[u]:
                        continue # (u, w) is already an edge
                    
                    # Proposed move: Remove (u, v), Add (u, w)
                    # This changes degrees: deg[u] stays same, deg[v] -1, deg[w] +1.
                    # This breaks regularity, which might be bad in the long run, 
                    # but we are minimizing max book, not regularity.
                    # However, breaking regularity might lead to higher books later.
                    # A better move: Swap (u, v) with (x, y) such that degrees are preserved.
                    # But that's complex. Let's stick to endpoint moves for simplicity and speed.
                    
                    # Calculate change in book sizes
                    # Edges affected:
                    # 1. (u, v) removed.
                    # 2. (u, w) added.
                    # 3. Edges (u, z) for z in N(u) \ {v, w}: book size changes?
                    #    Adding (u, w) creates triangles (u, w, z) for z in N(w) & N(u).
                    #    Removing (u, v) destroys triangles (u, v, z) for z in N(v) & N(u).
                    
                    # This is getting complicated to update incrementally correctly without errors.
                    # Given the small n=40, we can afford to recompute books for the whole graph
                    # after a proposed move if the move looks promising.
                    
                    # Heuristic: Only accept if the new edge (u, w) has a book size < worst_book_val.
                    # And ideally, it reduces the number of edges with book size == worst_book_val.
                    
                    # Estimate book size of new edge (u, w):
                    # common_uw = len(N(u) & N(w))
                    # Note: N(u) currently includes v. N(w) does not include u.
                    # The actual book size in the new graph will be slightly different because v is no longer connected to u, 
                    # but v might be connected to w.
                    # Let's just compute the exact book size of (u, w) in the *current* graph as a proxy.
                    common_uw = len(current_adj[u] & current_adj[w])
                    
                    if common_uw < worst_book_val:
                        # Perform the swap
                        # Remove (u, v)
                        current_adj[u].remove(v)
                        current_adj[v].remove(u)
                        current_edges.remove((u, v))
                        current_edges.remove((v, u))
                        
                        # Add (u, w)
                        current_adj[u].add(w)
                        current_adj[w].add(u)
                        current_edges.add((u, w))
                        current_edges.add((w, u))
                        
                        # Recompute books (O(m * n) which is 401 * 40 = 16000 ops, very fast)
                        current_books, current_max = compute_books(current_adj, current_edges)
                        
                        if current_max < worst_book_val:
                            worst_book_val = current_max
                            no_improve_count = 0
                            found_swap = True
                            break
                        else:
                            # Revert if it didn't improve the max
                            current_adj[u].remove(w)
                            current_adj[w].remove(u)
                            current_edges.remove((u, w))
                            current_edges.remove((w, u))
                            
                            current_adj[u].add(v)
                            current_adj[v].add(u)
                            current_edges.add((u, v))
                            current_edges.add((v, u))
                            
                            # Recompute books to be safe
                            current_books, current_max = compute_books(current_adj, current_edges)
                            no_improve_count += 1
                            continue
                if not found_swap:
                    no_improve_count += 1
                
                # Update best if we improved
                if current_max < best_score:
                    best_score = current_max
                    best_edges = list(current_edges)
                    
    if best_edges is None:
        # Fallback: K_{20,20} + 1 edge
        best_edges = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return [[u, v] for u, v in best_edges]
