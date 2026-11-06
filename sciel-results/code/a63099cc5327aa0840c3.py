import time
import random
import numpy as np

def construct():
    N = 40
    M = 401
    deadline = time.time() + 115
    
    def compute_books_from_adj(adj):
        """Compute book sizes for all edges from adjacency matrix."""
        books = {}
        max_b = 0
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i][j]:
                    # Count common neighbors
                    b = 0
                    for k in range(N):
                        if k != i and k != j and adj[i][k] and adj[j][k]:
                            b += 1
                    books[(i, j)] = b
                    if b > max_b:
                        max_b = b
        return books, max_b
    
    def get_edges_from_adj(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i][j]:
                    edges.append([i, j])
        return edges
    
    def init_bipartite_plus_one():
        """Initialize with balanced bipartite + 1 extra edge."""
        adj = [[0]*N for _ in range(N)]
        # Partition into two sets of 20
        left = list(range(20))
        right = list(range(20, 40))
        # Add all bipartite edges
        for u in left:
            for v in right:
                adj[u][v] = 1
                adj[v][u] = 1
        # Add one extra edge within a partition
        adj[left[0]][left[1]] = 1
        adj[left[1]][left[0]] = 1
        return adj
    
    def init_random():
        """Initialize with a random graph with M edges."""
        all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(all_pairs)
        adj = [[0]*N for _ in range(N)]
        for u, v in all_pairs[:M]:
            adj[u][v] = 1
            adj[v][u] = 1
        return adj
    
    def init_near_bipartite():
        """Initialize with near-bipartite structure with some randomization."""
        adj = [[0]*N for _ in range(N)]
        left = list(range(20))
        right = list(range(20, 40))
        
        # Add most bipartite edges but randomly omit a few
        bipartite_edges = [(u, v) for u in left for v in right]
        random.shuffle(bipartite_edges)
        
        # Add 395 bipartite edges (leave 5 out)
        for u, v in bipartite_edges[:395]:
            adj[u][v] = 1
            adj[v][u] = 1
        
        # Add 6 intra-partition edges
        intra_left = [(u, v) for u in left for v in left if u < v]
        intra_right = [(u, v) for u in right for v in right if u < v]
        all_intra = intra_left + intra_right
        random.shuffle(all_intra)
        
        for u, v in all_intra[:6]:
            adj[u][v] = 1
            adj[v][u] = 1
        
        return adj
    
    def get_non_edges(adj):
        non_edges = []
        for i in range(N):
            for j in range(i+1, N):
                if not adj[i][j]:
                    non_edges.append((i, j))
        return non_edges
    
    def evaluate_swap_incremental(adj, books, edge_remove, edge_add):
        """
        Evaluate a swap incrementally.
        Returns new_max_book and whether it's an improvement.
        Only edges incident to the 4 vertices can change.
        """
        u, v = edge_remove
        x, y = edge_add
        
        # Vertices affected
        affected_verts = {u, v, x, y}
        
        # Edges that could change: all edges incident to any of these vertices
        # plus the edge being removed and added
        affected_edges = set()
        for vert in affected_verts:
            for other in range(N):
                if other != vert:
                    a, b = min(vert, other), max(vert, other)
                    if adj[a][b] or (a == min(u,v) and b == max(u,v)):
                        affected_edges.add((a, b))
        
        # Also include the edge being added
        affected_edges.add((min(x,y), max(x,y)))
        
        # Save old values for affected edges
        old_books = {}
        for e in affected_edges:
            if e in books:
                old_books[e] = books[e]
        
        # Apply swap
        adj[u][v] = 0
        adj[v][u] = 0
        adj[x][y] = 1
        adj[y][x] = 1
        
        # Recompute books for affected edges
        new_max = 0
        for e in affected_edges:
            a, b = e
            if adj[a][b]:
                # Count common neighbors
                cnt = 0
                for k in range(N):
                    if k != a and k != b and adj[a][k] and adj[b][k]:
                        cnt += 1
                books[e] = cnt
                if cnt > new_max:
                    new_max = cnt
            else:
                if e in books:
                    del books[e]
        
        # Find overall max
        overall_max = 0
        for e, b in books.items():
            if b > overall_max:
                overall_max = b
        
        # Revert
        adj[u][v] = 1
        adj[v][u] = 1
        adj[x][y] = 0
        adj[y][x] = 0
        
        # Restore old books
        for e in affected_edges:
            if e in old_books:
                books[e] = old_books[e]
            elif e in books:
                del books[e]
        
        return overall_max
    
    def local_search(adj, books, max_book, deadline):
        """Local search with incremental updates."""
        current_max = max_book
        
        while time.time() < deadline:
            if current_max <= 7:
                break
            
            # Find the edge with maximum book
            max_edge = None
            max_val = -1
            for e, b in books.items():
                if b > max_val:
                    max_val = b
                    max_edge = e
            
            if max_val <= current_max:
                current_max = max_val
            
            if max_val <= 7:
                break
            
            u, v = max_edge
            
            # Get non-edges
            non_edges = get_non_edges(adj)
            if not non_edges:
                break
            
            # Try swaps - sample candidates
            random.shuffle(non_edges)
            batch = non_edges[:min(100, len(non_edges))]
            
            best_candidate = None
            best_new_max = max_val
            
            for x, y in batch:
                if (x == u and y == v) or (x == v and y == u):
                    continue
                
                new_max = evaluate_swap_incremental(adj, books, (u, v), (x, y))
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_candidate = (x, y)
            
            if best_candidate is not None and best_new_max < max_val:
                # Apply the swap
                x, y = best_candidate
                adj[u][v] = 0
                adj[v][u] = 0
                adj[x][y] = 1
                adj[y][x] = 1
                
                # Update books incrementally
                affected_verts = {u, v, x, y}
                affected_edges = set()
                for vert in affected_verts:
                    for other in range(N):
                        if other != vert:
                            a, b = min(vert, other), max(vert, other)
                            if adj[a][b]:
                                affected_edges.add((a, b))
                
                # Remove old books for edges that no longer exist
                for e in list(books.keys()):
                    a, b = e
                    if not adj[a][b]:
                        del books[e]
                
                # Recompute affected edges
                for e in affected_edges:
                    a, b = e
                    cnt = 0
                    for k in range(N):
                        if k != a and k != b and adj[a][k] and adj[b][k]:
                            cnt += 1
                    books[e] = cnt
                
                current_max = max(books.values()) if books else 0
            else:
                # Stuck, try a random perturbation
                if random.random() < 0.3 and non_edges:
                    # Random swap
                    re = random.choice(list(books.keys()))
                    ae = random.choice(non_edges)
                    if ae[0] != re[0] or ae[1] != re[1]:
                        u2, v2 = re
                        x2, y2 = ae
                        adj[u2][v2] = 0
                        adj[v2][u2] = 0
                        adj[x2][y2] = 1
                        adj[y2][x2] = 1
                        
                        # Recompute all books (expensive but rare)
                        books, current_max = compute_books_from_adj(adj)
                else:
                    break
        
        return current_max
    
    best_adj = None
    best_score = float('inf')
    
    # Try different initialization strategies
    inits = [
        init_bipartite_plus_one,
        init_near_bipartite,
        init_random,
        init_bipartite_plus_one,
        init_near_bipartite,
    ]
    
    for init_fn in inits:
        if time.time() >= deadline:
            break
        
        adj = init_fn()
        
        # Verify edge count
        edge_count = sum(adj[i][j] for i in range(N) for j in range(i+1, N))
        if edge_count != M:
            # Fix edge count
            non_edges = get_non_edges(adj)
            while edge_count > M and non_edges:
                e = random.choice(non_edges)
                adj[e[0]][e[1]] = 0
                adj[e[1]][e[0]] = 0
                non_edges.remove(e)
                edge_count -= 1
            while edge_count < M:
                all_pairs = [(i,j) for i in range(N) for j in range(i+1,N) if not adj[i][j]]
                if not all_pairs:
                    break
                e = random.choice(all_pairs)
                adj[e[0]][e[1]] = 1
                adj[e[1]][e[0]] = 1
                edge_count += 1
        
        books, max_book = compute_books_from_adj(adj)
        
        if max_book < best_score:
            best_score = max_book
            best_adj = [row[:] for row in adj]
        
        # Local search with remaining time
        remaining = deadline - time.time()
        if remaining > 5:
            new_max = local_search(adj, books, max_book, deadline)
            if new_max < best_score:
                best_score = new_max
                best_adj = [row[:] for row in adj]
    
    # Final fallback: if we never found a valid graph, return bipartite + 1
    if best_adj is None:
        best_adj = init_bipartite_plus_one()
    
    return get_edges_from_adj(best_adj)
