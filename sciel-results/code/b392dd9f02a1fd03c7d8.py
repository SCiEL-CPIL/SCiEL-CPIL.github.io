import random
import time
import math
from collections import defaultdict

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    n = 40
    target_edges = 401
    
    def build_graph_from_edges(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def compute_books(adj, edges):
        """Compute the book size for each edge (number of triangles containing it)."""
        books = {}
        for u, v in edges:
            common = adj[u] & adj[v]
            books[(u, v) if u < v else (v, u)] = len(common)
        return books
    
    def max_book(books):
        return max(books.values()) if books else 0
    
    # Start with K20,20 plus one edge
    edges_set = set()
    for a in range(20):
        for b in range(20, 40):
            edges_set.add((a, b))
    edges_set.add((0, 1))
    
    edges_list = [list(e) for e in edges_set]
    adj = build_graph_from_edges(edges_list)
    books = compute_books(adj, edges_list)
    current_max = max_book(books)
    best_edges = edges_list[:]
    best_max = current_max
    
    # Precompute some helpers
    def edge_key(u, v):
        return (u, v) if u < v else (v, u)
    
    # Simulated annealing with edge swaps
    temp = 50.0
    cooling = 0.9995
    min_temp = 0.5
    
    while time.time() < deadline and temp > min_temp:
        # Pick a random edge to remove
        idx = random.randint(0, len(edges_list) - 1)
        u, v = edges_list[idx]
        uk = edge_key(u, v)
        
        # Pick a random non-edge to add
        # Sample a few random non-edges
        added = False
        for _ in range(20):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a == b:
                continue
            ak = edge_key(a, b)
            if ak in edges_set:
                continue
            # Check it's not the same edge
            if ak == uk:
                continue
            
            # Perform swap: remove (u,v), add (a,b)
            new_u, new_v = a, b
            
            # Update books efficiently
            # Remove edge (u,v): its book was books[uk]
            old_book_uv = books[uk]
            
            # For each common neighbor w of u and v, the triangles (u,v,w) are removed
            common_uv = adj[u] & adj[v]
            
            # Remove (u,v) from graph
            adj[u].discard(v)
            adj[v].discard(u)
            edges_set.discard(uk)
            del books[uk]
            
            # Add (a,b) to graph
            adj[a].add(b)
            adj[b].add(a)
            edges_set.add(ak)
            
            # Compute new book for (a,b)
            common_ab = adj[a] & adj[b]
            new_book_ab = len(common_ab)
            books[ak] = new_book_ab
            
            # Update books for edges affected by the change
            # Edges (u,w) for w in common_uv lose one triangle
            for w in common_uv:
                ukw = edge_key(u, w)
                if ukw in books:
                    books[ukw] -= 1
            for w in common_uv:
                vkw = edge_key(v, w)
                if vkw in books:
                    books[vkw] -= 1
            
            # Edges (a,w) for w in common_ab gain one triangle (if they exist)
            for w in common_ab:
                awk = edge_key(a, w)
                if awk in books:
                    books[awk] += 1
            for w in common_ab:
                bwk = edge_key(b, w)
                if bwk in books:
                    books[bwk] += 1
            
            # Also, edges from u or v to a or b might be affected
            # Actually, the books for edges incident to u, v, a, b need careful handling
            # Let me just recompute books for all edges incident to u, v, a, b
            affected = set()
            for w in adj[u]:
                affected.add(edge_key(u, w))
            for w in adj[v]:
                affected.add(edge_key(v, w))
            for w in adj[a]:
                affected.add(edge_key(a, w))
            for w in adj[b]:
                affected.add(edge_key(b, w))
            
            # Recompute books for affected edges
            for ek in affected:
                if ek in books:
                    x, y = ek
                    books[ek] = len(adj[x] & adj[y])
            
            # Update edges_list
            edges_list[idx] = [a, b]
            
            new_max = max_book(books)
            
            if new_max < best_max:
                best_max = new_max
                best_edges = edges_list[:]
            
            if new_max < current_max or random.random() < math.exp((current_max - new_max) / temp):
                current_max = new_max
            else:
                # Revert
                edges_list[idx] = [u, v]
                adj[u].add(v)
                adj[v].add(u)
                edges_set.add(uk)
                books[uk] = old_book_uv
                
                adj[a].discard(b)
                adj[b].discard(a)
                edges_set.discard(ak)
                del books[ak]
                
                # Recompute affected books again
                for ek in affected:
                    if ek in books:
                        x, y = ek
                        books[ek] = len(adj[x] & adj[y])
            
            added = True
            break
        
        temp *= cooling
    
    # Return the best edges found
    return best_edges
