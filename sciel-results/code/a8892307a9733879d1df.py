import random
import time
import numpy as np

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    n = 40
    target_edges = 401
    
    def edge_key(u, v):
        return (u, v) if u < v else (v, u)
    
    def build_adj(edges_set):
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def compute_max_book(adj, edges_set):
        max_b = 0
        for u, v in edges_set:
            b = len(adj[u] & adj[v])
            if b > max_b:
                max_b = b
        return max_b
    
    def compute_all_books(adj, edges_set):
        books = {}
        for u, v in edges_set:
            books[(u, v) if u < v else (v, u)] = len(adj[u] & adj[v])
        return books
    
    best_edges = None
    best_max = 999
    
    # Strategy: Start from K_{20,20} (400 edges), remove some bipartite edges,
    # add intra-partition edges to reach 401 total, then optimize via local search.
    
    # First, try several structured constructions
    for num_intra in range(1, 20):
        num_remove = num_intra - 1
        
        # Build base K_{20,20}
        base = set()
        for a in range(20):
            for b in range(20, 40):
                base.add((a, b))
        
        all_bip = list(base)
        random.seed(42 + num_intra)
        random.shuffle(all_bip)
        remove_set = set(all_bip[:num_remove])
        
        # Choose intra-edges: use matchings within A and B
        a_vertices = list(range(20))
        b_vertices = list(range(20, 40))
        random.shuffle(a_vertices)
        random.shuffle(b_vertices)
        
        half = num_intra // 2
        rem = num_intra - half
        
        intra_edges = []
        for i in range(half):
            u = a_vertices[2*i]
            v = a_vertices[2*i+1]
            intra_edges.append((u, v))
        
        for i in range(rem):
            u = b_vertices[2*i]
            v = b_vertices[2*i+1]
            intra_edges.append((u, v))
        
        edges_set = set(base)
        edges_set.difference_update(remove_set)
        for e in intra_edges:
            edges_set.add(e)
        
        if len(edges_set) != target_edges:
            continue
        
        adj = build_adj(edges_set)
        cur_max = compute_max_book(adj, edges_set)
        
        if cur_max < best_max:
            best_max = cur_max
            best_edges = list(edges_set)
        
        if best_max <= 7:
            break
    
    # If we haven't found a good solution, try a different approach:
    # Randomized construction with optimization
    if best_max > 7:
        for trial in range(50):
            if time.time() > deadline - 20:
                break
            
            # Start with random graph close to target density
            edges_set = set()
            # Ensure we start with a reasonable structure
            # Mix of bipartite and some intra edges
            
            # Random bipartite-ish structure
            bip_edges = set()
            for a in range(20):
                for b in range(20, 40):
                    if random.random() < 0.95:
                        bip_edges.add((a, b))
            
            # Add some intra edges
            intra_count = target_edges - len(bip_edges)
            intra_candidates = set()
            for i in range(20):
                for j in range(i+1, 20):
                    intra_candidates.add((i, j))
            for i in range(20, 40):
                for j in range(i+1, 40):
                    intra_candidates.add((i, j))
            
            intra_list = list(intra_candidates)
            random.shuffle(intra_list)
            for i in range(min(intra_count, len(intra_list))):
                bip_edges.add(intra_list[i])
            
            # Trim or add to reach exactly 401
            if len(bip_edges) > target_edges:
                bip_list = list(bip_edges)
                random.shuffle(bip_list)
                for e in bip_list[:len(bip_edges) - target_edges]:
                    bip_edges.discard(e)
            
            while len(bip_edges) < target_edges:
                # Add random missing edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                k = edge_key(u, v)
                if k not in bip_edges:
                    bip_edges.add(k)
            
            edges_set = bip_edges
            adj = build_adj(edges_set)
            cur_max = compute_max_book(adj, edges_set)
            
            if cur_max < best_max:
                best_max = cur_max
                best_edges = list(edges_set)
            
            if best_max <= 7:
                break
    
    # Local search optimization on the best solution
    if best_edges and time.time() < deadline - 5:
        edges_set = set(tuple(e) for e in best_edges)
        adj = build_adj(edges_set)
        cur_max = compute_max_book(adj, edges_set)
        
        # Precompute books for efficiency
        books = compute_all_books(adj, edges_set)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Find edges with high book
            sorted_edges = sorted(books.items(), key=lambda x: -x[1])
            
            for edge, book_val in sorted_edges:
                if book_val < cur_max:
                    break
                
                if time.time() > deadline - 5:
                    break
                
                u, v = edge
                
                # Try to find a swap that reduces max book
                for _ in range(200):
                    if time.time() > deadline - 5:
                        break
                    
                    a = random.randint(0, n-1)
                    b = random.randint(0, n-1)
                    if a == b:
                        continue
                    ak = edge_key(a, b)
                    if ak in edges_set or ak == edge:
                        continue
                    
                    # Test swap: remove (u,v), add (a,b)
                    adj[u].discard(v)
                    adj[v].discard(u)
                    edges_set.discard(edge)
                    
                    adj[a].add(b)
                    adj[b].add(a)
                    edges_set.add(ak)
                    
                    # Recompute affected books
                    # Books affected: all edges incident to u, v, a, b
                    affected = set()
                    for x in adj[u]:
                        k = edge_key(u, x)
                        if k in edges_set:
                            affected.add(k)
                    for x in adj[v]:
                        k = edge_key(v, x)
                        if k in edges_set:
                            affected.add(k)
                    for x in adj[a]:
                        k = edge_key(a, x)
                        if k in edges_set:
                            affected.add(k)
                    for x in adj[b]:
                        k = edge_key(b, x)
                        if k in edges_set:
                            affected.add(k)
                    affected.add(ak)
                    
                    new_max = 0
                    for k in affected:
                        x, y = k
                        b_val = len(adj[x] & adj[y])
                        books[k] = b_val
                        if b_val > new_max:
                            new_max = b_val
                    
                    # Also check non-affected books (they don't change, so max is max(new_max, old_max_of_others))
                    # But we need the global max
                    # Quick check: if new_max >= cur_max, this swap doesn't help
                    if new_max < cur_max:
                        cur_max = new_max
                        improved = True
                        break
                    else:
                        # Revert
                        adj[a].discard(b)
                        adj[b].discard(a)
                        edges_set.discard(ak)
                        
                        adj[u].add(v)
                        adj[v].add(u)
                        edges_set.add(edge)
                        # Restore books
                        books = compute_all_books(adj, edges_set)
                        break
        
        if cur_max < best_max:
            best_max = cur_max
            best_edges = [list(e) for e in edges_set]
    
    # Second round of local search with 2-opt moves
    if best_edges and time.time() < deadline - 10:
        edges_set = set(tuple(e) for e in best_edges)
        adj = build_adj(edges_set)
        cur_max = compute_max_book(adj, edges_set)
        
        for _iter in range(3000):
            if time.time() > deadline - 10:
                break
            
            # Find max book edge
            max_b = 0
            max_edge = None
            for u, v in edges_set:
                b = len(adj[u] & adj[v])
                if b > max_b:
                    max_b = b
                    max_edge = (u, v)
            
            if max_edge is None or max_b < 2:
                break
            
            u, v = max_edge
            uk = edge_key(u, v)
            
            # Try 1-opt: swap one edge
            improved = False
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if a == b:
                    continue
                ak = edge_key(a, b)
                if ak in edges_set or ak == uk:
                    continue
                
                adj[u].discard(v)
                adj[v].discard(u)
                edges_set.discard(uk)
                
                adj[a].add(b)
                adj[b].add(a)
                edges_set.add(ak)
                
                new_max = compute_max_book(adj, edges_set)
                
                if new_max < cur_max:
                    cur_max = new_max
                    if cur_max < best_max:
                        best_max = cur_max
                        best_edges = [list(e) for e in edges_set]
                    improved = True
                    break
                else:
                    adj[a].discard(b)
                    adj[b].discard(a)
                    edges_set.discard(ak)
                    
                    adj[u].add(v)
                    adj[v].add(u)
                    edges_set.add(uk)
            
            if not improved:
                # Try 2-opt: remove two edges, add two edges
                improved_2 = False
                for _ in range(20):
                    if time.time() > deadline - 10:
                        break
                    
                    # Pick two edges to remove
                    e1_candidates = list(edges_set)
                    random.shuffle(e1_candidates)
                    if len(e1_candidates) < 2:
                        break
                    e1 = e1_candidates[0]
                    e1_key = edge_key(*e1)
                    e1_candidates.remove(e1)
                    e2 = e1_candidates[0]
                    e2_key = edge_key(*e2)
                    
                    # Pick two edges to add
                    new_e1 = None
                    new_e2 = None
                    for _t in range(50):
                        a1 = random.randint(0, n-1)
                        b1 = random.randint(0, n-1)
                        if a1 == b1:
                            continue
                        k1 = edge_key(a1, b1)
                        if k1 not in edges_set and k1 != e1_key and k1 != e2_key:
                            new_e1 = k1
                            break
                    
                    if new_e1 is None:
                        continue
                    
                    for _t in range(50):
                        a2 = random.randint(0, n-1)
                        b2 = random.randint(0, n-1)
                        if a2 == b2:
                            continue
                        k2 = edge_key(a2, b2)
                        if k2 not in edges_set and k2 != e1_key and k2 != e2_key and k2 != new_e1:
                            new_e2 = k2
                            break
                    
                    if new_e2 is None:
                        continue
                    
                    # Apply
                    for ek in [e1_key, e2_key]:
                        x, y = ek
                        adj[x].discard(y)
                        adj[y].discard(x)
                        edges_set.discard(ek)
                    
                    for ek in [new_e1, new_e2]:
                        x, y = ek
                        adj[x].add(y)
                        adj[y].add(x)
                        edges_set.add(ek)
                    
                    new_max = compute_max_book(adj, edges_set)
                    
                    if new_max < cur_max:
                        cur_max = new_max
                        if cur_max < best_max:
                            best_max = cur_max
                            best_edges = [list(e) for e in edges_set]
                        improved_2 = True
                        break
                    else:
                        # Revert
                        for ek in [new_e1, new_e2]:
                            x, y = ek
                            adj[x].discard(y)
                            adj[y].discard(x)
                            edges_set.discard(ek)
                        
                        for ek in [e1_key, e2_key]:
                            x, y = ek
                            adj[x].add(y)
                            adj[y].add(x)
                            edges_set.add(ek)
                
                if not improved_2:
                    break
    
    # Ensure we return exactly 401 edges
    if best_edges is None:
        # Fallback: just return K_{20,20} plus one edge
        best_edges = []
        for a in range(20):
            for b in range(20, 40):
                best_edges.append([a, b])
        best_edges.append([0, 1])
    
    return best_edges
