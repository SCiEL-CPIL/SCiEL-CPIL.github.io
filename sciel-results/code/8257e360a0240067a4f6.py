import random
import time

N = 40
M = 401

def _build_adj_and_books(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # book[u][v] = number of common neighbors of u and v (for u<v)
    book = {}
    for u in range(N):
        for v in adj[u]:
            if u < v:
                book[(u, v)] = len(adj[u] & adj[v])
    return adj, book

def _score(book):
    if not book:
        return 0
    return max(book.values())

def _remove_edge(edge, adj, book):
    u, v = edge
    adj[u].discard(v)
    adj[v].discard(u)
    book.pop((u, v), None)
    # recompute book values for edges incident to u or v
    for w in list(adj[u]):
        for x in (u, w):
            a, b = (x, w) if x < w else (w, x)
            if (a, b) != (u, v):
                book[(a, b)] = len(adj[a] & adj[b])
    for w in list(adj[v]):
        for x in (v, w):
            a, b = (x, w) if x < w else (w, x)
            if (a, b) != (u, v):
                book[(a, b)] = len(adj[a] & adj[b])
    # edges between u's and v's neighbors might change? No, removing (u,v)
    # only affects common-neighbor counts for pairs that include u or v.
    # But wait: for a pair (a,b) where neither is u nor v, the common
    # neighbors set is unchanged because u and v were not in adj[a]∩adj[b]
    # unless u or v was adjacent to both. Actually u and v could be common
    # neighbors of (a,b). So we need to recompute for all pairs where
    # u or v is a common neighbor. That's more expensive. For simplicity
    # and correctness, recompute all books incident to any vertex whose
    # neighbor set changed: only u and v changed. So recompute book[a][b]
    # for all a<b where u in adj[a]∩adj[b] or v in adj[a]∩adj[b].
    # To keep it simple and fast, recompute book for all pairs involving u or v
    # (already done above) PLUS all pairs (a,b) where both u and v are common neighbors? 
    # Actually removing (u,v) means u is no longer adjacent to v and v not to u.
    # For pair (a,b), common neighbors = adj[a] ∩ adj[b]. This set loses u iff 
    # u was in adj[a] and adj[b]. It loses v iff v was in adj[a] and adj[b].
    # So we need to recompute book[a][b] for all a<b such that 
    # (u in adj[a] and u in adj[b]) or (v in adj[a] and v in adj[b]).
    # This is still many pairs. For a 40-vertex graph, at most ~800 pairs.
    # Let's just recompute all books that could be affected: pairs where 
    # u or v is a common neighbor.
    affected = set()
    for a in range(N):
        if u in adj[a]:
            for b in range(a+1, N):
                if b != a and u in adj[b]:
                    affected.add((a, b) if a < b else (b, a))
    for a in range(N):
        if v in adj[a]:
            for b in range(a+1, N):
                if b != a and v in adj[b]:
                    affected.add((a, b) if a < b else (b, a))
    # Remove the edge itself from affected
    affected.discard((u, v) if u < v else (v, u))
    for a, b in affected:
        book[(a, b)] = len(adj[a] & adj[b])

def _add_edge(edge, adj, book):
    u, v = edge
    adj[u].add(v)
    adj[v].add(u)
    book[(u, v)] = len(adj[u] & adj[v])
    # recompute books for pairs where u or v is now a common neighbor
    affected = set()
    for a in range(N):
        if u in adj[a]:
            for b in range(a+1, N):
                if b != a and u in adj[b]:
                    affected.add((a, b) if a < b else (b, a))
    for a in range(N):
        if v in adj[a]:
            for b in range(a+1, N):
                if b != a and v in adj[b]:
                    affected.add((a, b) if a < b else (b, a))
    affected.discard((u, v) if u < v else (v, u))
    for a, b in affected:
        book[(a, b)] = len(adj[a] & adj[b])

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_edges = None
    best_score = float('inf')
    
    # Start with a random graph close to Turán
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Create a near-extremal graph: start with K20,20 + 1 edge, then randomly
        # swap some edges to diversify
        edges = []
        edge_set = set()
        # Base: complete bipartite K20,20
        for a in range(20):
            for b in range(20, 40):
                edges.append([a, b])
                edge_set.add((a, b))
        # Add one intra-part edge
        edges.append([0, 1])
        edge_set.add((0, 1))
        
        # Now do random edge swaps to diversify: remove a bipartite edge, add a random non-edge
        for _ in range(30):
            # Pick a random bipartite edge to remove
            bip_edges = [(a, b) for a in range(20) for b in range(20, 40) if (a, b) in edge_set]
            if not bip_edges:
                break
            rem_u, rem_v = random.choice(bip_edges)
            edge_set.discard((rem_u, rem_v))
            edges = [e for e in edges if not (e[0] == rem_u and e[1] == rem_v) and not (e[0] == rem_v and e[1] == rem_u)]
            
            # Add a random non-edge
            all_pairs = [(i, j) for i in range(N) for j in range(i+1, N) if (i, j) not in edge_set]
            if not all_pairs:
                break
            add_u, add_v = random.choice(all_pairs)
            edge_set.add((add_u, add_v))
            edges.append([add_u, add_v])
        
        if len(edges) != M:
            continue
        
        adj, book = _build_adj_and_books(edges)
        cur_score = _score(book)
        
        if cur_score < best_score:
            best_score = cur_score
            best_edges = [list(e) for e in edges]
        
        # Local search: iteratively remove the edge with the highest book value
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find all edges sorted by book value descending
            edge_list = [(k, v) for k, v in book.items()]
            edge_list.sort(key=lambda x: -x[1])
            
            for (eu, ev), val in edge_list:
                if time.time() > deadline:
                    break
                if val <= cur_score - 1:
                    break  # removing won't improve max book
                
                # Try removing this edge
                old_adj = [set(s) for s in adj]
                old_book = dict(book)
                old_edges_len = len(edges)
                
                _remove_edge((eu, ev), adj, book)
                edges.remove([eu, ev]) if [eu, ev] in edges else edges.remove([ev, eu])
                
                new_score = _score(book)
                
                if new_score < cur_score:
                    cur_score = new_score
                    improved = True
                    if cur_score < best_score:
                        best_score = cur_score
                        best_edges = [list(e) for e in edges]
                    # Keep the removal
                else:
                    # Revert
                    adj = old_adj
                    book = old_book
                    edges.append([eu, ev])
        
        if best_score <= 4:
            break
    
    return best_edges
