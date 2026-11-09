import numpy as np
import time
import random
import math

def construct():
    n = 40
    m = 401
    start_time = time.time()
    timeout = 145
    
    adj = np.zeros((n, n), dtype=np.int8)
    books = np.zeros((n, n), dtype=np.int32)
    edges = []
    
    def add_edge(u, v):
        if adj[u, v]:
            return
        adj[u, v] = 1
        adj[v, u] = 1
        edges.append((u, v))
        
        # Update books
        # book[u][v]
        common = np.dot(adj[u], adj[v])
        books[u, v] = common
        books[v, u] = common
        
        # Update neighbors
        # N(u) and N(v)
        # We need to update books[u][w] for w in N(u) \ {v}
        # books[u][w] += 1 if adj[v][w] == 1
        
        # Vectorized update is tricky with loops, but n=40 is small.
        # Loop is fine.
        for w in range(n):
            if w == v:
                continue
            if adj[u, w] and adj[v, w]:
                books[u, w] += 1
                books[w, u] += 1
            if adj[v, w] and adj[u, w]:
                books[v, w] += 1
                books[w, v] += 1
        # Note: The condition adj[u,w] and adj[v,w] is the same for both blocks.
        # So we can combine.
        # Actually, if adj[u,w] and adj[v,w] is true, we update both.
        # My previous code did:
        # if adj[u,w]: if adj[v,w]: update u,w
        # if adj[v,w]: if adj[u,w]: update v,w
        # This is redundant.
        # Let's fix add_edge logic.

    def add_edge_optimized(u, v):
        if adj[u, v]:
            return
        adj[u, v] = 1
        adj[v, u] = 1
        edges.append((u, v))
        
        common = np.dot(adj[u], adj[v])
        books[u, v] = common
        books[v, u] = common
        
        # For all w
        # If adj[u,w] and adj[v,w]:
        #   books[u,w] += 1
        #   books[v,w] += 1
        # Wait.
        # Adding (u,v).
        # v is added to N(u).
        # For any w in N(u):
        #   If v in N(w) (adj[v,w]==1), then v is a common neighbor of u and w.
        #   So books[u,w] increases by 1.
        # Similarly, u is added to N(v).
        # For any w in N(v):
        #   If u in N(w) (adj[u,w]==1), then u is a common neighbor of v and w.
        #   So books[v,w] increases by 1.
        
        # So:
        for w in range(n):
            if w == v:
                # w is v, skip for u-w update?
                # For u-v, we already set books[u,v].
                # Do we need to update books[u,v] via neighbor logic?
                # books[u,v] = |N(u) \cap N(v)|.
                # This was calculated by dot product.
                # The neighbor loop updates books[u,w] for w != v.
                continue
            if w == u:
                continue
            
            if adj[u, w] and adj[v, w]:
                books[u, w] += 1
                books[w, u] += 1
                
            if adj[v, w] and adj[u, w]:
                books[v, w] += 1
                books[w, v] += 1
                
        # This updates books[u,w] and books[v,w] correctly.

    def remove_edge_optimized(u, v):
        if not adj[u, v]:
            return
        adj[u, v] = 0
        adj[v, u] = 0
        edges.remove((u, v))
        
        books[u, v] = 0
        books[v, u] = 0
        
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u, w] and adj[v, w]:
                books[u, w] -= 1
                books[w, u] -= 1
            if adj[v, w] and adj[u, w]:
                books[v, w] -= 1
                books[w, v] -= 1

    # Init
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible)
    for u, v in all_possible[:m]:
        add_edge_optimized(u, v)
        
    def get_metrics():
        # Extract active books
        # adj is 0/1
        active_books = books[adj == 1]
        # Symmetric, so we double count?
        # books is symmetric. adj is symmetric.
        # books[adj==1] will contain both books[u,v] and books[v,u].
        # They are equal.
        # So max is correct.
        # sum_sq: sum(b^2). Since each edge is counted twice in the matrix extraction?
        # Yes, (u,v) and (v,u).
        # So sum_sq will be 2 * actual_sum_sq.
        # This is fine for comparison, as long as consistent.
        max_b = np.max(active_books) if len(active_books) > 0 else 0
        sum_sq = np.sum(active_books * active_books)
        return max_b, sum_sq

    best_max, best_sum_sq = get_metrics()
    best_edges = edges.copy()
    best_adj = adj.copy()
    best_books = books.copy()
    
    T = 50.0
    cooling = 0.9995
    min_T = 0.1
    
    while time.time() - start_time < timeout:
        if not edges:
            break
        u, v = random.choice(edges)
        
        found = False
        for _ in range(50):
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x != y and not adj[x, y]:
                found = True
                break
        if not found:
            continue
            
        curr_max, curr_sum_sq = get_metrics()
        
        remove_edge_optimized(u, v)
        add_edge_optimized(x, y)
        
        new_max, new_sum_sq = get_metrics()
        
        accepted = False
        if new_max < curr_max:
            accepted = True
        elif new_max == curr_max and new_sum_sq < curr_sum_sq:
            accepted = True
        else:
            if new_max > curr_max:
                penalty = (new_max - curr_max) * 100 + (new_sum_sq - curr_sum_sq)
            else:
                penalty = new_sum_sq - curr_sum_sq
            
            if penalty < 0:
                accepted = True
            else:
                prob = math.exp(-penalty / T)
                if random.random() < prob:
                    accepted = True
        
        if accepted:
            if new_max < best_max or (new_max == best_max and new_sum_sq < best_sum_sq):
                best_max = new_max
                best_sum_sq = new_sum_sq
                best_edges = edges.copy()
                best_adj = adj.copy()
                best_books = books.copy()
        else:
            remove_edge_optimized(x, y)
            add_edge_optimized(u, v)
            
        T *= cooling
        if T < min_T:
            T = min_T

    return [[u, v] for u, v in best_edges]
