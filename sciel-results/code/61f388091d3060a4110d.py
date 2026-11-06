import time
import random

def construct():
    start = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    n = 40
    m = 401
    
    def remaining():
        return DEADLINE - time.time()
    
    # ---- Representation ----
    # adj[u] is a bitmask of neighbors
    # edges stored as list of (u,v) with u<v
    # edge_set for O(1) membership
    
    def build_adj(edges):
        adj = [0]*n
        for u,v in edges:
            adj[u] |= (1<<v)
            adj[v] |= (1<<u)
        return adj
    
    def book_of(adj, u, v):
        return (adj[u] & adj[v]).bit_count()
    
    def max_and_sum(adj, edges):
        mx = 0
        s = 0
        for u,v in edges:
            b = (adj[u] & adj[v]).bit_count()
            if b > mx: mx = b
            s += b
        return mx, s
    
    # ---- Initial graph ----
    # Try a few different initial structures and pick the best, or just random.
    random.seed(20240501)
    
    all_edges = [(i,j) for i in range(n) for j in range(i+1,n)]
    
    # Initial: random
    random.shuffle(all_edges)
    edges = all_edges[:m]
    
    adj = build_adj(edges)
    edge_set = set(edges)
    mx, sm = max_and_sum(adj, edges)
    
    best_edges = list(edges)
    best_mx = mx
    best_sm = sm
    
    # Precompute list of all non-edges for random add
    # (recompute when edge_set changes? expensive. Instead, sample random and check membership)
    
    # ---- Local search ----
    # Move: remove a random edge, add a random non-edge. Accept if improves (mx, sm) or via SA.
    
    # To make it faster, we'll track mx and sm and update incrementally is complex;
    # instead recompute mx,sm each move (401 bit_counts, fast).
    
    T = 2.0  # SA temperature
    T_min = 0.05
    cooling = 0.9999
    
    # Precompute a list of all possible edges for sampling
    all_edges_list = [(i,j) for i in range(n) for j in range(i+1,n)]
    total_possible = len(all_edges_list)
    
    it = 0
    no_improve = 0
    while remaining() > 5.0:
        it += 1
        # pick edge to remove
        u,v = random.choice(edges)
        # pick non-edge to add
        # sample until find a non-edge
        for _ in range(30):
            a,b = random.choice(all_edges_list)
            if (a,b) not in edge_set:
                break
        else:
            continue
        if (a,b) == (u,v):
            continue
        
        # apply
        adj[u] &= ~(1<<v)
        adj[v] &= ~(1<<u)
        edge_set.discard((u,v))
        adj[a] |= (1<<b)
        adj[b] |= (1<<a)
        edge_set.add((a,b))
        
        # update edges list
        edges[edges.index((u,v))] = (a,b)  # O(n) but fine; actually edges.index is O(401)
        
        new_mx, new_sm = max_and_sum(adj, edges)
        
        # acceptance
        old_key = (mx, sm)
        new_key = (new_mx, new_sm)
        if new_key <= old_key:
            mx, sm = new_key
            no_improve = 0
            if new_key < (best_mx, best_sm):
                best_mx, best_sm = new_key
                best_edges = list(edges)
        else:
            # SA accept?
            delta = new_key[0] - old_key[0] + 0.01*(new_key[1]-old_key[1])
            if random.random() < 2.718281828 ** (-delta / max(T,1e-9)):
                mx, sm = new_key
            else:
                # revert
                adj[u] |= (1<<v)
                adj[v] |= (1<<u)
                edge_set.add((u,v))
                adj[a] &= ~(1<<b)
                adj[b] &= ~(1<<a)
                edge_set.discard((a,b))
                edges[edges.index((a,b))] = (u,v)
        no_improve += 1
        if no_improve > 200000:
            # reheat
            T = max(T, 1.0)
            no_improve = 0
        T = max(T_min, T * cooling)
    
    # Final targeted phase: reduce max book
    ...
    
    return [[u,v] for u,v in best_edges]
