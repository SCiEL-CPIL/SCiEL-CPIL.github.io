import time
import random
import math

N = 18
DEADLINE = 115.0  # Safety margin before 120s

def _build_adj(edges):
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_k4_free(adj):
    # K4-free iff for every vertex u, the subgraph induced on N(u) is triangle-free
    for u in range(N):
        nu = adj[u]
        v = nu
        while v:
            lb = v & (-v)
            i = lb.bit_length() - 1
            v -= lb
            ni_in_nu = adj[i] & nu
            # Check if ni_in_nu has any edge
            # i.e., is there j in ni_in_nu such that adj[j] & ni_in_nu != 0 (excluding j)
            w = ni_in_nu
            while w:
                lb2 = w & (-w)
                j = lb2.bit_length() - 1
                w -= lb2
                if adj[j] & ni_in_nu:
                    return False
    return True

def _max_triangle_free(adj):
    # Exact branch and bound for largest triangle-free induced subgraph
    full = (1 << N) - 1
    best = [0]
    
    # Precompute adjacency bitmasks
    # adj[i] is a bitmask of neighbors of i
    
    # Helper: check if adding v to current_mask keeps it triangle-free
    # This is true if (adj[v] & current_mask) is an independent set in the induced graph on current_mask?
    # No, it must be an independent set in the *original* graph G, because if x,y in adj[v] & current_mask are adjacent, 
    # then v-x, v-y, x-y form a triangle.
    # So we need: for all x in (adj[v] & current_mask), adj[x] & (adj[v] & current_mask) == 0.
    
    def is_compatible(v, mask):
        # Check if v can be added to mask without creating a triangle
        # v forms a triangle with x,y in mask if x-y is edge, x-v is edge, y-v is edge.
        # So we need no edge among neighbors of v in mask.
        nv_in_mask = adj[v] & mask
        # Check if nv_in_mask is independent
        nn = nv_in_mask
        while nn:
            lb = nn & (-nn)
            u = lb.bit_length() - 1
            nn -= lb
            if adj[u] & nv_in_mask:
                return False
        return True

    def bb(current_mask, avail_mask):
        cur_size = current_mask.bit_count()
        rem_size = avail_mask.bit_count()
        
        # Upper bound
        if cur_size + rem_size <= best[0]:
            return
            
        if avail_mask == 0:
            if cur_size > best[0]:
                best[0] = cur_size
            return
            
        # Heuristic: Pick a vertex from avail that has high degree in avail? 
        # Or just pick the lowest set bit for simplicity and consistency.
        lb = avail_mask & (-avail_mask)
        v = lb.bit_length() - 1
        new_avail = avail_mask - lb
        
        # Option 1: Exclude v
        bb(current_mask, new_avail)
        
        # Option 2: Include v
        # Check if v is compatible with current_mask
        if is_compatible(v, current_mask):
            # If we include v, do we restrict new_avail?
            # Vertices in new_avail that form a triangle with v and some u in current_mask are now invalid?
            # No, the set must be triangle-free. 
            # If we include v, any future vertex w added must not form a triangle with v and any u in current_mask.
            # This is handled by `is_compatible(w, current_mask | {v})` in the next recursion.
            # So we just pass `current_mask | lb` and `new_avail`.
            bb(current_mask | lb, new_avail)

    bb(0, full)
    return best[0]

def _edges_from_adj(adj):
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if adj[u] & (1 << v):
                edges.append([u, v])
    return edges

def _random_k4_free_graph():
    adj = [0] * N
    # Start with empty, add random edges, remove if K4 created
    # Or start with random and repair
    for u in range(N):
        for v in range(u+1, N):
            if random.random() < 0.5:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    # Repair: remove edges causing K4
    changed = True
    while changed:
        changed = False
        if not _is_k4_free(adj):
            # Find a K4 and remove an edge
            # This is slow if done naively. 
            # Given N=18, we can just try to find any K4.
            # For speed, we might just regenerate if not K4-free, but that's inefficient.
            # Let's try to remove a random edge and check again.
            # Better: iterate edges, remove one, check.
            for u in range(N):
                for v in range(u+1, N):
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        if _is_k4_free(adj):
                            changed = True
                            break
                if changed:
                    break
            if not changed:
                # Failed to repair by single edge removal? Unlikely for random graph.
                # Regenerate
                return _random_k4_free_graph()
    return adj

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Initial candidate: Turan graph T3(18)
    parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
    init_edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    init_edges.append([a, b])
    best_adj = _build_adj(init_edges)
    best_score = _max_triangle_free(best_adj)
    
    current_adj = best_adj[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    
    while time.time() - start_time < DEADLINE:
        # Generate neighbor: flip one random edge
        u = random.randint(0, N-1)
        v = random.randint(0, N-1)
        if u == v:
            continue
            
        new_adj = current_adj[:]
        if new_adj[u] & (1 << v):
            # Remove edge
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            # Add edge
            # Check if it creates a K4
            # K4 exists if u,v have 2 common neighbors that are connected?
            # Or simpler: just add and check _is_k4_free
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
        if not _is_k4_free(new_adj):
            continue
            
        new_score = _max_triangle_free(new_adj)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_adj = new_adj
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_adj = current_adj[:]
                
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart or reset T? For now, keep going.
            
    return _edges_from_adj(best_adj)
