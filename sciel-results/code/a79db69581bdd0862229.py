import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer for safety
    
    n = 18
    best_score = 18
    best_edges = []
    
    # Helper: Get edges from adjacency matrix
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    # Helper: Check if graph is K4-free using bitmask
    # Precompute neighbor masks
    def is_k4_free(adj):
        # Build neighbor masks
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i][j]:
                    mask |= (1 << j)
            masks[i] = mask
        
        # Check all 4-subsets
        # Optimization: Only check if the first 3 form a triangle
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                # u, v are adjacent
                common_uv = masks[u] & masks[v]
                # w must be in common_uv and w > v to avoid duplicates? 
                # No, just iterate w
                for w in range(v + 1, n):
                    if not (common_uv & (1 << w)):
                        continue
                    # u, v, w form a triangle
                    # Check if any x is connected to u, v, w
                    common_uvw = common_uv & masks[w]
                    # x must be distinct from u, v, w
                    # common_uvw contains neighbors of w that are also neighbors of u and v.
                    # It might contain u or v? 
                    # masks[w] contains u and v.
                    # common_uv contains u and v? 
                    # masks[u] contains v. masks[v] contains u.
                    # So common_uv contains u and v? 
                    # masks[u] has bit v. masks[v] has bit u.
                    # Intersection has bit u and bit v.
                    # So common_uvw has bits u, v, and any x.
                    # We need to check if there is an x != u, v, w.
                    # Remove u, v, w from common_uvw
                    check_mask = common_uvw & ~( (1 << u) | (1 << v) | (1 << w) )
                    if check_mask != 0:
                        return False
        return True

    # Helper: Find maximum triangle-free induced subgraph size
    # Uses precomputed independent set table
    def max_triangle_free_size(adj):
        # Build neighbor masks
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i][j]:
                    mask |= (1 << j)
            masks[i] = mask
        
        # Precompute `is_independent` for all 2^18 masks.
        # is_independent[mask] = True if no edges within mask.
        is_independent = [False] * (1 << n)
        is_independent[0] = True
        for mask in range(1, 1 << n):
            # Get lowest set bit
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            rest = mask ^ lsb
            # Check if v is connected to any node in rest
            # If not, and rest is independent, then mask is independent.
            # v is connected to any node in rest if (masks[v] & rest) != 0
            if is_independent[rest] and (masks[v] & rest) == 0:
                is_independent[mask] = True
            else:
                is_independent[mask] = False
        
        # Now, a set S is triangle-free iff for all v in S, (masks[v] & S) is independent.
        # We want to find max |S| satisfying this.
        
        # Iterate k from n down to 1.
        # For each k, check if any subset of size k is TF.
        for k in range(n, 0, -1):
            found = False
            # Iterate all subsets of size k
            # Using itertools.combinations
            for subset in itertools.combinations(range(n), k):
                # Convert to mask
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check if TF
                is_tf = True
                for v in subset:
                    # Get neighbors of v in subset
                    neigh_mask = masks[v] & mask
                    # Check if neigh_mask is independent
                    if not is_independent[neigh_mask]:
                        is_tf = False
                        break
                if is_tf:
                    return k
        return 0

    # Initial Construction: Random maximal K4-free graph
    def generate_random_k4_free_adj():
        adj = [[0]*n for _ in range(n)]
        
        # Start with random graph
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.5:
                    adj[u][v] = 1
                    adj[v][u] = 1
        
        # Remove edges to ensure K4-free
        # Iterate over all 4-tuples and remove an edge if it forms a K4
        # This is not efficient, but for n=18 it might be okay if we are smart.
        # Better: Check all 4-subsets. If K4, remove a random edge.
        # Repeat until no K4.
        
        for _ in range(1000):
            changed = False
            for subset in itertools.combinations(range(n), 4):
                u, v, w, x = subset
                if adj[u][v] and adj[u][w] and adj[u][x] and \
                   adj[v][w] and adj[v][x] and \
                   adj[w][x]:
                    # K4 found. Remove a random edge.
                    edges_in_k4 = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                    rem_u, rem_v = random.choice(edges_in_k4)
                    adj[rem_u][rem_v] = 0
                    adj[rem_v][rem_u] = 0
                    changed = True
            if not changed:
                break
        
        return adj

    # Local Search
    def flip_edge(adj, u, v):
        new_adj = [row[:] for row in adj]
        new_adj[u][v] ^= 1
        new_adj[v][u] = new_adj[u][v]
        return new_adj

    def score_graph(adj):
        if not is_k4_free(adj):
            return None
        return max_triangle_free_size(adj)

    # Start with Random Maximal K4-free
    adj = generate_random_k4_free_adj()
    current_score = score_graph(adj)
    if current_score is not None:
        best_score = current_score
        best_edges = get_edges_from_adj(adj)
    else:
        # Fallback
        adj = [[0]*n for _ in range(n)]
        current_score = n
        best_score = current_score
        best_edges = []

    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline - 5:
        # Pick random edge to flip
        u, v = random.sample(range(n), 2)
        new_adj = flip_edge(adj, u, v)
        
        if not is_k4_free(new_adj):
            continue
            
        new_score = max_triangle_free_size(new_adj)
        
        if new_score is not None:
            delta = new_score - current_score
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                current_score = new_score
                adj = new_adj
                if current_score < best_score:
                    best_score = current_score
                    best_edges = get_edges_from_adj(adj)
            
        T *= cooling
        if T < 0.01:
            T = 1.0 # Restart? Or just keep going?
            # Let's do a few restarts
            # Actually, just keep cooling.
            
    return best_edges
