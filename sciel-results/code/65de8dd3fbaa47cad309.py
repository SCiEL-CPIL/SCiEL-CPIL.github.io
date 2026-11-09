import random
import time

def _popcount(x):
    return bin(x).count('1')

def _is_k4_free_bitmask(adj):
    """Check if graph with adjacency bitmasks is K4-free."""
    n = 18
    # For each triangle (u, v, w), check if there is a common neighbor d
    # such that d is connected to u, v, w.
    # Actually, simpler: for each edge (u, v), let C = adj[u] & adj[v].
    # If there is an edge (a, b) in the subgraph induced by C, then {u, v, a, b} is K4.
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v] & ~((1 << u) | (1 << v))
                # Check if any two vertices in 'common' are connected
                # We can iterate through bits in common
                m = common
                while m:
                    # Extract lowest set bit
                    lsb = m & -m
                    a = lsb.bit_length() - 1
                    m ^= lsb
                    # Check if a is connected to any other vertex in common
                    if adj[a] & common:
                        return False
    return True

def _max_triangle_free_bitmask(adj):
    """
    Find the maximum number of vertices inducing a triangle-free subgraph.
    Uses branch and bound with bitmasks.
    """
    n = 18
    best = [0]
    
    # Precompute adjacencies for speed
    # adj is a list of integers, adj[i] has bit j set if edge (i,j) exists
    
    def branch(candidates, current_set, current_size):
        # Pruning
        if current_size + _popcount(candidates) <= best[0]:
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Heuristic: pick a vertex with high degree in candidates?
        # Or just pick the first one. Picking one with many conflicts might prune faster.
        # Let's pick the vertex with the most neighbors in 'current_set' that are connected to it?
        # Simpler: just pick the lowest index for determinism, or random?
        # For B&B, the order matters for speed. Let's try picking a vertex that maximizes constraints.
        # A vertex v that is connected to many vertices in current_set will remove many candidates if added.
        
        v_idx = -1
        max_conflicts = -1
        m = candidates
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            # Count how many u in current_set are connected to v
            # If we add v, we remove neighbors of v that are also neighbors of u in current_set
            # Let's just estimate by counting common neighbors with current_set
            # common_with_set = 0
            # cs = current_set
            # while cs:
            #   lsb2 = cs & -cs
            #   u = lsb2.bit_length() - 1
            #   cs ^= lsb2
            #   if adj[v] & (1 << u):
            #       common_with_set |= adj[v] & adj[u]
            # 
            # This is expensive. Let's just pick the first available vertex.
            # With N=18, even simple ordering works well with the strong bound.
            v_idx = v
            break # Just pick the first one for now to keep it simple and fast per step
            
        # If we want better ordering, we could sort candidates by degree in the induced subgraph of candidates.
        # But let's stick to simple first for now, as the pruning is strong.
        
        v = v_idx
        v_bit = 1 << v
        rest_candidates = candidates & ~v_bit
        
        # Option 1: Exclude v
        branch(rest_candidates, current_set, current_size)
        
        # Option 2: Include v
        # Check if v can be added to current_set without forming a triangle
        # v forms a triangle with u, w in current_set if adj[v][u] and adj[v][w] and adj[u][w]
        # We need to check if there exist u, w in current_set such that u-w is an edge and both connected to v.
        # Let N_v = adj[v] & current_set
        # If there is an edge between any two vertices in N_v, then adding v creates a triangle.
        N_v = adj[v] & current_set
        if N_v:
            # Check if N_v contains an edge
            # Iterate through bits in N_v
            m2 = N_v
            conflict = False
            while m2:
                lsb2 = m2 & -m2
                u = lsb2.bit_length() - 1
                m2 ^= lsb2
                if adj[u] & N_v:
                    conflict = True
                    break
            if conflict:
                # Cannot add v
                return # We already handled exclusion above
        
        # If no conflict, we can add v
        new_set = current_set | v_bit
        new_size = current_size + 1
        
        # Update candidates: remove any vertex w that would form a triangle with v and any u in current_set
        # w forms triangle with v and u if adj[v][w] and adj[u][w] and adj[v][u]
        # We know adj[v][u] is true for some u in N_v (if N_v is not empty, but we checked no triangle in N_v, 
        # wait. The condition for w is: w is connected to v, and w is connected to some u in current_set 
        # such that u is connected to v.
        # Let Forbidden = union over u in current_set where adj[v][u] of (adj[v] & adj[u])
        forbidden = 0
        m3 = N_v
        while m3:
            lsb3 = m3 & -m3
            u = lsb3.bit_length() - 1
            m3 ^= lsb3
            forbidden |= (adj[v] & adj[u])
        
        new_candidates = rest_candidates & ~forbidden
        
        branch(new_candidates, new_set, new_size)
        
    branch((1 << n) - 1, 0, 0)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 18
    full_mask = (1 << n) - 1
    
    # Initial graph: Turan T(18,3)
    # Parts: 0-5, 6-11, 12-17
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            part_i = 0 if i < 6 else (1 if i < 12 else 2)
            part_j = 0 if j < 6 else (1 if j < 12 else 2)
            if part_i != part_j:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    best_score = _max_triangle_free_bitmask(adj)
    best_adj = adj[:]
    
    # Local Search
    # We will try random edge swaps and single edge changes
    # To speed up, we can precompute lists of edges and non-edges
    # But since N=18, total edges = 153. We can just iterate or pick random.
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append((i, j))
                m ^= lsb
        return edges
        
    def get_non_edges(adj):
        non_edges = []
        for i in range(n):
            m = adj[i]
            for j in range(i + 1, n):
                if not (m & (1 << j)):
                    non_edges.append((i, j))
        return non_edges

    # Random restarts and local search
    attempts = 0
    while time.time() < deadline - 5:
        # Decide move type
        # 0: Remove edge
        # 1: Add edge (if K4-free)
        # 2: Swap (Remove one, Add one)
        # 3: Random Restart (New Turan or Random K4-free)
        
        move_type = random.randint(0, 3)
        
        if move_type == 3:
            # Random Restart
            # Generate a random K4-free graph
            # Start with empty graph, add random edges if K4-free
            adj_new = [0] * n
            # Try to add edges
            for _ in range(200):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v: continue
                if u > v: u, v = v, u
                if not (adj_new[u] & (1 << v)):
                    adj_new[u] |= (1 << v)
                    adj_new[v] |= (1 << u)
                    if not _is_k4_free_bitmask(adj_new):
                        adj_new[u] &= ~(1 << v)
                        adj_new[v] &= ~(1 << u)
            score = _max_triangle_free_bitmask(adj_new)
            if score < best_score:
                best_score = score
                best_adj = adj_new[:]
            # Continue search from this new graph
            adj = adj_new[:]
            continue
            
        # Get current edges and non-edges for the current adj
        # To save time, we might not want to rebuild these lists every iteration if possible,
        # but for N=18 it's fast enough.
        edges = get_edges(adj)
        non_edges = get_non_edges(adj)
        
        if move_type == 0 and edges:
            u, v = random.choice(edges)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            score = _max_triangle_free_bitmask(adj)
            if score < best_score:
                best_score = score
                best_adj = adj[:]
            else:
                # Revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        elif move_type == 1 and non_edges:
            u, v = random.choice(non_edges)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _is_k4_free_bitmask(adj):
                score = _max_triangle_free_bitmask(adj)
                if score < best_score:
                    best_score = score
                    best_adj = adj[:]
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
        elif move_type == 2 and edges and non_edges:
            u, v = random.choice(edges)
            x, y = random.choice(non_edges)
            # Remove u-v
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            # Add x-y
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            
            if _is_k4_free_bitmask(adj):
                score = _max_triangle_free_bitmask(adj)
                if score < best_score:
                    best_score = score
                    best_adj = adj[:]
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    adj[x] &= ~(1 << y)
                    adj[y] &= ~(1 << x)
            else:
                # Revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                adj[x] &= ~(1 << y)
                adj[y] &= ~(1 << x)
        
        # If we found a very good score, we might want to polish
        if best_score <= 9:
            # Intensive local search
            pass # The loop continues

    # Convert best_adj back to list of edges
    final_edges = []
    for i in range(n):
        m = best_adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                final_edges.append([i, j])
            m ^= lsb
            
    return final_edges
