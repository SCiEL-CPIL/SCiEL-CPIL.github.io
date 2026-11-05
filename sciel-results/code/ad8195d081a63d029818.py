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
    # Uses branch and bound with bitmasks
    def max_triangle_free_size(adj):
        # Build neighbor masks
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i][j]:
                    mask |= (1 << j)
            masks[i] = mask
        
        # We want to find the largest set S such that G[S] is triangle-free.
        # Triangle-free means: for any u in S, N(u) \cap S has no edges.
        # Or: No three vertices in S form a triangle.
        
        # We can use a recursive search.
        # State: current set S (bitmask), candidate vertices C (bitmask)
        # To ensure triangle-free, if we add v to S, then no two vertices in S \cap N(v) can be adjacent.
        # Actually, simpler: 
        # If S is triangle-free, then for any v in S, the subgraph induced by S \ {v} on N(v) must be triangle-free? No.
        
        # Let's use the property: A set S is triangle-free iff for all v in S, 
        # the set of neighbors of v in S contains no edge.
        # i.e., for all v in S, N(v) \cap S is an independent set in G.
        
        # Branch and Bound:
        # We build S vertex by vertex.
        # To speed up, we can precompute for each pair (u,v) if they are adjacent.
        
        # Alternative: Iterate k from n down to 1.
        # For a fixed k, check if there is a TF set of size k.
        # This is still hard.
        
        # Given n=18, we can use a heuristic or a fast exact solver.
        # Let's try a simple greedy + local search for the TF set size?
        # No, we need exact.
        
        # Let's use the fact that n=18 is small enough for a optimized DFS.
        
        # Precompute: for each v, the bitmask of its neighbors.
        # For each pair (u,v), the bitmask of common neighbors?
        
        # We will use a recursive function that takes a bitmask of available vertices 'avail'
        # and a bitmask of vertices in the current set 'curr'.
        # To maintain triangle-free property:
        # When adding a vertex v to curr, we must ensure that no two vertices in curr are adjacent to v?
        # No, we must ensure that v does not form a triangle with any two vertices in curr.
        # So, for all u, w in curr, u and w must not be both neighbors of v.
        # i.e., N(v) \cap curr must be an independent set.
        
        # This check is O(n^2) per step.
        
        # Let's optimize by pruning:
        # If we add v, the new available set is avail \ {v} \ {neighbors of v that would create triangles?}
        # Actually, if we add v, we cannot add any u such that u is adjacent to v AND u is adjacent to some w in curr?
        # No, u can be adjacent to v, as long as u is not adjacent to any w in curr \ {v}?
        
        # Let's just do a simple DFS with memoization or pruning.
        
        best = 0
        
        # Precompute: is_edge[u][v]
        is_edge = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    is_edge[i][j] = 1
        
        # Precompute: for each v, the list of neighbors
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    neighbors[i].append(j)
        
        # To check if a set S is triangle-free:
        # For each v in S, check if any edge exists in S \ {v} between two neighbors of v.
        
        # We'll use a recursive search that builds S.
        # To prune, we can bound the maximum possible size.
        
        # A faster approach for n=18:
        # Enumerate all subsets? 2^18 = 262k.
        # For each subset, check if TF.
        # Check TF:
        # For each v in subset:
        #   For each pair u,w in subset \ {v}:
        #      If is_edge[u][v] and is_edge[w][v] and is_edge[u][w]: return False
        # This is O(k^3) per subset.
        # Total: 262k * 18^3 ~ 1.5 * 10^9. Too slow in Python.
        
        # Optimized Check:
        # Precompute for each v, the bitmask of neighbors.
        # A subset S is TF iff for all v in S, (N(v) & S) has no edges.
        # "Has no edges" means: for all u in N(v) & S, (N(u) & S) & N(v) == 0?
        # No, (N(u) & S) should not contain any neighbor of v?
        # If u is in N(v) & S, and w is in N(v) & S, and u~w, then triangle.
        # So for a fixed v, the set T_v = N(v) & S must be an independent set.
        
        # We can precompute a table `is_independent[mask]` for all masks?
        # 2^18 masks.
        # is_independent[mask] = True if no edges within mask.
        # We can compute this:
        # is_independent[0] = True
        # is_independent[mask] = is_independent[mask without lowest bit] and 
        #                       (lowest_bit has no neighbors in mask without lowest bit)
        
        # Let's precompute `is_independent` for all 2^18 masks.
        # This takes 2^18 * 1 operations. Very fast.
        
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
        
        # We can iterate S from 1 to 2^n-1?
        # Or iterate by size?
        
        # Let's iterate by size k from n down to 1.
        # For each k, check if any subset of size k is TF.
        
        # To speed up, we can use the precomputed `is_independent`.
        
        # But checking all subsets of size k is still C(18, 9) ~ 48k.
        # For each subset, we check n conditions.
        # 48k * 18 ~ 864k operations.
        # This is fast enough!
        
        # So:
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

    # Initial Construction: Clebsch Graph on 16 vertices + 2 extra
    def generate_clebsch_adj():
        adj = [[0]*n for _ in range(n)]
        # Clebsch on 0..15
        for u in range(16):
            for v in range(u+1, 16):
                diff = u ^ v
                if bin(diff).count('1') == 2:
                    adj[u][v] = 1
                    adj[v][u] = 1
        # Vertices 16, 17
        # Start isolated
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

    # Start with Clebsch
    adj = generate_clebsch_adj()
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
