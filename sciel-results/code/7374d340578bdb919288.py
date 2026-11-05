import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18

    def build_adj_mask(edge_set):
        adj = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_tf_mask(mask, adj):
        """Check if vertex set represented by bitmask is triangle-free."""
        m = mask
        while m:
            u_bit = m & (-m)
            u = u_bit.bit_length() - 1
            m ^= u_bit
            common = adj[u] & mask
            while common:
                v_bit = common & (-common)
                v = v_bit.bit_length() - 1
                common ^= v_bit
                if v < u:
                    continue
                w_common = adj[u] & adj[v] & mask
                if w_common:
                    return False
        return True

    def max_tf_size_fast(adj):
        """Find the maximum size of a triangle-free induced subgraph."""
        # Try sizes from 12 down to 5. If we find one, return it.
        # If none found, return 4 (conservative, but likely we find one).
        for size in range(12, 4, -1):
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_tf_mask(mask, adj):
                    return size
                if time.time() > deadline - 10:
                    return 5  # conservative default to avoid timeout
        return 4

    def check_k4_free(adj, u, v):
        """Check if adding edge (u,v) would create a K4."""
        # A K4 containing (u,v) must have two other vertices w, z
        # such that u-v, u-w, u-z, v-w, v-z, w-z are all edges.
        # So w and z must be in the common neighborhood of u and v.
        # And w-z must be an edge.
        # So we just need to check if the common neighborhood of u and v
        # contains an edge.
        common_uv = adj[u] & adj[v] & ~(1 << u) & ~(1 << v)
        # Check if any two vertices in common_uv are adjacent
        m = common_uv
        while m:
            w_bit = m & (-m)
            w = w_bit.bit_length() - 1
            m ^= w_bit
            # Check if w has any neighbor in (common_uv & ~(1<<w))
            if adj[w] & common_uv & ~(1 << w):
                return False
        return True

    def random_k4_free_graph(rng):
        """Generate a random dense K4-free graph using greedy edge addition."""
        edges = set()
        all_possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_possible)
        
        adj = [0] * n
        for u, v in all_possible:
            if not check_k4_free(adj, u, v):
                continue
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        return edges

    def turan_k4_free_graph():
        """Generate a Turan-like K4-free graph T_3(18).
        This is a complete 3-partite graph with parts of size 6.
        It is K4-free.
        Max TF size is 12 (any two parts).
        """
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        edges = set()
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        edges.add((a, b) if a < b else (b, a))
        return edges

    rng = random.Random(42)
    
    best_score = 18
    best_edges = set()
    
    # Start with a few different initializations
    initializations = []
    
    # 1. Turan graph
    initializations.append(turan_k4_free_graph())
    
    # 2. Random K4-free graphs
    for _ in range(3):
        initializations.append(random_k4_free_graph(rng))
        
    # Evaluate initializations
    for edges in initializations:
        if time.time() > deadline - 10:
            break
        adj = build_adj_mask(edges)
        current_score = max_tf_size_fast(adj)
        if current_score < best_score:
            best_score = current_score
            best_edges = set(edges)

    # Local Search: Hill Climbing
    # We will iterate, trying to add edges to reduce the score.
    # We also try swapping vertices between "parts" conceptually?
    # Since we don't have explicit parts in the random graph, we just do edge additions.
    
    current_edges = set(best_edges)
    current_score = best_score
    adj = build_adj_mask(current_edges)
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        # Try to add a random non-edge
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            no_improve += 1
            continue
        u, v = min(u, v), max(u, v)
        
        if (u, v) in current_edges:
            no_improve += 1
            continue
            
        if not check_k4_free(adj, u, v):
            no_improve += 1
            continue
            
        # Tentatively add
        current_edges.add((u, v))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        new_score = max_tf_size_fast(adj)
        
        if new_score < current_score:
            current_score = new_score
            no_improve = 0
            if new_score < best_score:
                best_score = new_score
                best_edges = set(current_edges)
        else:
            # Revert
            current_edges.discard((u, v))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            no_improve += 1
            
    return [[u, v] for u, v in sorted(best_edges)]
