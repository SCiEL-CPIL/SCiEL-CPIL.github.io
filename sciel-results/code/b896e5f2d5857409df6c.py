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

    def max_tf_size_bnb(adj):
        """Find max size of triangle-free induced subgraph via branch and bound."""
        # A set S is triangle-free if for every edge (u,v) in S, there's no w in S
        # adjacent to both u and v.
        # We use B&B: build up the set, prune if adding a vertex creates a triangle
        # or if even completing with all remaining can't beat best.
        
        best = [0]
        
        def has_triangle(mask):
            """Check if induced subgraph on mask contains a triangle."""
            m = mask
            while m:
                u_bit = m & (-m)
                u = u_bit.bit_length() - 1
                m ^= u_bit
                # For each pair (u, v) with v > u in mask, check if they share a common neighbor in mask
                common_uv = adj[u] & mask
                while common_uv:
                    v_bit = common_uv & (-common_uv)
                    v = v_bit.bit_length() - 1
                    common_uv ^= v_bit
                    if v <= u:
                        continue
                    # Check if any w in mask is adjacent to both u and v
                    if adj[u] & adj[v] & mask:
                        return True
            return False

        def bnb(remaining_mask, current_mask, current_size):
            if current_size + popcount(remaining_mask) <= best[0]:
                return
            if current_mask and has_triangle(current_mask):
                return
            if remaining_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Try adding vertices
            # Heuristic: try vertices that are "safe" first
            m = remaining_mask
            while m:
                v_bit = m & (-m)
                v = v_bit.bit_length() - 1
                m ^= v_bit
                new_mask = current_mask | v_bit
                # Check if adding v creates a triangle
                if not has_triangle(new_mask):
                    bnb(remaining_mask & ~v_bit, new_mask, current_size + 1)
                # Also try NOT adding v (already handled by continuing loop)
            
            # Also try the case where we don't add any more vertices
            if current_size > best[0]:
                best[0] = current_size

        # Start with empty set
        bnb((1 << n) - 1, 0, 0)
        return best[0]

    def popcount(x):
        return bin(x).count('1')

    def check_k4_free(adj, u, v):
        """Check if adding edge (u,v) would create a K4."""
        common_uv = adj[u] & adj[v] & ~(1 << u) & ~(1 << v)
        m = common_uv
        while m:
            w_bit = m & (-m)
            w = w_bit.bit_length() - 1
            m ^= w_bit
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
        """Generate a Turan-like K4-free graph T_3(18)."""
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
    for _ in range(5):
        initializations.append(random_k4_free_graph(rng))
        
    # Evaluate initializations
    for edges in initializations:
        if time.time() > deadline - 15:
            break
        adj = build_adj_mask(edges)
        current_score = max_tf_size_bnb(adj)
        if current_score < best_score:
            best_score = current_score
            best_edges = set(edges)

    # Local Search
    current_edges = set(best_edges)
    current_score = best_score
    adj = build_adj_mask(current_edges)
    
    no_improve = 0
    max_no_improve = 30
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        
        # Try adding random non-edges
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            u, v = min(u, v), max(u, v)
            
            if (u, v) in current_edges:
                continue
            
            if not check_k4_free(adj, u, v):
                continue
            
            # Tentatively add
            current_edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            new_score = max_tf_size_bnb(adj)
            
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_edges = set(current_edges)
            else:
                # Revert
                current_edges.discard((u, v))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        # Try removing random edges
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            if not current_edges:
                break
            u, v = rng.choice(list(current_edges))
            
            # Tentatively remove
            current_edges.discard((u, v))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            new_score = max_tf_size_bnb(adj)
            
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_edges = set(current_edges)
            else:
                # Revert
                current_edges.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        if not improved:
            no_improve += 1
            
    return [[u, v] for u, v in sorted(best_edges)]
