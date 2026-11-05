import random
import time
import math

def construct():
    deadline = time.time() + 115  # Safety margin (evaluator kills at 150s)
    n = 40
    FULL = (1 << n) - 1

    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            bits = adj[i]
            while bits:
                lsb = bits & -bits
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                bits ^= lsb
        return edges

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def creates_c6_fast(adj, u, v):
        """
        Fast C6 check: is there a path u-a-b-c-d-v of length 5?
        Optimized bitmask operations.
        """
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)

        if nu == 0 or nv == 0:
            return False

        # We need to find a, b, c, d such that:
        # u-a, a-b, b-c, c-d, d-v are edges
        # a != d, a != v, d != u, b != c, etc.
        
        # Iterate a in N(u)
        bits_a = nu
        while bits_a:
            lsb_a = bits_a & -bits_a
            a = lsb_a.bit_length() - 1
            bits_a ^= lsb_a
            
            # d must be in N(v), d != a
            bits_d = nv & ~(1 << a)
            if bits_d == 0:
                continue
            
            # For each d, we need a path of length 3 from a to d in G-{u,v}
            # Path: a - b - c - d
            # b in N(a)\{u,v,d}, c in N(d)\{u,v,a}, b~c
            
            # To speed up, we can precompute N(a) and N(d) masks excluding u,v
            # Let's iterate d
            bits_d_cur = bits_d
            while bits_d_cur:
                lsb_d = bits_d_cur & -bits_d_cur
                d = lsb_d.bit_length() - 1
                bits_d_cur ^= lsb_d
                
                # b candidates: neighbors of a, excluding u, v, a, d
                b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if b_mask == 0:
                    continue
                
                # c candidates: neighbors of d, excluding u, v, a, d
                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if c_mask == 0:
                    continue
                
                # Check if there exists b in b_mask and c in c_mask such that b~c
                # Iterate over the smaller set to minimize iterations
                if b_mask.bit_count() < c_mask.bit_count():
                    bits_b = b_mask
                    while bits_b:
                        lsb_b = bits_b & -bits_b
                        b = lsb_b.bit_length() - 1
                        bits_b ^= lsb_b
                        # c must be in N(b) & c_mask, c != b
                        if adj[b] & c_mask & ~(1 << b):
                            return True
                else:
                    bits_c = c_mask
                    while bits_c:
                        lsb_c = bits_c & -bits_c
                        c = lsb_c.bit_length() - 1
                        bits_c ^= lsb_c
                        # b must be in N(c) & b_mask, b != c
                        if adj[c] & b_mask & ~(1 << c):
                            return True

        return False

    def is_c6_free(adj):
        """Full validation: check that no C6 exists in the graph."""
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    if creates_c6_fast(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        return False
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return True

    def get_missing_edges(adj):
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    missing.append((u, v))
        return missing

    def get_existing_edges(adj):
        existing = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    existing.append((u, v))
        return existing

    def greedy_construct(rng):
        """Greedy edge addition in random order."""
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)

        for u, v in all_edges:
            if not creates_c6_fast(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, rng):
        """
        Greedily add valid edges until no more can be added.
        Uses multiple passes with random ordering.
        """
        improved = True
        while improved:
            improved = False
            missing = get_missing_edges(adj)
            if not missing:
                break
            rng.shuffle(missing)
            for u, v in missing:
                if not creates_c6_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
        return adj

    def edge_swap_local_search(adj, rng, max_iters=20):
        """
        Try to swap one edge for another to increase edge count.
        Remove edge (u,v), then try to add a different edge (x,y).
        """
        current_m = count_edges(adj)
        best_m = current_m
        best_adj = adj[:]

        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break

            existing = get_existing_edges(adj)
            if not existing:
                break

            # Pick a random edge to remove
            u, v = rng.choice(existing)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

            # Now try to add as many edges as possible, prioritizing edges that were previously blocked
            # Simple approach: just run local search
            adj = local_search(adj, rng)
            m = count_edges(adj)

            if m > best_m:
                best_m = m
                best_adj = adj[:]
            else:
                # Revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                adj = local_search(adj, rng)

        return best_adj

    rng = random.Random(42)

    best_adj = None
    best_m = 0

    # Phase 1: Multiple random greedy constructions
    phase1_deadline = time.time() + 10
    while time.time() < phase1_deadline:
        adj = greedy_construct(rng)
        adj = local_search(adj, rng)
        m = count_edges(adj)
        if m > best_m:
            best_m = m
            best_adj = adj[:]

    current_adj = best_adj[:]
    current_m = best_m

    # Phase 2: Iterated Local Search with varied perturbations
    # We use a cooling schedule for acceptance
    temperature = 1.0
    cooling_factor = 0.995
    
    while time.time() < deadline - 5:
        perturbed = current_adj[:]
        
        # Choose perturbation type
        pert_type = rng.random()
        
        if pert_type < 0.3:
            # Remove 2-4 random edges
            num_removals = rng.randint(2, 4)
            existing = get_existing_edges(perturbed)
            if existing:
                rng.shuffle(existing)
                for i in range(min(num_removals, len(existing))):
                    u, v = existing[i]
                    perturbed[u] &= ~(1 << v)
                    perturbed[v] &= ~(1 << u)

        elif pert_type < 0.6:
            # Remove 5-10 random edges (larger shake)
            num_removals = rng.randint(5, 10)
            existing = get_existing_edges(perturbed)
            if existing:
                rng.shuffle(existing)
                for i in range(min(num_removals, len(existing))):
                    u, v = existing[i]
                    perturbed[u] &= ~(1 << v)
                    perturbed[v] &= ~(1 << u)

        else:
            # Edge Swap: Remove one edge, add one random missing edge if valid
            existing = get_existing_edges(perturbed)
            missing = get_missing_edges(perturbed)
            if existing and missing:
                u, v = rng.choice(existing)
                perturbed[u] &= ~(1 << v)
                perturbed[v] &= ~(1 << u)
                
                # Try to add a random missing edge
                x, y = rng.choice(missing)
                if not creates_c6_fast(perturbed, x, y):
                    perturbed[x] |= (1 << y)
                    perturbed[y] |= (1 << x)

        # Local search on perturbed
        perturbed = local_search(perturbed, rng)
        m = count_edges(perturbed)

        # Acceptance criterion
        if m >= best_m:
            if m > best_m:
                best_m = m
                best_adj = perturbed[:]
            current_adj = perturbed[:]
            current_m = m
        else:
            # Simulated Annealing style acceptance
            gap = best_m - m
            # Probability of accepting worse solution
            # p = exp(-gap / temperature)
            # To avoid underflow/overflow, we can use log or just a simple heuristic
            if gap <= 1:
                accept_prob = 0.5 * temperature
            elif gap <= 2:
                accept_prob = 0.2 * temperature
            else:
                accept_prob = 0.05 * temperature
            
            if rng.random() < accept_prob:
                current_adj = perturbed[:]
                current_m = m
        
        # Cool down
        temperature *= cooling_factor
        if temperature < 0.01:
            temperature = 0.01 # Keep a minimal chance of escape

    # Final validation
    if not is_c6_free(best_adj):
        # Fallback: try a fresh greedy construction
        adj = greedy_construct(rng)
        if is_c6_free(adj):
            best_adj = adj
        else:
            # Fallback: return empty graph
            return []

    return adj_to_edges(best_adj)
