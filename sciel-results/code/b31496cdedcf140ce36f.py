import random
import time

def construct():
    deadline = time.time() + 115  # Safety margin
    n = 40

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

    def creates_c6(adj, u, v):
        """
        Check if adding edge (u, v) creates a C6.
        Path: u - a - b - c - d - v
        """
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)

        if nu == 0 or nv == 0:
            return False

        bits_a = nu
        while bits_a:
            lsb_a = bits_a & -bits_a
            a = lsb_a.bit_length() - 1
            bits_a ^= lsb_a

            bits_d = nv & ~(1 << a)
            if bits_d == 0:
                continue

            bits_d_cur = bits_d
            while bits_d_cur:
                lsb_d = bits_d_cur & -bits_d_cur
                d = lsb_d.bit_length() - 1
                bits_d_cur ^= lsb_d

                # b in N(a) \ {u, v, a, d}
                b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if b_mask == 0:
                    continue

                # c in N(d) \ {u, v, a, d}
                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if c_mask == 0:
                    continue

                bits_b = b_mask
                while bits_b:
                    lsb_b = bits_b & -bits_b
                    b = lsb_b.bit_length() - 1
                    bits_b ^= lsb_b

                    # c in N(b) & c_mask, c != b
                    if adj[b] & c_mask & ~(1 << b):
                        return True

        return False

    def is_c6_free(adj):
        """Full validation: check that no C6 exists in the graph."""
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    if creates_c6(adj, u, v):
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

    # Greedy construction with random edge order
    def greedy_construct(rng):
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)

        for u, v in all_edges:
            if not creates_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    # Greedy completion: add as many valid edges as possible
    def greedy_complete(adj, rng):
        changed = True
        while changed:
            changed = False
            missing = get_missing_edges(adj)
            if not missing:
                break
            rng.shuffle(missing)
            for u, v in missing:
                if not creates_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
        return adj

    rng = random.Random(42)

    best_adj = None
    best_m = 0

    # Phase 1: Multiple random greedy constructions
    initial_time = time.time()
    while time.time() < initial_time + 15:
        adj = greedy_construct(rng)
        m = count_edges(adj)
        if m > best_m:
            best_m = m
            best_adj = adj[:]

    current_adj = best_adj[:]
    current_m = best_m

    # Phase 2: Iterated Local Search with perturbation
    time_start = time.time()
    while time.time() < deadline:
        # Perturb current best by removing some random edges
        perturbed = current_adj[:]
        
        num_removals = rng.randint(3, 8)
        existing = get_existing_edges(perturbed)
        if existing:
            rng.shuffle(existing)
            for i in range(min(num_removals, len(existing))):
                u, v = existing[i]
                perturbed[u] &= ~(1 << v)
                perturbed[v] &= ~(1 << u)
        
        # Greedy complete the perturbed graph
        perturbed = greedy_complete(perturbed, rng)
        
        m = count_edges(perturbed)
        
        if m > best_m:
            best_m = m
            best_adj = perturbed[:]
            current_adj = perturbed[:]
            current_m = m
        elif m > current_m - 3 and rng.random() < 0.3:
            # Accept worse solutions occasionally to escape local optima
            current_adj = perturbed[:]
            current_m = m

    # Final validation
    if not is_c6_free(best_adj):
        # Fallback: try a fresh greedy construction
        adj = greedy_construct(rng)
        if is_c6_free(adj):
            best_adj = adj

    return adj_to_edges(best_adj)
