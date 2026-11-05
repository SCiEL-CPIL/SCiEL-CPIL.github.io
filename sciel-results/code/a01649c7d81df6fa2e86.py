import random
import time
import math

def construct():
    deadline = time.time() + 118  # Safety margin (evaluator kills at 150s)
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
        Iterates over the smaller candidate set for speed.
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

                b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if b_mask == 0:
                    continue

                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if c_mask == 0:
                    continue

                # Iterate over the smaller set
                if b_mask.bit_count() < c_mask.bit_count():
                    bits_b = b_mask
                    while bits_b:
                        lsb_b = bits_b & -bits_b
                        b = lsb_b.bit_length() - 1
                        bits_b ^= lsb_b
                        if adj[b] & c_mask & ~(1 << b):
                            return True
                else:
                    bits_c = c_mask
                    while bits_c:
                        lsb_c = bits_c & -bits_c
                        c = lsb_c.bit_length() - 1
                        bits_c ^= lsb_c
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

    def final_polish(adj, rng):
        """
        Final high-precision polish: multiple passes of local search
        with different random orderings to squeeze out remaining edges.
        """
        for _ in range(5):
            if time.time() > deadline - 2:
                break
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

    rng = random.Random(42)

    best_adj = None
    best_m = 0

    # Phase 1: Multiple random greedy constructions
    phase1_deadline = time.time() + 8
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
    # Combines current program's SA acceptance with inspiration's perturbation types
    temperature = 1.0
    cooling_factor = 0.995
    min_temp = 0.01

    while time.time() < deadline - 3:
        perturbed = current_adj[:]

        pert_type = rng.random()

        if pert_type < 0.35:
            # Small perturbation: remove 2-4 random edges
            num_removals = rng.randint(2, 4)
            existing = get_existing_edges(perturbed)
            if existing:
                rng.shuffle(existing)
                for i in range(min(num_removals, len(existing))):
                    u, v = existing[i]
                    perturbed[u] &= ~(1 << v)
                    perturbed[v] &= ~(1 << u)

        elif pert_type < 0.55:
            # Medium perturbation: remove 5-10 random edges
            num_removals = rng.randint(5, 10)
            existing = get_existing_edges(perturbed)
            if existing:
                rng.shuffle(existing)
                for i in range(min(num_removals, len(existing))):
                    u, v = existing[i]
                    perturbed[u] &= ~(1 << v)
                    perturbed[v] &= ~(1 << u)

        elif pert_type < 0.75:
            # Swap-like perturbation: remove 1-3 edges, add 1-2 valid missing edges
            num_removals = rng.randint(1, 3)
            existing = get_existing_edges(perturbed)
            if existing:
                rng.shuffle(existing)
                for i in range(min(num_removals, len(existing))):
                    u, v = existing[i]
                    perturbed[u] &= ~(1 << v)
                    perturbed[v] &= ~(1 << u)

            missing = get_missing_edges(perturbed)
            if missing:
                num_adds = rng.randint(1, 2)
                rng.shuffle(missing)
                for i in range(min(num_adds, len(missing))):
                    u, v = missing[i]
                    if not creates_c6_fast(perturbed, u, v):
                        perturbed[u] |= (1 << v)
                        perturbed[v] |= (1 << u)

        else:
            # Large perturbation: remove 8-15 random edges
            num_removals = rng.randint(8, 15)
            existing = get_existing_edges(perturbed)
            if existing:
                rng.shuffle(existing)
                for i in range(min(num_removals, len(existing))):
                    u, v = existing[i]
                    perturbed[u] &= ~(1 << v)
                    perturbed[v] &= ~(1 << u)

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
        if temperature < min_temp:
            temperature = min_temp

    # Final polish on the best solution found
    if best_adj is not None:
        best_adj = final_polish(best_adj, rng)
        best_m = count_edges(best_adj)

    # Final validation
    if not is_c6_free(best_adj):
        # Fallback: try a fresh greedy construction
        adj = greedy_construct(rng)
        adj = local_search(adj, rng)
        if is_c6_free(adj):
            best_adj = adj
        else:
            # Fallback: return empty graph
            return []

    return adj_to_edges(best_adj)
