import time
import random

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    Structural change: spectral bipartite init + edge-swap kick mechanism.
    """
    N = 40
    deadline = time.time() + 115  # Safety margin

    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    best_edges_list = []
    best_count = 0

    def run_once(seed):
        nonlocal best_edges_list, best_count
        random.seed(seed)

        adj_bits = [0] * N
        edges_set = set()
        edges_list = []

        def add_edge(u, v):
            if u > v:
                u, v = v, u
            if adj_bits[u] & (1 << v):
                return False
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            edges_set.add((u, v))
            edges_list.append((u, v))
            return True

        def remove_edge(u, v):
            if u > v:
                u, v = v, u
            if not (adj_bits[u] & (1 << v)):
                return False
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            edges_set.discard((u, v))
            edges_list.remove((u, v))
            return True

        def can_add(u, v):
            """Check if adding edge (u,v) creates a C6 via length-4 path."""
            start_mask = 1 << u
            stack = [(u, 0, start_mask)]
            while stack:
                node, depth, mask = stack.pop()
                if depth == 4:
                    if adj_bits[node] & (1 << v):
                        return False
                    continue
                neighbors = adj_bits[node] & ~(1 << v) & ~mask
                while neighbors:
                    low = neighbors & (-neighbors)
                    nb = low.bit_length() - 1
                    neighbors ^= low
                    stack.append((nb, depth + 1, mask | (1 << nb)))
            return True

        def find_blocking_c6(u, v):
            start_mask = 1 << u
            stack = [(u, 0, start_mask, [u])]
            while stack:
                node, depth, mask, path = stack.pop()
                if depth == 4:
                    if adj_bits[node] & (1 << v):
                        c6_edges = []
                        for i in range(len(path) - 1):
                            c6_edges.append((path[i], path[i + 1]))
                        c6_edges.append((node, v))
                        return c6_edges
                    continue
                neighbors = adj_bits[node] & ~(1 << v) & ~mask
                while neighbors:
                    low = neighbors & (-neighbors)
                    nb = low.bit_length() - 1
                    neighbors ^= low
                    stack.append((nb, depth + 1, mask | (1 << nb), path + [nb]))
            return None

        def greedy_fill():
            """Try to add all possible missing edges in priority order."""
            for _ in range(5):  # multiple passes
                missing = [(u, v) for u, v in all_pairs if not (adj_bits[u] & (1 << v))]
                if not missing:
                    break
                # Sort by degree sum descending (prefer connecting high-degree nodes)
                missing.sort(key=lambda e: -(adj_bits[e[0]].bit_count() + adj_bits[e[1]].bit_count()))
                added = 0
                for u, v in missing:
                    if adj_bits[u] & (1 << v):
                        continue
                    if can_add(u, v):
                        add_edge(u, v)
                        added += 1
                if added == 0:
                    break

        def edge_swap_kick():
            """
            Edge-swap move: remove one edge (a,b), then try to add a different
            missing edge (c,d). This preserves edge count while changing structure.
            If we can then greedily fill and gain edges, accept.
            """
            if not edges_list:
                return False

            # Pick a random existing edge to remove
            rem_idx = random.randrange(len(edges_list))
            ra, rb = edges_list[rem_idx]

            # Save state
            saved_bits = adj_bits[:]
            saved_edges_list = edges_list[:]
            saved_edges_set = edges_set.copy()

            remove_edge(ra, rb)

            # Try to add a random missing edge
            missing = [(u, v) for u, v in all_pairs if not (adj_bits[u] & (1 << v))]
            if not missing:
                # Revert
                adj_bits[:] = saved_bits
                edges_list[:] = saved_edges_list
                edges_set.clear()
                edges_set.update(saved_edges_set)
                return False

            # Try several candidates
            random.shuffle(missing)
            swap_done = False
            for attempt in range(min(20, len(missing))):
                cu, cv = missing[attempt]
                if can_add(cu, cv):
                    add_edge(cu, cv)
                    swap_done = True
                    break

            if not swap_done:
                # Revert
                adj_bits[:] = saved_bits
                edges_list[:] = saved_edges_list
                edges_set.clear()
                edges_set.update(saved_edges_set)
                return False

            # Greedy fill after swap
            old_count = len(edges_list)
            greedy_fill()
            new_count = len(edges_list)

            # Accept if we gained edges, or with some probability if neutral
            if new_count > old_count:
                return True
            elif new_count == old_count:
                # Accept neutral swaps with some probability for diversity
                if random.random() < 0.3:
                    return True
            else:
                # Revert if we lost edges
                adj_bits[:] = saved_bits
                edges_list[:] = saved_edges_list
                edges_set.clear()
                edges_set.update(saved_edges_set)
                return False

            return True

        def multi_remove_kick():
            """Remove 2-4 edges from a blocking C6, then greedy fill."""
            missing = [(u, v) for u, v in all_pairs if not (adj_bits[u] & (1 << v))]
            if not missing or not edges_list:
                return False

            target_u, target_v = random.choice(missing)
            block_c6 = find_blocking_c6(target_u, target_v)
            if not block_c6:
                return False

            saved_bits = adj_bits[:]
            saved_edges_list = edges_list[:]
            saved_edges_set = edges_set.copy()

            num_removals = random.randint(2, 4)
            random.shuffle(block_c6)
            for _ in range(min(num_removals, len(block_c6))):
                e = block_c6.pop()
                remove_edge(e[0], e[1])

            greedy_fill()

            if len(edges_list) >= len(saved_edges_list):
                return True
            else:
                adj_bits[:] = saved_bits
                edges_list[:] = saved_edges_list
                edges_set.clear()
                edges_set.update(saved_edges_set)
                return False

        # === INITIALIZATION: Spectral bipartite approach ===
        # Split into two groups of 20, make each a dense subgraph,
        # add sparse cross-edges to avoid C6s
        group_a = list(range(20))
        group_b = list(range(20, 40))

        # Within group A: create a near-clique but remove some edges to be C6-free
        # A K20 contains many C6s, so we need a C6-free dense subgraph on 20 vertices.
        # Use a Turan-like structure: complete bipartite K10,10 has no odd cycles,
        # but C6 is even. Instead, use a 4-partite Turan graph T(20,4) = K5,5,5,5
        # which is C4-free and hence C6-free? No, T(n,r) is C4-free only for r>=3? 
        # Actually T(n,4) on 20 vertices: partition into 4 parts of 5, connect all cross-part.
        # This is C4-free? No, T(n,r) is C4-free iff... let me think.
        # Actually, the Turan graph T(n,r) has no (r+1)-clique but may have C6.
        # 
        # Simpler: use 4 disjoint K5s within each group of 20.
        # 4 K5s = 4*10 = 40 edges per group, 80 total.
        # Then add cross-edges carefully.

        # Alternative: use a random dense C6-free init
        # Start with 8 K5s (known C6-free, 104 edges) but perturb the partition
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    add_edge(base + a, base + b)

        # Now try to add cross-cluster edges
        greedy_fill()

        local_best = len(edges_list)
        local_best_bits = adj_bits[:]
        local_best_edges = edges_list[:]

        # === MAIN SEARCH LOOP ===
        temperature = 10.0
        cooling_rate = 0.97
        min_temperature = 0.05
        stagnation = 0
        max_stagnation = 15

        while time.time() < deadline:
            # Greedy fill
            old_count = len(edges_list)
            greedy_fill()
            new_count = len(edges_list)

            if new_count > local_best:
                local_best = new_count
                local_best_bits = adj_bits[:]
                local_best_edges = edges_list[:]
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= max_stagnation or (new_count == old_count and stagnation > 5):
                # Kick: choose between edge-swap and multi-remove
                kick_type = random.random()
                if kick_type < 0.5:
                    success = edge_swap_kick()
                else:
                    success = multi_remove_kick()

                if success:
                    # After kick, do greedy fill
                    greedy_fill()
                    if len(edges_list) > local_best:
                        local_best = len(edges_list)
                        local_best_bits = adj_bits[:]
                        local_best_edges = edges_list[:]
                    stagnation = 0
                    temperature *= cooling_rate
                    if temperature < min_temperature:
                        temperature = min_temperature
                else:
                    # Kick failed, cool down
                    temperature *= cooling_rate
                    if temperature < min_temperature:
                        temperature = min_temperature

                    # If stuck for too long, try a more aggressive reset
                    if stagnation > 30:
                        # Remove 5-8 random edges
                        saved_bits = adj_bits[:]
                        saved_edges_list = edges_list[:]
                        saved_edges_set = edges_set.copy()
                        num_rem = random.randint(5, 8)
                        for _ in range(num_rem):
                            if not edges_list:
                                break
                            e = random.choice(edges_list)
                            remove_edge(e[0], e[1])
                        greedy_fill()
                        if len(edges_list) >= len(saved_edges_list) - 2:
                            stagnation = 0
                        else:
                            adj_bits[:] = saved_bits
                            edges_list[:] = saved_edges_list
                            edges_set.clear()
                            edges_set.update(saved_edges_set)
                            temperature = max(temperature, 1.0)

        # Update global best
        if local_best > best_count:
            best_count = local_best
            final_edges = []
            for u in range(N):
                mask = local_best_bits[u]
                while mask:
                    low = mask & (-mask)
                    v = low.bit_length() - 1
                    mask ^= low
                    if u < v:
                        final_edges.append([u, v])
            best_edges_list = final_edges

    # Run multiple restarts with different seeds
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        run_once(42 + i * 7919)

    if not best_edges_list:
        # Fallback: 8 disjoint K5s
        result = []
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    result.append([base + a, base + b])
        return result

    return best_edges_list

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
