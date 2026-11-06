import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe within 60s

    # 1. Determine valid candidates (diagonal check)
    squares = set(k * k for k in range(1, 23))  # max sum 512, sqrt(512) ~ 22.6

    candidates = []
    for i in range(1, 257):
        if 2 * i not in squares:
            candidates.append(i)

    N = len(candidates)

    # 2. Build conflict masks
    masks = [0] * N
    for i in range(N):
        ci = candidates[i]
        m = 0
        for j in range(N):
            if i == j:
                continue
            cj = candidates[j]
            if ci + cj in squares:
                m |= (1 << j)
        masks[i] = m

    # Precompute degree
    degrees = [masks[i].bit_count() for i in range(N)]

    # 3. Greedy initialization with min-degree heuristic
    def greedy_solve():
        remaining = (1 << N) - 1
        selected = 0
        while remaining:
            best_node = -1
            min_deg = float('inf')
            r = remaining
            while r:
                lsb = r & (-r)
                idx = lsb.bit_length() - 1
                deg = (masks[idx] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_node = idx
                r ^= lsb
            if best_node == -1:
                break
            selected |= (1 << best_node)
            remaining &= ~(masks[best_node] | (1 << best_node))
        return selected, selected.bit_count()

    best_set, best_size = greedy_solve()

    # 4. Local Search with 2-swap moves (from inspiration)
    def local_search(current_set, current_size):
        nonlocal best_set, best_size

        current = current_set
        current_sz = current_size
        best_local = current
        best_local_sz = current_size

        # Phase 1: Greedy addition
        while True:
            if time.time() > deadline:
                break
            addable = []
            for i in range(N):
                bit = 1 << i
                if not (current & bit) and (masks[i] & current) == 0:
                    addable.append(i)
            if not addable:
                break
            best_add = min(addable, key=lambda i: (masks[i] & ~current).bit_count())
            current |= (1 << best_add)
            current_sz += 1
            if current_sz > best_local_sz:
                best_local = current
                best_local_sz = current_sz
            if current_sz > best_size:
                best_size = current_size
                best_set = current

        # Phase 2: Swaps - single and 2-swap
        max_iterations = 600
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            found_swap = False

            # Single swap: remove v, add u
            for _ in range(25):
                u_idx = random.randint(0, N - 1)
                u_bit = 1 << u_idx
                if current & u_bit:
                    continue
                r = current
                while r:
                    lsb = r & (-r)
                    v_idx = lsb.bit_length() - 1
                    v_bit = 1 << v_idx
                    if (masks[u_idx] & (current & ~v_bit)) == 0:
                        current = (current & ~v_bit) | u_bit
                        found_swap = True
                        break
                    r ^= lsb
                if found_swap:
                    break

            if not found_swap:
                # 2-swap: remove v1,v2, add u1,u2
                found_2swap = False
                for _ in range(12):
                    u1_idx = random.randint(0, N - 1)
                    u1_bit = 1 << u1_idx
                    if current & u1_bit:
                        continue
                    u2_idx = random.randint(0, N - 1)
                    u2_bit = 1 << u2_idx
                    if current & u2_bit or u2_idx == u1_idx:
                        continue
                    if masks[u1_idx] & u2_bit:
                        continue

                    r1 = current
                    while r1 and not found_2swap:
                        lsb1 = r1 & (-r1)
                        v1_idx = lsb1.bit_length() - 1
                        v1_bit = 1 << v1_idx
                        r2 = current & ~v1_bit
                        while r2 and not found_2swap:
                            lsb2 = r2 & (-r2)
                            v2_idx = lsb2.bit_length() - 1
                            v2_bit = 1 << v2_idx
                            new_current = current & ~v1_bit & ~v2_bit
                            if (masks[u1_idx] & new_current) == 0 and (masks[u2_idx] & new_current) == 0:
                                current = new_current | u1_bit | u2_bit
                                found_2swap = True
                                found_swap = True
                            r2 ^= lsb2
                        r1 ^= lsb1

                if not found_2swap:
                    # Try double removal + single add (net -1)
                    found_double = False
                    for _ in range(8):
                        u_idx = random.randint(0, N - 1)
                        u_bit = 1 << u_idx
                        if current & u_bit:
                            continue
                        r1 = current
                        while r1 and not found_double:
                            lsb1 = r1 & (-r1)
                            v1_idx = lsb1.bit_length() - 1
                            v1_bit = 1 << v1_idx
                            r2 = current & ~v1_bit
                            while r2 and not found_double:
                                lsb2 = r2 & (-r2)
                                v2_idx = lsb2.bit_length() - 1
                                v2_bit = 1 << v2_idx
                                new_current = current & ~v1_bit & ~v2_bit
                                if (masks[u_idx] & new_current) == 0:
                                    if random.random() < 0.35:
                                        current = new_current | u_bit
                                        current_sz -= 1
                                        found_double = True
                                r2 ^= lsb2
                            r1 ^= lsb1
                    if not found_double:
                        break

            # Track best
            if current_sz > best_local_sz:
                best_local = current
                best_local_sz = current_sz
            if current_sz > best_size:
                best_size = current_sz
                best_set = current

        # Phase 3: Final greedy addition
        while True:
            if time.time() > deadline:
                break
            addable = []
            for i in range(N):
                bit = 1 << i
                if not (current & bit) and (masks[i] & current) == 0:
                    addable.append(i)
            if not addable:
                break
            best_add = min(addable, key=lambda i: (masks[i] & ~current).bit_count())
            current |= (1 << best_add)
            current_sz += 1
            if current_sz > best_local_sz:
                best_local = current
                best_local_sz = current_sz
            if current_sz > best_size:
                best_size = current_sz
                best_set = current

        return best_local, best_local_sz

    # Run local search
    best_set, best_size = local_search(best_set, best_size)

    # 5. Efficient Branch and Bound with fast greedy coloring bound
    if time.time() < deadline - 2.0:
        import sys
        sys.setrecursionlimit(10000)

        # Fast greedy coloring bound: order nodes by degree descending,
        # assign each node the smallest color not used by already-colored neighbors
        # This is O(n^2) total for the coloring, but we cache it per remaining set
        # by using a simple heuristic: the number of colors in a greedy coloring
        # of the induced subgraph is an upper bound on the independence number.

        # Pre-sort nodes by degree for consistent ordering
        node_order = sorted(range(N), key=lambda i: -degrees[i])

        def fast_color_bound(remaining):
            """Fast greedy coloring upper bound on independence number."""
            if remaining == 0:
                return 0

            # Use a greedy coloring: process nodes in degree order
            # Assign each node the smallest color not used by its colored neighbors
            color_of = [-1] * N
            num_colors = 0
            r = remaining
            # Process in order of decreasing degree (pre-sorted)
            for idx in node_order:
                if not (remaining & (1 << idx)):
                    continue
                # Find colors used by already-colored neighbors
                neighbor_colors = set()
                m = masks[idx] & remaining
                # Check only already-colored neighbors
                mm = m
                while mm:
                    lsb = mm & (-mm)
                    nidx = lsb.bit_length() - 1
                    if color_of[nidx] >= 0:
                        neighbor_colors.add(color_of[nidx])
                    mm ^= lsb

                # Assign smallest available color
                c = 0
                while c in neighbor_colors:
                    c += 1
                color_of[idx] = c
                if c + 1 > num_colors:
                    num_colors = c + 1

            return num_colors

        # Even faster bound: just use the maximum clique size as a lower bound on chromatic number
        # Or use a simple greedy bound that's very fast

        def very_fast_bound(remaining):
            """Very fast upper bound: greedy coloring with early termination."""
            if remaining == 0:
                return 0

            # Simple bound: the independence number is at most the number of nodes
            # But we can do better with a quick greedy coloring
            color_of = {}
            num_colors = 0
            r = remaining
            # Process nodes in a fixed order for speed
            for idx in node_order:
                if not (remaining & (1 << idx)):
                    continue
                # Find colors used by colored neighbors
                used = set()
                m = masks[idx] & remaining
                mm = m
                while mm:
                    lsb = mm & (-mm)
                    nidx = lsb.bit_length() - 1
                    if nidx in color_of:
                        used.add(color_of[nidx])
                    mm ^= lsb
                c = 0
                while c in used:
                    c += 1
                color_of[idx] = c
                if c + 1 > num_colors:
                    num_colors = c + 1
            return num_colors

        # B&B with the fast bound
        def branch_and_bound(remaining, current_size, current_set, depth=0):
            nonlocal best_set, best_size

            if time.time() > deadline - 0.5:
                return

            if remaining == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
                return

            # Upper bound: current_size + independence number of remaining
            ub = very_fast_bound(remaining)
            if current_size + ub <= best_size:
                return

            # Pick a node to branch on - use max degree in remaining
            best_node = -1
            max_deg = -1
            r = remaining
            while r:
                lsb = r & (-r)
                idx = lsb.bit_length() - 1
                deg = (masks[idx] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_node = idx
                r ^= lsb

            if best_node == -1:
                return

            bit = 1 << best_node

            # Branch 1: Include best_node
            new_remaining_1 = remaining & ~(masks[best_node] | bit)
            branch_and_bound(new_remaining_1, current_size + 1, current_set | bit, depth + 1)

            # Branch 2: Exclude best_node
            if time.time() > deadline - 0.5:
                return
            new_remaining_2 = remaining & ~bit
            branch_and_bound(new_remaining_2, current_size, current_set, depth + 1)

        # Start B&B from scratch with the best known solution as lower bound
        initial_remaining = (1 << N) - 1
        branch_and_bound(initial_remaining, 0, 0)

    # 6. Multiple restarts of local search
    if time.time() < deadline - 5.0:
        for restart in range(4):
            if time.time() > deadline - 2.0:
                break
            perturbed = best_set
            num_remove = max(1, best_size // 10)
            for _ in range(num_remove):
                r = perturbed
                if r == 0:
                    break
                pos = random.randint(0, N - 1)
                if perturbed & (1 << pos):
                    perturbed &= ~(1 << pos)
            perturbed_size = perturbed.bit_count()
            perturbed, perturbed_size = local_search(perturbed, perturbed_size)
            if perturbed_size > best_size:
                best_set = perturbed
                best_size = perturbed_size

    # 7. Final polish
    if time.time() < deadline - 1.0:
        for i in range(N):
            bit = 1 << i
            if not (best_set & bit) and (masks[i] & best_set) == 0:
                best_set |= bit
                best_size += 1

    # Convert best_set bitset back to list of values
    result = []
    for i in range(N):
        if best_set & (1 << i):
            result.append(candidates[i])

    return result
