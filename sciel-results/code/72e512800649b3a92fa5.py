import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 105.0

    # Clebsch graph: vertices are 5-bit even-parity vectors
    vectors = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vectors.append(i)

    adj = [0] * 16
    clebsch_edges = []
    for u_idx in range(16):
        for v_idx in range(u_idx + 1, 16):
            xor = vectors[u_idx] ^ vectors[v_idx]
            if bin(xor).count('1') == 4:
                clebsch_edges.append([u_idx, v_idx])
                adj[u_idx] |= (1 << v_idx)
                adj[v_idx] |= (1 << u_idx)

    best_edges = clebsch_edges[:]
    best_count = 40

    # Fast 3-colorability with memoization on (color_masks_tuple, next_vertex_index)
    # We order vertices by descending degree for better pruning
    def make_checker(adj_list):
        n = 16
        degrees = [bin(adj_list[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        memo = {}

        def check3(adj_l, order, memo_store):
            # Returns True if 3-colorable
            color_masks = [0, 0, 0]
            uncolored = (1 << n) - 1

            def bt(uncolored_mask, cm0, cm1, cm2):
                if uncolored_mask == 0:
                    return True
                key = (uncolored_mask, cm0, cm1, cm2)
                if key in memo_store:
                    return memo_store[key]

                # Pick vertex with most constraints (highest degree in remaining)
                best_v = -1
                best_score = -1
                temp_mask = uncolored_mask
                while temp_mask:
                    lb = temp_mask & (-temp_mask)
                    v = lb.bit_length() - 1
                    temp_mask ^= lb
                    # Count how many colors are forbidden for v
                    nb = adj_l[v]
                    forb = 0
                    if cm0 & nb: forb += 1
                    if cm1 & nb: forb += 1
                    if cm2 & nb: forb += 1
                    score = forb * 100 + bin(adj_l[v] & uncolored_mask).count('1')
                    if score > best_score:
                        best_score = score
                        best_v = v

                v = best_v
                nb = adj_l[v]
                bit = 1 << v
                new_unc = uncolored_mask & ~bit

                # Try color 0
                if not (cm0 & nb):
                    if bt(new_unc, cm0 | bit, cm1, cm2):
                        memo_store[key] = True
                        return True
                # Try color 1
                if not (cm1 & nb):
                    if bt(new_unc, cm0, cm1 | bit, cm2):
                        memo_store[key] = True
                        return True
                # Try color 2
                if not (cm2 & nb):
                    if bt(new_unc, cm0, cm1, cm2 | bit):
                        memo_store[key] = True
                        return True

                memo_store[key] = False
                return False

            return bt((1 << n) - 1, 0, 0, 0)

        return check3

    checker = make_checker(adj)

    def is_3_colorable_fast(adj_mask):
        # Quick check: if max degree < 3, likely 3-colorable
        # Use the checker
        local_degrees = [bin(adj_mask[i]).count('1') for i in range(16)]
        order = sorted(range(16), key=lambda x: -local_degrees[x])
        memo = {}
        n = 16

        def bt(uncolored_mask, cm0, cm1, cm2):
            if uncolored_mask == 0:
                return True
            key = (uncolored_mask, cm0, cm1, cm2)
            if key in memo:
                return memo[key]

            best_v = -1
            best_score = -1
            temp = uncolored_mask
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                temp ^= lb
                nb = adj_mask[v]
                forb = (1 if cm0 & nb else 0) + (1 if cm1 & nb else 0) + (1 if cm2 & nb else 0)
                score = forb * 100 + bin(nb & uncolored_mask).count('1')
                if score > best_score:
                    best_score = score
                    best_v = v

            v = best_v
            nb = adj_mask[v]
            bit = 1 << v
            new_unc = uncolored_mask & ~bit

            if not (cm0 & nb):
                if bt(new_unc, cm0 | bit, cm1, cm2):
                    memo[key] = True
                    return True
            if not (cm1 & nb):
                if bt(new_unc, cm0, cm1 | bit, cm2):
                    memo[key] = True
                    return True
            if not (cm2 & nb):
                if bt(new_unc, cm0, cm1, cm2 | bit):
                    memo[key] = True
                    return True

            memo[key] = False
            return False

        return bt((1 << 16) - 1, 0, 0, 0)

    def can_add_edge(adj_mask, u, v):
        return (adj_mask[u] & adj_mask[v]) == 0

    # Local search: try adding edges to improve
    def local_search(adj_mask, current_edges, current_count):
        nonlocal best_edges, best_count
        n = 16
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            available = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj_mask[u] & (1 << v)):
                        if can_add_edge(adj_mask, u, v):
                            available.append((u, v))

            for u, v in available:
                if time.time() > deadline:
                    return
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                if not is_3_colorable_fast(adj_mask):
                    current_edges.append([u, v])
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = current_edges[:]
                    improved = True
                    break
                else:
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)

    # Phase 1: Local search from Clebsch
    local_search(adj[:], best_edges[:], best_count)

    # Phase 2: Randomized edge addition with backtracking
    all_pairs = [(u, v) for u in range(16) for v in range(u + 1, 16)]
    clebsch_set = set((min(u, v), max(u, v)) for u, v in clebsch_edges)
    candidate_edges = [(u, v) for u, v in all_pairs if (u, v) not in clebsch_set]

    # Phase 3: Branch and bound - try to find denser 4-chromatic triangle-free graphs
    def branch_and_bound(adj_mask, remaining_pairs, depth, current_count, target_count):
        nonlocal best_edges, best_count
        if time.time() > deadline:
            return
        if current_count >= best_count:
            # Check if 4-chromatic
            if not is_3_colorable_fast(adj_mask):
                if current_count > best_count:
                    best_count = current_count
                    best_edges = [[u, v] for u in range(16) for v in range(u + 1, 16)
                                   if adj_mask[u] & (1 << v)]

        if not remaining_pairs:
            return

        # Upper bound: current + remaining
        if current_count + len(remaining_pairs) <= best_count:
            return

        # Try adding first few edges
        for i in range(min(3, len(remaining_pairs))):
            u, v = remaining_pairs[i]
            if not (adj_mask[u] & (1 << v)):
                if can_add_edge(adj_mask, u, v):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    branch_and_bound(adj_mask, remaining_pairs[:i] + remaining_pairs[i+1:],
                                     depth + 1, current_count + 1, target_count)
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
            # Also try skipping
            branch_and_bound(adj_mask, remaining_pairs[:i] + remaining_pairs[i+1:],
                             depth + 1, current_count, target_count)

    # Phase 4: Systematic search - try all subsets of size > 40 from candidate edges
    # This is too large, so use randomized approach
    random.seed(42)
    for iteration in range(500):
        if time.time() > deadline:
            break

        # Start with Clebsch and add random edges
        adj_mask = adj[:]
        edges = best_edges[:]
        count = best_count

        # Add random candidate edges
        shuffled = candidate_edges[:]
        random.shuffle(shuffled)
        for u, v in shuffled[:20]:
            if time.time() > deadline:
                break
            if not (adj_mask[u] & (1 << v)):
                if can_add_edge(adj_mask, u, v):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges.append([u, v])
                    count += 1

        if count > best_count:
            if not is_3_colorable_fast(adj_mask):
                best_count = count
                best_edges = edges[:]
                local_search(adj_mask[:], edges[:], count)

    return best_edges
