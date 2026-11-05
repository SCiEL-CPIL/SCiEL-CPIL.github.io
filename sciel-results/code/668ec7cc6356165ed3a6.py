import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit

    N = 256
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency as bitsets
    # adj[i] is a bitmask of vertices j such that i+j is square
    adj = [0] * N
    forbidden = [False] * N

    for i in range(N):
        v = i + 1
        # Check self-loop
        if 2 * v in squares:
            forbidden[i] = True
        else:
            for sq in squares:
                u = sq - v
                if 1 <= u <= N:
                    j = u - 1
                    if not forbidden[j]:
                        adj[i] |= (1 << j)

    # List of valid candidates (0-indexed)
    candidates = [i for i in range(N) if not forbidden[i]]
    num_candidates = len(candidates)

    # Precompute candidate bitmasks for quick intersection
    cand_mask = 0
    for i in candidates:
        cand_mask |= (1 << i)

    def greedy_init():
        # Randomized greedy: pick min degree among available
        available_mask = cand_mask
        selected_mask = 0

        while available_mask:
            best_v = -1
            min_deg = N + 1
            m = available_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = (adj[v] & available_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break

            if best_v == -1:
                break

            selected_mask |= (1 << best_v)
            available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))

        return selected_mask

    def local_improve(mask):
        # Try to add any vertex
        improved = True
        while improved:
            improved = False
            not_in_mask = cand_mask & ~mask
            m = not_in_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if (adj[v] & mask) == 0:
                    mask |= lsb
                    improved = True
                    break
        return mask

    def local_search_swap(mask, max_iters=100):
        # 1-1 Swap and 1-2 Swap moves
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break

            mask = local_improve(mask)
            if not mask:
                break

            improved = False
            m = mask
            while m:
                lsb_u = m & -m
                u = lsb_u.bit_length() - 1
                m ^= lsb_u

                mask_wo_u = mask ^ lsb_u

                # Try 1-2 swap: remove u, add v1, v2
                # v1, v2 must be in candidates, not in mask_wo_u
                # and not conflict with mask_wo_u or each other
                # This is expensive, so we limit the search

                # First try simple 1-1 swap: remove u, add v
                # v must be in adj[u] \ mask, and (adj[v] & mask) == lsb_u
                potential_v = adj[u] & ~mask & cand_mask
                pv = potential_v
                while pv:
                    lsb_v = pv & -pv
                    v = lsb_v.bit_length() - 1
                    pv ^= lsb_v
                    if (adj[v] & mask) == lsb_u:
                        # Try the swap
                        new_mask = mask_wo_u | lsb_v
                        new_mask = local_improve(new_mask)
                        if new_mask.bit_count() > mask.bit_count():
                            mask = new_mask
                            improved = True
                            break
                if improved:
                    break

            if not improved:
                break

        return mask

    def local_search_12swap(mask, max_iters=20):
        """Try 1-2 swaps: remove one vertex, add two vertices."""
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break

            mask = local_improve(mask)
            if not mask:
                break

            improved = False
            m = mask
            while m and not improved:
                lsb_u = m & -m
                u = lsb_u.bit_length() - 1
                m ^= lsb_u

                mask_wo_u = mask ^ lsb_u

                # Find vertices that can be added to mask_wo_u
                # v must not conflict with mask_wo_u
                addable = cand_mask & ~mask_wo_u
                # v must not conflict with mask_wo_u
                av = addable
                addable_list = []
                while av:
                    lsb_v = av & -av
                    v = lsb_v.bit_length() - 1
                    av ^= lsb_v
                    if (adj[v] & mask_wo_u) == 0:
                        addable_list.append((v, lsb_v))

                # Try pairs from addable_list
                for i in range(len(addable_list)):
                    v1, lsb_v1 = addable_list[i]
                    for j in range(i + 1, len(addable_list)):
                        v2, lsb_v2 = addable_list[j]
                        # v1 and v2 must not conflict with each other
                        if (adj[v1] & lsb_v2) == 0 and (adj[v2] & lsb_v1) == 0:
                            new_mask = mask_wo_u | lsb_v1 | lsb_v2
                            new_mask = local_improve(new_mask)
                            if new_mask.bit_count() > mask.bit_count():
                                mask = new_mask
                                improved = True
                                break
                    if improved:
                        break

            if not improved:
                break

        return mask

    # Main Loop: Iterated Local Search
    best_mask = 0
    best_size = 0

    # Initial greedy
    mask = greedy_init()
    mask = local_improve(mask)
    best_mask = mask
    best_size = mask.bit_count()

    # ILS
    while time.time() - start_time < DEADLINE:
        # Perturb best_mask
        current = best_mask
        # Remove random vertices
        for _ in range(random.randint(1, 4)):
            if not current:
                break
            while True:
                r = random.randint(0, 255)
                if current & (1 << r):
                    current ^= (1 << r)
                    break

        # Now improve current
        current = local_improve(current)
        current = local_search_swap(current, max_iters=50)
        # Try 1-2 swaps as well
        current = local_search_12swap(current, max_iters=10)

        if current.bit_count() > best_size:
            best_size = current.bit_count()
            best_mask = current

        # Also try a fresh greedy with random weighting
        if random.random() < 0.4:
            def greedy_random():
                available_mask = cand_mask
                selected_mask = 0
                while available_mask:
                    best_v = -1
                    min_deg = N + 1
                    m = available_mask
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        m ^= lsb
                        deg = (adj[v] & available_mask).bit_count()
                        deg += random.uniform(0, 0.5)
                        if deg < min_deg:
                            min_deg = deg
                            best_v = v
                        if deg == 0:
                            break

                    if best_v == -1:
                        break
                    selected_mask |= (1 << best_v)
                    available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))
                return selected_mask

            mask_r = greedy_random()
            mask_r = local_improve(mask_r)
            mask_r = local_search_swap(mask_r, max_iters=50)
            mask_r = local_search_12swap(mask_r, max_iters=10)
            if mask_r.bit_count() > best_size:
                best_size = mask_r.bit_count()
                best_mask = mask_r

    # Convert best_mask to list of values (1-based)
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v + 1)
        m ^= lsb

    return sorted(result)
