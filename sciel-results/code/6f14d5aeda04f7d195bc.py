import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 128
    full_mask = (1 << N) - 1

    # Precompute LCMs and conflict bitsets
    lcms = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcms[i][j] = lcm(i, j)
            lcms[j][i] = lcms[i][j]

    # conflict[a][b] is a bitmask of c such that {a,b,c} is a bad triple
    conflict = [[0] * (N + 1) for _ in range(N + 1)]

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcms[a][b]
            for c in range(b + 1, N + 1):
                if lcms[a][c] == lab and lcms[b][c] == lab:
                    conflict[a][b] |= (1 << (c - 1))
                    conflict[b][a] |= (1 << (c - 1))
                    conflict[a][c] |= (1 << (b - 1))
                    conflict[c][a] |= (1 << (b - 1))
                    conflict[b][c] |= (1 << (a - 1))
                    conflict[c][b] |= (1 << (a - 1))

    # Precompute degree for each vertex
    degree = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if conflict[i][j]:
                degree[i] += 1
                degree[j] += 1

    def can_add(v, selected_mask):
        """Check if adding v to selected_mask is valid."""
        temp = selected_mask
        while temp:
            lsb = temp & -temp
            u_idx = lsb.bit_length() - 1
            u = u_idx + 1
            if conflict[u][v] & selected_mask:
                return False
            temp ^= lsb
        return True

    def greedy_select(order):
        """Greedy selection in given order."""
        selected_mask = 0
        for v in order:
            if time.time() > deadline:
                break
            if can_add(v, selected_mask):
                selected_mask |= (1 << (v - 1))
        return selected_mask

    def mask_to_list(mask):
        res = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            res.append(idx + 1)
            m ^= lsb
        return res

    # Phase 1: Get a good initial solution via multiple greedy strategies
    best_mask = 0
    best_size = 0

    # Ascending
    m = greedy_select(range(1, N + 1))
    best_size = len(mask_to_list(m))
    best_mask = m

    # Descending
    m = greedy_select(range(N, 0, -1))
    if len(mask_to_list(m)) > best_size:
        best_size = len(mask_to_list(m))
        best_mask = m

    # Degree-based ordering (prefer vertices with fewer conflicts)
    order = sorted(range(1, N + 1), key=lambda x: degree[x])
    m = greedy_select(order)
    if len(mask_to_list(m)) > best_size:
        best_size = len(mask_to_list(m))
        best_mask = m

    # Reverse degree-based
    order = sorted(range(1, N + 1), key=lambda x: -degree[x])
    m = greedy_select(order)
    if len(mask_to_list(m)) > best_size:
        best_size = len(mask_to_list(m))
        best_mask = m

    # Randomized restarts with different heuristics
    random.seed(42)
    for restart in range(300):
        if time.time() > deadline - 15:
            break
        # Mix of random and degree-based
        if restart % 3 == 0:
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif restart % 3 == 1:
            order = sorted(range(1, N + 1), key=lambda x: (degree[x], random.random()))
        else:
            order = sorted(range(1, N + 1), key=lambda x: (-degree[x], random.random()))
        m = greedy_select(order)
        sz = len(mask_to_list(m))
        if sz > best_size:
            best_size = sz
            best_mask = m

    # Phase 2: Local search - try to add more elements
    def try_fill(mask):
        """Try to add any remaining elements that can be added."""
        changed = True
        while changed:
            changed = False
            remaining = full_mask & ~mask
            temp = remaining
            while temp:
                lsb = temp & -temp
                v_idx = lsb.bit_length() - 1
                v = v_idx + 1
                if can_add(v, mask):
                    mask |= lsb
                    changed = True
                    break
                temp ^= lsb
        return mask

    best_mask = try_fill(best_mask)
    best_size = len(mask_to_list(best_mask))

    # Phase 3: Local search with 1-swap moves
    def local_search_1swap(mask, max_iter=500):
        """Improve mask using 1-swap moves."""
        current_mask = mask
        current_list = mask_to_list(current_mask)
        current_set = set(current_list)
        improved = True
        iter_count = 0

        while improved and iter_count < max_iter and time.time() < deadline - 5:
            improved = False
            iter_count += 1
            current_list = mask_to_list(current_mask)
            current_set = set(current_list)

            for u in current_list:
                if time.time() > deadline - 5:
                    break
                mask_without_u = current_mask & ~(1 << (u - 1))
                remaining = full_mask & ~mask_without_u
                temp = remaining
                while temp:
                    lsb = temp & -temp
                    v_idx = lsb.bit_length() - 1
                    v = v_idx + 1
                    if v not in current_set:
                        if can_add(v, mask_without_u):
                            new_mask = mask_without_u | lsb
                            # Try to fill
                            new_mask = try_fill(new_mask)
                            new_list = mask_to_list(new_mask)
                            if len(new_list) > len(current_list):
                                current_mask = new_mask
                                improved = True
                                break
                    temp ^= lsb
                if improved:
                    break

        return current_mask

    if time.time() < deadline - 10:
        best_mask = local_search_1swap(best_mask)
        best_size = len(mask_to_list(best_mask))

    # Final fill
    best_mask = try_fill(best_mask)
    best_size = len(mask_to_list(best_mask))

    return mask_to_list(best_mask)
