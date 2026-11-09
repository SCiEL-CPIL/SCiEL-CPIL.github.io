import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    N = 256

    # Precompute all 4-APs and 3-APs
    # For 4-APs: store for each element x, the list of masks (other 3 elements)
    # For 3-APs: store for each element x, the list of masks (other 2 elements)
    conflict_masks = [[] for _ in range(N + 1)]
    gain_masks = [[] for _ in range(N + 1)]
    
    # Precompute all 3-APs as tuples for later counting
    all_3aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 2):
            c = a + 2 * d
            if c > N:
                break
            b = a + d
            all_3aps.append((a, b, c))
            # x = a: needs b, c
            conflict_masks[a].append((1 << b) | (1 << c))
            # x = b: needs a, c
            conflict_masks[b].append((1 << a) | (1 << c))
            # x = c: needs a, b
            conflict_masks[c].append((1 << a) | (1 << b))

    # Precompute all 4-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 4):
            e = a + 3 * d
            if e > N:
                break
            b = a + d
            c = a + 2 * d
            # 4-AP: a, b, c, e
            # x = a: needs b, c, e
            conflict_masks[a].append((1 << b) | (1 << c) | (1 << e))
            # x = b: needs a, c, e
            conflict_masks[b].append((1 << a) | (1 << c) | (1 << e))
            # x = c: needs a, b, e
            conflict_masks[c].append((1 << a) | (1 << b) | (1 << e))
            # x = e: needs a, b, c
            conflict_masks[e].append((1 << a) | (1 << b) | (1 << c))

    def is_valid_add(x, M):
        """Check if adding x to set M (bitmask) creates a 4-AP."""
        for mask in conflict_masks[x]:
            if (M & mask) == mask:
                return False
        return True

    def count_3ap_gain(x, M):
        """Count 3-APs formed by adding x to set M."""
        gain = 0
        for mask in gain_masks[x]:
            if (M & mask) == mask:
                gain += 1
        return gain

    def count_total_3ap(M):
        """Count total 3-APs in set M using precomputed list."""
        count = 0
        for (a, b, c) in all_3aps:
            if (M & (1 << a)) and (M & (1 << b)) and (M & (1 << c)):
                count += 1
        return count

    def get_elements(M):
        """Get list of elements in bitmask M."""
        elems = []
        m = M
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            elems.append(x)
            m ^= lsb
        return elems

    best_M = 0
    best_score = 0

    # Multiple random restarts with greedy construction
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break

        M = 0
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)

        # Greedy: repeatedly add element with highest 3-AP gain that doesn't create 4-AP
        while remaining and time.time() < deadline - 15:
            best_x = -1
            best_gain = -1
            second_best_x = -1
            second_best_gain = -1

            for x in remaining:
                if is_valid_add(x, M):
                    g = count_3ap_gain(x, M)
                    if g > best_gain:
                        second_best_x = best_x
                        second_best_gain = best_gain
                        best_x = x
                        best_gain = g
                    elif g > second_best_gain:
                        second_best_x = x
                        second_best_gain = g

            if best_x == -1:
                break

            M |= (1 << best_x)
            remaining.remove(best_x)

        current_score = count_total_3ap(M)
        if current_score > best_score:
            best_score = current_score
            best_M = M

    # Local search on the best solution found
    if time.time() < deadline - 5:
        M = best_M
        current_score = best_score
        improved = True
        max_local_iters = 100

        for _ in range(max_local_iters):
            if time.time() > deadline - 3:
                break
            if not improved:
                break
            improved = False

            # Try adding any valid element
            for x in range(1, N + 1):
                if not (M & (1 << x)):
                    if is_valid_add(x, M):
                        new_M = M | (1 << x)
                        new_score = count_total_3ap(new_M)
                        if new_score > current_score:
                            M = new_M
                            current_score = new_score
                            improved = True
                            break

            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best_M = M
                continue

            # Try swaps: remove a, add b
            A = get_elements(M)
            B = [x for x in range(1, N + 1) if not (M & (1 << x))]

            for a in A:
                if time.time() > deadline - 3:
                    break
                M_temp = M & ~(1 << a)
                for b in B:
                    if is_valid_add(b, M_temp):
                        new_M = M_temp | (1 << b)
                        new_score = count_total_3ap(new_M)
                        if new_score > current_score:
                            M = new_M
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break

            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best_M = M

    # Convert bitmask to sorted list
    result = get_elements(best_M)
    return sorted(result)
