"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Precompute all 4-term arithmetic progressions (0-based indices)
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))

    total_aps = len(aps)

    # Precompute for each index, which APs it belongs to
    index_aps = [[] for _ in range(N)]
    for ap_id, (i1, i2, i3, i4) in enumerate(aps):
        for idx in (i1, i2, i3, i4):
            index_aps[idx].append(ap_id)

    def count_mono(coloring):
        """Count monochromatic 4-APs."""
        count = 0
        for (i1, i2, i3, i4) in aps:
            if coloring[i1] == coloring[i2] == coloring[i3] == coloring[i4]:
                count += 1
        return count

    def get_delta(coloring, idx):
        """Get the change in monochromatic 4-AP count if we flip coloring[idx]."""
        delta = 0
        for ap_id in index_aps[idx]:
            (i1, i2, i3, i4) = aps[ap_id]
            other = [i for i in (i1, i2, i3, i4) if i != idx]
            # Check if the other 3 are same color
            if coloring[other[0]] == coloring[other[1]] == coloring[other[2]]:
                if coloring[idx] == coloring[other[0]]:
                    delta -= 1  # was mono, won't be after flip
                else:
                    delta += 1  # wasn't mono, will be after flip
        return delta

    # Initial coloring: random
    best_coloring = [random.randint(0, 1) for _ in range(N)]
    best_count = count_mono(best_coloring)

    # Local search: try flipping each bit, accept if it improves
    improved = True
    while improved:
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            delta = get_delta(best_coloring, idx)
            if delta < 0:
                best_coloring[idx] ^= 1
                best_count += delta
                improved = True
        if time.time() > deadline:
            break

    # Try multiple random restarts if time permits
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        coloring = [random.randint(0, 1) for _ in range(N)]
        cur_count = count_mono(coloring)

        # Local search from this starting point
        improved = True
        while improved:
            improved = False
            for idx in range(N):
                if time.time() > deadline:
                    break
                delta = get_delta(coloring, idx)
                if delta < 0:
                    coloring[idx] ^= 1
                    cur_count += delta
                    improved = True
            if time.time() > deadline:
                break

        if cur_count < best_count:
            best_coloring = coloring[:]
            best_count = cur_count

    return best_coloring
# EVOLVE-BLOCK-END
