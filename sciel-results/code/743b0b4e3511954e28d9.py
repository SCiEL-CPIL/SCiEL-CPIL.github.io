import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s wall-clock

    n = 256
    # Precompute perfect squares up to 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1

    # Diagonal constraint: 2a must not be a perfect square
    # 2a = k^2 => a = k^2/2, k even, a = 2m^2
    diagonal_bad = set()
    for m in range(1, 12):
        a = 2 * m * m
        if 1 <= a <= n:
            diagonal_bad.add(a)

    # Valid candidates (pass diagonal test)
    candidates = [x for x in range(1, n + 1) if x not in diagonal_bad]
    m = len(candidates)

    # Map value -> index
    val_to_idx = {v: i for i, v in enumerate(candidates)}

    # Build bitmask adjacency
    # adj[i] is a bitmask of all j such that candidates[i] + candidates[j] is a perfect square
    adj = [0] * m
    for i in range(m):
        ai = candidates[i]
        mask = 0
        for j in range(m):
            if ai + candidates[j] in squares:
                mask |= (1 << j)
        adj[i] = mask

    # A vertex is "independent" from a mask if (adj[i] & mask) == 0
    # We want to find the largest independent set

    def can_add(idx, mask):
        """Check if vertex idx can be added to the set represented by mask."""
        return (adj[idx] & mask) == 0

    # Greedy initialization: process by increasing degree
    def greedy_init():
        # Count degrees
        deg = [bin(adj[i]).count('1') for i in range(m)]
        order = sorted(range(m), key=lambda i: (deg[i], candidates[i]))

        mask = 0
        for i in order:
            if can_add(i, mask):
                mask |= (1 << i)
        return mask

    def mask_to_set(mask):
        return [candidates[i] for i in range(m) if (mask >> i) & 1]

    def set_size(mask):
        return bin(mask).count('1')

    # Local search: 1-move (add a vertex) and 2-swap (remove one, add another)
    def local_search(mask, max_iters=5000):
        current = mask
        current_size = set_size(current)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 3:
                break

            # Try 1-move: add any vertex not in current
            for i in range(m):
                if not (current >> i) & 1:
                    if can_add(i, current):
                        current |= (1 << i)
                        current_size += 1
                        improved = True

            if not improved:
                # Try 2-swap: remove vertex i, add vertex j
                # For each vertex i in current, for each vertex j not in current:
                # Check if (current - i + j) is valid
                in_set = [i for i in range(m) if (current >> i) & 1]
                out_set = [i for i in range(m) if not (current >> i) & 1]

                for i in in_set:
                    mask_without_i = current & ~(1 << i)
                    for j in out_set:
                        if can_add(j, mask_without_i):
                            new_mask = mask_without_i | (1 << j)
                            # Check size: same size, so only do swap if it enables future additions
                            # Actually, do the swap and see if we can grow
                            # But for simplicity, accept any swap that doesn't reduce size
                            # and then try to add more
                            current = new_mask
                            improved = True
                            # Try to add more after swap
                            for k in range(m):
                                if not (current >> k) & 1:
                                    if can_add(k, current):
                                        current |= (1 << k)
                                        current_size += 1
                            break
                    if improved:
                        current_size = set_size(current)
                        break

        return current, set_size(current)

    # Run greedy init
    best_mask = greedy_init()
    best_size = set_size(best_mask)

    # Local search from greedy
    improved_mask, improved_size = local_search(best_mask)
    if improved_size > best_size:
        best_mask = improved_mask
        best_size = improved_size

    # Multiple random restarts
    random.seed(12345)
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1

        # Randomized greedy: shuffle order, process with some randomness
        order = list(range(m))
        random.shuffle(order)

        mask = 0
        for i in order:
            if can_add(i, mask):
                mask |= (1 << i)

        # Local search
        improved_mask, improved_size = local_search(mask, max_iters=2000)
        if improved_size > best_size:
            best_mask = improved_mask
            best_size = improved_size

    # Final polish: try to add more vertices
    for i in range(m):
        if not (best_mask >> i) & 1:
            if can_add(i, best_mask):
                best_mask |= (1 << i)
                best_size += 1

    result = mask_to_set(best_mask)
    if not result:
        return [1]
    return sorted(result)
