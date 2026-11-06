import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85
    N = 512
    B = 512

    def lcm(a, b):
        return a // gcd(a, b) * b

    # compat[i] = bitmask of j with lcm(i,j) <= B
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if lcm(i, j) <= B:
                mask |= (1 << j)
        compat[i] = mask

    def elapsed_ok():
        return time.time() < deadline

    def greedy_fill(mask):
        """Add compatible vertices greedily (in order 1..N). Returns new mask."""
        for v in range(1, N + 1):
            if mask & (1 << v):
                continue
            if (mask & ~compat[v]) == 0:
                mask |= (1 << v)
        return mask

    def greedy_fill_order(mask, order):
        """Add compatible vertices greedily in given order. Returns new mask."""
        for v in order:
            if mask & (1 << v):
                continue
            if (mask & ~compat[v]) == 0:
                mask |= (1 << v)
        return mask

    def initial_clique(order):
        mask = 0
        for v in order:
            if (mask & ~compat[v]) == 0:
                mask |= (1 << v)
        mask = greedy_fill(mask)
        return mask

    best_mask = 0
    best_size = 0

    # Precompute degree (number of compatible neighbors)
    deg = [0] * (N + 1)
    for i in range(1, N + 1):
        deg[i] = bin(compat[i]).count('1')

    # Try several orderings
    orders = []
    orders.append(list(range(1, N + 1)))
    orders.append(list(range(N, 0, -1)))
    deg_order = sorted(range(1, N + 1), key=lambda x: deg[x], reverse=True)
    orders.append(deg_order)
    orders.append(sorted(range(1, N + 1), key=lambda x: deg[x]))

    for order in orders:
        if not elapsed_ok():
            break
        mask = initial_clique(order)
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask

    # Local search: try swaps and re-greedy-fill
    def local_search(mask):
        nonlocal best_size, best_mask
        improved = True
        while improved and elapsed_ok():
            improved = False
            cur_size = bin(mask).count('1')
            clique = [v for v in range(1, N + 1) if mask & (1 << v)]
            # Try all swaps: remove v, add w, then greedy fill
            for i in range(len(clique)):
                if not elapsed_ok():
                    break
                v = clique[i]
                base_mask = mask & ~(1 << v)
                # Find all w that can replace v
                candidates = []
                for w in range(1, N + 1):
                    if mask & (1 << w):
                        continue
                    if (base_mask & ~compat[w]) == 0:
                        candidates.append(w)
                # Sort candidates by degree (prefer high-degree)
                candidates.sort(key=lambda w: deg[w], reverse=True)
                for w in candidates[:20]:  # Limit to top 20 candidates
                    new_mask = base_mask | (1 << w)
                    new_mask = greedy_fill(new_mask)
                    new_size = bin(new_mask).count('1')
                    if new_size > cur_size:
                        mask = new_mask
                        cur_size = new_size
                        improved = True
                        break
                if improved:
                    break
            # Also try pure additions (shouldn't happen after greedy_fill, but check)
            if not improved:
                new_mask = greedy_fill(mask)
                if bin(new_mask).count('1') > cur_size:
                    mask = new_mask
                    cur_size = bin(mask).count('1')
                    improved = True
            if bin(mask).count('1') > best_size:
                best_size = bin(mask).count('1')
                best_mask = mask
        return mask

    local_search(best_mask)

    # Multiple random restarts with local search
    rng = random.Random(42)
    restarts = 0
    while elapsed_ok() and restarts < 30:
        restarts += 1
        # Random perturbation: shuffle and pick random subset to start
        order = list(range(1, N + 1))
        rng.shuffle(order)
        mask = initial_clique(order)
        size = bin(mask).count('1')
        if size >= best_size - 2:  # Only improve if close to best
            mask = local_search(mask)
            size = bin(mask).count('1')
            if size > best_size:
                best_size = size
                best_mask = mask

    result = [v for v in range(1, N + 1) if best_mask & (1 << v)]
    return result
