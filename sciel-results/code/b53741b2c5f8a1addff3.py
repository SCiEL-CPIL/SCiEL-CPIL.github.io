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

    def initial_clique(order):
        mask = 0
        for v in order:
            if (mask & ~compat[v]) == 0:
                mask |= (1 << v)
        mask = greedy_fill(mask)
        return mask

    def get_divisors(m):
        divs = []
        for d in range(1, m + 1):
            if m % d == 0:
                divs.append(d)
        return divs

    # Strategy 1: Divisor-based cliques (deterministic, guaranteed valid)
    # Find the M in [1,512] with maximum number of divisors
    best_div_mask = 0
    best_div_size = 0
    for M in range(1, N + 1):
        divs = get_divisors(M)
        mask = 0
        for d in divs:
            mask |= (1 << d)
        size = bin(mask).count('1')
        if size > best_div_size:
            best_div_size = size
            best_div_mask = mask

    best_mask = best_div_mask
    best_size = best_div_size

    # Also try a few specific highly composite numbers explicitly
    for M in [360, 420, 480, 336, 240, 180, 120]:
        if M > N:
            continue
        divs = get_divisors(M)
        mask = 0
        for d in divs:
            mask |= (1 << d)
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask

    # Precompute degree
    deg = [0] * (N + 1)
    for i in range(1, N + 1):
        deg[i] = bin(compat[i]).count('1')

    # Strategy 2: Try several greedy orderings
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
                for w in candidates:  # Try ALL compatible candidates
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
            # Also try pure additions
            if not improved:
                new_mask = greedy_fill(mask)
                if bin(new_mask).count('1') > cur_size:
                    mask = new_mask
                    cur_size = bin(mask).count('1')
                    improved = True
            if bin(mask).count('1') > best_size:
                best_size = bin(mask).count('1')
                best_mask = mask
        # Final polish: one more pass with more aggressive search
        if elapsed_ok():
            cur_size = bin(mask).count('1')
            clique = [v for v in range(1, N + 1) if mask & (1 << v)]
            for i in range(len(clique)):
                if not elapsed_ok():
                    break
                v = clique[i]
                base_mask = mask & ~(1 << v)
                for w in range(1, N + 1):
                    if mask & (1 << w):
                        continue
                    if (base_mask & ~compat[w]) == 0:
                        new_mask = base_mask | (1 << w)
                        new_mask = greedy_fill(new_mask)
                        new_size = bin(new_mask).count('1')
                        if new_size > cur_size:
                            mask = new_mask
                            cur_size = new_size
                            break
            if bin(mask).count('1') > best_size:
                best_size = bin(mask).count('1')
                best_mask = mask
        return mask

    local_search(best_mask)

    # Multiple random restarts with local search
    rng = random.Random(42)
    restarts = 0
    while elapsed_ok() and restarts < 100:
        restarts += 1
        # Random perturbation: shuffle and pick random subset to start
        order = list(range(1, N + 1))
        rng.shuffle(order)
        mask = initial_clique(order)
        size = bin(mask).count('1')
        if size >= best_size - 3:  # Only improve if close to best
            mask = local_search(mask)
            size = bin(mask).count('1')
            if size > best_size:
                best_size = size
                best_mask = mask

    # Final verification and cleanup
    result = [v for v in range(1, N + 1) if best_mask & (1 << v)]
    # Verify the result is valid
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if lcm(result[i], result[j]) > B:
                # Fallback to divisor-based clique if invalid
                return [d for d in range(1, N+1) if best_div_mask & (1 << d)]
    return result
