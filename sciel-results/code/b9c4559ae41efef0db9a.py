"""
Size-efficient sorting networks on 18 wires.
Uses a structured local search with fast verification and greedy removal.
"""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

# Precompute sorted mask LUT: sorted mask has all 1s then all 0s (or all 0s)
# For N bits, sorted means there exists k such that bits N-1..N-k are 1 and rest 0.
# Actually, sorted means non-decreasing from wire 0 to wire N-1? 
# No: min goes to wire i, max to j. So after sorting, wire 0 has the min, wire N-1 has the max.
# So sorted means bit 0 <= bit 1 <= ... <= bit N-1.
# In binary, that means all 0s then all 1s.
# E.g., for N=3: 000, 001, 011, 111 are sorted.
_SORTED_LUT = np.zeros(1 << N, dtype=np.bool_)
for k in range(N + 1):
    # k ones at the end
    mask = (1 << k) - 1
    _SORTED_LUT[mask] = True


def _verify_full(comparators, n=N):
    """Full verification over all 2^n inputs using numpy."""
    if not comparators:
        return n <= 1
    total = 1 << n
    inputs = np.arange(total, dtype=np.uint32)
    for i, j in comparators:
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        swap_mask = (bit_i == 0) & (bit_j == 1)
        toggle = (1 << i) | (1 << j)
        inputs = np.where(swap_mask, inputs ^ toggle, inputs)
    if not np.all(_SORTED_LUT[inputs]):
        return False
    return True


def _verify(comparators, n=N):
    """Verify a network: full check with numpy LUT."""
    if not comparators:
        return n <= 1
    return _verify_full(comparators, n)


def _odd_even_transposition(n=N):
    """Generate odd-even transposition sorting network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators


def _bitonic_network(n):
    """Build a Batcher bitonic sorting network for n wires (n power of 2)."""
    def bitonic_merge(comps, lo, hi, direction):
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        bitonic_merge(comps, lo, lo + half, direction)
        bitonic_merge(comps, lo + half, hi, direction)
        step = half
        while step >= 1:
            for i in range(lo, lo + half, step):
                a = i
                b = i + step
                if b < hi:
                    if direction:
                        comps.append((a, b))
                    else:
                        comps.append((b, a))
            step //= 2

    def bitonic_sort(comps, lo, hi, direction):
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        bitonic_sort(comps, lo, lo + half, True)
        bitonic_sort(comps, lo + half, hi, False)
        bitonic_merge(comps, lo, hi, direction)

    comps = []
    # For non-power-of-2, we can pad to next power of 2 and take the first n wires
    # But simpler: use odd-even transposition as seed
    bitonic_sort(comps, 0, n, True)
    return comps


def _seed_network(n=N):
    """Generate a good seed network."""
    # Try bitonic for power of 2, else odd-even transposition
    if n & (n - 1) == 0:
        net = _bitonic_network(n)
        if _verify(net):
            return net
    return _odd_even_transposition(n)


def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Seed with odd-even transposition (153 comparators for N=18)
    seed = _seed_network(N)
    if not _verify(seed):
        seed = _odd_even_transposition(N)

    best = list(seed)
    best_len = len(best)

    if time.time() > deadline:
        return best

    random.seed(42)
    current = list(best)

    # Greedy removal: try to remove each comparator one by one
    def greedy_remove(net, budget_iters=500):
        changed = True
        iters = 0
        while changed and iters < budget_iters:
            iters += 1
            changed = False
            if time.time() > deadline:
                return net
            for idx in range(len(net)):
                if time.time() > deadline:
                    return net
                cand = net[:idx] + net[idx + 1:]
                if _verify(cand):
                    net = cand
                    changed = True
                    break
        return net

    current = greedy_remove(current)
    if len(current) < best_len:
        best = list(current)
        best_len = len(current)

    # Local search with multiple move types
    max_iterations = 100000
    consecutive_failures = 0
    max_consecutive_failures = 2000

    # Precompute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
    num_all_comps = len(all_comps)

    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if len(current) <= 70:
            break
        if consecutive_failures > max_consecutive_failures:
            random.seed(random.randint(0, 2 ** 32 - 1))
            consecutive_failures = 0
            current = list(best)

        mod_type = random.random()

        if mod_type < 0.50 and len(current) > 1:
            # Remove a random comparator
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx + 1:]
            if _verify(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
                    # Cascade removal
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        if len(current) <= 70:
                            break
                        idx2 = random.randint(0, len(current) - 1)
                        candidate2 = current[:idx2] + current[idx2 + 1:]
                        if _verify(candidate2):
                            current = candidate2
                            if len(current) < best_len:
                                best = list(current)
                                best_len = len(current)
                        else:
                            break
            else:
                consecutive_failures += 1

        elif mod_type < 0.65:
            # Add a random comparator
            if len(current) >= 4095:
                consecutive_failures += 1
                continue
            a, b = all_comps[random.randint(0, num_all_comps - 1)]
            # Insert at random position
            pos = random.randint(0, len(current))
            candidate = current[:pos] + [(a, b)] + current[pos:]
            if _verify(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1

        elif mod_type < 0.80:
            # Swap two comparators
            if len(current) < 2:
                continue
            i, j = random.sample(range(len(current)), 2)
            candidate = list(current)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _verify(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1

        elif mod_type < 0.90:
            # Replace a comparator with a different one
            if len(current) == 0:
                continue
            idx = random.randint(0, len(current) - 1)
            a, b = all_comps[random.randint(0, num_all_comps - 1)]
            candidate = list(current)
            candidate[idx] = (a, b)
            if _verify(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1

        else:
            # Batch removal
            if len(current) < 3:
                continue
            num_remove = random.randint(2, 3)
            if len(current) <= num_remove:
                continue
            remove_indices = set(random.sample(range(len(current)), num_remove))
            candidate = [c for idx, c in enumerate(current) if idx not in remove_indices]
            if _verify(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1

    return best

# EVOLVE-BLOCK-END
