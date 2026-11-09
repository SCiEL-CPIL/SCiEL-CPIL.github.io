"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    N = 256
    # Candidate numbers: 2..256
    nums = list(range(2, N + 1))
    n = len(nums)

    # Precompute compatibility: compatible[i][j] = (lcm(nums[i], nums[j]) > 256)
    # Use a bitmask or set for fast lookup
    # For each i, store set of j where compatible
    compat = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _lcm(nums[i], nums[j]) > 256:
                compat[i].add(j)
                compat[j].add(i)

    # Precompute reciprocal values as Fractions for exact arithmetic
    # But for speed in search, use float for bounds and exact at end
    recip = [1.0 / nums[i] for i in range(n)]

    # Sort by value descending (greedy preference: take high-value items first)
    order = sorted(range(n), key=lambda i: -recip[i])
    # Re-index so that order[0] is the best, order[1] next, etc.
    # We'll search in this order
    # Rebuild compat in the new order
    pos = [0] * n
    for k, i in enumerate(order):
        pos[i] = k

    # new_compat[k] = set of positions compatible with k
    new_compat = [set() for _ in range(n)]
    for i in range(n):
        for j in compat[i]:
            new_compat[pos[i]].add(pos[j])

    new_recip = [recip[i] for i in order]
    new_nums = [nums[i] for i in order]

    # Upper bound: for a set of remaining candidates, the max possible sum
    # is the sum of all reciprocals of remaining candidates (ignoring constraints)
    # Better: precompute suffix sums of reciprocals in order
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + new_recip[i]

    best_sum = 0.0
    best_set = []

    # Branch and bound
    # State: (idx, current_sum, current_set, available_mask)
    # available_mask: which candidates from idx onward are still compatible with all chosen
    # Use a bitmask for available candidates (n <= 255, fits in Python int)

    # Actually, let's use a different approach:
    # At each step, we decide to include or exclude the current candidate.
    # The upper bound at step idx with current sum s is: s + sum of reciprocals of all
    # available candidates from idx onward.

    # For efficiency, use iterative deepening or just greedy + local search.
    # Given time constraints, let's do a smart greedy with multiple restarts.

    # Strategy 1: Greedy - pick highest value compatible item, repeat
    def greedy():
        chosen = []
        available = set(range(n))
        while available:
            # Pick the highest value item from available
            best_idx = max(available, key=lambda i: new_recip[i])
            chosen.append(best_idx)
            available.discard(best_idx)
            # Remove all incompatible items
            to_remove = set()
            for j in available:
                if j not in new_compat[best_idx]:
                    to_remove.add(j)
            available -= to_remove
        return chosen

    # Strategy 2: Greedy with randomization
    import random
    random.seed(42)

    def greedy_randomized():
        chosen = []
        available = set(range(n))
        while available:
            # Pick from top few candidates
            sorted_avail = sorted(available, key=lambda i: -new_recip[i])
            top_k = min(5, len(sorted_avail))
            candidates = sorted_avail[:top_k]
            best_idx = random.choice(candidates)
            chosen.append(best_idx)
            available.discard(best_idx)
            to_remove = set()
            for j in available:
                if j not in new_compat[best_idx]:
                    to_remove.add(j)
            available -= to_remove
        return chosen

    def evaluate(chosen):
        s = 0.0
        for i in chosen:
            s += new_recip[i]
        return s

    # Try greedy first
    best_set = greedy()
    best_sum = evaluate(best_set)

    # Try multiple randomized greedy
    for trial in range(500):
        if time.time() > deadline:
            break
        candidate = greedy_randomized()
        s = evaluate(candidate)
        if s > best_sum:
            best_sum = s
            best_set = candidate

    # Local search: try to swap or add elements
    def local_search(initial_set, max_iters=100):
        current = list(initial_set)
        current_sum = evaluate(current)
        current_set = set(current)
        available = set(range(n)) - current_set

        # Remove incompatible from available
        for i in current:
            to_remove = set()
            for j in available:
                if j not in new_compat[i]:
                    to_remove.add(j)
            available -= to_remove

        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            # Try adding one element
            best_add = None
            best_add_gain = 0
            for j in available:
                if j not in current_set:
                    gain = new_recip[j]
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add = j
            if best_add is not None:
                # Check if adding best_add is feasible (it should be since available is pre-filtered)
                current.append(best_add)
                current_set.add(best_add)
                current_sum += best_add_gain
                # Update available
                available.discard(best_add)
                to_remove = set()
                for j in available:
                    if j not in new_compat[best_add]:
                        to_remove.add(j)
                available -= to_remove
                improved = True
                continue

            # Try swapping: remove one, add one
            for i in list(current):
                # What would be available if we removed i?
                # This is complex; skip for now
                pass

        return current, current_sum

    # Apply local search to best found
    refined, refined_sum = local_search(best_set)
    if refined_sum > best_sum:
        best_sum = refined_sum
        best_set = refined

    # Try local search on some random solutions
    for trial in range(100):
        if time.time() > deadline:
            break
        candidate = greedy_randomized()
        refined, refined_sum = local_search(candidate, max_iters=50)
        if refined_sum > best_sum:
            best_sum = refined_sum
            best_set = refined

    # Convert back to original numbers
    result = sorted([new_nums[i] for i in best_set])
    return result

# EVOLVE-BLOCK-END
