import time
import random
import math
from fractions import Fraction

N = 256
TIME_LIMIT = 50.0  # Safety margin before 60s kill

# Precompute AP pairs for each x
# ap_pairs[x] is a list of tuples (a, b) such that {x, a, b} is a 3-AP
ap_pairs = [[] for _ in range(N + 1)]

def precompute_ap_pairs():
    for x in range(1, N + 1):
        pairs = set()
        for d in range(1, (N - x) // 2 + 1):
            y = x + d
            z = x + 2 * d
            if y <= N and z <= N:
                pairs.add((y, z))
        for d in range(1, min(x - 1, N - x) + 1):
            y = x - d
            z = x + d
            if y >= 1 and z <= N:
                pairs.add((y, z))
        for d in range(1, x // 2 + 1):
            y = x - 2 * d
            z = x - d
            if y >= 1 and z <= N:
                pairs.add((y, z))
        ap_pairs[x] = list(pairs)

precompute_ap_pairs()

def can_add(bitset, x):
    for a, b in ap_pairs[x]:
        if (bitset & (1 << (a - 1))) and (bitset & (1 << (b - 1))):
            return False
    return True

def greedy_construct(order):
    bitset = 0
    score = Fraction(0, 1)
    for x in order:
        if can_add(bitset, x):
            bitset |= (1 << (x - 1))
            score += Fraction(1, x)
    return bitset, score

def bitset_to_list(bitset):
    return [x for x in range(1, N + 1) if bitset & (1 << (x - 1))]

def local_search_systematic(bitset, deadline):
    """
    Systematic local search using bitset. At each step, try ALL single-element
    removals and for each, find the BEST valid addition (smallest x = largest reciprocal).
    Accept if it improves. This is the "remove-and-best-add" refinement.
    """
    current = bitset
    current_score = Fraction(0, 1)
    for x in range(1, N + 1):
        if current & (1 << (x - 1)):
            current_score += Fraction(1, x)

    best = current
    best_score = current_score

    while time.time() < deadline:
        improved = False
        # Try all removals; for each, find best valid addition
        for removed in range(1, N + 1):
            if not (current & (1 << (removed - 1))):
                continue
            trial = current & ~(1 << (removed - 1))
            # Find the best (smallest x) valid addition
            best_add = None
            for x in range(1, N + 1):
                if x == removed:
                    continue
                if trial & (1 << (x - 1)):
                    continue
                if can_add(trial, x):
                    best_add = x
                    break
            if best_add is None:
                continue
            delta = Fraction(1, best_add) - Fraction(1, removed)
            if delta > 0:
                current = trial | (1 << (best_add - 1))
                current_score += delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best = current
                break  # restart scan after applying one improvement

        if not improved:
            break

    return best, best_score

def polish(bitset):
    """Final polish: add any remaining numbers that can be added (in increasing order)."""
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if not (bitset & (1 << (x - 1))):
                if can_add(bitset, x):
                    bitset |= (1 << (x - 1))
                    improved = True
    return bitset

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # Multiple greedy strategies for diverse starting points
    strategies = []
    strategies.append(list(range(1, N + 1)))  # smallest first
    strategies.append(list(range(N, 0, -1)))  # largest first

    rng = random.Random(0)
    for trial in range(15):
        order = list(range(1, N + 1))
        for i in range(len(order) - 1, 0, -1):
            if rng.random() < 0.3:
                j = max(0, i - rng.randint(1, 5))
                order[i], order[j] = order[j], order[i]
        strategies.append(order)

    # Run all strategies, keep best
    best_bitset = 0
    best_score = Fraction(0, 1)
    for order in strategies:
        if time.time() > deadline - 15:
            break
        bitset, score = greedy_construct(order)
        if score > best_score:
            best_score = score
            best_bitset = bitset

    # Systematic local search (remove-and-best-add) on the best solution
    best_bitset, best_score = local_search_systematic(best_bitset, deadline - 2)

    # Final polish
    best_bitset = polish(best_bitset)

    return bitset_to_list(best_bitset)

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0 / x for x in res))
