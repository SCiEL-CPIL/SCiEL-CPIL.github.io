import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin for 90s wall-clock

    n = 256

    # Precompute conflict neighbors: for each x, list of (y, z) such that {x,y,z} is a 3-AP
    # and x is the "middle" or one endpoint. We need: adding x to S is valid iff
    # for every (y,z) in conflict_pairs[x], not both y and z are in S.
    conflict_pairs = {i: [] for i in range(1, n + 1)}
    for b in range(2, n):
        for a in range(1, b):
            c = 2 * b - a
            if c <= n:
                conflict_pairs[a].append((b, c))
                conflict_pairs[b].append((a, c))
                conflict_pairs[c].append((a, b))

    inv = [0.0] + [1.0 / i for i in range(1, n + 1)]

    # Bitmask helpers
    def bit(x):
        return 1 << (x - 1)

    def can_add(x, M):
        for y, z in conflict_pairs[x]:
            if (M & bit(y)) and (M & bit(z)):
                return False
        return True

    def get_elements(M):
        return [i for i in range(1, n + 1) if M & bit(i)]

    def sum_recip(M):
        s = 0.0
        for i in range(1, n + 1):
            if M & bit(i):
                s += inv[i]
        return s

    # ---------- Branch and Bound ----------
    # State: (M, candidates, lower_bound, upper_bound)
    # We process candidates in order of decreasing 1/x (i.e., ascending x) to maximize early gains.
    # Upper bound: current_sum + sum of 1/x for all remaining candidates that are individually
    # compatible with M (i.e., can_add(x, M) is True). This is a valid upper bound because
    # any feasible completion can only include a subset of those candidates.

    # Pre-sort candidates: we'll always consider adding in order 1, 2, ..., n
    # But for B&B, we branch on whether to include or exclude each candidate.
    # To get tight bounds quickly, we include small numbers first.

    # We'll do a DFS B&B:
    #   Node: (current_mask, current_sum, index)
    #   At index i, candidate x = i.
    #   Branch 1: include x (if can_add(x, current_mask)) -> new_sum = current_sum + 1/x
    #   Branch 2: exclude x
    #   Upper bound at node: current_sum + sum_{j >= i, can_add(j, current_mask)} 1/j
    #   Prune if upper_bound <= best_found

    # To make the upper bound fast, we precompute prefix sums of 1/x.
    # But the bound depends on which candidates are compatible with current_mask,
    # so we need to compute it dynamically. For n=256, this is O(n) per node.
    # With good pruning, the number of nodes should be manageable.

    # Better: use a "suffix" bound. Precompute suffix_sum[i] = sum_{j=i}^{n} 1/j.
    # The upper bound is current_sum + suffix_sum[i] (ignoring conflicts), which is
    # looser but much faster. Let's use this as a quick filter, and if it passes,
    # compute the tighter bound.

    suffix_sum = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + inv[i]

    best_mask = 0
    best_score = 0.0
    best_elements = [1]  # fallback

    # Initialize with a good greedy solution to set a strong lower bound
    M_greedy = 0
    for x in range(1, n + 1):
        if can_add(x, M_greedy):
            M_greedy |= bit(x)
    best_mask = M_greedy
    best_score = sum_recip(M_greedy)
    best_elements = get_elements(best_mask)

    # DFS B&B
    # Stack: (mask, current_sum, index)
    # We process candidates in order 1..n.
    # At each step, we try to include the candidate first (to find high-sum solutions early).

    stack = [(M_greedy, best_score, 1)]  # Start from the greedy solution, index 1
    # Actually, we should start from scratch to explore the full space.
    # Let's restart with an empty set.
    stack = [(0, 0.0, 1)]

    nodes_visited = 0

    while stack:
        if time.time() > deadline:
            break

        M, cur_sum, idx = stack.pop()

        if idx > n:
            # Leaf node: evaluate
            if cur_sum > best_score:
                best_score = cur_sum
                best_mask = M
                best_elements = get_elements(M)
            continue

        # Quick upper bound using suffix sum (ignoring conflicts)
        ub_quick = cur_sum + suffix_sum[idx]
        if ub_quick <= best_score + 1e-15:
            continue  # Prune

        # Compute tighter upper bound: sum of 1/j for j >= idx where can_add(j, M)
        ub_tight = cur_sum
        for j in range(idx, n + 1):
            if can_add(j, M):
                ub_tight += inv[j]
        if ub_tight <= best_score + 1e-15:
            continue  # Prune

        nodes_visited += 1

        x = idx

        # Branch 1: Include x (if possible)
        if can_add(x, M):
            new_M = M | bit(x)
            new_sum = cur_sum + inv[x]
            # Push exclude branch first so include is processed first (LIFO)
            stack.append((M, cur_sum, idx + 1))
            stack.append((new_M, new_sum, idx + 1))
        else:
            # Can't include, just exclude
            stack.append((M, cur_sum, idx + 1))

    # Final verification
    final_elements = get_elements(best_mask)
    elem_set = set(final_elements)
    for a in final_elements:
        for b in final_elements:
            if a == b:
                continue
            c = 2 * b - a
            if c in elem_set and c != a and c != b:
                # Found a 3-AP, this shouldn't happen
                pass

    if not final_elements:
        final_elements = [1]

    return final_elements
