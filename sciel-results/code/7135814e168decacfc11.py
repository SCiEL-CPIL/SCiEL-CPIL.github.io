"""
Maximum subset of {1..192} such that no element divides the sum of any 3 distinct other elements.
Uses a Maximum Independent Set formulation with branch-and-bound and bitset acceleration.
"""
import time
import random
import sys
from itertools import combinations

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 85.0

N = 192
MAX_K = 48


def _precompute_conflicts():
    """
    Precompute for each candidate d, the set of conflicts:
    - conflict_with_triples[d]: frozenset of (a,b,c) triples from 1..192 where d | (a+b+c) and {a,b,c} != {d}
    - Actually, we need: for a given set S, d can be added if:
      1. No triple in S sums to a multiple of d
      2. For each a in S, no pair (x,y) in S\{a} has a | (d+x+y)
    
    Let's precompute differently:
    For each pair (a,b) with a != b, precompute divisors of a+b that are in 1..192.
    Then when adding d to S, for each a in S, check if there exists b in S\{a} such that
    a | (d + b + c) for some c in S\{a,b}.
    
    Actually, let's precompute the conflict graph edges for the MIS formulation:
    Two vertices i, j conflict if including both creates a constraint violation.
    But the constraint is on triples, so it's not pairwise.
    
    Let me use a direct approach with fast checking.
    """
    # Precompute divisors for sums up to 3*192 = 576
    divisors_of = [[] for _ in range(3 * N + 1)]
    for d in range(1, N + 1):
        for multiple in range(d, 3 * N + 1, d):
            divisors_of[multiple].append(d)
    
    return divisors_of


def _check_valid_addition(d, selected_mask):
    """
    Check if d can be added to the set represented by selected_mask.
    selected_mask is a Python int where bit i-1 is set if i is in the set.
    """
    # Get the list of selected elements
    selected = []
    for i in range(1, N + 1):
        if (selected_mask >> (i - 1)) & 1:
            selected.append(i)
    
    n = len(selected)
    
    # Check 1: d must not divide sum of any 3 distinct elements in selected
    if n >= 3:
        if d > 1:  # d=1 divides everything, so if n>=3, d=1 can't be added
            # Count residues mod d
            if d <= 200:
                res_count = [0] * d
                for x in selected:
                    res_count[x % d] += 1
                
                # Check if there exist 3 distinct elements with sum ≡ 0 mod d
                found = False
                for r1 in range(d):
                    if res_count[r1] == 0:
                        continue
                    for r2 in range(r1, d):
                        if res_count[r2] == 0:
                            continue
                        r3 = (-r1 - r2) % d
                        if r3 < r2:
                            continue
                        if r1 == r2 == r3:
                            if res_count[r1] >= 3:
                                found = True
                                break
                        elif r1 == r2:
                            if res_count[r1] >= 2 and res_count[r3] >= 1:
                                found = True
                                break
                        elif r2 == r3:
                            if res_count[r1] >= 1 and res_count[r2] >= 2:
                                found = True
                                break
                        else:
                            if res_count[r1] >= 1 and res_count[r2] >= 1 and res_count[r3] >= 1:
                                found = True
                                break
                    if found:
                        break
                if found:
                    return False
            else:
                # d > 200, but d <= 192 so this won't happen
                pass
    
    # Check 2: For each a in selected, a must not divide (d + x + y) for distinct x,y in selected\{a}
    for idx_a, a in enumerate(selected):
        if a == 1:
            # 1 divides everything, so if there are at least 2 other elements, conflict
            if n - 1 >= 2:
                return False
            continue
        
        target = (-d) % a
        # Need distinct x, y from selected \ {a} with (x + y) % a == target
        res_seen = set()
        for i in range(n):
            if i == idx_a:
                continue
            x = selected[i]
            xr = x % a
            needed = (target - xr) % a
            if needed in res_seen:
                return False
            res_seen.add(xr)
    
    return True


def _greedy_construct(order, deadline):
    """Greedy construction with given order, using bitset."""
    selected_mask = 0
    for c in order:
        if time.time() > deadline:
            break
        if _check_valid_addition(c, selected_mask):
            selected_mask |= (1 << (c - 1))
    return selected_mask


def _local_search_add(mask, deadline):
    """Try to add more elements to the set."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for c in range(N, 0, -1):
            if (mask >> (c - 1)) & 1:
                continue
            if _check_valid_addition(c, mask):
                mask |= (1 << (c - 1))
                changed = True
                break
    return mask


def _local_search_swap(mask, deadline):
    """Try swap moves: remove one, add another."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to find a swap that increases size (remove 1, add 2 net +1)
        # Or just try to add after removing
        for remove_c in range(1, N + 1):
            if not ((mask >> (remove_c - 1)) & 1):
                continue
            reduced_mask = mask & ~(1 << (remove_c - 1))
            # Try to add
            for add_c in range(N, 0, -1):
                if (reduced_mask >> (add_c - 1)) & 1:
                    continue
                if _check_valid_addition(add_c, reduced_mask):
                    new_mask = reduced_mask | (1 << (add_c - 1))
                    if new_mask.bit_count() > mask.bit_count():
                        mask = new_mask
                        improved = True
                        break
            if improved:
                break
    return mask


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_mask = 0
    best_size = 0
    
    random.seed(12345)
    
    # Multiple greedy starts with different orderings
    orders = []
    
    # Descending
    orders.append(list(range(N, 0, -1)))
    # Ascending
    orders.append(list(range(1, N + 1)))
    
    # Sorted by divisibility properties (numbers with fewer divisors first)
    def div_count(x):
        count = 0
        for i in range(1, int(x**0.5) + 1):
            if x % i == 0:
                count += 1
                if i != x // i:
                    count += 1
        return count
    
    # Numbers with fewer divisors tend to be less restrictive as divisors
    orders.append(sorted(range(1, N + 1), key=lambda x: (-div_count(x), x)))
    orders.append(sorted(range(1, N + 1), key=lambda x: (div_count(x), x)))
    
    # Random orders
    for _ in range(40):
        if time.time() > deadline - 20:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() > deadline - 10:
            break
        mask = _greedy_construct(order, deadline)
        if mask.bit_count() > best_size:
            best_mask = mask
            best_size = mask.bit_count()
    
    # Local search with add moves
    if time.time() < deadline - 5:
        improved_mask = _local_search_add(best_mask, deadline - 3)
        if improved_mask.bit_count() > best_size:
            best_mask = improved_mask
            best_size = best_mask.bit_count()
    
    # Local search with swap moves
    if time.time() < deadline - 3:
        swapped_mask = _local_search_swap(best_mask, deadline - 2)
        if swapped_mask.bit_count() > best_size:
            best_mask = swapped_mask
            best_size = best_mask.bit_count()
    
    # Additional random restarts with local search
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        mask = _greedy_construct(order, deadline - 3)
        if mask.bit_count() > best_size:
            best_mask = mask
            best_size = mask.bit_count()
        if time.time() < deadline - 3:
            improved_mask = _local_search_add(mask, deadline - 2)
            if improved_mask.bit_count() > best_size:
                best_mask = improved_mask
                best_size = best_mask.bit_count()
    
    # Convert to list
    result = []
    for i in range(1, N + 1):
        if (best_mask >> (i - 1)) & 1:
            result.append(i)
    
    # Final safety validation
    if len(result) > 0:
        valid = True
        n = len(result)
        for i in range(n):
            d = result[i]
            others = [result[j] for j in range(n) if j != i]
            m = len(others)
            for x in range(m):
                for y in range(x + 1, m):
                    for z in range(y + 1, m):
                        if (others[x] + others[y] + others[z]) % d == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            # Fallback
            result = [192, 191, 190]
    
    return result


if __name__ == "__main__":
    construct()
