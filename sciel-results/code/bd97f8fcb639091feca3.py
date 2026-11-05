# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _fast_check_new_element(x, existing):
    """Check that x is not the mean of any subset of >=2 elements in existing,
    and no element in existing is the mean of any subset containing x."""
    n = len(existing)
    if n < 2:
        return True
    
    # Part 1: x must not be the mean of any subset of >=2 from existing
    # For k=2: need a+b == 2*x, i.e., a+b == 2x
    target2 = 2 * x
    for i in range(n):
        for j in range(i+1, n):
            if existing[i] + existing[j] == target2:
                return False
    # For k=3: need a+b+c == 3*x
    target3 = 3 * x
    if n >= 3:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if existing[i] + existing[j] + existing[k] == target3:
                        return False
    # For k=4: need a+b+c+d == 4*x
    target4 = 4 * x
    if n >= 4:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if existing[i] + existing[j] + existing[k] + existing[l] == target4:
                            return False
    # For k>=5: less likely but check up to n if feasible
    for k in range(5, min(n, 10) + 1):
        target = k * x
        # Use a set for faster lookup: check if any combination sums to target
        # For small n this is fine; for larger n we skip k>=5 as very rare
        if n <= 15:
            for combo in combinations(existing, k):
                if sum(combo) == target:
                    return False
    # Part 2: no element a in existing should be the mean of a subset of >=2 from (existing - a) + {x}
    # This means: for each a in existing, check subsets of (existing without a) that include x
    # i.e., subset S of existing (not containing a) such that sum(S) + x = |S| * a (if x is in the subset)
    # OR subset T of existing (not containing a, not containing x) such that sum(T) + x = (|T|+1)*a
    # Simpler: for each a, the other elements are (existing - a) + [x]. Check if a is mean of any subset of size>=2 from that set.
    # But this is expensive. Let's do it smarter:
    # a is mean of subset R of size r from (existing\a) ∪ {x}: sum(R) = r*a
    # R can contain x or not.
    # If R doesn't contain x: sum(R) = r*a where R ⊆ existing\a, |R|>=2
    #   This was already checked before adding x (it's the same condition as before).
    # If R contains x: R = {x} ∪ S where S ⊆ existing\a, |S| >= 1, |R| = |S|+1 >= 2
    #   sum(R) = x + sum(S) = (|S|+1)*a
    #   => sum(S) = (|S|+1)*a - x
    #   For each S ⊆ existing\a, |S|>=1: check if sum(S) = (|S|+1)*a - x
    for i, a in enumerate(existing):
        others_without_a = existing[:i] + existing[i+1:]
        m = len(others_without_a)
        # S is a non-empty subset of others_without_a
        # |S| ranges from 1 to m
        # Need sum(S) = (|S|+1)*a - x
        # For |S|=1: need some y in others_without_a such that y = 2*a - x
        #   i.e., y + x = 2*a, which means a is the mean of {x, y}
        for y in others_without_a:
            if x + y == 2 * a:
                return False
        # For |S|=2: need y1+y2 = 3*a - x
        target_s2 = 3 * a - x
        if m >= 2:
            for j in range(m):
                for k2 in range(j+1, m):
                    if others_without_a[j] + others_without_a[k2] == target_s2:
                        return False
        # For |S|=3: need y1+y2+y3 = 4*a - x
        target_s3 = 4 * a - x
        if m >= 3:
            for j in range(m):
                for k2 in range(j+1, m):
                    for l in range(k2+1, m):
                        if others_without_a[j] + others_without_a[k2] + others_without_a[l] == target_s3:
                            return False
        # For |S|>=4: less common, check if m is small
        if m <= 10:
            for s in range(4, m+1):
                target_s = (s+1)*a - x
                for combo in combinations(others_without_a, s):
                    if sum(combo) == target_s:
                        return False
    return True


def _exact_verify(selected):
    """Exact verification using DP for all subset sizes."""
    n = len(selected)
    if n < 2:
        return True
    for idx, a in enumerate(selected):
        others = [x for j, x in enumerate(selected) if j != idx]
        m = len(others)
        # Check if a is the mean of any subset of others of size k>=2
        # i.e., sum(subset) = k*a for some k in 2..m
        # Use DP: dp[s][t] = True if we can pick s elements summing to t
        # But this is expensive for large m. Use a different approach:
        # For each k from 2 to m, check if any k-subset sums to k*a
        # For small k (2,3,4), enumerate. For larger k, use DP or skip.
        for k in range(2, m+1):
            target = k * a
            if k <= 4:
                found = False
                if k == 2:
                    for i in range(m):
                        for j in range(i+1, m):
                            if others[i] + others[j] == target:
                                found = True
                                break
                        if found:
                            break
                elif k == 3:
                    for i in range(m):
                        for j in range(i+1, m):
                            for l in range(j+1, m):
                                if others[i] + others[j] + others[l] == target:
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                else:  # k == 4
                    for i in range(m):
                        for j in range(i+1, m):
                            for l in range(j+1, m):
                                for p in range(l+1, m):
                                    if others[i] + others[j] + others[l] + others[p] == target:
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                if found:
                    return False
            else:
                # For k >= 5, use a simple check: the sum of any k elements
                # Minimum possible sum of k elements: sum of k smallest
                # Maximum possible sum of k elements: sum of k largest
                others_sorted = sorted(others)
                min_sum = sum(others_sorted[:k])
                max_sum = sum(others_sorted[-k:])
                if target < min_sum or target > max_sum:
                    continue
                # Try to find a subset of size k summing to target
                # Use a greedy/DP approach for small m
                if m <= 20:
                    # Simple DP: dp[j][s] = number of ways to pick j elements summing to s
                    # This can be expensive, so use a set-based approach
                    # Actually for m<=20 and k>=5, enumerate combinations
                    for combo in combinations(others, k):
                        if sum(combo) == target:
                            return False
                # For m > 20 and k >= 5, skip (very rare)
    return True


def _greedy_construct(order, max_elements=48, max_val=512, deadline=None):
    """Build a valid set by greedily adding elements from the given order."""
    selected = []
    for x in order:
        if x > max_val:
            continue
        if len(selected) >= max_elements:
            break
        if deadline is not None and time.time() > deadline:
            break
        if _fast_check_new_element(x, selected):
            selected.append(x)
    return selected


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    best = []
    
    # Try multiple orderings
    random.seed(42)
    
    # Ordering 1: natural order 1..512
    order1 = list(range(1, 513))
    result1 = _greedy_construct(order1, 48, 512, deadline)
    if len(result1) > len(best):
        best = result1
    
    # Ordering 2: even numbers first, then odds
    if time.time() < deadline:
        evens = [x for x in range(1, 513) if x % 2 == 0]
        odds = [x for x in range(1, 513) if x % 2 == 1]
        random.shuffle(evens)
        random.shuffle(odds)
        order2 = evens + odds
        result2 = _greedy_construct(order2, 48, 512, deadline)
        if len(result2) > len(best):
            best = result2
    
    # Ordering 3: random shuffle
    if time.time() < deadline:
        order3 = list(range(1, 513))
        random.shuffle(order3)
        result3 = _greedy_construct(order3, 48, 512, deadline)
        if len(result3) > len(best):
            best = result3
    
    # Ordering 4: multiples of 3 first
    if time.time() < deadline:
        mult3 = [x for x in range(1, 513) if x % 3 == 0]
        rest3 = [x for x in range(1, 513) if x % 3 != 0]
        random.shuffle(mult3)
        random.shuffle(rest3)
        order4 = mult3 + rest3
        result4 = _greedy_construct(order4, 48, 512, deadline)
        if len(result4) > len(best):
            best = result4
    
    # Ordering 5: random with bias toward larger numbers
    if time.time() < deadline:
        order5 = list(range(1, 513))
        # Bias: larger numbers first
        random.shuffle(order5)
        order5.sort(key=lambda x: -random.random())
        result5 = _greedy_construct(order5, 48, 512, deadline)
        if len(result5) > len(best):
            best = result5
    
    # Try more random restarts if time permits
    restart_count = 5
    for r in range(restart_count):
        if time.time() > deadline:
            break
        order = list(range(1, 513))
        random.shuffle(order)
        result = _greedy_construct(order, 48, 512, deadline)
        if len(result) > len(best):
            best = result
    
    # Local search: try to extend best with remaining elements
    if len(best) < 48 and time.time() < deadline:
        best_set = set(best)
        remaining = [x for x in range(1, 513) if x not in best_set]
        random.shuffle(remaining)
        for x in remaining:
            if len(best) >= 48:
                break
            if time.time() > deadline:
                break
            if _fast_check_new_element(x, best):
                best.append(x)
    
    # Final exact verification
    if best and time.time() < deadline - 2:
        if not _exact_verify(best):
            # Fallback: use the natural order result
            fallback = _greedy_construct(list(range(1, 513)), 48, 512, deadline)
            if _exact_verify(fallback):
                best = fallback
    
    return best
# EVOLVE-BLOCK-END
