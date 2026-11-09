import time
import random

def construct():
    start = time.time()
    DEADLINE = 55.0  # safety margin before 60s kill
    random.seed(42)

    def is_valid(subset):
        # Check no subset of subset sums to 200 using bitset DP
        reachable = 1  # bit 0 set
        for x in subset:
            new_reachable = reachable | (reachable << x)
            if (new_reachable >> 200) & 1:
                return False
            reachable = new_reachable
        return True

    def greedy(order, existing=None):
        if existing is None:
            existing = set()
        reachable = 1  # bit 0
        selected = set(existing)
        for x in order:
            if x in selected:
                continue
            new_reachable = reachable | (reachable << x)
            if not ((new_reachable >> 200) & 1):
                selected.add(x)
                reachable = new_reachable
        return selected

    # Build conflict graph: edge (i,j) if i+j < 200 (i.e., they could be part of a sum-200 triple with some k)
    # More precisely: i and j conflict if 200 - i - j is in [1,100] and 200-i-j > 0
    # This means if we include both i and j, we must exclude k = 200-i-j
    # For a pairwise conflict graph for MIS heuristic:
    # Two elements "conflict" if including both forces exclusion of a third
    # Actually for the greedy, let's use a different approach:
    
    # Compute "conflict degree" for each element: how many other elements it pairs with to sum to < 200
    # An element x conflicts with y if x + y < 200 (they could be extended to 200)
    # But that's too broad. Let's use: x conflicts with y if there exists z in [1,100] such that x+y+z = 200
    # i.e., z = 200 - x - y, and 1 <= z <= 100, z != x, z != y
    
    conflict_degree = {}
    for x in range(1, 101):
        deg = 0
        for y in range(1, 101):
            if y == x:
                continue
            z = 200 - x - y
            if 1 <= z <= 100 and z != x and z != y:
                deg += 1
        conflict_degree[x] = deg
    
    # Greedy MIS: prioritize lowest degree (least conflicting) vertices
    # But we also need to ensure the subset sum constraint, not just pairwise
    # So we'll use the degree as a tiebreaker in the greedy order
    
    # Strategy 1: Sort by conflict degree (ascending - pick least conflicting first)
    order_by_degree = sorted(range(1, 101), key=lambda x: (conflict_degree[x], x))
    result1 = greedy(order_by_degree)
    
    # Strategy 2: Sort by conflict degree (descending - pick most conflicting first, they block more)
    order_by_degree_desc = sorted(range(1, 101), key=lambda x: (-conflict_degree[x], x))
    result2 = greedy(order_by_degree_desc)
    
    # Strategy 3: Original greedy from top
    result3 = greedy(range(100, 0, -1))
    
    # Strategy 4: Greedy from bottom
    result4 = greedy(range(1, 101))
    
    best = max(result1, result2, result3, result4, key=len)
    
    # Try random orderings with degree-based bias
    for i in range(500):
        if time.time() - start > DEADLINE:
            break
        # Weighted random: lower degree elements have higher probability
        weights = [1.0 / (conflict_degree[x] + 1) for x in range(1, 101)]
        order = list(range(1, 101))
        random.shuffle(order)
        res = greedy(order)
        if len(res) > len(best):
            best = res
        if len(best) >= 60:  # likely near-optimal
            break
    
    # Local search: try to add elements to best set
    for x in range(1, 101):
        if time.time() - start > DEADLINE:
            break
        if x in best:
            continue
        candidate = best | {x}
        if is_valid(candidate):
            best = candidate
    
    # Second pass local search
    for x in range(1, 101):
        if time.time() - start > DEADLINE:
            break
        if x in best:
            continue
        candidate = best | {x}
        if is_valid(candidate):
            best = candidate
    
    # Try removing one element and adding two
    for x in sorted(best, key=lambda v: -conflict_degree[v]):
        if time.time() - start > DEADLINE:
            break
        reduced = best - {x}
        for y in range(1, 101):
            if y in reduced or y == x:
                continue
            cand1 = reduced | {y}
            if is_valid(cand1) and len(cand1) > len(best):
                best = cand1
                break
            for z in range(1, 101):
                if z in reduced or z == x or z == y:
                    continue
                cand2 = reduced | {y, z}
                if is_valid(cand2) and len(cand2) > len(best):
                    best = cand2
                    break

    # Final verification
    assert len(best) > 0
    assert all(1 <= x <= 100 for x in best)
    assert len(best) == len(set(best))
    
    # Verify no subset sums to 200
    reachable = 1
    for x in sorted(best):
        new_reachable = reachable | (reachable << x)
        if (new_reachable >> 200) & 1:
            return list(best)  # Should not happen
        reachable = new_reachable

    return list(best)
