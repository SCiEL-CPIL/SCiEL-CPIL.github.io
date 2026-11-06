import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit (evaluator kills at 90s)
    
    n = 192
    universe = set(range(1, n + 1))
    
    # Precompute all triples of distinct elements from 1..192
    # For each triple (x, y, z) with x < y < z, compute s = x+y+z
    # For each divisor d of s, if d is in universe and d is not in {x,y,z},
    # then d cannot coexist with {x,y,z} in a valid set.
    # We store for each element d, the set of triples (as frozensets or tuples) that violate d.
    
    # Actually, let's store conflicts differently.
    # conflict[d] = set of frozensets {x,y,z} such that d | (x+y+z) and d not in {x,y,z}.
    # Then a set S is valid if and only if for every d in S, no triple in conflict[d] is a subset of S.
    
    # Let's precompute this.
    # Number of triples is C(192, 3) = 192*191*190/6 = 1,165,840.
    # This is feasible to precompute once.
    
    # conflict_map: dict from int to set of frozensets
    conflict_map = {i: set() for i in range(1, n + 1)}
    
    # Generate all triples
    # To save memory and time, we can just store the triples that cause conflicts.
    # For each triple (x,y,z), s = x+y+z.
    # Find all divisors d of s such that 1 <= d <= 192 and d not in {x,y,z}.
    # Add frozenset({x,y,z}) to conflict_map[d].
    
    # Optimization: We can iterate x, y, z.
    # But 1.1M triples * (divisor check) might be slow in Python if done naively.
    # Divisor check: loop d from 1 to sqrt(s).
    # s is at most 190+191+192 = 573.
    # sqrt(573) ~ 24.
    # So ~48 operations per triple.
    # 1.1M * 48 = 55M operations. Might take a few seconds.
    # Let's try to optimize.
    
    # Alternative:
    # For each d in 1..192:
    #   For each multiple m of d (m = 2d, 3d, ... up to 573):
    #     Find all triples (x,y,z) in 1..192, x<y<z, x+y+z=m, d not in {x,y,z}.
    #     Add frozenset({x,y,z}) to conflict_map[d].
    
    # This might be faster if we precompute pair sums?
    # For fixed d and m:
    #   We need x+y = m-z.
    #   Iterate z from 1 to 192.
    #   If z == d, skip.
    #   rem = m - z.
    #   We need x < y, x+y=rem, x,y != d, x,y != z, x,y in 1..192.
    #   Also x < y < z? No, x<y<z is just for unique triple generation.
    #   If we generate all x,y,z such that x+y+z=m, we might generate duplicates if we don't enforce order.
    #   But we want to add the frozenset.
    #   So we can just find all sets {x,y,z} summing to m.
    
    # Let's stick to the triple iteration, but optimize the divisor finding.
    # Precompute divisors for all numbers up to 573.
    max_sum = 3 * n
    divisors = [[] for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(i, max_sum + 1, i):
            divisors[j].append(i)
    
    # Now iterate triples
    # To avoid O(N^3) being too slow, note that we only care about d in 1..192.
    # So for a triple sum s, we only check divisors d of s where d <= 192.
    
    # Let's do it.
    # x from 1 to 192
    # y from x+1 to 192
    # z from y+1 to 192
    # s = x+y+z
    # for d in divisors[s]:
    #   if d <= n and d not in (x,y,z):
    #     conflict_map[d].add(frozenset((x,y,z)))
    
    # This loop is ~1.1M iterations.
    # Inner loop is small.
    # Should be fast enough (< 2s).
    
    for x in range(1, n + 1):
        for y in range(x + 1, n + 1):
            for z in range(y + 1, n + 1):
                s = x + y + z
                # Check divisors
                for d in divisors[s]:
                    if d <= n:
                        if d != x and d != y and d != z:
                            conflict_map[d].add(frozenset((x, y, z)))
    
    # Now we have conflict_map.
    # A set S is valid if for all d in S, no triple in conflict_map[d] is a subset of S.
    
    # Helper to check if a set S is valid
    def is_valid(S):
        S_set = set(S)
        for d in S_set:
            for triple in conflict_map[d]:
                if triple.issubset(S_set):
                    return False
        return True
    
    # Helper to check if adding x to S is valid
    # Only need to check:
    # 1. x does not form a violating triple with any 2 elements in S.
    #    i.e., for any triple T in conflict_map[x], if T contains x? No, T does not contain x by definition.
    #    Wait, conflict_map[x] contains triples {a,b,c} such that x | (a+b+c).
    #    So if S already contains {a,b,c}, then adding x is invalid? No, x is the divisor.
    #    The condition is: "No selected integer may divide the sum of any k=3 DISTINCT OTHER selected integers."
    #    So if x is in S, we check if x divides sum of any 3 others in S.
    #    So when adding x to S, we must check:
    #    a) Does x divide the sum of any 3 elements in S?
    #       This means: is there a triple T in S such that x | sum(T)?
    #       This is equivalent to: is there a triple T in conflict_map[x] such that T is a subset of S?
    #    b) For each y in S, does y divide the sum of any 3 elements in (S U {x}) \ {y}?
    #       The new triples involving x are {x, u, v} where u,v in S\{y}.
    #       So we check: for each y in S, is there a triple T in conflict_map[y] such that:
    #          - T is a subset of (S U {x})
    #          - x in T (otherwise it was already checked)
    #          - T does not contain y (by definition of conflict_map)
    #       So T must be of the form {x, u, v} with u,v in S.
    #       And {x, u, v} in conflict_map[y].
    
    def can_add(S, x):
        S_set = set(S)
        # Check 1: x divides sum of any 3 in S?
        for triple in conflict_map[x]:
            if triple.issubset(S_set):
                return False
        
        # Check 2: For each y in S, does y divide sum of {x, u, v} for u,v in S?
        # We can optimize by checking specific triples.
        # For each y in S:
        #   We need to know if there exist u, v in S, u != v, u != x, v != x, u != y, v != y
        #   such that {x, u, v} in conflict_map[y].
        #   Since conflict_map[y] is a set of frozensets, we can iterate through them?
        #   Or iterate u, v in S?
        #   Iterating u, v in S is O(K^2). K is ~50. 2500 checks.
        #   For each pair (u,v), check if frozenset({x,u,v}) in conflict_map[y].
        #   This is O(1) lookup.
        #   So O(K^2) per y, total O(K^3) ~ 125k operations.
        #   This is fast.
        
        for y in S_set:
            if y == x:
                continue
            # Check all pairs u, v in S
            # u < v to avoid duplicates
            S_list = list(S_set)
            for i in range(len(S_list)):
                u = S_list[i]
                if u == y or u == x:
                    continue
                for j in range(i + 1, len(S_list)):
                    v = S_list[j]
                    if v == y or v == x:
                        continue
                    tri = frozenset((x, u, v))
                    if tri in conflict_map[y]:
                        return False
        return True
    
    # Helper to check if removing r and adding x to S is valid
    # New set S' = (S - {r}) U {x}
    # We can just call is_valid(S')?
    # is_valid is O(K * |conflict|).
    # |conflict| can be large.
    # But K is small.
    # Let's see how big conflict_map[d] is.
    # For d=1, every triple sums to a multiple of 1. So conflict_map[1] has C(192,3) triples.
    # is_valid checks for d=1: if any triple in S is in conflict_map[1].
    # Since all triples are in conflict_map[1], if |S| >= 3, is_valid returns False for d=1.
    # So 1 can never be in a valid set of size >= 3.
    # This is a huge insight.
    # Similarly for small d.
    
    # Let's use is_valid for swaps, it's simpler and correctness is paramount.
    # If it's too slow, we can optimize later.
    
    # Greedy Construction
    # Start with empty set.
    # Try to add elements in some order.
    # Which order?
    # Larger elements are "safer" because they have fewer multiples in the sum range?
    # Sum of 3 others is at most 3*192 = 576.
    # Multiples of d in [1, 576] are 576/d.
    # So larger d has fewer multiples, thus fewer conflicts.
    # So we should try to add large elements first.
    
    S = set()
    candidates = list(range(n, 0, -1))
    random.shuffle(candidates) # Maybe mix? Or strictly descending?
    # Let's try strictly descending first.
    candidates = list(range(n, 0, -1))
    
    for x in candidates:
        if can_add(S, x):
            S.add(x)
        # Check time
        if time.time() - start_time > time_limit * 0.3:
            break
    
    # Local Search
    # Try to add more elements.
    # Then try swaps.
    
    # 1. Greedy Add
    # We might have missed some elements because the order was fixed.
    # Let's do a random restart of greedy additions?
    # Or just try to add any remaining element that fits.
    
    remaining = [x for x in range(1, n + 1) if x not in S]
    random.shuffle(remaining)
    
    for x in remaining:
        if can_add(S, x):
            S.add(x)
        if time.time() - start_time > time_limit * 0.5:
            break
    
    # 2. Local Search: Swaps
    # While time permits:
    #   Pick random r in S, random x not in S.
    #   If is_valid((S - {r}) U {x}):
    #       S = (S - {r}) U {x}
    #   Else if we can improve? No, swap keeps size same.
    #   We want to maximize cardinality.
    #   So swaps are only useful if they allow future additions.
    #   This is hard to predict.
    #   Better strategy:
    #   1. Try to add any x not in S.
    #   2. If no x can be added, try a swap (r, x) such that after swap, some other y can be added?
    #      Or just random walk in the space of valid sets of same size to find a "hole" that allows addition?
    #      This is complex.
    #      
    #   Simpler:
    #   1. Try to add.
    #   2. If stuck, do a random valid swap.
    #   3. Repeat.
    #   This is a random walk.
    
    # Let's implement a simple loop.
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try to add
        # Pick random candidates
        not_in_S = [x for x in range(1, n + 1) if x not in S]
        if not_in_S:
            # Shuffle to avoid bias
            random.shuffle(not_in_S)
            for x in not_in_S:
                if can_add(S, x):
                    S.add(x)
                    improved = True
                    break
        
        if improved:
            continue
        
        # If no addition, try swap
        if S:
            # Pick random r in S
            r = random.choice(list(S))
            # Pick random x not in S
            if not_in_S:
                x = random.choice(not_in_S)
                new_S = (S - {r}) | {x}
                if is_valid(new_S):
                    S = new_S
                    # Don't mark improved, as size is same.
                    # But we changed the set.
                    pass
    
    return list(S)

# Test briefly
if __name__ == "__main__":
    res = construct()
    print(len(res))
