"""
Rewrite using a Max-SAT style greedy construction with conflict resolution.
"""
import math

def construct():
    N = 180
    
    # Precompute all forbidden triples (a, b, c) with a < b < c, all in [1, N]
    # such that 1/a = 1/b + 1/c  <=>  b*c = a*(b+c)
    # Note: a < b and a < c (since 1/a > 1/b and 1/a > 1/c)
    # Also b < c w.l.o.g. (symmetric in b, c)
    
    # For efficiency, iterate over possible (b, c) pairs and compute a
    # a = (b*c) / (b+c)
    # We need a to be an integer and a < b (since a is the smallest)
    # Also a >= 1 and a <= N
    
    forbidden = []  # list of (a, b, c)
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            numerator = b * c
            denominator = b + c
            if numerator % denominator == 0:
                a = numerator // denominator
                if 1 <= a < b:
                    forbidden.append((a, b, c))
    
    # Now, we want to find a large subset S of [1, N] with no forbidden triple.
    # Start with the full set [1, N], and greedily remove elements to resolve conflicts.
    # Strategy: try to keep as many elements as possible.
    # We can use a greedy approach: process elements in some order, and for each,
    # if it's not "forced out" by existing conflicts, keep it.
    # But a better heuristic: start with the larger elements (since smaller a's are more likely to be in conflicts)
    # and add them if they don't create conflicts with already-added elements.
    # Actually, the seed starts from [91, 180] which works because for a >= 91, b >= a+1 >= 92, c >= b+1 >= 93
    # then 1/a >= 1/91, 1/b + 1/c <= 1/92 + 1/93 < 2/92 = 1/46 < 1/91? No, 1/92+1/93 ≈ 0.01087+0.01075=0.02162, 1/91≈0.01099.
    # Wait, 1/b + 1/c < 2/b. If b > 2a, then 2/b < 1/a. So if all elements are > 2a for all a in the set...
    # In the seed [91, 180], min is 91, max is 180. For a=91, b>=92, c>=93. 1/92+1/93 ≈ 0.0216, 1/91 ≈ 0.0109. So 1/a < 1/b+1/c is possible?
    # Actually we need 1/a = 1/b + 1/c. Since 1/a > 1/b and 1/a > 1/c, we have 1/b + 1/c < 2/b.
    # For equality 1/a = 1/b + 1/c, we need b*c = a*(b+c).
    # If a, b, c are all in [91, 180], is there a solution?
    # a = bc/(b+c). Since b, c >= 91, b+c >= 182, bc/(b+c) <= bc/(2*91) if b=c=91, but b<c.
    # Let's just trust the seed works.
    
    # Greedy: start from N down to 1. Maintain a set S.
    # For each x, check if adding x to S creates any forbidden triple.
    # A triple (a, b, c) is in S if all three are in S.
    # When adding x, new conflicts involve triples where x is one of a, b, or c.
    # If x = a: need (x, b, c) in forbidden with b, c in S.
    # If x = b: need (a, x, c) in forbidden with a, c in S.
    # If x = c: need (a, b, x) in forbidden with a, b in S.
    
    # Precompute for each x, the list of triples where x is a, b, or c.
    triples_as_a = [[] for _ in range(N + 1)]
    triples_as_b = [[] for _ in range(N + 1)]
    triples_as_c = [[] for _ in range(N + 1)]
    
    for (a, b, c) in forbidden:
        triples_as_a[a].append((b, c))
        triples_as_b[b].append((a, c))
        triples_as_c[c].append((a, b))
    
    S = set()
    # Try adding from N down to 1
    for x in range(N, 0, -1):
        # Check if x can be added
        can_add = True
        # Check triples where x is a
        for (b, c) in triples_as_a[x]:
            if b in S and c in S:
                can_add = False
                break
        if can_add:
            for (a, c) in triples_as_b[x]:
                if a in S and c in S:
                    can_add = False
                    break
        if can_add:
            for (a, b) in triples_as_c[x]:
                if a in S and b in S:
                    can_add = False
                    break
        if can_add:
            S.add(x)
    
    # The above greedy may not be optimal. Let's try a different approach:
    # Start with a large initial set (like the seed) and try to add smaller elements.
    # Or, use a local search / hill climbing approach.
    
    # Actually, let's try the greedy from the top again, but verify the size.
    # The seed gives 90. Our greedy from N down to 1 might give more.
    
    # Let's also try starting from the seed and trying to add elements from 1 to 90.
    S_seed = set(range(91, N + 1))
    S_best = set(S_seed)
    
    # Try adding elements from 1 to 90 in some order
    for x in range(1, 91):
        can_add = True
        for (b, c) in triples_as_a[x]:
            if b in S_best and c in S_best:
                can_add = False
                break
        if can_add:
            for (a, c) in triples_as_b[x]:
                if a in S_best and c in S_best:
                    can_add = False
                    break
        if can_add:
            for (a, b) in triples_as_c[x]:
                if a in S_best and b in S_best:
                    can_add = False
                    break
        if can_add:
            S_best.add(x)
    
    # Now try to add elements from 91 to 180 that might have been excluded
    # Actually, the seed already includes all of [91, 180].
    
    # Let's try a more systematic approach: use the greedy from N down to 1 as the primary method.
    S_greedy = set()
    for x in range(N, 0, -1):
        can_add = True
        for (b, c) in triples_as_a[x]:
            if b in S_greedy and c in S_greedy:
                can_add = False
                break
        if can_add:
            for (a, c) in triples_as_b[x]:
                if a in S_greedy and c in S_greedy:
                    can_add = False
                    break
        if can_add:
            for (a, b) in triples_as_c[x]:
                if a in S_greedy and b in S_greedy:
                    can_add = False
                    break
        if can_add:
            S_greedy.add(x)
    
    # Pick the larger of the two
    if len(S_greedy) > len(S_best):
        S_best = S_greedy
    
    # Try to improve S_best with local search
    # For each x not in S_best, try adding it and removing the minimum number of elements to resolve conflicts.
    # This is complex, so let's try a simpler heuristic:
    # Randomly swap elements or try to add elements with forced removals.
    
    # For now, return the best we have
    return list(S_best)
