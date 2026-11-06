"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

def construct():
    n = 192
    k = 3
    
    # Precompute for each integer a (1..n), the set of triples (x,y,z) of distinct
    # other selected integers such that a divides x+y+z.
    # But we need a dynamic check. Let's think about the constraint differently.
    
    # Constraint: For no selected integer a, and no 3 distinct other selected integers
    # x, y, z (all != a), does a divide (x+y+z).
    
    # This is a complex hypergraph independent set problem. The current seed does a
    # greedy approach from largest to smallest, checking all constraints.
    
    # Let's try a different strategy:
    # 1. Start with a large set of candidates.
    # 2. Iteratively remove elements that cause violations.
    # 3. Or use a smarter greedy: prioritize numbers that are less likely to divide sums.
    
    # Observation: If we pick numbers that are all larger than n/2, then the sum of any 3
    # is at most 3*n, and each number is > n/2. For a to divide x+y+z where x,y,z > n/2,
    # we need x+y+z >= a. Since a <= n and x+y+z can be up to 3n, this is possible.
    
    # Better observation: If all selected numbers are in (n/2, n], then for any a in the set,
    # the sum of any 3 others is at least 3*(n/2+1) ≈ 1.5n and at most 3n.
    # a divides sum means sum = m*a for some integer m. Since sum < 3n and a > n/2,
    # m can be 1, 2, or 3.
    # m=1: sum = a, but sum > 1.5n > a (since a <= n), impossible.
    # m=2: sum = 2a. Possible if 2a is in range [1.5n, 3n] and 2a <= 3n (true since a<=n).
    # m=3: sum = 3a. Possible if 3a <= 3n (true) and 3a >= 1.5n (true since a > n/2).
    
    # So even in (n/2, n], we can have violations when sum = 2a or 3a.
    
    # Let's try an iterative removal approach:
    # Start with all numbers 1..n, or a subset, then remove conflicting elements.
    
    # Actually, let's try a greedy that picks numbers in a specific order and checks.
    # The seed picks from largest to smallest. Let's try smallest to largest, or a mixed order.
    
    # Key insight: Small numbers are more likely to divide sums. So maybe we should avoid small numbers.
    # Let's try starting from the top but with a smarter check.
    
    # Precompute: for each a, and each pair (x,y) with x != y != a, what z values would make a | (x+y+z)?
    # z ≡ -(x+y) (mod a), z != a, z != x, z != y, 1 <= z <= n.
    
    # For efficiency, let's use a different approach:
    # Greedy: try to add elements one by one. For each candidate, check if adding it violates
    # any existing constraint.
    
    # When we add a new element 'new', we need to check:
    # 1. For each existing 'a' in selected: does 'a' divide the sum of any 3 distinct others including 'new'?
    #    The 3 others must be distinct from 'a'. If 'new' is one of them, we need 2 others from selected (excluding 'a' and 'new').
    #    So for each a, for each pair (x,y) from selected (excluding a and new), check if a | (x+y+new).
    # 2. For 'new' itself: does 'new' divide the sum of any 3 distinct others from selected (excluding new)?
    #    So for each triple (x,y,z) from selected (excluding new), check if new | (x+y+z).
    
    # This is O(|selected|^2) for check 1 and O(|selected|^3) for check 2 per candidate.
    # With |selected| up to 48 and candidates up to 192, this might be slow but let's try.
    
    # Actually, let's precompute more cleverly.
    
    # Let's try a random restart greedy:
    # Try multiple random orders and pick the best result.
    
    def check_valid(selected):
        """Check if the selected set is valid."""
        s = set(selected)
        for a in selected:
            others = [v for v in selected if v != a]
            for triple in combinations(others, 3):
                if sum(triple) % a == 0:
                    return False
        return True
    
    from itertools import combinations
    
    # Greedy with multiple orders
    best = []
    
    # Try descending order (like seed)
    selected = []
    for x in range(n, 0, -1):
        candidate = selected + [x]
        valid = True
        # Check all constraints
        for a in candidate:
            others = [v for v in candidate if v != a]
            for triple in combinations(others, 3):
                if sum(triple) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
    
    best = max(best, selected, key=len)
    
    # Try ascending order
    selected = []
    for x in range(1, n + 1):
        candidate = selected + [x]
        valid = True
        for a in candidate:
            others = [v for v in candidate if v != a]
            for triple in combinations(others, 3):
                if sum(triple) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
    
    best = max(best, selected, key=len)
    
    # Try random orders
    random.seed(42)
    for _ in range(50):
        order = list(range(1, n + 1))
        random.shuffle(order)
        selected = []
        for x in order:
            candidate = selected + [x]
            valid = True
            for a in candidate:
                others = [v for v in candidate if v != a]
                for triple in combinations(others, 3):
                    if sum(triple) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                selected.append(x)
        best = max(best, selected, key=len)
    
    return best

# EVOLVE-BLOCK-END
