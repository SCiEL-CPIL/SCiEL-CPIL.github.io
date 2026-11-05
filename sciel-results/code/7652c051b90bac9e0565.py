"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from functools import lru_cache
from collections import defaultdict

def construct():
    """
    Return a nonempty subset of [1, 180] with no THREE DISTINCT elements a,b,c
    satisfying 1/a = 1/b + 1/c. Maximize cardinality.
    
    We model this as a Maximum Independent Set problem in a 3-uniform hypergraph.
    Vertices are integers 1..180. A hyperedge is a triple {a,b,c} (distinct) such that
    1/a = 1/b + 1/c. We want the largest subset with no hyperedge fully contained.
    
    Approach:
    1. Precompute all forbidden triples.
    2. Use a greedy heuristic to get a good initial solution.
    3. Use local search / branch and bound to improve.
    
    Given time constraints (60s), we'll use a greedy construction with careful ordering,
    followed by local improvement.
    """
    N = 180
    numbers = list(range(1, N + 1))
    
    # Precompute all forbidden triples {a, b, c} with a, b, c distinct and 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c  =>  1/a = (b+c)/(bc)  =>  bc = a(b+c)  =>  bc - ab - ac = 0
    # => (a-b)(a-c) = a^2  ... let's just iterate.
    # For given b, c: 1/a = 1/b + 1/c => a = bc/(b+c). So a must be integer, a != b, a != c.
    # Also need b != c for distinctness? The problem says THREE DISTINCT elements.
    # So a, b, c must all be distinct.
    
    forbidden = set()
    # For each pair (b, c) with b < c to avoid duplicates, check if a = bc/(b+c) is integer
    # and a is in [1, N] and a != b and a != c.
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    # Store the triple as a sorted tuple for consistency
                    triple = tuple(sorted((a, b, c)))
                    forbidden.add(triple)
    
    # Convert to a conflict structure: for each vertex, which other vertices form a forbidden triple with it
    # Actually, for MIS in hypergraph, we need to ensure no hyperedge is fully in our set.
    # Let's build: for each pair (i, j), list of k such that {i,j,k} is forbidden.
    # This helps in local search.
    
    pair_conflicts = defaultdict(list)
    for (a, b, c) in forbidden:
        pair_conflicts[(a, b)].append(c)
        pair_conflicts[(a, c)].append(b)
        pair_conflicts[(b, c)].append(a)
        # Also reverse order for lookup convenience
        pair_conflicts[(b, a)].append(c)
        pair_conflicts[(c, a)].append(b)
        pair_conflicts[(c, b)].append(a)
    
    # Greedy construction: add numbers in an order that minimizes conflicts
    # Strategy: start with large numbers (they have fewer conflicts since 1/a = 1/b+1/c
    # with a < b, c means a is smaller, so large a have fewer representations)
    # Actually, if a is large, 1/a is small, so 1/b + 1/c must be small, meaning b, c large.
    # So large numbers might be safer to include.
    
    # Let's try: include all numbers from N down to 1, skipping those that would form a forbidden triple.
    # Greedy: process in descending order. Maintain set S. For each x from N down to 1,
    # check if adding x creates any forbidden triple with two elements already in S.
    
    S = set()
    # For efficient checking: when adding x, check all pairs (u, v) in S such that {x, u, v} is forbidden.
    # We can precompute for each x, the set of pairs (u, v) that with x form a forbidden triple.
    
    # conflicts_with[x] = list of (u, v) such that {x, u, v} is a forbidden triple
    conflicts_with = defaultdict(list)
    for (a, b, c) in forbidden:
        conflicts_with[a].append((b, c))
        conflicts_with[b].append((a, c))
        conflicts_with[c].append((a, b))
    
    # Greedy pass: descending order
    for x in range(N, 0, -1):
        # Check if adding x conflicts with any pair in S
        can_add = True
        for (u, v) in conflicts_with[x]:
            if u in S and v in S:
                can_add = False
                break
        if can_add:
            S.add(x)
    
    # Local search improvement: try to swap elements or add/remove
    # Let's try a simple local search: iterate through numbers not in S, see if we can add them
    # by removing some conflicting elements.
    
    # Better local search: hill climbing with swaps
    # Actually, let's try to add numbers in ascending order now, as a second pass
    # This might fill in gaps.
    
    # Alternative approach: use a different greedy ordering.
    # Let's try multiple greedy strategies and keep the best.
    
    def greedy(order):
        S_local = set()
        for x in order:
            can_add = True
            for (u, v) in conflicts_with[x]:
                if u in S_local and v in S_local:
                    can_add = False
                    break
            if can_add:
                S_local.add(x)
        return S_local
    
    # Try different orderings
    best_set = set(S)
    best_size = len(S)
    
    # Descending
    S1 = greedy(range(N, 0, -1))
    if len(S1) > best_size:
        best_set = S1
        best_size = len(S1)
    
    # Ascending
    S2 = greedy(range(1, N + 1))
    if len(S2) > best_size:
        best_set = S2
        best_size = len(S2)
    
    # Random orderings
    import random
    random.seed(42)
    for trial in range(20):
        order = list(range(1, N + 1))
        random.shuffle(order)
        S3 = greedy(order)
        if len(S3) > best_size:
            best_set = S3
            best_size = len(S3)
    
    # Local improvement on best_set
    # Try to add each number not in best_set by removing minimal conflicting numbers
    # For simplicity, try single-element swaps: replace x in S with y not in S if it increases size (impossible)
    # or maintains size but allows more additions later.
    
    # Let's try: for each y not in S, check if we can add y by removing some elements from S.
    # This is complex. Instead, let's try a simple heuristic:
    # Iterate through numbers 1..N. If not in S, try to add it. If conflict, 
    # remove the "least useful" conflicting elements.
    
    # Simpler: restart with a better greedy.
    # The key insight: numbers > N/2 = 90 cannot be the 'a' in 1/a = 1/b + 1/c with b,c < a
    # because 1/b + 1/c > 1/a if b,c < a. Wait, 1/b + 1/c >= 1/(a-1) + 1/(a-1) = 2/(a-1) > 1/a for a > 2.
    # So if a > 90, then b and c must be < a, but 1/b + 1/c > 1/a, so no solution with a > 90?
    # Wait: 1/a = 1/b + 1/c. If a > b and a > c, then 1/a < 1/b and 1/a < 1/c, so 1/b + 1/c > 1/a.
    # So a cannot be the largest. Thus in any forbidden triple, a is NOT the largest.
    # So the largest element in a forbidden triple is either b or c.
    
    # This means: if we take all numbers from 91 to 180, can they form a forbidden triple?
    # In a forbidden triple {a, b, c}, one of them is the "a" (the one in the denominator on the left).
    # The other two are b and c. We established a < b and a < c.
    # So if all three numbers are > 90, then a > 90, b > 90, c > 90.
    # But a < b and a < c, so b and c are the larger ones.
    # Is it possible? Let's check: smallest a > 90 is 91.
    # 1/91 = 1/b + 1/c with b, c > 91.
    # 1/b + 1/c <= 1/92 + 1/92 = 2/92 = 1/46 > 1/91.
    # So it's possible in principle. Let's check if there are actual solutions.
    
    # The seed program returns [91, ..., 180] which has size 90.
    # Let's see if we can do better.
    
    # Let's verify the seed: are there any forbidden triples within [91, 180]?
    # If not, size 90 is valid.
    
    # Let's check our forbidden set for triples all in [91, 180]
    triples_in_high = [t for t in forbidden if min(t) >= 91]
    # If empty, then [91..180] is valid.
    
    # Let's try to extend: can we add some numbers < 91 to the set [91..180]?
    # If we add x < 91, we need to ensure no forbidden triple {x, u, v} with u, v in [91..180] or {x, u, v} with one in [91..180] and one < 91.
    
    # Let's try: start with [91..180] and try to add numbers from 1 to 90 in some order.
    
    S_base = set(range(91, N + 1))
    
    # Try to add numbers 1..90 in descending order (larger first, fewer conflicts?)
    for x in range(90, 0, -1):
        can_add = True
        for (u, v) in conflicts_with[x]:
            if u in S_base and v in S_base:
                can_add = False
                break
        if can_add:
            S_base.add(x)
    
    if len(S_base) > best_size:
        best_set = S_base
        best_size = len(S_base)
    
    # Also try ascending
    S_base2 = set(range(91, N + 1))
    for x in range(1, 91):
        can_add = True
        for (u, v) in conflicts_with[x]:
            if u in S_base2 and v in S_base2:
                can_add = False
                break
        if can_add:
            S_base2.add(x)
    
    if len(S_base2) > best_size:
        best_set = S_base2
        best_size = len(S_base2)
    
    # Local search: try to improve by swapping
    # For each y not in best_set, for each x in best_set, check if (best_set - {x} + {y}) is valid
    # This is O(N^2 * conflicts) which might be slow. Let's do a limited version.
    
    # Actually, let's just return the best we found.
    return sorted(best_set)

# EVOLVE-BLOCK-END
