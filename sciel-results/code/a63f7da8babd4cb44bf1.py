"""Rewritten program: maximize subset of [1,180] with no distinct triple a,b,c s.t. 1/a=1/b+1/c."""
# EVOLVE-BLOCK-START
def construct():
    N = 180
    # Find all forbidden triples (a,b,c) distinct in [1,N] with 1/a = 1/b + 1/c
    # c = a*b/(b-a), need b > a, c integer in [1,N], c distinct from a,b
    forbidden = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if 1 <= c <= N and c != a and c != b:
                    forbidden.add(frozenset((a, b, c)))
    # Also consider permutations: the equation is symmetric in b,c (both on RHS)
    # 1/a = 1/b + 1/c is symmetric in b and c, so we've covered all by fixing a as the "left" element
    # But a could also be the larger... no: 1/a = 1/b + 1/c means 1/a > 1/b and 1/a > 1/c, so a < b and a < c.
    # So a is always the smallest. Good, our enumeration with a < b and c > a covers all.
    # However, we need c > a as well. Let's verify: since 1/a = 1/b + 1/c > 1/b, so a < b. Similarly a < c.
    # So a is min. Our loop has a < b; c = ab/(b-a). Is c > a? c/a = b/(b-a) > 1 iff b > b-a iff a > 0. Yes.
    # So all good.

    # Build adjacency: for each element, list of triples it participates in
    from collections import defaultdict
    triples_of = defaultdict(list)
    for tri in forbidden:
        for x in tri:
            triples_of[x].append(tri)

    # Greedy: start with all elements, remove to break triples
    included = set(range(1, N + 1))
    # Local search: try to add back removed elements / swap to improve
    # Use a simple approach: iteratively try to add any excluded element that doesn't create a triple
    excluded = set(range(1, N + 1)) - included

    def is_valid(s):
        for tri in forbidden:
            if tri.issubset(s):
                return False
        return True

    # Start with all, then greedily remove
    included = set(range(1, N + 1))
    # Sort elements by degree (number of triples) descending, remove if needed
    # Better: use a max-flow-like or just local search
    # Local search: start with all, while there's a bad triple, remove the element with highest "cost"
    # Cost = number of triples it's in (removing it breaks many)
    changed = True
    while changed:
        changed = False
        # Find a violating triple
        bad = None
        for tri in forbidden:
            if tri.issubset(included):
                bad = tri
                break
        if bad is None:
            break
        # Remove the element in bad with highest degree (in forbidden triples) to break most future issues
        # Actually we want to minimize total removals. Remove the one that appears in most triples among remaining.
        best = None
        best_cnt = -1
        for x in bad:
            cnt = sum(1 for t in triples_of[x] if t.issubset(included))
            if cnt > best_cnt:
                best_cnt = cnt
                best = x
        included.discard(best)
        changed = True

    # Now try to add back excluded elements
    improved = True
    while improved:
        improved = False
        for x in sorted(excluded, reverse=True):
            if x in included:
                continue
            if is_valid(included | {x}):
                included.add(x)
                excluded.discard(x)
                improved = True

    # Local search with swaps: try to swap an included element for two excluded
    # Simpler: try adding each excluded one at a time again (already done)
    # Try 1-for-1 swaps
    for x in list(included):
        for y in list(excluded):
            new_inc = (included - {x}) | {y}
            if is_valid(new_inc) and len(new_inc) > len(included):
                pass  # same size, no gain
            # Only useful if it allows adding more later; skip for simplicity

    return sorted(included)
# EVOLVE-BLOCK-END
