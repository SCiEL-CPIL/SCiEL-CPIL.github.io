"""Combined program: precomputed triples + fast greedy + local search + swaps."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under the 90s kill
    N = 180
    rng = random.Random(20240517)

    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # a = b*c / (b+c), so (b+c) | (b*c).
    # Since 1/a = 1/b + 1/c > 1/c, we have a < c. Similarly a < b.
    # So a is always the smallest of the three distinct elements.
    
    triples = []  # list of (a, b, c) with a < b < c
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if a < b and 1 <= a <= N:
                    triples.append((a, b, c))
    
    # For fast checking, build: for each a, the set of (b, c) pairs that form a triple with a.
    triple_map = {}
    for a, b, c in triples:
        if a not in triple_map:
            triple_map[a] = set()
        triple_map[a].add((b, c))
    
    # Build reverse maps: for each element x, list of (other1, other2) such that
    # {x, other1, other2} forms a forbidden triple.
    # When we add element x to a set S, we need to check if any triple containing x
    # has both other elements already in S.
    # For element x, the triples it participates in are:
    #   - As 'a': triples where x is the smallest: (x, b, c) for (b,c) in triple_map[x]
    #   - As 'b': triples where x is the middle: (a, x, c) 
    #   - As 'c': triples where x is the largest: (a, b, x)
    
    # Build conflict_pairs[x] = list of (y, z) such that {x, y, z} is a forbidden triple
    conflict_pairs = {x: [] for x in range(1, N + 1)}
    for a, b, c in triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    def has_triple(S):
        """Check if set S contains any forbidden triple."""
        for a in S:
            if a in triple_map:
                for (b, c) in triple_map[a]:
                    if b in S and c in S:
                        return True
        return False
    
    def count_conflicts(x, S):
        """Count how many forbidden triples would be formed if x is added to S."""
        count = 0
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                count += 1
        return count
    
    # Greedy construction with a given order
    def greedy_order(order):
        S = set()
        for x in order:
            if count_conflicts(x, S) == 0:
                S.add(x)
        return S
    
    # Phase 1: Multiple greedy constructions with different orderings
    best = set(range(91, N + 1))  # baseline: 90 elements
    best_size = len(best)
    
    # Descending order
    cand = greedy_order(list(range(N, 0, -1)))
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)
    
    # Ascending order
    cand = greedy_order(list(range(1, N + 1)))
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)
    
    # Random orderings
    for _ in range(200):
        if time.time() >= deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        cand = greedy_order(order)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)
    
    # Phase 2: Local search - try to add elements
    for _ in range(500):
        if time.time() >= deadline:
            break
        improved = False
        for x in range(1, N + 1):
            if x in best:
                continue
            if count_conflicts(x, best) == 0:
                best.add(x)
                best_size += 1
                improved = True
                break
        if not improved:
            break
    
    # Phase 3: Swap search - remove one element, add another
    # For a swap (remove y, add x), we need:
    #   - x not in best
    #   - y in best
    #   - After removing y and adding x, no new conflicts
    # The conflicts of x in (best - {y}) must be 0
    # We can compute this by counting conflicts of x in best, then subtracting
    # conflicts that involved y.
    
    for _ in range(50000):
        if time.time() >= deadline:
            break
        if not best:
            break
        x = rng.randint(1, N)
        if x in best:
            continue
        y = rng.choice(list(best))
        # Check if x can be added after removing y
        # Count conflicts of x in (best - {y})
        # This equals count_conflicts(x, best) minus conflicts where y is involved
        conflicts_in_best = count_conflicts(x, best)
        # Conflicts involving y: for each (u, v) in conflict_pairs[x],
        # if y == u or y == v, and the other is in best, that's a conflict in best
        # but not in (best - {y})
        conflicts_with_y = 0
        for (u, v) in conflict_pairs[x]:
            if u == y and v in best:
                conflicts_with_y += 1
            elif v == y and u in best:
                conflicts_with_y += 1
        if conflicts_in_best - conflicts_with_y == 0:
            best.discard(y)
            best.add(x)
    
    # Phase 4: More local search after swaps
    for _ in range(500):
        if time.time() >= deadline:
            break
        improved = False
        for x in range(1, N + 1):
            if x in best:
                continue
            if count_conflicts(x, best) == 0:
                best.add(x)
                best_size += 1
                improved = True
                break
        if not improved:
            break
    
    # Phase 5: Random restarts with diverse seeds
    for _ in range(5000):
        if time.time() >= deadline:
            break
        size = rng.randint(60, 120)
        s = set(rng.sample(range(1, N + 1), size))
        if has_triple(s):
            continue
        remaining = [x for x in range(1, N + 1) if x not in s]
        rng.shuffle(remaining)
        for x in remaining:
            if time.time() >= deadline:
                break
            if count_conflicts(x, s) == 0:
                s.add(x)
        if len(s) > best_size:
            best = s
            best_size = len(s)
    
    # Final verification
    if has_triple(best):
        best = set(range(91, N + 1))
    
    return list(best)
# EVOLVE-BLOCK-END
