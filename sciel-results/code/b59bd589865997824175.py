"""Rewritten program for the suite task."""
import time
import random

# EVOLVE-BLOCK-START

def _conflict_pairs(n=180):
    """Return a list of frozenset pairs {a,b} that appear in some forbidden triple 1/a=1/b+1/c with distinct a,b,c in [1,n]."""
    pairs = set()
    for b in range(1, n + 1):
        for c in range(b + 1, n + 1):
            # 1/a = 1/b + 1/c = (b+c)/(b*c)  =>  a = b*c/(b+c)
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= n and a != b and a != c:
                    # The triple is (a, b, c) with all distinct
                    # All three pairs conflict
                    pairs.add(frozenset((a, b)))
                    pairs.add(frozenset((a, c)))
                    pairs.add(frozenset((b, c)))
    return pairs


def _validate(s, n=180):
    """Check that no three distinct elements a,b,c in s satisfy 1/a = 1/b + 1/c."""
    sset = set(s)
    for a in s:
        for b in s:
            if b == a:
                continue
            for c in s:
                if c == a or c == b:
                    continue
                # Check 1/a = 1/b + 1/c  <=>  b*c = a*(b+c)
                if b * c == a * (b + c):
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = 50.0  # safety margin before 60s hard limit
    n = 180

    # Precompute conflict pairs
    conflict_pairs = _conflict_pairs(n)

    # Build adjacency: for each vertex, set of neighbors
    adj = [set() for _ in range(n + 1)]
    for p in conflict_pairs:
        a, b = tuple(p)
        adj[a].add(b)
        adj[b].add(a)

    # Greedy: pick vertices in order of increasing degree (fewest conflicts first)
    # This tends to include "safe" numbers (large numbers have fewer conflicts)
    order = sorted(range(1, n + 1), key=lambda x: len(adj[x]))

    def greedy_init(order_list):
        chosen = set()
        for v in order_list:
            if all(u not in chosen for u in adj[v]):
                chosen.add(v)
        return chosen

    best_set = greedy_init(order)
    best_size = len(best_set)

    # Local search: try to add any vertex not in set that has no conflicts with set
    # Then try swaps: remove one, add one or two
    def try_add(set_s):
        improved = False
        added_any = True
        while added_any:
            added_any = False
            for v in range(1, n + 1):
                if v not in set_s and all(u not in set_s for u in adj[v]):
                    set_s.add(v)
                    added_any = True
                    improved = True
        return set_s, improved

    def local_search(set_s, max_iters=200):
        current = set(set_s)
        for _ in range(max_iters):
            if time.time() - start_time > deadline - 5:
                break
            # Try adding
            current, _ = try_add(current)
            # Try single swaps: remove v, add w
            improved = False
            candidates_to_remove = list(current)
            random.shuffle(candidates_to_remove)
            for v in candidates_to_remove:
                if len(current) <= 1:
                    break
                # If we remove v, what can we add?
                new_set = current - {v}
                # Try to add multiple vertices
                for w in range(1, n + 1):
                    if w not in new_set and all(u not in new_set for u in adj[w]):
                        new_set.add(w)
                        # Continue adding greedily
                        changed = True
                        while changed:
                            changed = False
                            for w2 in range(1, n + 1):
                                if w2 not in new_set and all(u not in new_set for u in adj[w2]):
                                    new_set.add(w2)
                                    changed = True
                if len(new_set) > len(current):
                    current = new_set
                    improved = True
                    break
            if not improved:
                # Try 2-for-1 swaps
                improved = False
                for v in list(current):
                    new_set = current - {v}
                    # Try to add
                    changed = True
                    while changed:
                        changed = False
                        for w in range(1, n + 1):
                            if w not in new_set and all(u not in new_set for u in adj[w]):
                                new_set.add(w)
                                changed = True
                    if len(new_set) > len(current):
                        current = new_set
                        improved = True
                        break
                if not improved:
                    break
        return current

    # Run local search
    best_set = local_search(best_set)
    best_size = len(best_set)

    # ILS: perturb and re-optimize
    rng = random.Random(42)
    iterations = 0
    while time.time() - start_time < deadline - 2:
        iterations += 1
        # Perturb: remove k random elements
        k = rng.randint(2, min(6, max(1, len(best_set) // 10)))
        perturbed = set(best_set)
        to_remove = rng.sample(list(perturbed), min(k, len(perturbed)))
        for v in to_remove:
            perturbed.discard(v)

        # Re-optimize
        candidate = local_search(perturbed, max_iters=100)

        if len(candidate) > best_size:
            best_set = candidate
            best_size = len(candidate)

        # Also try a fresh greedy with random tie-breaking
        if iterations % 5 == 0:
            random_order = list(range(1, n + 1))
            # Sort by degree with random tie-breaking
            random_order.sort(key=lambda x: (len(adj[x]), rng.random()))
            fresh = greedy_init(random_order)
            fresh = local_search(fresh, max_iters=100)
            if len(fresh) > best_size:
                best_set = fresh
                best_size = len(fresh)

    # Final validation
    result = sorted(best_set)
    if not _validate(result, n):
        # Fallback to safe answer
        result = list(range(91, 181))

    return result

# EVOLVE-BLOCK-END
