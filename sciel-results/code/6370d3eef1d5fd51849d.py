import random
import time
from itertools import combinations

N = 16
DEADLINE = 140.0  # safety margin before 150s kill


def _all_pairs(n):
    return list(combinations(range(n), 2))


# Precompute all triples and the pairs they contain
_ALL_TRIPLES = list(combinations(range(N), 3))


def _independence_number(triples):
    """Compute the independence number of the hypergraph via branch and bound.
    Independence number = largest subset of points containing no triple."""
    if not triples:
        return N
    # Represent triples as frozensets for fast membership
    tsets = [frozenset(t) for t in triples]
    points = set(range(N))

    best = [0]

    # For each point, precompute which triples contain it
    point_to_triples = {p: [] for p in range(N)}
    for idx, t in enumerate(tsets):
        for p in t:
            point_to_triples[p].append(idx)

    def search(remaining, chosen_size):
        if not remaining:
            if chosen_size > best[0]:
                best[0] = chosen_size
            return
        # Upper bound: chosen + |remaining|
        if chosen_size + len(remaining) <= best[0]:
            return
        # Pick a point with the most covering triples (branching heuristic)
        # Try branching: include the point or exclude it.
        # Choose point with max impact
        p = max(remaining, key=lambda x: len(point_to_triples[x]))
        # Branch 1: include p
        # After including p, we must exclude all points that would complete a triple with p
        # i.e., for any triple containing p, the other two points must not both be in remaining
        # Actually, including p means: no triple can be fully in (chosen ∪ {p}) ∪ future
        # Simpler: including p removes from consideration any triple that is subset of remaining ∪ {p}
        # We need to ensure no triple is fully contained in the chosen set.
        # Standard approach: if we include p, we must remove from 'remaining' any set of points
        # that together with p form a triple. Specifically, for each triple t containing p,
        # if the other 2 points of t are in remaining, we cannot include both of them.
        # This is complex; use a simpler recursive formulation.
        #
        # Alternative cleaner formulation:
        # State: set S of candidate points, set T of triples fully contained in S.
        # Independence number = max size of subset with no triple.
        # Recursion: pick a triple t in T. Either exclude one of its points.
        #
        # Let me use a different, well-known branch-and-bound for hypergraph independence.

    # Better approach: branch on a triple
    def search2(candidates, active_triples):
        """candidates: set of points still available
           active_triples: list of indices of triples fully contained in candidates
           We want max independent set."""
        if not active_triples:
            # No constraint, take all
            return len(candidates)
        # Upper bound
        if len(candidates) <= best[0]:
            return len(candidates)
        # Pick a triple with minimum support (fewest points in candidates)
        t_idx = min(active_triples, key=lambda i: len(tsets[i] & candidates))
        t = tsets[t_idx]
        # Branch: for each point in t ∩ candidates, exclude that point
        # (at least one point of each active triple must be excluded)
        pts_in_t = [p for p in t if p in candidates]
        max_val = 0
        for p in pts_in_t:
            new_candidates = candidates - {p}
            new_active = []
            for i in active_triples:
                if tsets[i] <= new_candidates:
                    new_active.append(i)
            val = search2(new_candidates, new_active)
            if val > max_val:
                max_val = val
            # Update global best
            if max_val > best[0]:
                best[0] = max_val
        return max_val

    search2(frozenset(range(N)), list(range(len(tsets))))
    return best[0]


def _independence_number_fast(triples):
    """Faster independence number using a greedy upper bound + local search verification."""
    if not triples:
        return N
    tsets = [frozenset(t) for t in triples]
    n_triples = len(tsets)

    # Greedy: repeatedly pick a point, remove all triples it's in, until no triples left.
    # This gives a lower bound on independence number.
    def greedy_lb():
        remaining = set(range(N))
        active = set(range(n_triples))
        count = 0
        while active:
            # Pick point in most active triples
            if not remaining:
                break
            point_counts = {p: 0 for p in remaining}
            for i in active:
                for p in tsets[i]:
                    if p in remaining:
                        point_counts[p] += 1
            if max(point_counts.values()) == 0:
                count += len(remaining)
                break
            p = max(remaining, key=lambda x: point_counts[x])
            count += 1
            remaining.discard(p)
            # Remove all triples containing p
            new_active = set()
            for i in active:
                if p not in tsets[i]:
                    new_active.add(i)
            active = new_active
        return count

    lb = greedy_lb()

    # Try to verify if lb is optimal via small branch and bound
    # Use the branch on triple approach but with time limit
    best = [lb]
    point_to_triples = {p: [] for p in range(N)}
    for idx, t in enumerate(tsets):
        for p in t:
            point_to_triples[p].append(idx)

    def bounded_search(candidates, active_triples, depth=0):
        if depth > 8:  # limit recursion depth for speed
            return len(candidates)
        if not active_triples:
            return len(candidates)
        if len(candidates) <= best[0]:
            return len(candidates)
        # Pick triple with fewest points in candidates
        t_idx = min(active_triples, key=lambda i: len(tsets[i] & candidates))
        t = tsets[t_idx]
        pts_in_t = [p for p in t if p in candidates]
        max_val = 0
        for p in pts_in_t:
            new_candidates = candidates - {p}
            new_active = [i for i in active_triples if tsets[i] <= new_candidates]
            val = bounded_search(new_candidates, new_active, depth + 1)
            if val > max_val:
                max_val = val
        if max_val > best[0]:
            best[0] = max_val
        return max_val

    bounded_search(frozenset(range(N)), list(range(n_triples)))
    return best[0]


def _is_linear(triples):
    """Check if no pair of points appears in more than one triple."""
    pair_count = {}
    for t in triples:
        for a, b in combinations(t, 2):
            key = (min(a, b), max(a, b))
            pair_count[key] = pair_count.get(key, 0) + 1
            if pair_count[key] > 1:
                return False
    return True


def _can_add(triples, new_triple, pair_used):
    """Check if adding new_triple maintains linearity, given pair_used set."""
    for a, b in combinations(new_triple, 2):
        key = (min(a, b), max(a, b))
        if key in pair_used:
            return False
    return True


def construct():
    start_time = time.time()
    random.seed(42)

    best_system = []
    best_score = N  # independence number, lower is better

    # Precompute all pairs for linearity check
    # pair_used will be a set of (min, max) pairs

    def try_improve(system, pair_used, triples_list, current_score):
        nonlocal best_system, best_score
        if current_score < best_score:
            best_score = current_score
            best_system = list(system)
            return True
        return False

    # Start with a greedy construction
    def greedy_build():
        system = []
        pair_used = set()
        # Randomly order triples
        shuffled = list(_ALL_TRIPLES)
        random.shuffle(shuffled)
        for t in shuffled:
            if _can_add(system, t, pair_used):
                system.append(t)
                for a, b in combinations(t, 2):
                    pair_used.add((min(a, b), max(a, b)))
        return system, pair_used

    # Initial greedy
    system, pair_used = greedy_build()
    score = _independence_number_fast(system)
    try_improve(system, pair_used, system, score)

    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.9995

    while time.time() - start_time < DEADLINE:
        T = max(T_min, T * cooling)

        # Choose a move
        move_type = random.random()
        new_system = list(system)
        new_pair_used = set(pair_used)

        if move_type < 0.4 and len(new_system) < 80:
            # Add a random triple
            t = random.choice(_ALL_TRIPLES)
            if _can_add(new_system, t, new_pair_used):
                new_system.append(t)
                for a, b in combinations(t, 2):
                    new_pair_used.add((min(a, b), max(a, b)))
            else:
                continue
        elif move_type < 0.6 and len(new_system) > 0:
            # Remove a random triple
            idx = random.randint(0, len(new_system) - 1)
            removed = new_system[idx]
            new_system.pop(idx)
            for a, b in combinations(removed, 2):
                new_pair_used.discard((min(a, b), max(a, b)))
        else:
            # Swap: remove one, add another
            if len(new_system) == 0:
                continue
            idx = random.randint(0, len(new_system) - 1)
            removed = new_system[idx]
            new_system.pop(idx)
            for a, b in combinations(removed, 2):
                new_pair_used.discard((min(a, b), max(a, b)))
            t = random.choice(_ALL_TRIPLES)
            if not _can_add(new_system, t, new_pair_used):
                # Try a few more
                added = False
                for _ in range(10):
                    t = random.choice(_ALL_TRIPLES)
                    if _can_add(new_system, t, new_pair_used):
                        added = True
                        break
                if not added:
                    continue
            new_system.append(t)
            for a, b in combinations(t, 2):
                new_pair_used.add((min(a, b), max(a, b)))

        # Evaluate new system
        new_score = _independence_number_fast(new_system)
        delta = new_score - score
        if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-9)):
            system = new_system
            pair_used = new_pair_used
            score = new_score
            try_improve(system, pair_used, system, score)

    # Final check
    if best_system and _is_linear(best_system):
        return [list(t) for t in best_system]

    # Fallback: return a valid linear system
    system, pair_used = greedy_build()
    if system and _is_linear(system):
        return [list(t) for t in system]

    # Absolute fallback: 5 disjoint triples
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
