import time
import random
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = 82  # safety margin before 90s kill

    N = 180

    # Precompute all reciprocal Schur triples (a, b, c) with 1/a = 1/b + 1/c, a,b,c distinct
    # 1/a = 1/b + 1/c => bc = a(b+c) => (b-a)(c-a) = a^2
    # For each a, factor pairs (d, e) of a^2 with d*e = a^2, d != e
    # then b = a + d, c = a + e, with b,c in [1, N] and all distinct

    triples = []
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                e = a2 // d
                if d == e:
                    continue
                # Two cases: (b-a, c-a) = (d, e) or (e, d)
                b1 = a + d
                c1 = a + e
                if b1 <= N and c1 <= N and b1 != c1 and b1 != a and c1 != a:
                    triples.append((a, b1, c1))
                b2 = a + e
                c2 = a + d
                if b2 <= N and c2 <= N and b2 != c2 and b2 != a and c2 != a:
                    if (a, b2, c2) != (a, b1, c1):
                        triples.append((a, b2, c2))

    # Deduplicate
    seen = set()
    unique_triples = []
    for t in triples:
        fs = frozenset(t)
        if len(fs) == 3:
            if fs not in seen:
                seen.add(fs)
                unique_triples.append(tuple(sorted(fs)))

    num_triples = len(unique_triples)

    # Build element -> set of triple indices containing it
    elem_triple_indices = {i: [] for i in range(1, N + 1)}
    for idx, t in enumerate(unique_triples):
        for x in t:
            if x in elem_triple_indices:
                elem_triple_indices[x].append(idx)

    def is_valid(S):
        in_set = set(S)
        for t in unique_triples:
            if all(x in in_set for x in t):
                return False
        return True

    def count_violations(S):
        """Return list of violated triple indices."""
        in_set = set(S)
        viol = []
        for idx, t in enumerate(unique_triples):
            if all(x in in_set for x in t):
                viol.append(idx)
        return viol

    def fix_by_removal(S):
        """Remove minimal elements to make S valid. Greedy: remove element in most violated triples."""
        changed = True
        while changed:
            changed = False
            in_set = set(S)
            viol_count = {i: 0 for i in in_set}
            for t in unique_triples:
                if all(x in in_set for x in t):
                    for x in t:
                        viol_count[x] += 1
            max_v = 0
            worst = None
            for x in in_set:
                v = viol_count.get(x, 0)
                if v > max_v:
                    max_v = v
                    worst = x
            if worst is not None and max_v > 0:
                S.discard(worst)
                changed = True
        return S

    def try_add_element(S, x):
        """Try to add x to S, returning the fixed set if it improves size."""
        trial = set(S)
        trial.add(x)
        fixed = fix_by_removal(trial)
        return fixed

    def hill_climb(S, max_rounds=30):
        """Hill climb by trying to add each excluded element."""
        S = set(S)
        for _ in range(max_rounds):
            if time.time() > DEADLINE:
                break
            excluded = [i for i in range(1, N + 1) if i not in S]
            if not excluded:
                break
            improved = False
            # Sort excluded by how few constraints they create (heuristic)
            # Elements that appear in fewer triples are safer to add
            def constraint_count(x):
                return len(elem_triple_indices.get(x, []))
            excluded.sort(key=constraint_count)
            for x in excluded:
                if time.time() > DEADLINE:
                    break
                fixed = try_add_element(S, x)
                if len(fixed) > len(S):
                    S = fixed
                    improved = True
            if not improved:
                break
        return S

    def swap_search(S, max_rounds=20):
        """Try swapping: remove one element, add another."""
        S = set(S)
        for _ in range(max_rounds):
            if time.time() > DEADLINE:
                break
            excluded = [i for i in range(1, N + 1) if i not in S]
            if not excluded:
                break
            improved = False
            # Try removing each element in S and adding each excluded element
            # Limit to avoid O(n^2) per round
            for x_remove in list(S):
                if time.time() > DEADLINE:
                    break
                for x_add in excluded:
                    if time.time() > DEADLINE:
                        break
                    trial = set(S)
                    trial.discard(x_remove)
                    trial.add(x_add)
                    fixed = fix_by_removal(trial)
                    if len(fixed) > len(S):
                        S = fixed
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return S

    # Strategy 1: Start from full set and greedily remove
    S_full = set(range(1, N + 1))
    S_greedy = fix_by_removal(S_full)
    best_S = hill_climb(S_greedy)
    best_size = len(best_S)

    # Strategy 2: Try multiple upper intervals
    for start_val in range(60, 90, 5):
        if time.time() > DEADLINE:
            break
        base_S = set(range(start_val, N + 1))
        S_interval = hill_climb(base_S)
        if len(S_interval) > best_size:
            best_S = S_interval
            best_size = len(best_S)

    # Strategy 3: Random initialization with hill climbing
    rng = random.Random(42)
    for restart in range(10):
        if time.time() > DEADLINE:
            break
        # Random subset of varying sizes
        size = rng.randint(int(N * 0.5), int(N * 0.85))
        random_set = set(rng.sample(range(1, N + 1), size))
        fixed = fix_by_removal(random_set)
        fixed = hill_climb(fixed)
        if len(fixed) > best_size:
            best_S = fixed
            best_size = len(best_S)

    # Strategy 4: Swap search from best solution
    if time.time() < DEADLINE - 5:
        S_swapped = swap_search(best_S)
        if len(S_swapped) > best_size:
            best_S = S_swapped
            best_size = len(best_S)

    # Strategy 5: More random restarts with swap
    for restart in range(5):
        if time.time() > DEADLINE:
            break
        size = rng.randint(int(N * 0.6), int(N * 0.9))
        random_set = set(rng.sample(range(1, N + 1), size))
        fixed = fix_by_removal(random_set)
        fixed = hill_climb(fixed)
        fixed = swap_search(fixed)
        if len(fixed) > best_size:
            best_S = fixed
            best_size = len(best_S)

    # Final verification and cleanup
    final = sorted(best_S)
    if not final:
        final = list(range(73, 180))

    # Double-check validity
    in_final = set(final)
    for t in unique_triples:
        if all(x in in_final for x in t):
            final.remove(t[0])
            in_final.discard(t[0])

    if not final:
        final = [1]

    return final
