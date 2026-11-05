import time
import random

def construct():
    start = time.time()
    DEADLINE = 85  # safety margin before 90s kill

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
        in_set = set(S)
        count = 0
        for t in unique_triples:
            if all(x in in_set for x in t):
                count += 1
        return count

    def fix_by_removal(S):
        """Remove minimal elements to make S valid. Greedy: remove element in most violated triples."""
        changed = True
        while changed:
            changed = False
            in_set = set(S)
            viol = {i: 0 for i in in_set}
            for t in unique_triples:
                if all(x in in_set for x in t):
                    for x in t:
                        viol[x] += 1
            max_v = 0
            worst = None
            for x in in_set:
                v = viol.get(x, 0)
                if v > max_v:
                    max_v = v
                    worst = x
            if worst is not None and max_v > 0:
                S.discard(worst)
                changed = True
        return S

    # Strategy 1: Start from full set and greedily remove
    S_full = set(range(1, N + 1))
    S_greedy = fix_by_removal(S_full)
    best_S = set(S_greedy)
    best_size = len(best_S)

    # Strategy 2: Start from upper interval and try to add lower elements
    base_S = set(range(73, 180))  # {73, ..., 179}, size 107
    S_interval = set(base_S)
    if len(S_interval) > best_size:
        best_S = S_interval
        best_size = len(best_S)

    # Hill climbing: try adding each excluded element
    for _ in range(50):
        if time.time() > DEADLINE:
            break
        excluded = [i for i in range(1, N + 1) if i not in best_S]
        if not excluded:
            break
        improved = False
        for x in excluded:
            if time.time() > DEADLINE:
                break
            trial = set(best_S)
            trial.add(x)
            fixed = fix_by_removal(trial)
            if len(fixed) > best_size:
                best_S = fixed
                best_size = len(fixed)
                improved = True
        if not improved:
            break

    # Simulated annealing / random search with swap moves
    rng = random.Random(42)
    for iteration in range(3000):
        if time.time() > DEADLINE:
            break
        if best_size >= N:
            break
        excluded = [i for i in range(1, N + 1) if i not in best_S]
        if not excluded:
            break
        
        # Randomly choose between add and swap
        if len(excluded) > 0 and (len(best_S) == 0 or rng.random() < 0.5):
            # Add move
            x = rng.choice(excluded)
            trial = set(best_S)
            trial.add(x)
            fixed = fix_by_removal(trial)
            if len(fixed) > best_size:
                best_S = fixed
                best_size = len(best_S)
        else:
            # Swap move
            x_out = rng.choice(list(best_S))
            x_in = rng.choice(excluded)
            trial = set(best_S)
            trial.discard(x_out)
            trial.add(x_in)
            fixed = fix_by_removal(trial)
            if len(fixed) > best_size:
                best_S = fixed
                best_size = len(best_S)

    # Try different starting points with random initialization
    for restart in range(8):
        if time.time() > DEADLINE:
            break
        # Random subset of size ~N*0.7
        random_set = set(rng.sample(range(1, N + 1), int(N * 0.7)))
        fixed = fix_by_removal(random_set)
        # Hill climb from this point with swaps
        for _ in range(30):
            if time.time() > DEADLINE:
                break
            excluded = [i for i in range(1, N + 1) if i not in fixed]
            if not excluded:
                break
            improved = False
            for x in excluded:
                if time.time() > DEADLINE:
                    break
                trial = set(fixed)
                trial.add(x)
                fixed_trial = fix_by_removal(trial)
                if len(fixed_trial) > len(fixed):
                    fixed = fixed_trial
                    improved = True
            if not improved:
                break
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
