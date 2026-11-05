import time
import random
from collections import defaultdict

def construct():
    start = time.time()
    DEADLINE = 82  # safety margin before 90s kill

    N = 180

    # Precompute all reciprocal Schur triples (a, b, c) with 1/a = 1/b + 1/c, a,b,c distinct
    # 1/a = 1/b + 1/c => bc = a(b+c) => (b-a)(c-a) = a^2
    # So for each a, factor pairs (d, e) of a^2 with d*e = a^2, d != e
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
                b2 = a + e
                c2 = a + d
                # Both orderings give the same triple {a, b, c}
                if b1 <= N and c1 <= N and b1 != c1 and b1 != a and c1 != a:
                    triples.append((a, b1, c1))
                # b2, c2 is just swap of b1, c1, so same triple
                # But we need to make sure we don't miss cases where d > e
                # Actually since we iterate d up to sqrt(a2), we get each pair once
                # But d and e are the two factors, so b=a+d, c=a+e covers it
                # Wait, we also need to consider that d could be the larger factor
                # No, since d*e=a2 and we take d <= e, b=a+d <= a+e=c
                # So this is fine

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

    # Build conflict graph: edge between i and j if they appear together in some triple
    # Actually, for independent set, we need: cannot pick two elements that are in the same triple
    # Wait, no. The constraint is: no three distinct elements a,b,c in S with 1/a=1/b+1/c
    # This is a 3-uniform hypergraph constraint, not a 2-uniform graph constraint.
    # So we can't directly use standard MIS on a 2-uniform graph.
    # 
    # However, we can model it as: for each triple (a,b,c), at least one of a,b,c must be excluded.
    # This is a hitting set / set cover problem, or equivalently, we want to maximize the set S
    # such that for every triple, S does not contain all three elements.
    #
    # Let's use a different approach: greedy + local search on the actual constraint.

    # For each element, list the triples it participates in
    elem_triples = defaultdict(list)
    for idx, t in enumerate(unique_triples):
        for x in t:
            elem_triples[x].append(idx)

    def count_violations(S):
        """Count number of triples fully contained in S."""
        in_set = set(S)
        viol = 0
        for t in unique_triples:
            if all(x in in_set for x in t):
                viol += 1
        return viol

    def is_valid(S):
        in_set = set(S)
        for t in unique_triples:
            if all(x in in_set for x in t):
                return False
        return True

    def violation_count_for_element(S, x):
        """Count how many triples containing x are fully in S."""
        in_set = set(S)
        cnt = 0
        for t_idx in elem_triples.get(x, []):
            t = unique_triples[t_idx]
            if all(y in in_set for y in t):
                cnt += 1
        return cnt

    def fix_by_removal(S):
        """Remove minimal elements to make S valid."""
        S = set(S)
        changed = True
        while changed:
            changed = False
            in_set = set(S)
            # Find all violated triples
            violated = []
            for t in unique_triples:
                if all(x in in_set for x in t):
                    violated.append(t)
            if not violated:
                break
            # Find the element that appears in the most violated triples
            elem_count = defaultdict(int)
            for t in violated:
                for x in t:
                    elem_count[x] += 1
            worst = max(elem_count, key=lambda x: elem_count[x])
            S.discard(worst)
            changed = True
        return S

    # Strategy: Use a greedy approach with multiple random orderings
    # Then apply local search (add/remove) to improve

    best_S = set(range(73, 180))  # Known good baseline, size 107
    best_size = len(best_S)

    # Greedy from random orderings
    rng = random.Random(42)
    for restart in range(200):
        if time.time() > DEADLINE:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S = set()
        for x in order:
            trial = S | {x}
            if is_valid(trial):
                S = trial
        if len(S) > best_size:
            best_S = S
            best_size = len(S)

    # Local search from best
    for _ in range(100):
        if time.time() > DEADLINE:
            break
        improved = False
        # Try adding each excluded element
        excluded = [i for i in range(1, N + 1) if i not in best_S]
        rng.shuffle(excluded)
        for x in excluded:
            if time.time() > DEADLINE:
                break
            trial = set(best_S)
            trial.add(x)
            if is_valid(trial):
                best_S = trial
                best_size += 1
                improved = True
                break
        if not improved:
            # Try swapping: remove one, add another
            found_swap = False
            for x in list(best_S):
                if time.time() > DEADLINE:
                    break
                for y in excluded:
                    if time.time() > DEADLINE:
                        break
                    trial = set(best_S)
                    trial.discard(x)
                    trial.add(y)
                    if is_valid(trial) and len(trial) > best_size:
                        best_S = trial
                        found_swap = True
                        break
                if found_swap:
                    break
            if not found_swap:
                break

    # Simulated annealing style
    if time.time() < DEADLINE - 5:
        current_S = set(best_S)
        current_size = len(current_S)
        T = 10.0
        for iteration in range(5000):
            if time.time() > DEADLINE:
                break
            if current_size >= N:
                break
            # Random move: add, remove, or swap
            move_type = rng.random()
            if move_type < 0.5:
                # Add
                excluded = [i for i in range(1, N + 1) if i not in current_S]
                if not excluded:
                    continue
                x = rng.choice(excluded)
                new_S = current_S | {x}
            elif move_type < 0.7:
                # Remove
                if not current_S:
                    continue
                x = rng.choice(list(current_S))
                new_S = current_S - {x}
            else:
                # Swap
                excluded = [i for i in range(1, N + 1) if i not in current_S]
                if not excluded or not current_S:
                    continue
                x = rng.choice(list(current_S))
                y = rng.choice(excluded)
                new_S = (current_S - {x}) | {y}

            new_viol = count_violations(new_S)
            old_viol = count_violations(current_S)
            new_size = len(new_S)
            old_size = len(current_S)

            # Accept if better or with probability based on "energy" difference
            delta_size = new_size - old_size
            delta_viol = new_viol - old_viol
            # We want to minimize violations and maximize size
            # Use a combined score: size - 100*violations
            old_score = old_size - 100 * old_viol
            new_score = new_size - 100 * new_viol
            delta = new_score - old_score
            if delta >= 0 or rng.random() < (1.0 / (1.0 + delta)) if delta < 0 else True:
                current_S = new_S
                current_size = new_size
                if current_size > best_size and is_valid(current_S):
                    best_S = set(current_S)
                    best_size = current_size
            T *= 0.999

    # Final verification
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
