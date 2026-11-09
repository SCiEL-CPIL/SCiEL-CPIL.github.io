import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140
    random.seed(12345)

    n = 16
    all_triples = list(combinations(range(n), 3))
    T = len(all_triples)  # 560
    triple_idx = {t: i for i, t in enumerate(all_triples)}

    # Build list of K(2,2,2) constraints: each is a set of 8 triple indices
    # 3 disjoint pairs -> 8 transversal triples
    constraints = []
    seen = set()
    for p1 in combinations(range(n), 2):
        for p2 in combinations(range(n), 2):
            if set(p1) & set(p2):
                continue
            for p3 in combinations(range(n), 2):
                if set(p1) & set(p3) or set(p2) & set(p3):
                    continue
                # 8 transversal triples
                tris = []
                for a in p1:
                    for b in p2:
                        for c in p3:
                            tris.append(tuple(sorted((a, b, c))))
                key = tuple(sorted(tris))
                if key in seen:
                    continue
                seen.add(key)
                constraints.append([triple_idx[t] for t in tris])

    C = len(constraints)

    # Initialize: all triples present
    present = [True] * T

    # For each constraint, count how many of its 8 triples are present
    def constraint_count(ci):
        return sum(present[i] for i in constraints[ci])

    # Greedy: while any constraint has count == 8, remove a triple from it
    # Prefer removing the triple that participates in the most violated constraints
    # (to fix multiple constraints at once)

    # Precompute: for each triple, which constraints it belongs to
    triple_to_constraints = [[] for _ in range(T)]
    for ci, c in enumerate(constraints):
        for ti in c:
            triple_to_constraints[ti].append(ci)

    # Track violation count per constraint (8 = violated)
    viol = [8] * C  # initially all 8 present

    # Count per triple: number of constraints where this triple is present
    # and the constraint is currently violated (count == 8)
    # Actually, we want: for each triple, how many violated constraints it's in

    # Greedy removal loop
    # Use a priority: remove triple that appears in the most violated constraints
    # To be efficient, maintain counts

    # For each triple, count of violated constraints it's in
    triple_viol_count = [0] * T
    for ti in range(T):
        for ci in triple_to_constraints[ti]:
            if viol[ci] == 8:
                triple_viol_count[ti] += 1

    removed = 0
    while True:
        if time.time() > deadline - 5:
            break
        # Find a violated constraint
        # Pick the triple with max triple_viol_count among present triples in violated constraints
        best_t = -1
        best_count = 0
        # Sample-based: find any violated constraint, then pick best triple in it
        # For speed, scan for violated constraints
        found = False
        for ci in range(C):
            if viol[ci] == 8:
                found = True
                # Among the 8 triples in this constraint, pick the one with max triple_viol_count
                for ti in constraints[ci]:
                    if present[ti] and triple_viol_count[ti] > best_count:
                        best_count = triple_viol_count[ti]
                        best_t = ti
                break

        if not found:
            break  # No more violated constraints

        # Remove best_t
        present[best_t] = False
        removed += 1

        # Update violation counts for constraints containing best_t
        for ci in triple_to_constraints[best_t]:
            viol[ci] -= 1
            # Update triple_viol_count for other triples in this constraint
            if viol[ci] < 8:
                # This constraint is no longer violated
                for ti in constraints[ci]:
                    if present[ti]:
                        triple_viol_count[ti] -= 1

    # Now try to add back removed triples if possible (local search)
    # A triple can be added if no constraint containing it would go to count 8
    # i.e., for all constraints c containing ti, viol[c] < 8 (since adding increases by 1)

    # Try to add triples back, in order of how many constraints they satisfy
    # (i.e., triples that are in few constraints are easier to add)
    added_back = 0
    # Sort all triples by number of constraints (ascending) - easier to add
    order = sorted(range(T), key=lambda ti: len(triple_to_constraints[ti]))

    # Multiple passes
    for _pass in range(50):
        if time.time() > deadline - 3:
            break
        progress = False
        for ti in order:
            if present[ti]:
                continue
            # Check if adding ti keeps all constraints < 8
            can_add = True
            for ci in triple_to_constraints[ti]:
                if viol[ci] >= 8:
                    can_add = False
                    break
            if can_add:
                present[ti] = True
                added_back += 1
                progress = True
                for ci in triple_to_constraints[ti]:
                    viol[ci] += 1

        if not progress:
            break

    # Swap-based local search: try replacing one present triple with another
    # to allow more additions
    # Simpler: just try adding in different orders
    # Try random restarts of the add-back phase

    best_count = sum(present)

    # Try a few random perturbations and re-optimize
    for _restart in range(20):
        if time.time() > deadline - 3:
            break

        # Start from current solution
        cur_present = present[:]
        cur_viol = viol[:]

        # Randomly remove a few triples
        present_indices = [i for i in range(T) if cur_present[i]]
        if len(present_indices) < 10:
            break
        for _ in range(random.randint(3, 8)):
            ri = random.choice(present_indices)
            cur_present[ri] = False
            for ci in triple_to_constraints[ri]:
                cur_viol[ci] -= 1
            if ri in present_indices:
                present_indices.remove(ri)

        # Now try to add back greedily in random order
        random.shuffle(order)
        for ti in order:
            if cur_present[ti]:
                continue
            can_add = True
            for ci in triple_to_constraints[ti]:
                if cur_viol[ci] >= 8:
                    can_add = False
                    break
            if can_add:
                cur_present[ti] = True
                for ci in triple_to_constraints[ti]:
                    cur_viol[ci] += 1

        cur_count = sum(cur_present)
        if cur_count > best_count:
            best_count = cur_count
            present = cur_present
            viol = cur_viol

    # Build result
    result = [list(all_triples[i]) for i in range(T) if present[i]]
    return result
