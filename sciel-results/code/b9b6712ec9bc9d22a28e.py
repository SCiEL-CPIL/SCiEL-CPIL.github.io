import time
import random
import numpy as np

N = 16
DEADLINE = 320.0
N_TOTAL = N * N

_ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distances
_D2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i, p1 in enumerate(_ALL_POINTS):
    for j, p2 in enumerate(_ALL_POINTS):
        _D2[i, j] = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute isosceles relation masks
# iso_mask[i][j] = bitmask of points k such that (i,j,k) forms an isosceles triangle
iso_mask = [[0] * N_TOTAL for _ in range(N_TOTAL)]

for i in range(N_TOTAL):
    for j in range(i + 1, N_TOTAL):
        di_j = _D2[i, j]
        mask = 0
        for k in range(N_TOTAL):
            if k == i or k == j:
                continue
            di_k = _D2[i, k]
            dj_k = _D2[j, k]
            if di_k == dj_k or di_k == di_j or dj_k == di_j:
                mask |= (1 << k)
        iso_mask[i][j] = mask
        iso_mask[j][i] = mask


def _is_no_isosceles(S_mask, S_list):
    """Check if set S is no-isosceles."""
    n = len(S_list)
    if n < 3:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            a = S_list[i]
            b = S_list[j]
            if iso_mask[a][b] & S_mask:
                return False
    return True


def _get_blocked_mask(S_list):
    """
    Return a bitmask of all points blocked by pairs in S_list.
    A point p is blocked if it forms an isosceles triangle with some pair (a,b) in S_list.
    """
    blocked = 0
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            a = S_list[i]
            b = S_list[j]
            blocked |= iso_mask[a][b]
    return blocked


def _is_maximal(S_mask, S_list):
    """
    Check if S is maximal: no point outside S can be added without violating no-isosceles.
    This is equivalent to saying every point outside S is blocked by S.
    """
    if len(S_list) < 2:
        return False

    blocked = _get_blocked_mask(S_list)
    full_mask = (1 << N_TOTAL) - 1
    return (blocked | S_mask) == full_mask


def _can_add(p, S_mask, S_list):
    """
    Check if point p can be added to S while maintaining no-isosceles.
    """
    if S_mask & (1 << p):
        return False
    blocked = _get_blocked_mask(S_list)
    return not (blocked & (1 << p))


def _local_search_reduce(S_list, deadline_margin=10):
    """
    Iteratively try to remove points from S while maintaining maximality.
    Tries single removals, then pair removals.
    """
    S = list(S_list)
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)

    changed = True
    while changed:
        changed = False
        if time.time() > DEADLINE - deadline_margin:
            break

        n_S = len(S)

        # Try single removals
        for i in range(n_S):
            S_test = S[:i] + S[i+1:]
            S_mask_test = S_mask & ~(1 << S[i])

            if _is_maximal(S_mask_test, S_test):
                S = S_test
                S_mask = S_mask_test
                changed = True
                break

        if changed:
            continue

        # Try pair removals
        for i in range(n_S):
            if time.time() > DEADLINE - deadline_margin:
                break
            for j in range(i + 1, n_S):
                S_test = S[:i] + S[i+1:j] + S[j+1:]
                S_mask_test = S_mask & ~(1 << S[i]) & ~(1 << S[j])

                if _is_maximal(S_mask_test, S_test):
                    S = S_test
                    S_mask = S_mask_test
                    changed = True
                    break
            if changed:
                break

    return S


def _repair_maximality(S_list, deadline_margin=10):
    """
    Given a valid set S (not necessarily maximal), greedily add points to make it maximal.
    """
    S = list(S_list)
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)

    if _is_maximal(S_mask, S):
        return S

    while True:
        if time.time() > DEADLINE - deadline_margin:
            break

        blocked = _get_blocked_mask(S)
        unblocked = []
        for p in range(N_TOTAL):
            if S_mask & (1 << p):
                continue
            if not (blocked & (1 << p)):
                unblocked.append(p)

        if not unblocked:
            break

        best_p = None
        best_gain = -1

        for p in unblocked:
            new_blocked = blocked
            for a in S:
                new_blocked |= iso_mask[p][a]

            gain = 0
            for q in unblocked:
                if new_blocked & (1 << q):
                    gain += 1

            if gain > best_gain:
                best_gain = gain
                best_p = p

        if best_p is None:
            break

        S.append(best_p)
        S_mask |= (1 << best_p)

    return S


def _generate_initial_set(deadline_margin=10):
    """
    Generate a random maximal set.
    Start with random points, ensure validity, then repair maximality.
    """
    S = []
    S_mask = 0
    core_size = random.randint(3, 5)
    candidates = random.sample(range(N_TOTAL), core_size)
    for p in candidates:
        if _can_add(p, S_mask, S):
            S.append(p)
            S_mask |= (1 << p)

    S = _repair_maximality(S, deadline_margin)
    return S


def _min_vertex_cover_bipartite(S_list, deadline_margin=5):
    """
    Attempt to find a minimum vertex cover in the bipartite graph induced by S_list.
    Partition by parity of x+y. Edges represent isosceles constraints.
    Use greedy + local search on the bipartite structure.
    """
    # This is a simplified version - in a real implementation we'd use MILP
    # For now, just do local search with parity-aware perturbations
    S = list(S_list)
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)

    # Classify points by parity
    even = [p for p in S if (_ALL_POINTS[p][0] + _ALL_POINTS[p][1]) % 2 == 0]
    odd = [p for p in S if (_ALL_POINTS[p][0] + _ALL_POINTS[p][1]) % 2 == 1]

    # Try swapping points between parity classes
    improved = True
    while improved:
        improved = False
        if time.time() > DEADLINE - deadline_margin:
            break

        # Try removing one point from each class and adding a better one
        for i in range(len(even)):
            for j in range(len(odd)):
                S_test = [p for p in S if p != even[i] and p != odd[j]]
                S_mask_test = 0
                for x in S_test:
                    S_mask_test |= (1 << x)

                # Try adding back one point from each class
                candidates_even = [p for p in range(N_TOTAL) if (_ALL_POINTS[p][0] + _ALL_POINTS[p][1]) % 2 == 0 and not (S_mask_test & (1 << p))]
                candidates_odd = [p for p in range(N_TOTAL) if (_ALL_POINTS[p][0] + _ALL_POINTS[p][1]) % 2 == 1 and not (S_mask_test & (1 << p))]

                best_combo = None
                best_size = len(S_test) + 2

                for pe in candidates_even[:5]:
                    for po in candidates_odd[:5]:
                        S_combo = S_test + [pe, po]
                        S_combo_mask = S_mask_test | (1 << pe) | (1 << po)

                        if _is_no_isosceles(S_combo_mask, S_combo) and _is_maximal(S_combo_mask, S_combo):
                            if len(S_combo) < best_size:
                                best_size = len(S_combo)
                                best_combo = S_combo

                if best_combo and len(best_combo) < len(S):
                    S = best_combo
                    S_mask = 0
                    for x in S:
                        S_mask |= (1 << x)
                    even = [p for p in S if (_ALL_POINTS[p][0] + _ALL_POINTS[p][1]) % 2 == 0]
                    odd = [p for p in S if (_ALL_POINTS[p][0] + _ALL_POINTS[p][1]) % 2 == 1]
                    improved = True
                    break

            if improved:
                break

    return S


def construct():
    start_time = time.time()

    best_set = None
    best_size = N_TOTAL + 1

    rng = random.Random(42)

    # Strategy 1: Multiple initializations with ILS
    for attempt in range(100):
        if time.time() - start_time > DEADLINE - 30:
            break

        # Generate initial maximal set
        S = _generate_initial_set(deadline_margin=10)

        # Local search to reduce size
        S = _local_search_reduce(S, deadline_margin=10)

        # Parity-aware refinement
        S = _min_vertex_cover_bipartite(S, deadline_margin=5)

        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]

        if best_size <= 8:
            break

        # Perturbation: Remove k random points
        k = random.randint(2, 4)
        if len(S) > k:
            remove_indices = rng.sample(range(len(S)), k)
            remove_set = set(S[i] for i in remove_indices)

            S_perturbed = [x for x in S if x not in remove_set]

            # Repair maximality
            S_repaired = _repair_maximality(S_perturbed, deadline_margin=10)

            # Local search again
            S_repaired = _local_search_reduce(S_repaired, deadline_margin=10)

            # Parity-aware refinement
            S_repaired = _min_vertex_cover_bipartite(S_repaired, deadline_margin=5)

            if len(S_repaired) < best_size:
                best_size = len(S_repaired)
                best_set = S_repaired[:]
            elif len(S_repaired) == best_size and rng.random() < 0.5:
                best_set = S_repaired[:]

    # Final check
    if best_set is None:
        S = []
        S_mask = 0
        for p in range(N_TOTAL):
            if _can_add(p, S_mask, S):
                S.append(p)
                S_mask |= (1 << p)
        best_set = _local_search_reduce(S, deadline_margin=5)

    return [_ALL_POINTS[i] for i in best_set]
