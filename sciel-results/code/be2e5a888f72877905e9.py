import itertools
import random
import time
import numpy as np
from numpy.random import default_rng

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit

    points = list(range(16))
    n_points = 16

    # Precompute all triples
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    n_triples = len(all_triples)

    # Precompute transversals: each is a tuple of 8 triple indices
    transversals = []
    for six_pts in itertools.combinations(points, 6):
        sp = list(six_pts)
        # Generate all 15 perfect matchings on 6 points
        matchings = set()
        for perm in itertools.permutations(range(6)):
            m = frozenset([
                tuple(sorted((sp[perm[0]], sp[perm[1]]))),
                tuple(sorted((sp[perm[2]], sp[perm[3]]))),
                tuple(sorted((sp[perm[4]], sp[perm[5]])))
            ])
            m = tuple(sorted(m))
            matchings.add(m)

        for m in matchings:
            (p1, p2, p3) = m
            a, b = p1
            c, d = p2
            e, f = p3
            t_indices = []
            for x in (a, b):
                for y in (c, d):
                    for z in (e, f):
                        t = tuple(sorted((x, y, z)))
                        t_indices.append(triple_to_idx[t])
            t_indices = tuple(sorted(t_indices))
            transversals.append(t_indices)

    n_transversals = len(transversals)

    # Build mapping from each triple index to the list of transversal indices it belongs to
    triple_to_transversals = [[] for _ in range(n_triples)]
    for ti, trans in enumerate(transversals):
        for t_idx in trans:
            triple_to_transversals[t_idx].append(ti)

    # Convert to numpy arrays for faster operations
    # Represent current state as a boolean numpy array
    current_set = np.ones(n_triples, dtype=bool)

    # For each transversal, track how many of its 8 triples are currently present
    conflict_count = np.array([8] * n_transversals, dtype=np.int32)

    def is_valid():
        return np.all(conflict_count < 8)

    def find_best_removal():
        """Find the triple to remove that breaks the most conflicts."""
        conflicting = np.where(conflict_count == 8)[0]
        if len(conflicting) == 0:
            return None

        # Score each triple in conflicting transversals by how many conflicting transversals it's in
        best_score = -1
        best_triple = None

        # Sample if too many conflicting transversals
        if len(conflicting) > 300:
            sample = np.random.choice(conflicting, size=300, replace=False)
        else:
            sample = conflicting

        for trans_idx in sample:
            trans = transversals[trans_idx]
            for t_idx in trans:
                # Count how many of t_idx's transversals are currently conflicting
                score = 0
                for tj in triple_to_transversals[t_idx]:
                    if conflict_count[tj] == 8:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_triple = t_idx

        return best_triple

    # Phase 1: Aggressive removal to get a valid configuration
    while time.time() < deadline - 50:
        if is_valid():
            break

        removal = find_best_removal()
        if removal is None:
            break

        t_idx = removal
        current_set[t_idx] = False
        for trans_j in triple_to_transversals[t_idx]:
            conflict_count[trans_j] -= 1

    # Phase 2: Local search with swaps to maximize count
    rng = np.random.default_rng(42)

    best_count = int(np.sum(current_set))
    best_state = current_set.copy()
    best_conflict = conflict_count.copy()

    # First, try adding back triples greedily
    absent_triples = np.where(~current_set)[0]
    rng.shuffle(absent_triples)

    for t_idx in absent_triples:
        if time.time() > deadline - 30:
            break

        can_add = True
        for trans_j in triple_to_transversals[t_idx]:
            if conflict_count[trans_j] == 7:
                can_add = False
                break

        if can_add:
            current_set[t_idx] = True
            for trans_j in triple_to_transversals[t_idx]:
                conflict_count[trans_j] += 1

    current_count = int(np.sum(current_set))
    if current_count > best_count:
        best_count = current_count
        best_state = current_set.copy()
        best_conflict = conflict_count.copy()

    # Phase 3: Iterated local search with random restarts and swaps
    for iteration in range(200):
        if time.time() > deadline - 15:
            break

        if not is_valid():
            # Fix conflicts
            removal = find_best_removal()
            if removal is not None:
                current_set[removal] = False
                for trans_j in triple_to_transversals[removal]:
                    conflict_count[trans_j] -= 1
            continue

        # Try a random perturbation: remove k random present triples, then greedily add back
        present = np.where(current_set)[0]
        absent = np.where(~current_set)[0]

        if len(present) < 3 or len(absent) < 3:
            break

        k = rng.integers(1, 4)  # Remove 1-3 triples
        to_remove = rng.choice(present, size=min(k, len(present)), replace=False)

        # Save state
        old_state = current_set.copy()
        old_conflict = conflict_count.copy()

        # Remove
        for t_idx in to_remove:
            current_set[t_idx] = False
            for trans_j in triple_to_transversals[t_idx]:
                conflict_count[trans_j] -= 1

        # Greedily add back triples
        rng.shuffle(absent)
        for t_idx in absent:
            if time.time() > deadline - 15:
                break
            can_add = True
            for trans_j in triple_to_transversals[t_idx]:
                if conflict_count[trans_j] == 7:
                    can_add = False
                    break
            if can_add:
                current_set[t_idx] = True
                for trans_j in triple_to_transversals[t_idx]:
                    conflict_count[trans_j] += 1

        new_count = int(np.sum(current_set))
        if new_count > best_count:
            best_count = new_count
            best_state = current_set.copy()
            best_conflict = conflict_count.copy()
        else:
            # Revert
            current_set = old_state
            conflict_count = old_conflict

    # Use the best state found
    final_set = best_state

    # Convert back to list of triples
    result = [list(all_triples[i]) for i in range(n_triples) if final_set[i]]

    return result
