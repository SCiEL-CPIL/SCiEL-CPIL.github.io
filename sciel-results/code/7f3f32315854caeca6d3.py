import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    rng = np.random.default_rng(12345)
    random.seed(12345)

    # --- Step 1: Generate candidate vectors from multiple norm classes ---
    # We want vectors where min_pairwise_dist >= max_norm
    # i.e., for all i != j: |v_i - v_j|^2 >= max(||v||^2)
    # |v_i - v_j|^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    # So we need: ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j >= max_norm_sq
    # => v_i.v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2

    # Let's use a mixed-norm approach. We'll generate candidates and then
    # filter based on a global max_norm_sq constraint.

    candidates = []

    # Norm-squared = 2: two +/-1s
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(v)

    # Norm-squared = 3: one +/-2 and one +/-1 (or three +/-1s)
    # Three +/-1s
    for i, j, k in itertools.combinations(range(D), 3):
        for s in itertools.product([1, -1], repeat=3):
            v = [0] * D
            v[i] = s[0]
            v[j] = s[1]
            v[k] = s[2]
            candidates.append(v)

    # Norm-squared = 4: +/-2 e_i (one 2)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Norm-squared = 4: four +/-1s
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            candidates.append(v)

    # Norm-squared = 5: five +/-1s
    for indices in itertools.combinations(range(D), 5):
        for signs in itertools.product([1, -1], repeat=5):
            v = [0] * D
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    N = len(candidates)

    # Compute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1)

    # For a kissing config, we need min dist >= max norm
    # Let's try to find the best max_norm_sq we can support
    # Actually, let's just compute the conflict graph for each possible max_norm_sq
    # and pick the one that gives the largest independent set.

    # But that's expensive. Instead, let's use a unified approach:
    # We'll normalize the constraint. For a given set of vectors, the condition is:
    # For all i != j: ||v_i - v_j||^2 >= max(norms_sq)
    # => 2*v_i.v_j <= norms_sq[i] + norms_sq[j] - max_norm_sq

    # Let's compute the dot product matrix
    dots = candidates @ candidates.T

    # We'll try different max_norm_sq values and find the best independent set
    # The max_norm_sq is determined by the vectors we include.
    # Strategy: try max_norm_sq = 2, 3, 4, 5 and see which gives the best result.

    best_size = 0
    best_indices = []

    for max_norm_sq in [2, 3, 4, 5]:
        if time.time() > deadline - 20.0:
            break

        # Filter candidates: only those with norms_sq <= max_norm_sq
        # Actually no - we can include vectors with norms_sq > max_norm_sq too,
        # but then the constraint becomes tighter. Let me reconsider.
        #
        # The condition is: min_{i!=j} ||v_i - v_j||^2 >= max_i ||v_i||^2
        # Let M = max_i ||v_i||^2 over the selected set.
        # Then for all i != j: ||v_i - v_j||^2 >= M
        # => 2*v_i.v_j <= norms_sq[i] + norms_sq[j] - M

        # So M depends on which vectors we select. This is circular.
        #
        # Alternative: fix M and only consider vectors with norms_sq <= M.
        # Then the constraint is: 2*v_i.v_j <= norms_sq[i] + norms_sq[j] - M
        # And M = max norms_sq over selected.
        #
        # If we restrict to norms_sq <= M, then M is at most the max norm in our pool.
        # Let's try: for each candidate M in {2,3,4,5}, build conflict graph where
        # two vectors conflict if 2*dots[i][j] > norms_sq[i] + norms_sq[j] - M
        # (with strict inequality meaning conflict, i.e., we need <=)
        #
        # Wait: we need ||v_i - v_j||^2 >= M, which is norms_sq[i]+norms_sq[j]-2*dots[i][j] >= M
        # => 2*dots[i][j] <= norms_sq[i]+norms_sq[j]-M
        # Conflict if 2*dots[i][j] > norms_sq[i]+norms_sq[j]-M

        # Only consider vectors with norms_sq <= M (to keep M as the max)
        valid_mask = norms_sq <= max_norm_sq
        valid_indices = np.where(valid_mask)[0]
        Nv = len(valid_indices)

        if Nv == 0:
            continue

        # Build conflict matrix for valid vectors
        sub_dots = dots[np.ix_(valid_indices, valid_indices)]
        sub_norms = norms_sq[valid_indices]

        # Conflict condition: 2*sub_dots[i][j] > sub_norms[i] + sub_norms[j] - max_norm_sq
        conflict_matrix = 2 * sub_dots > (sub_norms[:, None] + sub_norms[None, :] - max_norm_sq)
        np.fill_diagonal(conflict_matrix, False)

        # Build adjacency list
        neighbors = [list(np.where(conflict_matrix[i])[0]) for i in range(Nv)]

        # Now find maximum independent set using randomized greedy + local search
        # Use simulated annealing with swap moves

        best_local_size = 0
        best_local_set = set()

        # Multiple random restarts
        num_restarts = 0
        while time.time() < deadline - 10.0 and num_restarts < 50:
            num_restarts += 1

            # Random greedy initialization
            available = set(range(Nv))
            selected_set = set()

            # Random order
            order = list(available)
            rng.shuffle(order)

            for v in order:
                if v in available:
                    # Check if v conflicts with any already selected
                    can_add = True
                    for n in neighbors[v]:
                        if n in selected_set:
                            can_add = False
                            break
                    if can_add:
                        selected_set.add(v)
                        available.discard(v)
                        for n in neighbors[v]:
                            available.discard(n)

            current_size = len(selected_set)
            current_set = selected_set

            if current_size > best_local_size:
                best_local_size = current_size
                best_local_set = set(current_set)

            # Local search: try swaps
            temp = 2.0
            cooling = 0.95
            max_iters = 2000

            for it in range(max_iters):
                if time.time() > deadline - 10.0:
                    break

                if len(current_set) == 0:
                    break

                # Pick a random vertex to potentially remove
                removed = rng.choice(list(current_set))
                current_set.discard(removed)

                # Try to add up to 2 vertices that conflict with removed
                candidates_to_add = [n for n in neighbors[removed] if n not in current_set]
                rng.shuffle(candidates_to_add)

                added = 0
                new_set = set(current_set)
                for c in candidates_to_add:
                    if added >= 2:
                        break
                    # Check if c conflicts with any in new_set
                    can_add = True
                    for n in neighbors[c]:
                        if n in new_set:
                            can_add = False
                            break
                    if can_add:
                        new_set.add(c)
                        added += 1

                if added > 1:
                    # Accept (improvement)
                    current_set = new_set
                elif added == 1:
                    # Neutral - accept with some probability
                    if rng.random() < 0.5:
                        current_set = new_set
                else:
                    # Worse - accept with probability exp(-delta/temp)
                    delta = 1 - added  # = 1
                    if rng.random() < np.exp(-delta / temp):
                        current_set = new_set

                # Restore if we didn't accept
                # Actually we need to be more careful. Let me redo this.
                # The above is a bit messy. Let me use a cleaner SA approach.
                break  # Skip the messy SA, use a cleaner approach below

            # Cleaner local search: 1-swap and 2-swap
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                # Try 1-swap: remove one, add one
                for v in list(current_set):
                    test_set = set(current_set)
                    test_set.discard(v)
                    # Find candidates that conflict with v
                    for c in neighbors[v]:
                        if c not in test_set:
                            # Check c doesn't conflict with test_set
                            conflict = False
                            for n in neighbors[c]:
                                if n in test_set:
                                    conflict = True
                                    break
                            if not conflict:
                                test_set.add(c)
                                if len(test_set) > len(current_set):
                                    current_set = test_set
                                    improved = True
                                    break
                    if improved:
                        break

                if improved:
                    continue

                # Try 2-swap: remove one, add two
                for v in list(current_set):
                    test_set = set(current_set)
                    test_set.discard(v)
                    candidates = [c for c in neighbors[v] if c not in test_set]
                    rng.shuffle(candidates)
                    added = 0
                    new_set = set(test_set)
                    for c in candidates:
                        if added >= 2:
                            break
                        conflict = False
                        for n in neighbors[c]:
                            if n in new_set:
                                conflict = True
                                break
                        if not conflict:
                            new_set.add(c)
                            added += 1
                    if added > 1:
                        current_set = new_set
                        improved = True
                        break

            if len(current_set) > best_local_size:
                best_local_size = len(current_set)
                best_local_set = set(current_set)

        # Map back to original indices
        if best_local_size > best_size:
            best_size = best_local_size
            best_indices = [valid_indices[i] for i in best_local_set]

    # Final verification
    if best_indices:
        # Verify the set is valid
        idx = np.array(best_indices)
        sub_vecs = candidates[idx]
        sub_norms = norms_sq[idx]
        M = np.max(sub_norms)
        sub_dots = sub_vecs @ sub_vecs.T
        # Check: for all i != j, 2*dots[i][j] <= norms_sq[i]+norms_sq[j]-M
        n_check = len(idx)
        valid = True
        for i in range(n_check):
            for j in range(i + 1, n_check):
                if 2 * sub_dots[i][j] > sub_norms[i] + sub_norms[j] - M:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            # Remove conflicts greedily
            cleaned = []
            used = set()
            for i in best_indices:
                conflict = False
                for j in cleaned:
                    if 2 * dots[i][j] > norms_sq[i] + norms_sq[j] - M:
                        conflict = True
                        break
                if not conflict:
                    cleaned.append(i)
                    used.add(i)
            best_indices = cleaned
            best_size = len(best_indices)

    if best_size == 0:
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)

    return candidates[best_indices]

# EVOLVE-BLOCK-END
