"""Construct kissing configuration in R^11 using W4 vectors (two non-zero entries of magnitude 1)."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np


def _generate_w4_vectors():
    """Generate all possible W4 vectors: vectors with exactly two non-zero entries, each ±1."""
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    return np.array(vectors, dtype=np.int64)


def _pairwise_distance_ok(v1, v2):
    """Check if the squared distance between v1 and v2 is at least 4 (norm squared of max vector)."""
    diff = v1 - v2
    return np.dot(diff, diff) >= 4


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 seconds safety margin

    # Generate all candidate W4 vectors
    candidates = _generate_w4_vectors()
    n_candidates = len(candidates)

    # Greedy construction with conflict tracking
    # Two W4 vectors conflict if their squared distance < 4
    # For W4 vectors, distance < 4 means they share at least one coordinate with same sign
    # Let's build a conflict graph implicitly

    selected = []
    in_set = [False] * n_candidates

    # For efficiency, precompute conflict relationships lazily
    # Two vectors v_a and v_b conflict if dist(v_a, v_b)^2 < 4
    # dist^2 = sum of (v_a[i] - v_b[i])^2
    # If they share no coordinates: dist^2 = 2 + 2 = 4 (OK, not conflict)
    # If they share 1 coordinate:
    #   Same sign at shared: dist^2 = 0 + 1 + 1 = 2 (conflict)
    #   Opposite sign at shared: dist^2 = 4 + 1 + 1 = 6 (OK)
    # If they share 2 coordinates (same vector): dist^2 = 0 (conflict, but we handle uniqueness)

    # So conflict happens when two vectors share exactly one coordinate AND have the same sign there.

    # Let's index: vector index -> (i, j, si, sj)
    vec_info = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    vec_info.append((i, j, si, sj))

    # Map from (coord, sign) to list of vector indices that have that coordinate with that sign
    coord_sign_to_vecs = {}
    for idx, (i, j, si, sj) in enumerate(vec_info):
        key1 = (i, si)
        key2 = (j, sj)
        if key1 not in coord_sign_to_vecs:
            coord_sign_to_vecs[key1] = []
        coord_sign_to_vecs[key1].append(idx)
        if key2 not in coord_sign_to_vecs:
            coord_sign_to_vecs[key2] = []
        coord_sign_to_vecs[key2].append(idx)

    # For each vector, precompute the set of conflicting vectors
    # v_a conflicts with v_b if they share exactly one coordinate with same sign
    # We can compute conflicts by: for each (coord, sign) pair, all vectors in that group
    # conflict with each other... wait no. Two vectors in the same (coord, sign) group
    # share that coordinate with same sign. They may or may not share the other coordinate.
    # If they share the other coordinate too (same vector), they're identical.
    # If they share only this coordinate, they conflict.
    # If they share neither other coordinate, they only share this one -> conflict.

    # Actually, let me just compute pairwise conflicts directly for selected vectors.
    # With greedy, we check each new candidate against all selected.

    # For speed, let's use the structure: a candidate v=(i,j,si,sj) conflicts with selected
    # vector w=(i',j',si',sj') if:
    # - They share exactly one coordinate with same sign, OR
    # - They are the same vector

    # Let's build a fast lookup: for each selected vector, we know its (i,j,si,sj).
    # For a new candidate, check against each selected vector.

    # Better: maintain for each (coord, sign) the count of selected vectors having that.
    # A new candidate (i,j,si,sj) conflicts with a selected vector if:
    #   The selected vector has (i,si) but not (j,sj), OR
    #   The selected vector has (j,sj) but not (i,si), OR
    #   The selected vector is exactly (i,j,si,sj)

    # Let's use a different approach: for each selected vector, store its two (coord,sign) keys.
    # For a new candidate, it conflicts if there exists a selected vector sharing exactly one key.

    # Maintain: key_to_count[(coord,sign)] = number of selected vectors with that key
    # And for each selected vector, its two keys.

    # A new candidate with keys k1=(i,si), k2=(j,sj):
    # Conflicts with selected vectors that have k1 but not k2, or k2 but not k1, or both.
    # Actually, if a selected vector has both k1 and k2, it's the same vector.
    # If it has k1 only, it conflicts.
    # If it has k2 only, it conflicts.
    # If it has neither, no conflict.

    # So: conflict_count = (count of selected with k1) + (count of selected with k2) - 2*(count with both k1 and k2)
    # Wait, that's not quite right. Let me think again.

    # Let S1 = selected vectors that have k1
    # Let S2 = selected vectors that have k2
    # Let S12 = selected vectors that have both k1 and k2 (i.e., the exact same vector)
    # Vectors in S1 \ S12 have k1 but not k2 -> conflict
    # Vectors in S2 \ S12 have k2 but not k1 -> conflict
    # Vectors in S12 have both -> same vector -> conflict
    # So total conflicts = |S1 \ S12| + |S2 \ S12| + |S12| = |S1| + |S2| - |S12|

    key_to_count = {}
    selected_keys = []  # list of (k1, k2) for each selected vector

    for idx in range(n_candidates):
        if time.time() > deadline:
            break

        i, j, si, sj = vec_info[idx]
        k1 = (i, si)
        k2 = (j, sj)

        c1 = key_to_count.get(k1, 0)
        c2 = key_to_count.get(k2, 0)

        # Count how many selected vectors have both k1 and k2
        # Since k1 and k2 uniquely determine the vector (i,j,si,sj) with i<j,
        # there can be at most one such vector.
        # Check if this exact vector is already selected.
        # Actually, we need to check if any selected vector has both keys.
        # Since we're building incrementally, we can track this.

        # For simplicity and correctness, let's just check directly against selected vectors
        # when the conflict count is small, or use the formula.

        # Let me use a simpler approach: for each candidate, check against all selected.
        # This is O(n^2) in the worst case but with early termination.

        is_conflict = False
        for (si_sel, sj_sel, ki_sel, kj_sel) in selected_keys:
            # Check if candidate (i,j,si,sj) conflicts with (ki_sel, kj_sel, si_sel, sj_sel)
            # They conflict if they share exactly one coordinate with same sign
            shared = 0
            if i == ki_sel and si == si_sel:
                shared += 1
            if i == kj_sel and si == sj_sel:
                shared += 1
            if j == ki_sel and sj == si_sel:
                shared += 1
            if j == kj_sel and sj == sj_sel:
                shared += 1

            if shared == 1:
                is_conflict = True
                break
            if shared == 2:
                # Same vector
                is_conflict = True
                break

        if not is_conflict:
            selected.append(idx)
            selected_keys.append((si, sj, k1, k2))
            key_to_count[k1] = key_to_count.get(k1, 0) + 1
            key_to_count[k2] = key_to_count.get(k2, 0) + 1

    # Build the result array
    if not selected:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    result = candidates[selected]
    return result
# EVOLVE-BLOCK-END
