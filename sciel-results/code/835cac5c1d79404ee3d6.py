import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def calculate_threshold_fast(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using bitmask representation.
    Uses precomputed transition tables for speed.
    """
    start = FULL_MASK
    if start == 0 or (start & (start - 1)) == 0:
        return 0

    # Precompute subset transitions for both letters
    # For each subset (bitmask), compute the image under a and b
    # We'll compute on-the-fly but cache within this call

    visited = [-1] * (1 << N)
    visited[start] = 0
    q = deque([start])

    while q:
        curr = q.popleft()
        curr_dist = visited[curr]

        if (curr & (curr - 1)) == 0:  # Power of 2 = single state
            return curr_dist

        # Apply letter 'a'
        next_a = 0
        temp = curr
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            next_a |= (1 << a_trans[idx])
            temp ^= lsb
        if visited[next_a] == -1:
            visited[next_a] = curr_dist + 1
            q.append(next_a)

        # Apply letter 'b'
        next_b = 0
        temp = curr
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            next_b |= (1 << b_trans[idx])
            temp ^= lsb
        if visited[next_b] == -1:
            visited[next_b] = curr_dist + 1
            q.append(next_b)

    return -1  # Not synchronizing


def precompute_subset_transitions(a_trans, b_trans):
    """Precompute the image of every subset under a and b. Returns two lists indexed by bitmask."""
    n_subsets = 1 << N
    trans_a = [0] * n_subsets
    trans_b = [0] * n_subsets

    for mask in range(n_subsets):
        ta = 0
        tb = 0
        for i in range(N):
            if mask & (1 << i):
                ta |= (1 << a_trans[i])
                tb |= (1 << b_trans[i])
        trans_a[mask] = ta
        trans_b[mask] = tb

    return trans_a, trans_b


def calculate_threshold_precomputed(trans_a, trans_b):
    """BFS using precomputed transition tables."""
    start = FULL_MASK
    visited = [-1] * (1 << N)
    visited[start] = 0
    q = deque([start])

    while q:
        curr = q.popleft()
        curr_dist = visited[curr]

        if (curr & (curr - 1)) == 0:
            return curr_dist

        na = trans_a[curr]
        if visited[na] == -1:
            visited[na] = curr_dist + 1
            q.append(na)

        nb = trans_b[curr]
        if visited[nb] == -1:
            visited[nb] = curr_dist + 1
            q.append(nb)

    return -1


def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    best_score = 0
    best_a = None
    best_b = None

    def try_automaton(a, b):
        nonlocal best_score, best_a, best_b
        score = calculate_threshold_fast(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
            return True
        return False

    # Strategy 1: Systematic search over structured families
    # Family: a = cycle, b = identity with controlled defects

    a_cycle = [(i + 1) % N for i in range(N)]

    # Single defect
    for k in range(N):
        for v in range(N):
            if v == k:
                continue
            if time.time() > deadline:
                break
            b = list(range(N))
            b[k] = v
            try_automaton(a_cycle, b)

    if time.time() > deadline:
        if best_a is None:
            best_a = a_cycle
            best_b = [0] * N
        return [best_a, best_b]

    # Two defects
    for k1 in range(N):
        for k2 in range(k1 + 1, N):
            for v1 in range(N):
                for v2 in range(N):
                    if time.time() > deadline:
                        break
                    if v1 == k1 and v2 == k2:
                        continue
                    b = list(range(N))
                    b[k1] = v1
                    b[k2] = v2
                    if len(set(b)) == N:
                        continue
                    try_automaton(a_cycle, b)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if time.time() > deadline:
        if best_a is None:
            best_a = a_cycle
            best_b = [0] * N
        return [best_a, best_b]

    # Strategy 2: Steepest-ascent hill climbing on b (with a fixed as cycle)
    random.seed(42)

    # Start from best found so far or a random non-permutation
    if best_b is not None:
        current_b = best_b[:]
    else:
        current_b = [random.randint(0, N - 1) for _ in range(N)]
        if len(set(current_b)) == N:
            current_b[random.randint(0, N - 1)] = current_b[random.randint(0, N - 1)]

    a_current = a_cycle[:]

    # Steepest ascent: repeatedly try all single-position changes, take the best improvement
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        best_neighbor_score = best_score
        best_neighbor_b = None
        best_neighbor_idx = -1
        best_neighbor_val = -1

        for i in range(N):
            for new_val in range(N):
                if new_val == current_b[i]:
                    continue
                old_val = current_b[i]
                current_b[i] = new_val
                if len(set(current_b)) == N:
                    current_b[i] = old_val
                    continue
                score = calculate_threshold_fast(a_current, current_b)
                current_b[i] = old_val
                if score > best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor_b = current_b[:]
                    best_neighbor_idx = i
                    best_neighbor_val = new_val
                    if time.time() > deadline - 5.0:
                        break
            if time.time() > deadline - 5.0:
                break

        if best_neighbor_b is not None:
            current_b = best_neighbor_b
            best_score = best_neighbor_score
            best_a = a_current[:]
            best_b = current_b[:]
            improved = True

    # Strategy 3: Multi-start random search with local refinement
    for restart in range(80):
        if time.time() > deadline - 5.0:
            break

        # Random b
        b_rand = [random.randint(0, N - 1) for _ in range(N)]
        if len(set(b_rand)) == N:
            b_rand[random.randint(0, N - 1)] = b_rand[random.randint(0, N - 1)]

        score = calculate_threshold_fast(a_cycle, b_rand)
        if score > best_score:
            best_score = score
            best_a = a_cycle[:]
            best_b = b_rand[:]

        # Local refinement from this start
        if score >= best_score - 2:  # Only refine promising starts
            current_b2 = b_rand[:]
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                best_local = None
                best_local_score = score
                for i in range(N):
                    for nv in range(N):
                        if nv == current_b2[i]:
                            continue
                        old = current_b2[i]
                        current_b2[i] = nv
                        if len(set(current_b2)) == N:
                            current_b2[i] = old
                            continue
                        s = calculate_threshold_fast(a_cycle, current_b2)
                        current_b2[i] = old
                        if s > best_local_score:
                            best_local_score = s
                            best_local = (i, nv)
                if best_local is None:
                    break
                i, nv = best_local
                current_b2[i] = nv
                score = best_local_score
                if score > best_score:
                    best_score = score
                    best_a = a_cycle[:]
                    best_b = current_b2[:]

    # Strategy 4: Try different 'a' structures with hill climbing
    if time.time() < deadline - 10.0:
        a_structures = [
            list(range(N)),                          # identity
            [N - 1 - i for i in range(N)],           # reversal
            [(i * 5) % N for i in range(N)],         # permutation
            [(i * 7) % N for i in range(N)],         # another permutation
            [(i * 3) % N for i in range(N)],         # another permutation
            [(i * 11) % N for i in range(N)],        # another permutation
            [(i * 2 + 1) % N for i in range(N)],     # another permutation
            [(i * 4 + 3) % N for i in range(N)],     # another permutation
        ]

        for a_struct in a_structures:
            if time.time() > deadline - 5.0:
                break
            # Use current best b as starting point
            if best_b is not None:
                b_start = best_b[:]
            else:
                b_start = [random.randint(0, N - 1) for _ in range(N)]
                if len(set(b_start)) == N:
                    b_start[random.randint(0, N - 1)] = b_start[random.randint(0, N - 1)]

            score = calculate_threshold_fast(a_struct, b_start)
            if score > best_score:
                best_score = score
                best_a = a_struct[:]
                best_b = b_start[:]

            # Hill climb b for this a
            current_b3 = b_start[:]
            improved3 = True
            while improved3 and time.time() < deadline - 5.0:
                improved3 = False
                best_nb = None
                best_ns = score
                for i in range(N):
                    for nv in range(N):
                        if nv == current_b3[i]:
                            continue
                        old = current_b3[i]
                        current_b3[i] = nv
                        if len(set(current_b3)) == N:
                            current_b3[i] = old
                            continue
                        s = calculate_threshold_fast(a_struct, current_b3)
                        current_b3[i] = old
                        if s > best_ns:
                            best_ns = s
                            best_nb = (i, nv)
                if best_nb is None:
                    break
                i, nv = best_nb
                current_b3[i] = nv
                score = best_ns
                if score > best_score:
                    best_score = score
                    best_a = a_struct[:]
                    best_b = current_b3[:]
                improved3 = True

    # Strategy 5: Double-position hill climbing for more diverse moves
    if time.time() < deadline - 5.0 and best_b is not None:
        current_b5 = best_b[:]
        current_a5 = best_a[:]
        score5 = best_score
        improved5 = True
        while improved5 and time.time() < deadline - 3.0:
            improved5 = False
            best_nb5 = None
            best_ns5 = score5
            # Try a subset of double-position changes to keep it tractable
            for i1 in range(0, N, 2):  # Step by 2 to reduce combinations
                for i2 in range(i1 + 1, N, 2):
                    for nv1 in range(0, N, 2):
                        for nv2 in range(0, N, 2):
                            if nv1 == current_b5[i1] and nv2 == current_b5[i2]:
                                continue
                            old1 = current_b5[i1]
                            old2 = current_b5[i2]
                            current_b5[i1] = nv1
                            current_b5[i2] = nv2
                            if len(set(current_b5)) == N:
                                current_b5[i1] = old1
                                current_b5[i2] = old2
                                continue
                            s = calculate_threshold_fast(current_a5, current_b5)
                            current_b5[i1] = old1
                            current_b5[i2] = old2
                            if s > best_ns5:
                                best_ns5 = s
                                best_nb5 = (i1, nv1, i2, nv2)
                            if time.time() > deadline - 3.0:
                                break
                        if time.time() > deadline - 3.0:
                            break
                    if time.time() > deadline - 3.0:
                        break
                if time.time() > deadline - 3.0:
                    break
            if best_nb5 is None:
                break
            i1, nv1, i2, nv2 = best_nb5
            current_b5[i1] = nv1
            current_b5[i2] = nv2
            score5 = best_ns5
            if score5 > best_score:
                best_score = score5
                best_a = current_a5[:]
                best_b = current_b5[:]
            improved5 = True

    # Fallback
    if best_a is None:
        best_a = a_cycle
        best_b = [0] * N

    return [best_a, best_b]
