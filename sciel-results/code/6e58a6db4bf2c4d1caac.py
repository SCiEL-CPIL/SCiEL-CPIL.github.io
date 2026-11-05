import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 256
    best_A = []
    best_B = []
    best_score = 0
    rng = random.Random(12345)

    def can_add(x, S, S_sums, S_diffs, other_diffs):
        if x + x in S_sums:
            return False
        for a in S:
            if x + a in S_sums:
                return False
            if (x - a) in other_diffs:
                return False
            if (a - x) in other_diffs:
                return False
        return True

    def get_diffs(S):
        diffs = set()
        L = len(S)
        for i in range(L):
            for j in range(L):
                if i != j:
                    diffs.add(S[i] - S[j])
        return diffs

    def get_sums(S):
        sums = set()
        L = len(S)
        for i in range(L):
            for j in range(i, L):
                sums.add(S[i] + S[j])
        return sums

    def add_element(x, S, S_sums, S_diffs):
        for a in S:
            S_sums.add(x + a)
            S_diffs.add(x - a)
            S_diffs.add(a - x)
        S_sums.add(x + x)
        S.append(x)

    def remove_element(x, S, S_sums, S_diffs):
        idx = S.index(x)
        S.pop(idx)
        for a in S:
            S_sums.discard(x + a)
            S_diffs.discard(x - a)
            S_diffs.discard(a - x)
        S_sums.discard(x + x)

    # Greedy construction with random restarts and local search
    for trial in range(5000):
        if time.time() > deadline:
            break

        elements = list(range(1, n + 1))
        rng.shuffle(elements)

        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()

        for x in elements:
            if time.time() > deadline:
                break

            can_A = can_add(x, A, sums_A, diffs_A, diffs_B)
            can_B = can_add(x, B, sums_B, diffs_B, diffs_A)

            if can_A and can_B:
                if len(A) <= len(B):
                    add_element(x, A, sums_A, diffs_A)
                else:
                    add_element(x, B, sums_B, diffs_B)
            elif can_A:
                add_element(x, A, sums_A, diffs_A)
            elif can_B:
                add_element(x, B, sums_B, diffs_B)

        score = len(A) + len(B)
        if score > best_score:
            best_A = A[:]
            best_B = B[:]
            best_score = score

        # Local Search
        improved = True
        local_deadline = min(deadline, start_time + 55.0)

        while improved and time.time() < local_deadline:
            improved = False

            # Recompute auxiliary sets
            sums_A = get_sums(A)
            sums_B = get_sums(B)
            diffs_A = get_diffs(A)
            diffs_B = get_diffs(B)

            # 1. Try to add unassigned elements
            assigned = set(A) | set(B)
            unassigned = [x for x in range(1, n + 1) if x not in assigned]

            for x in unassigned:
                if time.time() > local_deadline:
                    break
                if can_add(x, A, sums_A, diffs_A, diffs_B):
                    add_element(x, A, sums_A, diffs_A)
                    improved = True
                    break
                if can_add(x, B, sums_B, diffs_B, diffs_A):
                    add_element(x, B, sums_B, diffs_B)
                    improved = True
                    break

            if improved:
                continue

            # 2. Try moving from A to B
            for i, x in enumerate(A):
                if time.time() > local_deadline:
                    break
                A_new = A[:i] + A[i+1:]
                B_new = B + [x]

                # Check if A_new is Sidon
                s_a_new = get_sums(A_new)
                if len(s_a_new) != len(A_new) * (len(A_new) + 1) // 2:
                    continue

                # Check if B_new is Sidon
                s_b_new = get_sums(B_new)
                if len(s_b_new) != len(B_new) * (len(B_new) + 1) // 2:
                    continue

                # Check diff disjointness
                d_a_new = get_diffs(A_new)
                d_b_new = get_diffs(B_new)
                if not (d_a_new & d_b_new):
                    if max(len(A_new), len(B_new)) < max(len(A), len(B)) or \
                       (max(len(A_new), len(B_new)) == max(len(A), len(B)) and rng.random() < 0.1):
                        A = A_new
                        B = B_new
                        improved = True
                        break

            if improved:
                continue

            # 3. Try moving from B to A
            for i, x in enumerate(B):
                if time.time() > local_deadline:
                    break
                B_new = B[:i] + B[i+1:]
                A_new = A + [x]

                s_a_new = get_sums(A_new)
                if len(s_a_new) != len(A_new) * (len(A_new) + 1) // 2:
                    continue

                s_b_new = get_sums(B_new)
                if len(s_b_new) != len(B_new) * (len(B_new) + 1) // 2:
                    continue

                d_a_new = get_diffs(A_new)
                d_b_new = get_diffs(B_new)
                if not (d_a_new & d_b_new):
                    if max(len(A_new), len(B_new)) < max(len(A), len(B)) or \
                       (max(len(A_new), len(B_new)) == max(len(A), len(B)) and rng.random() < 0.1):
                        A = A_new
                        B = B_new
                        improved = True
                        break

            if improved:
                continue

            # 4. Try swaps between A and B
            for i, x in enumerate(A):
                if time.time() > local_deadline:
                    break
                for j, y in enumerate(B):
                    A_new = A[:i] + A[i+1:] + [y]
                    B_new = B[:j] + B[j+1:] + [x]

                    s_a_new = get_sums(A_new)
                    if len(s_a_new) != len(A_new) * (len(A_new) + 1) // 2:
                        continue

                    s_b_new = get_sums(B_new)
                    if len(s_b_new) != len(B_new) * (len(B_new) + 1) // 2:
                        continue

                    d_a_new = get_diffs(A_new)
                    d_b_new = get_diffs(B_new)
                    if not (d_a_new & d_b_new):
                        if max(len(A_new), len(B_new)) < max(len(A), len(B)) or \
                           (max(len(A_new), len(B_new)) == max(len(A), len(B)) and rng.random() < 0.05):
                            A = A_new
                            B = B_new
                            improved = True
                            break
                if improved:
                    break

            if improved:
                continue

            # 5. Try swapping an element in A with an unassigned element
            for i, x in enumerate(A):
                if time.time() > local_deadline:
                    break
                for z in unassigned:
                    A_new = A[:i] + A[i+1:] + [z]

                    s_a_new = get_sums(A_new)
                    if len(s_a_new) != len(A_new) * (len(A_new) + 1) // 2:
                        continue

                    d_a_new = get_diffs(A_new)
                    if not (d_a_new & diffs_B):
                        if rng.random() < 0.1:
                            A = A_new
                            improved = True
                            break
                if improved:
                    break

            if improved:
                continue

            # 6. Try swapping an element in B with an unassigned element
            for j, y in enumerate(B):
                if time.time() > local_deadline:
                    break
                for z in unassigned:
                    B_new = B[:j] + B[j+1:] + [z]

                    s_b_new = get_sums(B_new)
                    if len(s_b_new) != len(B_new) * (len(B_new) + 1) // 2:
                        continue

                    d_b_new = get_diffs(B_new)
                    if not (d_b_new & diffs_A):
                        if rng.random() < 0.1:
                            B = B_new
                            improved = True
                            break
                if improved:
                    break

        # Update best
        score = len(A) + len(B)
        if score > best_score:
            best_A = A[:]
            best_B = B[:]
            best_score = score

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
