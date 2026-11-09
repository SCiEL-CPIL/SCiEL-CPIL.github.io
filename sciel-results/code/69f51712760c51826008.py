import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 256

    A = set()
    remaining = set(range(1, N + 1))

    def is_valid_add(x, A):
        # Check 4-AP
        # x is 1st
        for d in range(1, (N - x) // 3 + 1):
            if x + d in A and x + 2 * d in A and x + 3 * d in A:
                return False
        # x is 2nd
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            if x - d in A and x + d in A and x + 2 * d in A:
                return False
        # x is 3rd
        for d in range(1, min(x // 2, N - x) + 1):
            if x - 2 * d in A and x - d in A and x + d in A:
                return False
        # x is 4th
        for d in range(1, (x - 1) // 3 + 1):
            if x - 3 * d in A and x - 2 * d in A and x - d in A:
                return False
        return True

    def count_3ap_gain(x, A):
        gain = 0
        # x is 1st
        for d in range(1, (N - x) // 2 + 1):
            if x + d in A and x + 2 * d in A:
                gain += 1
        # x is 2nd
        for d in range(1, min(x - 1, N - x) + 1):
            if x - d in A and x + d in A:
                gain += 1
        # x is 3rd
        for d in range(1, x // 2 + 1):
            if x - 2 * d in A and x - d in A:
                gain += 1
        return gain

    # Greedy Phase
    while remaining and time.time() < deadline - 40:
        best_x = -1
        best_score = -1
        for x in remaining:
            if is_valid_add(x, A):
                score = count_3ap_gain(x, A)
                if score > best_score:
                    best_score = score
                    best_x = x
        if best_x == -1:
            break
        A.add(best_x)
        remaining.remove(best_x)

    def count_total_3ap(A):
        count = 0
        for b in A:
            for d in range(1, min(b - 1, N - b) + 1):
                if b - d in A and b + d in A:
                    count += 1
        return count

    current_score = count_total_3ap(A)

    # Local Search Phase: iterative 1-swap
    # We'll do multiple passes, each trying to find an improving swap
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False

        # Compute value (loss) of each element in A
        A_values = {}
        for a in A:
            loss = 0
            for d in range(1, (N - a) // 2 + 1):
                if a + d in A and a + 2 * d in A:
                    loss += 1
            for d in range(1, min(a - 1, N - a) + 1):
                if a - d in A and a + d in A:
                    loss += 1
            for d in range(1, a // 2 + 1):
                if a - 2 * d in A and a - d in A:
                    loss += 1
            A_values[a] = loss

        # Compute gain of each element in remaining (if valid to add)
        R_values = {}
        for b in remaining:
            if is_valid_add(b, A):
                gain = count_3ap_gain(b, A)
                R_values[b] = gain
            else:
                R_values[b] = -1

        # Sort: remove low-value elements, add high-gain elements
        sorted_A = sorted(A_values.keys(), key=lambda x: A_values[x])
        sorted_R = sorted(R_values.keys(), key=lambda x: R_values[x], reverse=True)

        # Try swaps: top K from A (lowest value) x top K from R (highest gain)
        K = 10
        swap_found = False
        for a in sorted_A[:K]:
            if swap_found:
                break
            for b in sorted_R[:K]:
                if b not in remaining:
                    continue
                # Check if b is valid in A - {a}
                A.discard(a)
                if b in A:
                    A.add(a)
                    continue
                if is_valid_add(b, A):
                    gain_b = count_3ap_gain(b, A)
                    new_score = current_score - A_values[a] + gain_b
                    if new_score > current_score:
                        # Accept swap
                        A.add(b)
                        remaining.discard(b)
                        remaining.add(a)
                        current_score = new_score
                        improved = True
                        swap_found = True
                        break
                A.add(a)  # restore a

        if not swap_found and time.time() < deadline - 10:
            # Try more extensive search: all pairs but with early termination
            # Only try if we have time
            for a in sorted_A:
                if time.time() > deadline - 5:
                    break
                if swap_found:
                    break
                loss_a = A_values[a]
                # Skip if loss is too high (no element can beat it easily)
                max_possible_gain = len(A)  # rough upper bound
                if loss_a >= max_possible_gain:
                    continue

                for b in sorted_R:
                    if b not in remaining:
                        continue
                    if R_values[b] <= loss_a:
                        continue  # can't improve

                    A.discard(a)
                    if is_valid_add(b, A):
                        gain_b = count_3ap_gain(b, A)
                        new_score = current_score - loss_a + gain_b
                        if new_score > current_score:
                            A.add(b)
                            remaining.discard(b)
                            remaining.add(a)
                            current_score = new_score
                            improved = True
                            swap_found = True
                            break
                    A.add(a)

    # Final cleanup: ensure we have a valid set
    # Verify no 4-AP exists
    A_list = sorted(A)
    for i in range(len(A_list)):
        for j in range(i + 1, len(A_list)):
            for k in range(j + 1, len(A_list)):
                a1, a2, a3 = A_list[i], A_list[j], A_list[k]
                d = a2 - a1
                if d == 0:
                    continue
                a4 = a3 + d
                if a4 in A:
                    # Found a 4-AP, remove the last element
                    A.discard(a4)
                    break

    return sorted(A)
