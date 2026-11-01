"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def construct(q, n, radius):
    # Fallback for general case
    if not (q == 3 and n == 6 and radius == 1):
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(6, 1)
    N = q ** n  # 729
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute spheres as integers (bitmasks) - each sphere covers 1 + n*(q-1) = 13 words
    # For q=3, n=6: each sphere covers 1 + 6*2 = 13 words
    spheres = np.zeros(N, dtype=np.int64)
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        spheres[i] = mask

    full_mask = (1 << N) - 1
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    best_code = None
    best_size = N

    # Helper to prune a code using bitset operations
    def prune(code_indices):
        current = list(code_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(current)):
                # Check if current[i] is redundant
                union_others = 0
                for j in range(len(current)):
                    if i == j:
                        continue
                    union_others |= spheres[current[j]]
                    if union_others == full_mask:
                        break
                if union_others == full_mask:
                    current.pop(i)
                    changed = True
                    break
        return current

    # Start from tetracode product (81 words) and prune
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            init_code.append(word_to_idx[w])

    pruned_init = prune(init_code)
    if len(pruned_init) < best_size:
        best_size = len(pruned_init)
        best_code = [all_words[i] for i in pruned_init]

    # Greedy restarts with pruning
    num_restarts = 300
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        random.seed(r)
        candidates = list(range(N))
        random.shuffle(candidates)

        covered = 0
        chosen = []

        while covered != full_mask:
            best_gain = -1
            best_idx = -1

            for idx in candidates:
                new_covered = spheres[idx] & ~covered
                gain = bin(new_covered).count('1')

                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_gain == 0:
                break

            chosen.append(best_idx)
            covered |= spheres[best_idx]

        pruned_greedy = prune(chosen)

        if len(pruned_greedy) < best_size:
            best_size = len(pruned_greedy)
            best_code = [all_words[i] for i in pruned_greedy]

        # Simulated annealing polish on the best code found so far
        if time.time() < deadline - 5.0 and best_code is not None:
            polish_start = time.time()
            polish_deadline = polish_start + 3.0

            # Convert best_code to indices
            code_set = set()
            for w in best_code:
                code_set.add(word_to_idx[w])

            # Compute coverage deficit: number of uncovered words
            current_covered = 0
            for idx in code_set:
                current_covered |= spheres[idx]
            uncovered_count = N - bin(current_covered).count('1')

            if uncovered_count == 0:
                # Try to remove redundant words via SA
                current_list = list(code_set)
                temperature = 0.5
                cooling = 0.95

                while time.time() < polish_deadline:
                    if not current_list:
                        break

                    # Randomly try to remove a word
                    idx_to_remove = random.choice(current_list)
                    new_list = [x for x in current_list if x != idx_to_remove]

                    # Check if still covering
                    new_covered = 0
                    for idx in new_list:
                        new_covered |= spheres[idx]
                    new_uncovered = N - bin(new_covered).count('1')

                    if new_uncovered == 0:
                        # Accept: we removed a redundant word
                        current_list = new_list
                        temperature *= cooling
                    else:
                        # Reject
                        temperature *= cooling

                if len(current_list) < best_size:
                    best_size = len(current_list)
                    best_code = [all_words[i] for i in current_list]

    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in best_code]
# EVOLVE-BLOCK-END
