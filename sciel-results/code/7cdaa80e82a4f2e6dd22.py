import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n

    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbor lists
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)

    # --- Phase 1: Randomized Greedy Initialization ---
    # Try multiple random seeds and keep the best (smallest) code
    best_greedy_code = None
    best_greedy_size = total_words

    for seed in [42, 123, 777, 2024, 31337]:
        random.seed(seed)
        cov_count = np.zeros(total_words, dtype=np.int32)
        is_code = np.zeros(total_words, dtype=bool)
        code_list = []

        potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
        uncovered_remaining = total_words

        while uncovered_remaining > 0 and time.time() < deadline - 60.0:
            # Add some randomness to the selection
            top_k = min(20, total_words)
            # Get the top potential candidates
            if top_k >= total_words:
                c = np.argmax(potential)
            else:
                # Sample from top candidates
                top_indices = np.argpartition(potential, -top_k)[-top_k:]
                top_potentials = potential[top_indices]
                # Weighted random choice
                weights = top_potentials.astype(np.float64)
                weights = weights / weights.sum()
                c = top_indices[random.choices(range(len(top_indices)), weights=weights, k=1)[0]]

            if potential[c] == 0:
                break
            code_list.append(c)
            is_code[c] = True
            potential[c] = 0
            neigh_c = neighbors[c]
            for x in neigh_c:
                if cov_count[x] == 0:
                    cov_count[x] = 1
                    uncovered_remaining -= 1
                    potential[neighbors[x]] -= 1
                else:
                    cov_count[x] += 1

        if len(code_list) < best_greedy_size:
            best_greedy_size = len(code_list)
            best_greedy_code = code_list[:]
            best_cov_count = cov_count.copy()
            best_is_code = is_code.copy()

        if time.time() > deadline - 55.0:
            break

    current_code = best_greedy_code[:]
    current_size = len(current_code)
    best_code = current_code[:]
    best_size = current_size

    # --- Phase 2: Iterated Local Search ---
    current_cov_count = best_cov_count.copy()
    current_is_code = best_is_code.copy()
    current_code_set = set(current_code)

    random.seed(42)
    stuck_counter = 0
    no_improve_counter = 0
    max_no_improve = 500

    while time.time() < deadline - 5.0:
        # 1. Try to remove a word
        if current_size == 0:
            break
        idx = random.randint(0, current_size - 1)
        c = current_code[idx]

        if np.all(current_cov_count[neighbors[c]] > 1):
            current_cov_count[neighbors[c]] -= 1
            current_is_code[c] = False
            current_code_set.remove(c)
            current_code.pop(idx)
            current_size -= 1
            stuck_counter = 0
            no_improve_counter = 0

            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
        else:
            stuck_counter += 1
            no_improve_counter += 1

        # 2. If stuck, try a swap or 2-opt
        if stuck_counter > 3 and time.time() < deadline - 10.0 and current_size > 0:
            # Try 1-opt swap
            c1_idx = random.randint(0, current_size - 1)
            c1 = current_code[c1_idx]

            # Find risky points: x in neighbors[c1] where cov_count[x] == 1
            risky_points = [x for x in neighbors[c1] if current_cov_count[x] == 1]

            if not risky_points:
                # Should have been caught in step 1, but just in case
                current_cov_count[neighbors[c1]] -= 1
                current_is_code[c1] = False
                current_code_set.remove(c1)
                current_code.pop(c1_idx)
                current_size -= 1
                stuck_counter = 0
                no_improve_counter = 0
                if current_size < best_size:
                    best_size = current_size
                    best_code = current_code[:]
            else:
                # Find c2 not in code such that c2 covers all risky points.
                candidates = set(neighbors[risky_points[0]].tolist())
                candidates.discard(c1)

                for x in risky_points[1:]:
                    candidates &= set(neighbors[x].tolist())
                    if not candidates:
                        break

                valid_candidates = [c2 for c2 in candidates if c2 not in current_code_set]

                if valid_candidates:
                    c2 = random.choice(valid_candidates)

                    current_cov_count[neighbors[c1]] -= 1
                    current_is_code[c1] = False
                    current_code_set.remove(c1)
                    current_code[c1_idx] = c2

                    current_cov_count[neighbors[c2]] += 1
                    current_is_code[c2] = True
                    current_code_set.add(c2)

                    stuck_counter = 0
                    no_improve_counter = 0
                else:
                    # Try 2-opt: Remove c1 and another c2, add one c3
                    if current_size >= 2 and stuck_counter > 8:
                        c2_idx = random.randint(0, current_size - 1)
                        if c2_idx == c1_idx:
                            c2_idx = (c2_idx + 1) % current_size
                        c2 = current_code[c2_idx]

                        # Collect risky points for c1 and c2
                        risky_c1 = [x for x in neighbors[c1] if current_cov_count[x] == 1]
                        risky_c2 = [x for x in neighbors[c2] if current_cov_count[x] == 1]
                        all_risky = risky_c1 + risky_c2

                        if all_risky:
                            # Find c3 that covers all all_risky
                            candidates_2opt = set(neighbors[all_risky[0]].tolist())
                            candidates_2opt.discard(c1)
                            candidates_2opt.discard(c2)

                            for x in all_risky[1:]:
                                candidates_2opt &= set(neighbors[x].tolist())
                                if not candidates_2opt:
                                    break

                            valid_candidates_2opt = [c3 for c3 in candidates_2opt if c3 not in current_code_set]

                            if valid_candidates_2opt:
                                c3 = random.choice(valid_candidates_2opt)

                                # Update counts
                                current_cov_count[neighbors[c1]] -= 1
                                current_cov_count[neighbors[c2]] -= 1
                                current_cov_count[neighbors[c3]] += 1

                                # Update sets
                                current_is_code[c1] = False
                                current_is_code[c2] = False
                                current_is_code[c3] = True
                                current_code_set.remove(c1)
                                current_code_set.remove(c2)
                                current_code_set.add(c3)

                                # Update list
                                indices_to_remove = sorted([c1_idx, c2_idx], reverse=True)
                                for idx_rm in indices_to_remove:
                                    current_code.pop(idx_rm)
                                current_code.append(c3)
                                current_size -= 1
                                stuck_counter = 0
                                no_improve_counter = 0

                                if current_size < best_size:
                                    best_size = current_size
                                    best_code = current_code[:]

            stuck_counter = 0

        # 3. If no improvement for a long time, restart from a perturbed state
        if no_improve_counter > max_no_improve and time.time() < deadline - 15.0:
            # Perturb: randomly swap 2-5 words
            perturb_size = min(random.randint(2, 5), current_size)
            for _ in range(perturb_size):
                if current_size == 0:
                    break
                idx1 = random.randint(0, current_size - 1)
                c1 = current_code[idx1]
                risky = [x for x in neighbors[c1] if current_cov_count[x] == 1]
                if not risky:
                    current_cov_count[neighbors[c1]] -= 1
                    current_is_code[c1] = False
                    current_code_set.remove(c1)
                    current_code.pop(idx1)
                    current_size -= 1
                else:
                    cands = set(neighbors[risky[0]].tolist())
                    cands.discard(c1)
                    for x in risky[1:]:
                        cands &= set(neighbors[x].tolist())
                        if not cands:
                            break
                    vc = [c for c in cands if c not in current_code_set]
                    if vc:
                        c2 = random.choice(vc)
                        current_cov_count[neighbors[c1]] -= 1
                        current_is_code[c1] = False
                        current_code_set.remove(c1)
                        current_code[idx1] = c2
                        current_cov_count[neighbors[c2]] += 1
                        current_is_code[c2] = True
                        current_code_set.add(c2)
            no_improve_counter = 0
            stuck_counter = 0

    # --- Phase 3: Polish ---
    final_code = best_code[:]
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov_count[neighbors[idx]] += 1

    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(len(final_code) - 1, -1, -1):
            c = final_code[i]
            if np.all(final_cov_count[neighbors[c]] > 1):
                final_cov_count[neighbors[c]] -= 1
                final_code.pop(i)
                improved = True

    if not np.all(final_cov_count > 0):
        return [list(digits_of[i]) for i in best_code]

    return [list(digits_of[i]) for i in final_code]
