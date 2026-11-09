import time
import numpy as np
import random

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Precompute all odd divisors of 51975 > 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # ascending order
    all_moduli = divisors  # all available moduli

    def score_from_cover_count(cover_count):
        return int(np.sum(cover_count == 0))

    def full_greedy_init(moduli_order, base, deadline, cover_count=None):
        """Full greedy: at each step, pick the (m,r) pair that covers most uncovered."""
        if cover_count is None:
            cover_count = np.zeros(base, dtype=np.int32)
        classes = []
        used = set()

        while len(classes) < len(moduli_order):
            if time.time() > deadline - 5.0:
                break

            best_gain = -1
            best_m = None
            best_r = None

            for m in moduli_order:
                if m in used:
                    continue
                # For this modulus, find best residue
                # Vectorize: for each r, count uncovered in r::m
                # cover_count[r::m] == 0 gives boolean, sum gives count
                # Do this efficiently
                for r in range(m):
                    cnt = int(np.sum(cover_count[r::m] == 0))
                    if cnt > best_gain:
                        best_gain = cnt
                        best_m = m
                        best_r = r

            if best_m is None or best_gain <= 0:
                break

            classes.append([best_m, best_r])
            cover_count[best_r::best_m] += 1
            used.add(best_m)

        return classes, cover_count

    def local_search(classes, cover_count, base, deadline):
        """Local search with add/remove/change/swap moves using cover_count."""
        best_classes = [c[:] for c in classes]
        best_cover = cover_count.copy()
        best_score = score_from_cover_count(best_cover)
        used_moduli = set(m for m, r in best_classes)
        unused_moduli = [m for m in all_moduli if m not in used_moduli]

        rng = random.Random(12345)

        improved = True
        stall_count = 0
        max_stall = 500

        while improved and time.time() < deadline - 10.0:
            improved = False
            stall_count += 1

            if stall_count > max_stall:
                break

            # Try a random sequence of moves
            for _ in range(20):
                if time.time() > deadline - 10.0:
                    break

                move_type = rng.choice(['change', 'add', 'remove'])

                if move_type == 'change' and best_classes:
                    idx = rng.randint(0, len(best_classes) - 1)
                    m, r_old = best_classes[idx]
                    r_new = rng.randint(0, m - 1)
                    if r_new == r_old:
                        continue

                    # Apply
                    best_cover[r_old::m] -= 1
                    best_cover[r_new::m] += 1
                    new_score = score_from_cover_count(best_cover)

                    if new_score <= best_score:
                        best_classes[idx] = [m, r_new]
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                            stall_count = 0
                    else:
                        # Revert
                        best_cover[r_new::m] -= 1
                        best_cover[r_old::m] += 1

                elif move_type == 'add' and unused_moduli:
                    m = rng.choice(unused_moduli)
                    r = rng.randint(0, m - 1)

                    # Apply
                    best_cover[r::m] += 1
                    new_score = score_from_cover_count(best_cover)

                    if new_score <= best_score:
                        best_classes.append([m, r])
                        used_moduli.add(m)
                        unused_moduli.remove(m)
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                            stall_count = 0
                    else:
                        # Revert
                        best_cover[r::m] -= 1

                elif move_type == 'remove' and best_classes:
                    idx = rng.randint(0, len(best_classes) - 1)
                    m, r = best_classes[idx]

                    # Apply
                    best_cover[r::m] -= 1
                    new_score = score_from_cover_count(best_cover)

                    if new_score <= best_score:
                        best_classes.pop(idx)
                        used_moduli.remove(m)
                        unused_moduli.append(m)
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                            stall_count = 0
                    else:
                        # Revert
                        best_cover[r::m] += 1

        # Final polish: for each class, try all residues to find the best
        if time.time() < deadline - 5.0:
            for idx in range(len(best_classes)):
                if time.time() > deadline - 5.0:
                    break
                m, r_old = best_classes[idx]
                best_r_for_m = r_old
                best_cnt = -1
                for r in range(m):
                    cnt = int(np.sum(best_cover[r::m] == 0))
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_r_for_m = r
                if best_r_for_m != r_old:
                    best_cover[r_old::m] -= 1
                    best_cover[best_r_for_m::m] += 1
                    best_classes[idx] = [m, best_r_for_m]

        return best_classes, best_score

    best_score = base
    best_classes = []

    # Strategy 1: Full greedy with ascending divisors
    if time.time() < deadline - 30.0:
        classes, cover_count = full_greedy_init(divisors, base, deadline)
        score = score_from_cover_count(cover_count)
        if score < best_score:
            best_score = score
            best_classes = classes

    # Strategy 2: Full greedy with descending divisors
    if time.time() < deadline - 30.0:
        div_desc = list(reversed(divisors))
        classes, cover_count = full_greedy_init(div_desc, base, deadline)
        score = score_from_cover_count(cover_count)
        if score < best_score:
            best_score = score
            best_classes = classes

    # Strategy 3: Randomized greedy attempts
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 40.0 and attempts < 8:
        attempts += 1
        div_rand = list(divisors)
        rng.shuffle(div_rand)
        classes, cover_count = full_greedy_init(div_rand, base, deadline)
        score = score_from_cover_count(cover_count)
        if score < best_score:
            best_score = score
            best_classes = classes

    # Local search refinement on best solution
    if best_classes and time.time() < deadline - 15.0:
        best_cover = np.zeros(base, dtype=np.int32)
        for m, r in best_classes:
            best_cover[r::m] += 1
        best_classes, best_score = local_search(best_classes, best_cover, base, deadline)

    # Final polish: exhaustive residue optimization for each class
    if best_classes and time.time() < deadline - 3.0:
        cover_count = np.zeros(base, dtype=np.int32)
        for m, r in best_classes:
            cover_count[r::m] += 1

        # Iterate a few times, each time optimizing each class's residue
        for iteration in range(3):
            if time.time() > deadline - 3.0:
                break
            for idx in range(len(best_classes)):
                if time.time() > deadline - 3.0:
                    break
                m, r_old = best_classes[idx]
                best_r = r_old
                best_cnt = -1
                for r in range(m):
                    cnt = int(np.sum(cover_count[r::m] == 0))
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_r = r
                if best_r != r_old:
                    cover_count[r_old::m] -= 1
                    cover_count[best_r::m] += 1
                    best_classes[idx] = [m, best_r]

        final_score = score_from_cover_count(cover_count)
        if final_score < best_score:
            best_score = final_score

    return best_classes if best_classes else []
