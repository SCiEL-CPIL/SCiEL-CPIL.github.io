"""Targeted zero-cancellation search: directly optimize which square terms vanish."""
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 85.0  # safety margin before 90s kill

    def coeffs_from_state(state):
        """state: list of 12 (pos, sign, mag) tuples -> ascending coeff list."""
        max_deg = max(p for p, s, m in state)
        coeffs = [0] * (max_deg + 1)
        for p, s, m in state:
            coeffs[p] = s * m
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    def is_valid(coeffs):
        if not (1 <= len(coeffs) <= 41):
            return False
        if coeffs[-1] == 0:
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    def square_nonzero(coeffs):
        sq = np.convolve(np.array(coeffs, dtype=np.int64), np.array(coeffs, dtype=np.int64))
        return int(np.count_nonzero(sq))

    def make_state(pos_sign_mag_list):
        return tuple(sorted(pos_sign_mag_list, key=lambda x: x[0]))

    # ---- Phase 1: Generate diverse initial candidates with structure ----
    random.seed(42)
    initial_states = []

    # Consecutive all +1
    initial_states.append(make_state([(i, 1, 1) for i in range(12)]))

    # Symmetric around center
    center = 20
    for offset_pattern in [
        [-11,-9,-7,-5,-3,-1,0,1,3,5,7,9],
        [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10],
        [-13,-11,-9,-7,-5,-3,1,3,5,7,9,11],
    ]:
        initial_states.append(make_state([(center + o, 1, 1) for o in offset_pattern]))

    # Alternating signs on consecutive
    initial_states.append(make_state([(i, 1 if i % 2 == 0 else -1, 1) for i in range(12)]))

    # Alternating signs on even-spaced
    base = 20
    offs = [-11,-9,-7,-5,-3,-1,0,1,3,5,7,9]
    initial_states.append(make_state([(base + o, 1 if k % 2 == 0 else -1, 1) for k, o in enumerate(offs)]))

    # Pairs with cancellation: (x^a + x^b)^2 style building blocks
    # Try 6 pairs that might cancel
    pair_bases = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11)]
    initial_states.append(make_state([(p, 1, 1) for p, _ in pair_bases]))

    # Spread with some clustering
    for _ in range(20):
        # Random positions with some clustering tendency
        pos = []
        while len(pos) < 12:
            if pos and random.random() < 0.3:
                # Cluster near existing
                anchor = random.choice(pos)
                new_p = max(0, min(40, anchor + random.randint(-2, 2)))
            else:
                new_p = random.randrange(41)
            if new_p not in pos:
                pos.append(new_p)
        pos.sort()
        signs = [random.choice([-1, 1]) for _ in range(12)]
        mags = [random.choice([1, 1, 1, 2, 2, 3]) for _ in range(12)]
        initial_states.append(make_state([(pos[i], signs[i], mags[i]) for i in range(12)]))

    # ---- Phase 2: Local search with targeted zero-finding ----
    def local_search(state, time_limit=5.0):
        """Intensive local search: try to zero out specific square coefficients."""
        cur_coeffs = coeffs_from_state(state)
        cur_sq = np.convolve(np.array(cur_coeffs, dtype=np.int64), np.array(cur_coeffs, dtype=np.int64))
        cur_score = int(np.count_nonzero(cur_sq))
        best_state = state
        best_score = cur_score

        t_start = time.time()
        steps = 0
        while time.time() - t_start < time_limit:
            steps += 1
            if steps % 1000 == 0 and time.time() > DEADLINE:
                break

            # Identify a nonzero square coefficient to target for zeroing
            nonzero_indices = np.nonzero(cur_sq)[0]
            if len(nonzero_indices) == 0:
                break

            # Prioritize: target coefficients with small magnitude (easier to zero)
            target_idx = nonzero_indices[np.argmin(np.abs(cur_sq[nonzero_indices]))]

            # Find which pairs of polynomial terms contribute to this square coefficient
            # Square coeff at degree d = sum_{i+j=d} c_i * c_j
            # For our sparse poly, only terms at positions in our support contribute
            positions = [p for p, s, m in state]
            contributing_pairs = []
            for i in range(12):
                for j in range(i, 12):
                    if positions[i] + positions[j] == target_idx:
                        contributing_pairs.append((i, j))

            if not contributing_pairs:
                # Can't directly affect this coefficient; try another target
                continue

            # Try adjusting one of the contributing terms
            i, j = random.choice(contributing_pairs)
            new_list = [tuple(t) for t in state]

            move_type = random.randint(1, 4)
            if move_type == 1:
                # Flip sign of term i
                new_list[i] = (new_list[i][0], -new_list[i][1], new_list[i][2])
            elif move_type == 2:
                # Change magnitude of term i
                new_mag = random.choice([1, 1, 2, 2, 3, 4, 5, 6, 8])
                if new_mag != new_list[i][2]:
                    new_list[i] = (new_list[i][0], new_list[i][1], new_mag)
            elif move_type == 3:
                # Move term i to a nearby position
                new_pos = max(0, min(40, new_list[i][0] + random.randint(-3, 3)))
                if new_pos not in [t[0] for t in new_list if t is not new_list[i]]:
                    new_list[i] = (new_pos, new_list[i][1], new_list[i][2])
            else:
                # Flip both i and j
                new_list[i] = (new_list[i][0], -new_list[i][1], new_list[i][2])
                new_list[j] = (new_list[j][0], -new_list[j][1], new_list[j][2])

            new_state = make_state(new_list)
            if len(set(p for p, s, m in new_state)) != 12:
                continue

            new_coeffs = coeffs_from_state(new_state)
            new_sq = np.convolve(np.array(new_coeffs, dtype=np.int64), np.array(new_coeffs, dtype=np.int64))
            new_score = int(np.count_nonzero(new_sq))

            if new_score < best_score:
                best_score = new_score
                best_state = new_state
                cur_state = new_state
                cur_coeffs = new_coeffs
                cur_sq = new_sq
                cur_score = new_score
            elif new_score == cur_score and random.random() < 0.3:
                # Accept neutral moves to explore
                cur_state = new_state
                cur_coeffs = new_coeffs
                cur_sq = new_sq
                cur_score = new_score

        return best_state, best_score

    # Evaluate all initial states
    best_state = None
    best_score = float('inf')
    for st in initial_states:
        sc = square_nonzero(coeffs_from_state(st))
        if sc < best_score:
            best_score = sc
            best_state = st

    # ---- Phase 3: Iterated local search with restarts ----
    t_start = time.time()
    restarts = 0
    while time.time() - t_start < DEADLINE - t_start and time.time() < DEADLINE - 5:
        restarts += 1
        # Local search from current best
        improved_state, improved_score = local_search(best_state, time_limit=3.0)
        if improved_score < best_score:
            best_score = improved_score
            best_state = improved_state

        # Generate a perturbed version for next restart
        if time.time() > DEADLINE - 10:
            break

        perturbed_list = [tuple(t) for t in best_state]
        # Perturb 1-3 random terms
        for _ in range(random.randint(1, 3)):
            i = random.randrange(12)
            pert = random.randint(1, 3)
            if pert == 1:
                new_pos = random.randrange(41)
                if new_pos not in [t[0] for t in perturbed_list if t is not perturbed_list[i]]:
                    perturbed_list[i] = (new_pos, perturbed_list[i][1], perturbed_list[i][2])
            elif pert == 2:
                perturbed_list[i] = (perturbed_list[i][0], -perturbed_list[i][1], perturbed_list[i][2])
            else:
                new_mag = random.choice([1, 2, 3, 4, 5])
                perturbed_list[i] = (perturbed_list[i][0], perturbed_list[i][1], new_mag)

        perturbed_state = make_state(perturbed_list)
        if len(set(p for p, s, m in perturbed_state)) != 12:
            continue
        perturbed_score = square_nonzero(coeffs_from_state(perturbed_state))
        if perturbed_score < best_score:
            best_score = perturbed_score
            best_state = perturbed_state

    # ---- Finalize ----
    result = coeffs_from_state(best_state)
    if not is_valid(result):
        result = [1] * 12
    return result
