"""Combined program: structured (pos,sign,mag) state + fast SA over support/sign/magnitude."""
import time
import random
import numpy as np

def construct():
    start = time.time()
    DEADLINE = start + 80.0  # safety margin before 90s kill

    # ---- helpers -----------------------------------------------------------
    def coeffs_from_state(state):
        """state: list of 12 (pos, sign, mag) tuples -> ascending coeff list."""
        max_deg = max(p for p, s, m in state)
        coeffs = [0] * (max_deg + 1)
        for p, s, m in state:
            coeffs[p] = s * m
        # trim trailing zeros
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

    def square_nonzero(state):
        coeffs = np.array(coeffs_from_state(state), dtype=np.int64)
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    # ---- state construction ------------------------------------------------
    def make_state(pos_sign_mag_list):
        return tuple(sorted(pos_sign_mag_list, key=lambda x: x[0]))

    # ---- initial states ----------------------------------------------------
    random.seed(12345)
    initial_states = []

    # S1: consecutive, all +1, mag 1
    initial_states.append(make_state([(i, 1, 1) for i in range(12)]))

    # S2: symmetric offsets around center, all +1
    center = 20
    offsets = [-11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9]
    initial_states.append(make_state([(center + o, 1, 1) for o in offsets]))

    # S3: powers-of-2 spread (max cancellation potential), all +1
    spread = sorted(random.sample(range(41), 12))
    initial_states.append(make_state([(p, 1, 1) for p in spread]))

    # S4: alternating signs on consecutive support
    initial_states.append(make_state([(i, 1 if i % 2 == 0 else -1, 1) for i in range(12)]))

    # S5: alternating signs on even-spaced support
    base = 20
    offs = [-11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9]
    initial_states.append(make_state([(base + o, 1 if k % 2 == 0 else -1, 1) for k, o in enumerate(offs)]))

    # S6: small cluster + spread mix
    mix_pos = sorted([0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40])
    initial_states.append(make_state([(p, 1, 1) for p in mix_pos]))

    # S7-S16: random
    for _ in range(10):
        pos = sorted(random.sample(range(41), 12))
        signs = [random.choice([-1, 1]) for _ in range(12)]
        mags = [random.choice([1, 1, 1, 2, 3]) for _ in range(12)]
        initial_states.append(make_state([(pos[i], signs[i], mags[i]) for i in range(12)]))

    # ---- evaluate initial states ------------------------------------------
    best_score = float('inf')
    best_state = None
    for st in initial_states:
        sc = square_nonzero(st)
        if sc < best_score:
            best_score = sc
            best_state = st

    # ---- simulated annealing ----------------------------------------------
    cur_state = best_state
    cur_score = best_score

    T0 = 8.0
    T_end = 0.02
    total_steps = 60000

    for step in range(total_steps):
        if time.time() > DEADLINE:
            break
        progress = step / total_steps
        T = T0 * (T_end / T0) ** progress

        # generate a move
        new_list = [tuple(t) for t in cur_state]
        move = random.randint(1, 6)
        if move == 1:      # move one position
            i = random.randrange(12)
            new_pos = random.randrange(41)
            new_list[i] = (new_pos, new_list[i][1], new_list[i][2])
        elif move == 2:    # flip one sign
            i = random.randrange(12)
            new_list[i] = (new_list[i][0], -new_list[i][1], new_list[i][2])
        elif move == 3:    # flip two signs
            i, j = random.sample(range(12), 2)
            new_list[i] = (new_list[i][0], -new_list[i][1], new_list[i][2])
            new_list[j] = (new_list[j][0], -new_list[j][1], new_list[j][2])
        elif move == 4:    # change one magnitude
            i = random.randrange(12)
            new_mag = random.choice([1, 1, 2, 2, 3, 4, 5])
            new_list[i] = (new_list[i][0], new_list[i][1], new_mag)
        elif move == 5:    # swap two positions (keep signs/mags with slot)
            i, j = random.sample(range(12), 2)
            new_list[i] = (new_list[j][0], new_list[i][1], new_list[i][2])
            new_list[j] = (new_list[i][0], new_list[j][1], new_list[j][2])
        else:              # move two positions
            i, j = random.sample(range(12), 2)
            new_list[i] = (random.randrange(41), new_list[i][1], new_list[i][2])
            new_list[j] = (random.randrange(41), new_list[j][1], new_list[j][2])

        new_state = make_state(new_list)
        if len(set(p for p, s, m in new_state)) != 12:
            continue  # collision: skip

        new_score = square_nonzero(new_state)
        delta = new_score - cur_score
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-9)):
            cur_state = new_state
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best_state = cur_state

    # ---- finalize ----------------------------------------------------------
    result = coeffs_from_state(best_state)
    if not is_valid(result):
        result = [1] * 12
    return result
