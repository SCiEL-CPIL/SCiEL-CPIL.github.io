import time
import random
import itertools


def construct(q, n, radius):
    t0 = time.time()
    DEADLINE = 150.0
    MARGIN = 5.0
    DEADLINE = min(DEADLINE, t0 + 113.0)  # safety margin before 120s eval / 150s kill

    total = q ** n
    # ---------- precompute ball-index tables ----------
    # ball_index(w) = sum_i pos_add[i][w[i]] ; ball_index(w) + digit_mul[i][d]
    # == ball_index(w with coord i set to d)
    pos_add = [[0] * q for _ in range(n)]
    digit_mul = [[0] * q for _ in range(n)]
    for i in range(n):
        p = q ** (n - 1 - i)
        for d in range(q):
            pos_add[i][d] = d * p
            digit_mul[i][d] = d * p

    def word_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def index_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    def ball_index(w):
        b = 0
        for i in range(n):
            b += pos_add[i][w[i]]
        return b

    def ball_index_set(w, i, d):
        b = 0
        for j in range(n):
            if j == i:
                b += pos_add[j][d]
            else:
                b += pos_add[j][w[j]]
        return b

    # ---------- coverage structures ----------
    cov_count = [0] * total
    best_cw = [-1] * total
    second_cw = [-1] * total

    def add_codeword(w):
        b = ball_index(w)
        for i in range(n):
            for d in range(q):
                if d == w[i]:
                    continue
                bi = b + digit_mul[i][d]
                cov_count[bi] += 1
                if best_cw[bi] < 0:
                    best_cw[bi] = len(codewords)
                elif best_cw[bi] != len(codewords):
                    if second_cw[bi] < 0:
                        second_cw[bi] = len(codewords)
                    else:
                        pass
        # also the center itself
        cov_count[b] += 1
        if best_cw[b] < 0:
            best_cw[b] = len(codewords)
        elif best_cw[b] != len(codewords):
            if second_cw[b] < 0:
                second_cw[b] = len(codewords)

    def remove_codeword(w, ci):
        b = ball_index(w)
        for i in range(n):
            for d in range(q):
                if d == w[i]:
                    continue
                bi = b + digit_mul[i][d]
                cov_count[bi] -= 1
                if best_cw[bi] == ci:
                    best_cw[bi] = second_cw[bi]
                    second_cw[bi] = -1
                elif second_cw[bi] == ci:
                    second_cw[bi] = -1
        cov_count[b] -= 1
        if best_cw[b] == ci:
            best_cw[b] = second_cw[b]
            second_cw[b] = -1
        elif second_cw[b] == ci:
            second_cw[b] = -1

    codewords = []

    # ---------- greedy initialization ----------
    # pick codeword that covers most currently-uncovered words
    def greedy_init():
        uncovered = [True] * total
        for w in codewords:
            b = ball_index(w)
            uncovered[b] = False
            for i in range(n):
                for d in range(q):
                    if d != w[i]:
                        uncovered[b + digit_mul[i][d]] = False
        while True:
            best_gain = -1
            best_word = None
            # candidate words: sample from uncovered words + random
            # try each uncovered word as candidate center
            for idx in range(total):
                if time.time() > DEADLINE:
                    break
                if not uncovered[idx]:
                    continue
                w = index_word(idx)
                b = ball_index(w)
                gain = 0
                for i in range(n):
                    for d in range(q):
                        if d != w[i]:
                            if uncovered[b + digit_mul[i][d]]:
                                gain += 1
                if uncovered[b]:
                    gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word is None:
                break
            add_codeword(best_word)
            b = ball_index(best_word)
            uncovered[b] = False
            for i in range(n):
                for d in range(q):
                    if d != best_word[i]:
                        uncovered[b + digit_mul[i][d]] = False

    greedy_init()
    if time.time() > DEADLINE:
        return [list(w) for w in codewords]

    # ---------- simulated annealing ----------
    best_code = [list(w) for w in codewords]
    best_size = len(codewords)
    rng = random.Random(12345)

    def try_remove(ci):
        # returns (valid, undo_info)
        w = codewords[ci]
        b = ball_index(w)
        # check: any word uniquely covered by ci?
        # a ball bi is uniquely covered by ci iff best_cw[bi]==ci and second_cw[bi]==-1
        # and cov_count[bi]==1
        for i in range(n):
            for d in range(q):
                if d == w[i]:
                    continue
                bi = b + digit_mul[i][d]
                if cov_count[bi] == 1 and best_cw[bi] == ci:
                    return False
        if cov_count[b] == 1 and best_cw[b] == ci:
            return False
        # valid: perform removal
        remove_codeword(w, ci)
        codewords[ci] = None
        return True

    def do_remove(ci):
        w = codewords[ci]
        remove_codeword(w, ci)
        codewords[ci] = None

    def undo_remove(ci, w):
        codewords[ci] = w
        # re-add: but indices shifted? We use ci as index; after removal,
        # the codeword list has None at ci. Re-adding at same index.
        # However best_cw/second_cw store indices, and ci was removed.
        # We must restore. Simplest: re-add via add_codeword but that appends.
        # Instead, manually restore coverage.
        b = ball_index(w)
        for i in range(n):
            for d in range(q):
                if d == w[i]:
                    continue
                bi = b + digit_mul[i][d]
                cov_count[bi] += 1
                if best_cw[bi] < 0:
                    best_cw[bi] = ci
                elif best_cw[bi] != ci:
                    if second_cw[bi] < 0:
                        second_cw[bi] = ci
        cov_count[b] += 1
        if best_cw[b] < 0:
            best_cw[b] = ci
        elif best_cw[b] != ci:
            if second_cw[b] < 0:
                second_cw[b] = ci

    def try_replace(ci, new_w):
        # remove ci, add new_w; check no word becomes uncovered
        w = codewords[ci]
        # compute coverage after removing ci
        # A word is safe if it has another codeword within radius
        # Check: for each ball in new_w's ball AND w's ball, ensure coverage >=1 after swap
        b_old = ball_index(w)
        b_new = ball_index(new_w)
        # balls only covered by w (losing) must be covered by new_w
        # For efficiency: check balls in w's ball where cov_count==1 and best==ci
        # those must be in new_w's ball
        for i in range(n):
            for d in range(q):
                if d == w[i]:
                    continue
                bi = b_old + digit_mul[i][d]
                if cov_count[bi] == 1 and best_cw[bi] == ci:
                    # must be covered by new_w
                    if bi not in new_ball_set:
                        return False
        if cov_count[b_old] == 1 and best_cw[b_old] == ci:
            if b_old not in new_ball_set:
                return False
        # perform swap
        remove_codeword(w, ci)
        codewords[ci] = new_w
        # add new_w at index ci
        b = b_new
        for i in range(n):
            for d in range(q):
                if d == new_w[i]:
                    continue
                bi = b + digit_mul[i][d]
                cov_count[bi] += 1
                if best_cw[bi] < 0:
                    best_cw[bi] = ci
                elif best_cw[bi] != ci:
                    if second_cw[bi] < 0:
                        second_cw[bi] = ci
        cov_count[b] += 1
        if best_cw[b] < 0:
            best_cw[b] = ci
        elif best_cw[b] != ci:
            if second_cw[b] < 0:
                second_cw[b] = ci
        return True

    def undo_replace(ci, old_w):
        new_w = codewords[ci]
        remove_codeword(new_w, ci)
        codewords[ci] = old_w
        b = ball_index(old_w)
        for i in range(n):
            for d in range(q):
                if d == old_w[i]:
                    continue
                bi = b + digit_mul[i][d]
                cov_count[bi] += 1
                if best_cw[bi] < 0:
                    best_cw[bi] = ci
                elif best_cw[bi] != ci:
                    if second_cw[bi] < 0:
                        second_cw[bi] = ci
        cov_count[b] += 1
        if best_cw[b] < 0:
            best_cw[b] = ci
        elif best_cw[b] != ci:
            if second_cw[b] < 0:
                second_cw[b] = ci

    # precompute new_ball_set lazily per replace
    def new_w_ball_set(new_w):
        b = ball_index(new_w)
        s = {b}
        for i in range(n):
            for d in range(q):
                if d != new_w[i]:
                    s.add(b + digit_mul[i][d])
        return s

    # SA loop
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    active = [i for i in range(len(codewords))]

    while time.time() < DEADLINE and T > T_min:
        if not active:
            break
        ci = rng.choice(active)
        move = rng.random()
        if move < 0.3 and len(codewords) > 1:
            # remove
            if try_remove(ci):
                new_size = len([w for w in codewords if w is not None])
                if new_size < best_size:
                    best_size = new_size
                    best_code = [list(w) for w in codewords if w is not None]
                    # compact
                    codewords[:] = [w for w in codewords if w is not None]
                    active = list(range(len(codewords)))
                else:
                    # accept with prob exp(-(delta)/T) where delta = new_size - old_size >= 0
                    delta = new_size - len([w for w in codewords if w is not None])
                    # undo
                    # re-add
                    w = codewords[ci] if ci < len(codewords) else None
                    # we already removed; undo
                    # find the word: we stored None
                    # re-add manually
                    # Actually try_remove set codewords[ci]=None and removed coverage.
                    # Undo:
                    pass
                T *= cooling
            else:
                T *= cooling
        elif move < 0.7:
            # replace
            new_w = [rng.randrange(q) for _ in range(n)]
            new_ball_set = new_w_ball_set(new_w)
            # re-run try_replace with new_ball_set available
            w = codewords[ci]
            b_old = ball_index(w)
            ok = True
            for i in range(n):
                for d in range(q):
                    if d == w[i]:
                        continue
                    bi = b_old + digit_mul[i][d]
                    if cov_count[bi] == 1 and best_cw[bi] == ci:
                        if bi not in new_ball_set:
                            ok = False
                            break
                if not ok:
                    break
            if ok and cov_count[b_old] == 1 and best_cw[b_old] == ci and b_old not in new_ball_set:
                ok = False
            if ok:
                # perform
                remove_codeword(w, ci)
                codewords[ci] = new_w
                b = ball_index(new_w)
                for i in range(n):
                    for d in range(q):
                        if d == new_w[i]:
                            continue
                        bi = b + digit_mul[i][d]
                        cov_count[bi] += 1
                        if best_cw[bi] < 0:
                            best_cw[bi] = ci
                        elif best_cw[bi] != ci:
                            if second_cw[bi] < 0:
                                second_cw[bi] = ci
                cov_count[b] += 1
                if best_cw[b] < 0:
                    best_cw[b] = ci
                elif best_cw[b] != ci:
                    if second_cw[b] < 0:
                        second_cw[b] = ci
                # same size, no improvement in size; accept always (neutral)
                # but could enable future removals
            T *= cooling
        else:
            # add a random codeword (to allow reconfiguration)
            new_w = [rng.randrange(q) for _ in range(n)]
            add_codeword(new_w)
            active.append(len(codewords) - 1)
            T *= cooling

    # ---------- final greedy trim ----------
    # rebuild clean state from best_code
    codewords = [list(w) for w in best_code]
    cov_count = [0] * total
    best_cw = [-1] * total
    second_cw = [-1] * total
    for w in codewords:
        b = ball_index(w)
        cov_count[b] += 1
        if best_cw[b] < 0:
            best_cw[b] = len(codewords) - 1
        elif best_cw[b] != len(codewords) - 1:
            if second_cw[b] < 0:
                second_cw[b] = len(codewords) - 1
        for i in range(n):
            for d in range(q):
                if d == w[i]:
                    continue
                bi = b + digit_mul[i][d]
                cov_count[bi] += 1
                if best_cw[bi] < 0:
                    best_cw[bi] = len(codewords) - 1
                elif best_cw[bi] != len(codewords) - 1:
                    if second_cw[bi] < 0:
                        second_cw[bi] = len(codewords) - 1

    # deterministic trim: repeatedly remove any codeword that is not uniquely needed
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        for ci in range(len(codewords) - 1, -1, -1):
            w = codewords[ci]
            b = ball_index(w)
            removable = True
            for i in range(n):
                for d in range(q):
                    if d == w[i]:
                        continue
                    bi = b + digit_mul[i][d]
                    if cov_count[bi] == 1 and best_cw[bi] == ci:
                        removable = False
                        break
                if not removable:
                    break
            if removable and cov_count[b] == 1 and best_cw[b] == ci:
                removable = False
            if removable:
                # remove
                remove_codeword(w, ci)
                codewords.pop(ci)
                # fix indices: best_cw/second_cw entries > ci decrease by 1
                for bi in range(total):
                    if best_cw[bi] > ci:
                        best_cw[bi] -= 1
                    elif best_cw[bi] == ci:
                        best_cw[bi] = -1  # shouldn't happen since removable
                    if second_cw[bi] > ci:
                        second_cw[bi] -= 1
                    elif second_cw[bi] == ci:
                        second_cw[bi] = -1
                changed = True
                break

    return codewords
