"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 12
M = 40
FULL = (1 << N) - 1

def _is_sub(a, b):
    return (a & b) == a

def _score_from_mask(masks):
    s = 0
    for i in range(M):
        a = masks[i]
        for j in range(i + 1, M):
            b = masks[j]
            if a == b:
                continue
            if (a & b) == a:
                s += 1
            elif (a & b) == b:
                s += 1
    return s

def _init_masks():
    # Build a diverse starting family: mix of small and large sets
    masks = []
    # Add empty set and full set
    masks.append(0)
    masks.append(FULL)
    # Add a chain: {0}, {0,1}, {0,1,2}, ... up to 11
    for k in range(1, N):
        m = (1 << k) - 1
        masks.append(m)
    # Add random sets of various sizes
    random.seed(12345)
    while len(masks) < M:
        size = random.randint(0, N)
        if size == 0:
            m = 0
        elif size == N:
            m = FULL
        else:
            chosen = random.sample(range(N), size)
            m = 0
            for v in chosen:
                m |= (1 << v)
        masks.append(m)
    return masks

def _neighbor(masks):
    # Pick a random set and a random move
    i = random.randrange(M)
    move = random.random()
    if move < 0.5:
        # Flip one bit (add or remove a vertex)
        bit = random.randrange(N)
        masks[i] ^= (1 << bit)
        return i
    elif move < 0.75:
        # Replace with a random set of similar size
        cur = masks[i]
        size = bin(cur).count('1')
        new_size = max(0, min(N, size + random.randint(-2, 2)))
        if new_size == 0:
            new_m = 0
        elif new_size == N:
            new_m = FULL
        else:
            chosen = random.sample(range(N), new_size)
            new_m = 0
            for v in chosen:
                new_m |= (1 << v)
        masks[i] = new_m
        return i
    else:
        # Swap two sets' contents (helps escape local optima)
        j = random.randrange(M)
        masks[i], masks[j] = masks[j], masks[i]
        return i, j

def _score_change(masks, idx):
    # Score contribution of idx with all others (excluding self)
    a = masks[idx]
    s = 0
    for j in range(M):
        if j == idx:
            continue
        b = masks[j]
        if a == b:
            continue
        if (a & b) == a:
            s += 1
        elif (a & b) == b:
            s += 1
    return s

def construct():
    deadline = time.time() + 80.0
    best_score = -1
    best_masks = None

    for restart in range(3):
        masks = _init_masks()
        # Perturb for subsequent restarts
        if restart > 0:
            for _ in range(20):
                i = random.randrange(M)
                bit = random.randrange(N)
                masks[i] ^= (1 << bit)

        score = _score_from_mask(masks)
        if score > best_score:
            best_score = score
            best_masks = list(masks)

        T = 2.0
        T_min = 0.01
        cooling = 0.9995

        while time.time() < deadline:
            if T < T_min:
                break
            old = _neighbor(masks)
            if isinstance(old, tuple):
                i, j = old
                # Compute delta for swap: score change
                # Before: contributions of i and j with everyone else (excluding each other counted once)
                # After swap: i has old j, j has old i
                # Delta = (new_contrib_i + new_contrib_j) - (old_contrib_i + old_contrib_j)
                # where contrib excludes the pair (i,j) itself
                a_old = masks[i]
                b_old = masks[j]
                # old contributions of i (with all except j) and j (with all except i)
                old_ci = 0
                old_cj = 0
                for k in range(M):
                    if k == i or k == j:
                        continue
                    c = masks[k]
                    if a_old != c and ((a_old & c) == a_old or (a_old & c) == c):
                        old_ci += 1
                    if b_old != c and ((b_old & c) == b_old or (b_old & c) == c):
                        old_cj += 1
                # pair (i,j) itself
                pair_old = 0
                if a_old != b_old and ((a_old & b_old) == a_old or (a_old & b_old) == b_old):
                    pair_old = 1
                # new: i has b_old, j has a_old
                new_ci = 0
                new_cj = 0
                for k in range(M):
                    if k == i or k == j:
                        continue
                    c = masks[k]
                    if b_old != c and ((b_old & c) == b_old or (b_old & c) == c):
                        new_ci += 1
                    if a_old != c and ((a_old & c) == a_old or (a_old & c) == c):
                        new_cj += 1
                pair_new = 0
                if b_old != a_old and ((b_old & a_old) == b_old or (b_old & a_old) == a_old):
                    pair_new = 1
                delta = (new_ci + new_cj + pair_new) - (old_ci + old_cj + pair_old)
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    score += delta
                else:
                    masks[i], masks[j] = a_old, b_old
                    i = None
            else:
                i = old
                a_old = masks[i]
                # old contribution of i
                old_c = 0
                for k in range(M):
                    if k == i:
                        continue
                    b = masks[k]
                    if a_old != b and ((a_old & b) == a_old or (a_old & b) == b):
                        old_c += 1
                # new contribution
                a_new = masks[i]
                new_c = 0
                for k in range(M):
                    if k == i:
                        continue
                    b = masks[k]
                    if a_new != b and ((a_new & b) == a_new or (a_new & b) == b):
                        new_c += 1
                delta = new_c - old_c
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    score += delta
                else:
                    masks[i] = a_old
            if score > best_score:
                best_score = score
                best_masks = list(masks)
            T *= cooling

    # Convert to lists of ints
    result = []
    for m in best_masks:
        result.append([v for v in range(N) if (m >> v) & 1])
    return result
# EVOLVE-BLOCK-END
