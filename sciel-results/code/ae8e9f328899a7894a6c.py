import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s wall-clock limit

    p = 1009
    # Precompute modular inverses for all residues 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)

    best_start = 2
    best_len = 1  # [2,2] is always valid: 2*2 = 4 != 1 mod 1009

    for s in range(1, p):
        if time.time() > deadline:
            break
        # Self-inverse residues 1 and 1008 cannot appear in any valid interval
        if s == 1 or s == p - 1:
            continue
        current_set = set()
        for e in range(s, p):
            if time.time() > deadline:
                break
            if e == 1 or e == p - 1:
                break  # contiguous interval can't skip self-inverse
            if inv[e] in current_set:
                break  # e and inv[e] would both be in interval
            current_set.add(e)
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s

    return [best_start, best_len]
