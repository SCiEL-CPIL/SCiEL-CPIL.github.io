"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_check = min(len(new), len(block))
            overlap = 0
            for o in range(max_check, -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    S = list(word)
    best_S = S[:]
    best_len = len(S)

    # Initialize counts
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i + n])
        counts[p] = counts.get(p, 0) + 1

    # Greedy removal from right to left, multiple passes
    while time.time() < deadline:
        made_progress = False
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break

            # Old windows containing position i
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)

            valid = True
            old_perms = []
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j + n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
                old_perms.append(p)

            if not valid:
                continue

            # Decrement old
            for p in old_perms:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]

            # New straddling windows
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 1 - n)
            for k in range(start_new, end_new + 1):
                wc = []
                for idx in range(k, k + n):
                    s_idx = idx if idx < i else idx + 1
                    wc.append(S[s_idx])
                p_new = tuple(wc)
                counts[p_new] = counts.get(p_new, 0) + 1

            S.pop(i)
            made_progress = True

        if len(S) < best_len:
            best_S = S[:]
            best_len = len(S)

        if not made_progress:
            break

    return "".join(map(str, best_S))
# EVOLVE-BLOCK-END
