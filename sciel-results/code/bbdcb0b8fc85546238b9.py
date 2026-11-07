import time
import random

# ---------- precomputed divisibility table (max possible sum = 30*256 = 7680) ----------
MAX_SUM = 7680
DIV = [[False] * (MAX_SUM + 1) for _ in range(MAX_SUM + 1)]
for a in range(1, MAX_SUM + 1):
    row = DIV[a]
    for b in range(a, MAX_SUM + 1, a):
        row[b] = True

# ---------- candidate pool: mix of large and small (coprime-friendly) numbers ----------
# Large numbers (128-256) have few multiples; small numbers (1-60) are coprime-friendly
# and can grow the set because the constraint is on subset SUMS, not the elements.
candidates_base = list(range(1, 61)) + list(range(64, 257))


def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before the 90s wall-clock kill

    best_set = []
    max_restarts = 200000

    for i in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break

        random.shuffle(candidates_base)

        current_set = []
        sums = set()
        sums_list = []  # parallel list of the same elements, for fast iteration

        for c in candidates_base:
            # --- cheap pre-filter: reject before building the new-sum list ---
            # (1) c itself must not divide / be divided by any existing sum
            ok = True
            for s in sums_list:
                if DIV[c][s] or DIV[s][c]:
                    ok = False
                    break
            if not ok:
                continue

            # (2) collision: some existing sum s must not satisfy s+c == another existing sum t
            #     i.e. c != t - s for any s, t in sums. Check cheaply.
            if len(sums_list) > 1:
                # build difference set only when the set is large enough to matter
                diff = set()
                sl = sums_list
                for s in sl:
                    for t in sl:
                        d = t - s
                        if d == c:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue

            # --- build new sums ---
            new_sums_list = [c] + [s + c for s in sums_list]

            # --- check new sums against old sums ---
            ok = True
            for n in new_sums_list:
                for o in sums_list:
                    if DIV[n][o] or DIV[o][n]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            # --- check new sums against each other ---
            ns_sorted = sorted(new_sums_list)
            for idx_a in range(len(ns_sorted)):
                a = ns_sorted[idx_a]
                row_a = DIV[a]
                for idx_b in range(idx_a + 1, len(ns_sorted)):
                    b = ns_sorted[idx_b]
                    if row_a[b]:  # b is a multiple of a
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            # --- accept ---
            current_set.append(c)
            for ns in new_sums_list:
                sums.add(ns)
                sums_list.append(ns)

            if len(current_set) > len(best_set):
                best_set = current_set.copy()

    return [int(x) for x in best_set]
