"""New design: greedy seed + local search insertion of elements."""
import time

def construct():
    deadline = time.time() + 80.0

    def all_sums_distinct(seq):
        s = set()
        for i in range(len(seq)):
            total = 0
            for j in range(i, len(seq)):
                total += seq[j]
                if total in s:
                    return False
                s.add(total)
        return True

    # --- Phase 1: Greedy seed ---
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    best = list(values)

    if time.time() > deadline - 10.0:
        return best

    # --- Phase 2: Local search insertion ---
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        used = set(best)
        unused = [x for x in range(1, 257) if x not in used]
        if not unused:
            break
        for val in unused:
            if time.time() > deadline - 5.0:
                break
            for pos in range(len(best) + 1):
                candidate = best[:pos] + [val] + best[pos:]
                if all_sums_distinct(candidate):
                    best = candidate
                    improved = True
                    break
            if improved:
                break

    return best
