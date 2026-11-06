import time

# EVOLVE-BLOCK-START
def construct():
    """Return a strictly increasing sequence in [1,256] whose nonempty
    consecutive-block sums are all distinct, maximizing length.
    Backtracking with prefix-sum pruning and a wall-clock deadline guard."""
    deadline = time.time() + 80.0
    MAXV = 256

    best = [1]
    best_len = 1

    def dfs(seq, sums):
        nonlocal best, best_len
        n = len(seq)
        if n > best_len:
            best_len = n
            best = seq[:]
        if n >= MAXV:
            return
        last = seq[-1]
        for x in range(last + 1, MAXV + 1):
            # New block sums that end at x (going backwards over seq):
            # x, x+last, x+last+..., each strictly increasing.
            new_sums = []
            t = x
            ok = True
            for a in reversed(seq):
                if t in sums:
                    ok = False
                    break
                new_sums.append(t)
                t += a
            if not ok:
                continue
            for s in new_sums:
                sums.add(s)
            seq.append(x)
            dfs(seq, sums)
            seq.pop()
            for s in new_sums:
                sums.discard(s)
            if time.time() > deadline:
                return

    dfs([1], set())
    return best
# EVOLVE-BLOCK-END
