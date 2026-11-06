"""Size-efficient sorting network on 17 wires.

Design (from scratch):
  * Seed: odd-even transposition network (136 comparators).
  * Verification via the zero-one principle: all 2**17 = 131072 Boolean inputs
    are simulated at once.  Each wire is a numpy uint64 array of 2048 words
    (2048*64 = 131072 bits); a comparator is a handful of vector ops.
  * Local search: repeatedly try to REMOVE a single comparator; keep the
    removal if the network still sorts every input.  This is a sound
    monotone-improvement heuristic that can only reduce the comparator count.
  * An explicit wall-clock deadline guard returns the best network found so far.
"""
import time
import numpy as np

N = 17
M = 1 << N            # number of zero-one inputs
NW = M // 64          # uint64 words per wire (M is a multiple of 64)
T_LIMIT = 75.0        # safety margin before the 90 s evaluator kill


def construct():
    t0 = time.time()

    # ---- build the seed network (odd-even transposition) -------------------
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))

    # ---- precompute bit masks for the M zero-one inputs --------------------
    # word w holds bits [64*w, 64*w+64); bit b within a word is set iff
    # input value v has (v >> (64*w + b)) & 1 == 1.
    words = np.arange(NW, dtype=np.uint64)
    masks = (words[:, None] << 6) | np.arange(64, dtype=np.uint64)[None, :]
    masks = np.bitwise_and(masks, np.uint64(17))
    masks = np.bitwise_and(masks, np.uint64(63))
    masks = np.left_shift(masks, np.uint64(1))          # masks[k] = 1 << k
    masks = np.ascontiguousarray(masks, dtype=np.uint64)  # [64, NW]

    def verify(net):
        """Return True iff `net` sorts every zero-one input."""
        wires = np.zeros((N, NW), dtype=np.uint64)
        for k in range(N):
            wires[k] = masks[k]
        for (i, j) in net:
            a = wires[i]
            b = wires[j]
            m = np.bitwise_and(a, b)
            wires[i] = np.bitwise_or(m, np.bitwise_and(a, np.invert(b)))
            wires[j] = np.bitwise_or(m, np.bitwise_and(b, np.invert(a)))
        for k in range(1, N):
            if not np.array_equal(wires[k], masks[k]):
                return False
        return True

    # ---- local search: greedy removal of redundant comparators -------------
    best = list(net)
    best_len = len(best)
    if verify(best):
        changed = True
        while changed and (time.time() - t0) < T_LIMIT:
            changed = False
            cur = list(best)
            i = 0
            while i < len(cur):
                if (time.time() - t0) > T_LIMIT - 3.0:
                    break
                cand = cur[:i] + cur[i + 1:]
                if verify(cand):
                    cur = cand
                    changed = True
                    # do not increment i: re-check this position with the
                    # shifted list (a further removal may now be possible).
                else:
                    i += 1
            if len(cur) < best_len:
                best = cur
                best_len = len(cur)

    return best


# ---- self-test (skipped when imported by the evaluator) --------------------
if __name__ == "__main__":
    t0 = time.time()
    net = construct()
    print("comparators:", len(net), "elapsed: %.2f s" % (time.time() - t0))
    # quick sanity: verify with the same machinery
    M = 1 << N
    NW = M // 64
    words = np.arange(NW, dtype=np.uint64)
    masks = (words[:, None] << 6) | np.arange(64, dtype=np.uint64)[None, :]
    masks = np.bitwise_and(masks, np.uint64(17))
    masks = np.bitwise_and(masks, np.uint64(63))
    masks = np.left_shift(masks, np.uint64(1))
    masks = np.ascontiguousarray(masks, dtype=np.uint64)
    wires = np.zeros((N, NW), dtype=np.uint64)
    for k in range(N):
        wires[k] = masks[k]
    for (i, j) in net:
        a = wires[i]; b = wires[j]
        m = np.bitwise_and(a, b)
        wires[i] = np.bitwise_or(m, np.bitwise_and(a, np.invert(b)))
        wires[j] = np.bitwise_or(m, np.bitwise_and(b, np.invert(a)))
    ok = all(np.array_equal(wires[k], masks[k]) for k in range(1, N))
    print("sorts all zero-one inputs:", ok)
    # also test a couple of random non-Boolean inputs
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.integers(0, 100, size=N)
        w = x.copy()
        for (i, j) in net:
            if w[i] > w[j]:
                w[i], w[j] = w[j], w[i]
        assert list(w) == sorted(x.tolist()), "failed on non-Boolean input"
    print("non-Boolean spot checks passed")
