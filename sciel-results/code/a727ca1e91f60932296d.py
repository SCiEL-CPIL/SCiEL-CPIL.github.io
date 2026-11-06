# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    # ---------------- Core vectorized add/remove (fast) ----------------
    def _add(chosen, chosen_set, counts, x):
        counts[2 * x] += 1
        if chosen:
            counts[x + chosen] += 1
        chosen.append(x)
        chosen_set.add(x)

    def _remove(chosen, chosen_set, counts, x):
        counts[2 * x] -= 1
        if chosen:
            counts[x + chosen] -= 1
        chosen.remove(x)
        chosen_set.discard(x)

    def _validity(chosen, counts, candidates):
        """Boolean array: which candidates are insertable (sums stay <= 2)."""
        if len(candidates) == 0:
            return np.zeros(0, dtype=bool)
        v = counts[2 * candidates] < 2
        if chosen:
            v &= np.all(counts[candidates[:, None] + chosen[None, :]] < 2, axis=1)
        return v

    def _saturate(chosen, chosen_set, counts, deadline):
        """Insert every compatible element until none remain (grows the set)."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            cands = np.setdiff1d(np.arange(1, 1001), np.array(sorted(chosen_set)))
            if len(cands) == 0:
                break
            valid = _validity(np.array(sorted(chosen_set)), counts, cands)
            for x in cands[valid]:
                _add(chosen, chosen_set, counts, int(x))
                changed = True

    # ---------------- Mian-Chowla (Sidon) core builders ----------------
    def _mc_forward():
        S = [1]
        Sset = {1}
        for x in range(2, 1001):
            ok = True
            for a in S:
                if (x - a) in Sset and (x - a) >= a:
                    ok = False
                    break
            if ok:
                S.append(x)
                Sset.add(x)
        return S

    def _mc_random(seed):
        rng = random.Random(seed)
        order = list(range(1, 1001))
        rng.shuffle(order)
        S = []
        Sset = set()
        for x in order:
            ok = True
            for a in S:
                if (x - a) in Sset and (x - a) >= a:
                    ok = False
                    break
            if ok:
                S.append(x)
                Sset.add(x)
        return S

    # ---------------- Local search: insert + swap (can grow) ----------------
    def _local_search(chosen, time_limit):
        chosen = list(chosen)
        chosen_set = set(chosen)
        counts = np.zeros(2001, dtype=np.int32)
        for i, a in enumerate(chosen):
            counts[2 * a] += 1
            for b in chosen[i + 1:]:
                counts[a + b] += 1

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            n = len(chosen)
            cands = np.setdiff1d(np.arange(1, 1001), np.array(sorted(chosen_set)))
            if len(cands) == 0:
                break

            # (a) pure insert
            valid = _validity(np.array(sorted(chosen_set)), counts, cands)
            if np.any(valid):
                x = int(cands[valid][0])
                _add(chosen, chosen_set, counts, x)
                improved = True
                continue

            # (b) swap: remove one, then try to insert a bigger one
            for i in range(n):
                old = chosen[i]
                _remove(chosen, chosen_set, counts, old)
                rem = np.array(sorted(chosen_set))
                rem_cands = np.setdiff1d(np.arange(old + 1, 1001), rem)
                if len(rem_cands) > 0:
                    rv = _validity(rem, counts, rem_cands)
                    if np.any(rv):
                        x = int(rem_cands[rv][0])
                        _add(chosen, chosen_set, counts, x)
                        improved = True
                        break
                if not improved:
                    _add(chosen, chosen_set, counts, old)
        return chosen

    # ---------------- Drive multiple cores ----------------
    best_set = []
    best_size = 0

    def _consider(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)

    # Core 1: deterministic forward Mian-Chowla
    S = _mc_forward()
    Sset = set(S)
    counts = np.zeros(2001, dtype=np.int32)
    for i, a in enumerate(S):
        counts[2 * a] += 1
        for b in S[i + 1:]:
            counts[a + b] += 1
    _saturate(S, Sset, counts, deadline - 25)
    _consider(S)

    # Random Sidon cores, saturate + local search each
    t = 0
    seed = 1
    while time.time() < deadline - 12:
        S = _mc_random(seed)
        seed += 1
        Sset = set(S)
        counts = np.zeros(2001, dtype=np.int32)
        for i, a in enumerate(S):
            counts[2 * a] += 1
            for b in S[i + 1:]:
                counts[a + b] += 1
        _saturate(S, Sset, counts, deadline - 10)
        if time.time() < deadline - 6:
            S = _local_search(S, min(time.time() + 4, deadline - 5))
        _consider(S)
        t += 1

    # Final polish on the best set
    if time.time() < deadline - 3:
        best_set = _local_search(best_set, deadline - 1)
        best_size = len(best_set)

    return best_set
# EVOLVE-BLOCK-END
