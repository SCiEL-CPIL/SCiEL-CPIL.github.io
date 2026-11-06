"""Fresh design: greedy + local search over residue classes with distinct moduli 2..128."""
import math
import random
import time

MAXIMUM = 128
MODULI = list(range(2, MAXIMUM + 1))


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _is_pairwise_disjoint(pairs):
    n = len(pairs)
    for i in range(n):
        ri, mi = pairs[i]
        for j in range(i + 1, n):
            rj, mj = pairs[j]
            if _gcd(mi, mj) % (ri - rj) == 0:
                return False
    return True


def _greedy(moduli_order, seed=0):
    rng = random.Random(seed)
    chosen = []
    used_mods = set()
    for m in moduli_order:
        if m in used_mods:
            continue
        candidates = list(range(m))
        rng.shuffle(candidates)
        best_r = None
        for r in candidates:
            ok = True
            for (r2, m2) in chosen:
                if _gcd(m, m2) % (r - r2) == 0:
                    ok = False
                    break
            if ok:
                best_r = r
                break
        if best_r is not None:
            chosen.append([best_r, m])
            used_mods.add(m)
    return chosen


def _local_search(pairs, seed=0, time_limit=55.0):
    rng = random.Random(seed)
    start = time.time()
    best = [p[:] for p in pairs]
    best_len = len(best)
    cur = [p[:] for p in pairs]
    cur_len = len(cur)
    all_mods = set(MODULI)
    in_set = set(m for (_, m) in cur)

    # candidate residues per modulus: 0,1,2,3,4,5,6,7,8,9,10,11
    cand_res = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    def valid_with(pairs, r, m):
        for (r2, m2) in pairs:
            if m2 == m:
                continue
            if _gcd(m, m2) % (r - r2) == 0:
                return False
        return True

    while time.time() - start < time_limit:
        improved = False

        # 1) Try to add a missing modulus with a valid residue.
        missing = [m for m in MODULI if m not in in_set]
        rng.shuffle(missing)
        for m in missing:
            cands = cand_res[:]
            rng.shuffle(cands)
            for r in cands:
                if valid_with(cur, r, m):
                    cur.append([r, m])
                    in_set.add(m)
                    cur_len += 1
                    improved = True
                    break
            if improved:
                break

        # 2) Try to swap a residue of an existing modulus to free capacity.
        idxs = list(range(len(cur)))
        rng.shuffle(idxs)
        for i in idxs:
            r, m = cur[i]
            cands = [c for c in cand_res if c != r]
            rng.shuffle(cands)
            for nr in cands:
                # check new residue against all others
                ok = True
                for j in range(len(cur)):
                    if j == i:
                        continue
                    r2, m2 = cur[j]
                    if _gcd(m, m2) % (nr - r2) == 0:
                        ok = False
                        break
                if ok:
                    # does this help? Try an add pass right after.
                    cur[i][0] = nr
                    # attempt to add a missing modulus now
                    missing2 = [m2 for m2 in MODULI if m2 not in in_set]
                    rng.shuffle(missing2)
                    added = False
                    for m2 in missing2:
                        cands2 = cand_res[:]
                        rng.shuffle(cands2)
                        for r2 in cands2:
                            if valid_with(cur, r2, m2):
                                cur.append([r2, m2])
                                in_set.add(m2)
                                cur_len += 1
                                added = True
                                break
                        if added:
                            break
                    if added:
                        improved = True
                        break
                    # if no add, revert the swap (keep search simple)
                    cur[i][0] = r
            if improved:
                break

        if not improved:
            # try removing a modulus that is "blocking" many candidates, then re-add
            # (simple heuristic: drop the one with smallest modulus first, then re-greedy-add)
            # Only do this occasionally to avoid churn.
            if len(cur) > 1 and rng.random() < 0.3:
                # remove a random element
                i = rng.randrange(len(cur))
                r, m = cur[i]
                cur.pop(i)
                in_set.discard(m)
                cur_len -= 1
                # now try to add several
                missing = [m2 for m2 in MODULI if m2 not in in_set]
                rng.shuffle(missing)
                added_any = False
                for m2 in missing:
                    cands = cand_res[:]
                    rng.shuffle(cands)
                    for r2 in cands:
                        if valid_with(cur, r2, m2):
                            cur.append([r2, m2])
                            in_set.add(m2)
                            cur_len += 1
                            added_any = True
                            break
                    if cur_len > cur_len - 1 and added_any:
                        # stop adding once we've recovered
                        if cur_len >= len(cur) - 1:
                            break
            if not improved:
                # no progress this round; if we've been stuck, break
                break

        if cur_len > best_len:
            best_len = cur_len
            best = [p[:] for p in cur]

        if best_len == len(MODULI):
            break

    return best


def construct():
    deadline = time.time() + 55.0  # safety margin under 60s/90s limits

    # Initial greedy with a few orderings.
    candidates = []
    base = _greedy(MODULI, seed=0)
    candidates.append(base)

    # Try reverse order and a few shuffles.
    for s in (1, 2, 3, 4, 5):
        order = MODULI[:]
        random.Random(s).shuffle(order)
        candidates.append(_greedy(order, seed=s))

    best = max(candidates, key=len)

    # Local search from the best greedy (and a couple of others if time permits).
    for start_pairs in [best, candidates[0]]:
        if time.time() > deadline:
            break
        ls = _local_search(start_pairs, seed=42, time_limit=deadline - time.time() - 1.0)
        if len(ls) > len(best):
            best = ls

    if time.time() > deadline:
        # return best found so far
        pass

    # Final safety: ensure validity and nonempty.
    if not best:
        best = [[1, 2]]
    if not _is_pairwise_disjoint(best):
        # rebuild via a conservative greedy as fallback
        fb = _greedy(MODULI, seed=999)
        best = fb if fb else [[1, 2]]

    return best
