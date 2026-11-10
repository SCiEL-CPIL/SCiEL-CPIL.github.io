"""Rewritten solver: multi-ordering greedy + 1-for-2 local search for
pairwise-disjoint infinite residue classes with distinct moduli in 2..128."""
import time
import random
import math

MAXM = 128
DEADLINE_FRACTION = 0.85  # safety margin before the 90s kill
_START = None


def _is_disjoint(r1, m1, r2, m2, gcd_table):
    g = gcd_table[m1][m2]
    return (r1 - r2) % g != 0


def _build_candidates():
    """Return list of (residue, modulus) candidates covering all moduli 2..128.
    Multiple residues per modulus give the greedy/search more options."""
    cands = []
    for m in range(2, MAXM + 1):
        seen = set()
        # deterministic heuristics
        for r in (0, 1, m - 1, (m - 1) // 2, m // 2, (m + 1) // 2, 2 * m // 3):
            r %= m
            if r not in seen:
                seen.add(r)
                cands.append((r, m))
        # a few random residues to diversify
        for _ in range(3):
            r = random.randrange(m)
            if r not in seen:
                seen.add(r)
                cands.append((r, m))
    return cands


def _greedy(cands, order, gcd_table, used_mod, selected, time_limit):
    """Greedy: insert candidates in `order`; keep a class if disjoint from all
    selected. Returns (selected, used_mod)."""
    for (r, m) in order:
        if time.time() > time_limit:
            break
        if m in used_mod:
            continue
        ok = True
        for (r2, m2) in selected:
            if not _is_disjoint(r, m, r2, m2, gcd_table):
                ok = False
                break
        if ok:
            selected.append((r, m))
            used_mod.add(m)
    return selected, used_mod


def _try_insert_one(selected, used_mod, r, m, gcd_table):
    if m in used_mod:
        return False
    for (r2, m2) in selected:
        if not _is_disjoint(r, m, r2, m2, gcd_table):
            return False
    return True


def _local_search(selected, used_mod, cands_by_mod, gcd_table, time_limit):
    """1-for-2 swap: drop one selected class, then try to insert two new
    (previously unused) moduli. Accept only if size grows."""
    improved = True
    while improved:
        improved = False
        if time.time() > time_limit:
            break
        n = len(selected)
        for drop_i in range(n):
            if time.time() > time_limit:
                break
            r_d, m_d = selected[drop_i]
            base = [x for x in selected if x is not selected[drop_i]]
            base_used = set(m2 for (_, m2) in base)
            # candidate moduli not currently used and != m_d
            cand_mods = [m for m in range(2, MAXM + 1)
                         if m not in base_used and m != m_d]
            # try to insert two disjoint classes among these
            best_pair = None
            for m1 in cand_mods:
                if best_pair is not None and len(best_pair) == 2:
                    break
                for r1 in cands_by_mod[m1]:
                    if not _try_insert_one(base, base_used, r1, m1, gcd_table):
                        continue
                    base.append((r1, m1))
                    base_used.add(m1)
                    for m2 in cand_mods:
                        if m2 == m1 or m2 in base_used:
                            continue
                        for r2 in cands_by_mod[m2]:
                            if _try_insert_one(base, base_used, r2, m2, gcd_table):
                                base.append((r2, m2))
                                base_used.add(m2)
                                best_pair = (m1, m2)
                                # revert to base-with-one
                                base.pop()
                                base_used.discard(m2)
                                break
                        if best_pair is not None:
                            break
                    # revert the one we added
                    base.pop()
                    base_used.discard(m1)
                    if best_pair is not None:
                        break
            if best_pair is not None:
                # reconstruct: base + the two classes (recompute residues)
                new_selected = list(base)
                new_used = set(base_used)
                for m in best_pair:
                    if m in new_used:
                        continue
                    for r in cands_by_mod[m]:
                        if _try_insert_one(new_selected, new_used, r, m, gcd_table):
                            new_selected.append((r, m))
                            new_used.add(m)
                            break
                if len(new_selected) > n:
                    selected = new_selected
                    used_mod = new_used
                    improved = True
                    break
    return selected, used_mod


def construct():
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 90.0 * DEADLINE_FRACTION

    # gcd table
    gcd_table = [[0] * (MAXM + 1) for _ in range(MAXM + 1)]
    for a in range(2, MAXM + 1):
        for b in range(2, MAXM + 1):
            gcd_table[a][b] = math.gcd(a, b)

    # candidate pool (built once; random part fixed by seed for reproducibility)
    random.seed(12345)
    cands = _build_candidates()
    cands_by_mod = {}
    for (r, m) in cands:
        cands_by_mod.setdefault(m, []).append(r)

    best = []
    best_size = 0

    def consider(sel):
        nonlocal best, best_size
        if len(sel) > best_size:
            best_size = len(sel)
            best = [list(x) for x in sel]

    # seed solution: power-of-2 family (guaranteed valid, 7 classes)
    seed_sel = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    consider(seed_sel)

    # multi-ordering greedy restarts
    all_mods = list(range(2, MAXM + 1))
    orders = []
    orders.append(list(all_mods))                 # ascending
    orders.append(list(reversed(all_mods)))       # descending
    orders.append(sorted(all_mods, key=lambda m: m.bit_count(), reverse=True))
    rnd = random.Random(999)
    for _ in range(4):
        o = list(all_mods)
        rnd.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > deadline:
            break
        # order the candidate list by the chosen modulus order, preserving
        # candidate diversity (iterate moduli, then their residues)
        ordered = []
        for m in order:
            for r in cands_by_mod[m]:
                ordered.append((r, m))
        sel, used = _greedy(ordered, ordered, gcd_table, set(), [], deadline)
        # local search
        sel, used = _local_search(sel, used, cands_by_mod, gcd_table, deadline)
        consider(sel)

    # final local-search polish on the current best
    if best and time.time() < deadline:
        sel = [tuple(x) for x in best]
        used = set(m for (_, m) in sel)
        sel, used = _local_search(sel, used, cands_by_mod, gcd_table, deadline)
        consider(sel)

    if not best:
        best = [[1, 2]]
    return best


# Self-test (not used by evaluator, but keeps module importable & sanity-checked)
if __name__ == "__main__":
    sol = construct()
    print("size:", len(sol))
    print(sol)
    # verify disjointness
    for i in range(len(sol)):
        for j in range(i + 1, len(sol)):
            r1, m1 = sol[i]
            r2, m2 = sol[j]
            g = math.gcd(m1, m2)
            assert (r1 - r2) % g != 0, (i, j, r1, m1, r2, m2)
    # verify distinct moduli and canonical residues
    mods = [m for (_, m) in sol]
    assert len(set(mods)) == len(mods)
    for r, m in sol:
        assert 0 <= r < m and 2 <= m <= 128
    print("OK")
