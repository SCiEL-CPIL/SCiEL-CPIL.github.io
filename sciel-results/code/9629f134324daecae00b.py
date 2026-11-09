"""Difference triangle set (7,5) – combined: interleaved greedy + targeted reduction +
difference-signature genetic algorithm + row-replacement SA, with diff-count tightening."""
import time
import random
import math

_START = time.time()
_DEADLINE = _START + 145.0


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    """Return set of all positive differences in a row."""
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _scope(rows):
    return max(max(r) for r in rows)


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if r[j] <= r[i]:
                    return False
        for d in _diffs_of_row(r):
            if d in used:
                return False
            used.add(d)
    return True


class DiffTracker:
    """Fast difference tracking using a bytearray for O(1) membership."""
    __slots__ = ('max_diff', 'used', 'count')

    def __init__(self, max_diff):
        self.max_diff = max_diff
        self.used = bytearray(max_diff + 1)
        self.count = 0

    def add(self, diffs):
        u = self.used
        md = self.max_diff
        for d in diffs:
            if d <= md and not u[d]:
                u[d] = 1
                self.count += 1

    def remove(self, diffs):
        u = self.used
        md = self.max_diff
        for d in diffs:
            if d <= md and u[d]:
                u[d] = 0
                self.count -= 1

    def check(self, diffs):
        u = self.used
        md = self.max_diff
        for d in diffs:
            if d <= md and u[d]:
                return False
        return True

    def copy(self):
        nt = DiffTracker(self.max_diff)
        nt.used = bytearray(self.used)
        nt.count = self.count
        return nt


def _greedy_row_fast(tracker, k, max_mark):
    """Greedy: build a row of k+1 entries starting at 0, smallest valid x at each step."""
    row = [0]
    for _ in range(k):
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            new_diffs = {x - a for a in row}
            if tracker.check(new_diffs):
                row.append(x)
                tracker.add(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None
    return row


def _interleaved_greedy(n_rows, k, max_mark, rnd):
    """Interleaved greedy: extend the row with smallest current max at each step."""
    tracker = DiffTracker(max_mark + 1)
    rows = [[0] for _ in range(n_rows)]

    for _ in range(n_rows * k):
        # pick row with smallest current max among incomplete
        i = 0
        best_last = rows[0][-1]
        for idx in range(1, n_rows):
            if len(rows[idx]) < k + 1 and rows[idx][-1] < best_last:
                i = idx
                best_last = rows[idx][-1]
        if len(rows[i]) >= k + 1:
            # all complete?
            if all(len(r) == k + 1 for r in rows):
                break
            # find any incomplete
            i = None
            for idx in range(n_rows):
                if len(rows[idx]) < k + 1:
                    i = idx
                    break
            if i is None:
                break
        row = rows[i]
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            new_diffs = {x - a for a in row}
            if tracker.check(new_diffs):
                row.append(x)
                tracker.add(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None, None

    if all(len(r) == k + 1 for r in rows):
        return rows, tracker
    return None, None


def _multi_seed_greedy(n_rows, k, num_seeds, time_budget, rnd):
    """Generate multiple interleaved greedy solutions with different seeds, return best."""
    deadline = time.time() + time_budget
    best_rows = None
    best_scope = float('inf')
    for seed in range(num_seeds):
        if time.time() > deadline:
            break
        r2 = random.Random(seed * 7919 + 13)
        rows, _ = _interleaved_greedy(n_rows, k, 200, r2)
        if rows is not None:
            sc = _scope(rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [r[:] for r in rows]
    return best_rows, best_scope


def _reconstruct_row(signature, k, max_mark):
    """
    Given a set of k*(k+1)/2 difference values (the row's full difference set),
    greedily reconstruct a row of k+1 increasing integers starting at 0 whose
    difference set equals `signature`. Return row or None if impossible.
    We build the row entry by entry; at each step the new entry x must produce
    differences {x - a : a in row} all in signature, and we consume them.
    """
    remaining = set(signature)
    row = [0]
    for _ in range(k):
        # candidate x: must be > row[-1]; the differences x - a for a in row must all be in remaining
        # Smallest feasible x: try x = row[-1] + 1 upward
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            ok = True
            new_diffs = []
            for a in row:
                d = x - a
                if d <= 0 or d not in remaining:
                    ok = False
                    break
                new_diffs.append(d)
            if ok:
                row.append(x)
                for d in new_diffs:
                    remaining.discard(d)
                found = True
                break
            x += 1
        if not found:
            return None
    # all differences must be consumed
    if remaining:
        return None
    return row


def _signature_from_row(row):
    return _diffs_of_row(row)


def _ga_signature(n_rows, k, lb, max_mark, time_budget, rnd, seed_rows=None):
    """
    Genetic algorithm over difference-signature partitions.
    Universe U = {1..max_mark} (we only need 105 values; we work with the full set 1..max_mark
    but require each signature to be exactly the row's 15 diffs).
    Actually we restrict universe to a candidate set: the union of diffs from a good seed,
    or simply 1..max_mark. We maintain a partition of a fixed pool P of size n_rows*15.
    P is chosen as the multiset-free set: the 105 smallest values that appear across seed rows,
    or 1..105 if no seed. Each signature is a 15-subset; the 7 signatures partition P.
    Fitness: number of signatures that embed into a valid row + (-scope).
    """
    n_sig = n_rows * k * (k + 1) // 2  # total diffs needed = 105
    # Build pool P
    if seed_rows is not None:
        union = set()
        for r in seed_rows:
            union |= _signature_from_row(r)
        # take the smallest n_sig values from union; if union too small, pad with 1..
        vals = sorted(union)
        if len(vals) < n_sig:
            # pad
            extra = set(range(1, max_mark + 1)) - set(vals)
            vals += sorted(extra)[:n_sig - len(vals)]
        pool = sorted(vals[:n_sig])
    else:
        pool = list(range(1, n_sig + 1))

    # Initial partition: assign pool values round-robin into n_rows signatures of size 15
    def make_partition(pool, rnd):
        sigs = [set() for _ in range(n_rows)]
        # shuffle pool
        p = pool[:]
        rnd.shuffle(p)
        for i, v in enumerate(p):
            sigs[i % n_rows].add(v)
        return sigs

    def embed_all(sigs, max_mark):
        """Try to embed each signature; return (rows, scope) or None."""
        rows = []
        for sig in sigs:
            row = _reconstruct_row(sig, k, max_mark)
            if row is None:
                return None
            rows.append(row)
        return rows, _scope(rows)

    def fitness(sigs, max_mark):
        # count embeddable + penalize scope
        cnt = 0
        best_sc = 0
        for sig in sigs:
            row = _reconstruct_row(sig, k, max_mark)
            if row is not None:
                cnt += 1
                best_sc = max(best_sc, max(row))
        # higher is better: embed count dominates, then lower scope
        return cnt * 10000 - best_sc

    deadline = time.time() + time_budget
    pop_size = 60
    elite = 6

    # Seed population: include the seed rows' signatures if available
    population = []
    if seed_rows is not None:
        seed_sigs = [_signature_from_row(r) for r in seed_rows]
        # check they partition a valid set (they should, since rows valid)
        population.append(seed_sigs)
    while len(population) < pop_size:
        population.append(make_partition(pool, rnd))

    best_sigs = None
    best_fit = float('-inf')
    for sigs in population:
        f = fitness(sigs, max_mark)
        if f > best_fit:
            best_fit = f
            best_sigs = [s.copy() for s in sigs]

    gen = 0
    while time.time() < deadline and gen < 4000:
        gen += 1
        # Evaluate current population, keep elites
        scored = []
        for sigs in population:
            f = fitness(sigs, max_mark)
            scored.append((f, sigs))
        scored.sort(key=lambda x: -x[0])
        for f, sigs in scored:
            if f > best_fit:
                best_fit = f
                best_sigs = [s.copy() for s in sigs]

        new_pop = [ [s.copy() for s in sigs] for f, sigs in scored[:elite] ]
        # check if best fully embeds
        rows, sc = embed_all(best_sigs, max_mark)
        if rows is not None and sc <= max_mark:
            # success; return
            return rows, sc

        while len(new_pop) < pop_size:
            p1 = scored[rnd.randint(0, len(scored) - 1)][1]
            p2 = scored[rnd.randint(0, len(scored) - 1)][1]
            # crossover: swap whole signatures between two parents
            child = [s.copy() for s in p1]
            # swap a few signature slots
            for _ in range(rnd.randint(1, 3)):
                a = rnd.randint(0, n_rows - 1)
                b = rnd.randint(0, n_rows - 1)
                child[a], child[b] = p2[a].copy(), p2[b].copy()
            # mutation: swap one value between two signatures (preserves partition)
            if rnd.random() < 0.9:
                for _ in range(rnd.randint(1, 2)):
                    a = rnd.randint(0, n_rows - 1)
                    b = rnd.randint(0, n_rows - 1)
                    if a == b:
                        continue
                    va = rnd.choice(list(child[a]))
                    vb = rnd.choice(list(child[b]))
                    child[a].discard(va)
                    child[b].discard(vb)
                    child[a].add(vb)
                    child[b].add(va)
            new_pop.append(child)
        population = new_pop

    # Return best embeddable if any, else None
    if best_sigs is not None:
        rows, sc = embed_all(best_sigs, max_mark)
        if rows is not None:
            return rows, sc
    return None, float('inf')


def _tighten_polish(rows, n_rows, k, time_limit, max_mark):
    """
    Deterministic tightening with diff-count arrays: repeatedly reduce each element
    to its smallest valid value. Uses int diff-counts for fast bookkeeping.
    """
    start_time = time.time()
    current = [list(r) for r in rows]
    mv = max(max(r) for r in current)
    size = max_mark + k * (k + 1) + 50
    diff_counts = [0] * (size + 1)

    def build_counts():
        for d in diff_counts:
            pass
        diff_counts[:] = [0] * (size + 1)
        for row in current:
            n = len(row)
            for i in range(n):
                ai = row[i]
                for j in range(i + 1, n):
                    d = row[j] - ai
                    if d <= size:
                        diff_counts[d] += 1

    build_counts()

    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current[r]
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                if lower_bound >= old_val:
                    continue
                # old diffs involving pos
                old_diffs = [abs(old_val - row[i]) for i in range(len(row)) if i != pos]
                for new_val in range(lower_bound, old_val):
                    new_diffs = [abs(new_val - row[i]) for i in range(len(row)) if i != pos]
                    # remove old
                    for d in old_diffs:
                        if d <= size:
                            diff_counts[d] -= 1
                    valid = True
                    for d in new_diffs:
                        if d > size:
                            valid = False
                            break
                        if diff_counts[d] > 0:
                            valid = False
                            break
                    if valid:
                        row[pos] = new_val
                        for d in new_diffs:
                            diff_counts[d] += 1
                        improved = True
                        break
                    else:
                        for d in old_diffs:
                            if d <= size:
                                diff_counts[d] += 1
    return current


def _sa_row_replacement(rows, n_rows, k, max_mark, time_budget, rnd):
    """Simulated Annealing with row-replacement moves (rebuild a row given the others' diffs)."""
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    # diffs of all rows
    all_diffs = set()
    for r in best:
        all_diffs |= _diffs_of_row(r)

    deadline = time.time() + time_budget
    T = 8.0
    cooling = 0.999
    min_T = 0.05

    def rebuild_row(ri, tracker, max_mark):
        """Rebuild row ri greedily given tracker (which excludes row ri's diffs)."""
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            found = False
            while x <= max_mark:
                new_diffs = {x - a for a in row}
                if tracker.check(new_diffs):
                    row.append(x)
                    tracker.add(new_diffs)
                    found = True
                    break
                x += 1
            if not found:
                return None
        return row

    while time.time() < deadline and T > min_T:
        ri = rnd.randint(0, n_rows - 1)
        old_row = best[ri]
        old_diffs = _diffs_of_row(old_row)
        # build tracker of other rows
        tracker = DiffTracker(max_mark + 1)
        for idx in range(n_rows):
            if idx != ri:
                tracker.add(_diffs_of_row(best[idx]))
        new_row = rebuild_row(ri, tracker, max_mark)
        if new_row is None:
            T *= cooling
            continue
        new_diffs = _diffs_of_row(new_row)
        # ensure no conflict (should be fine since tracker excluded old)
        # compute new scope
        other_max = 0
        for idx in range(n_rows):
            if idx != ri:
                other_max = max(other_max, max(best[idx]))
        new_scope = max(other_max, max(new_row))
        delta = new_scope - best_scope
        if delta <= 0 or rnd.random() < math.exp(-max(0, delta) / T):
            best[ri] = new_row
            if new_scope < best_scope:
                best_scope = new_scope
        T *= cooling
    return best, best_scope


def construct(n_rows, k):
    rnd = random.Random(12345)
    best_rows = None
    best_scope = float('inf')

    lb = n_rows * k * (k + 1) // 2  # 105 for (7,5)
    max_mark = 200

    # Phase 1: multi-seed interleaved greedy for diverse starts
    rows, sc = _multi_seed_greedy(n_rows, k, 12, 15.0, rnd)
    if rows is not None:
        best_rows = rows
        best_scope = sc

    if _dead():
        return best_rows

    # Phase 2: tighten the greedy solution
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 10.0
        if remaining > 3.0:
            best_rows = _tighten_polish(best_rows, n_rows, k, min(remaining, 30.0), max_mark)
            best_scope = _scope(best_rows)

    if _dead():
        return best_rows

    # Phase 3: difference-signature genetic algorithm (structural change)
    remaining = _DEADLINE - time.time() - 20.0
    if remaining > 5.0:
        ga_rows, ga_sc = _ga_signature(n_rows, k, lb, max_mark, min(remaining, 60.0), rnd,
                                       seed_rows=best_rows)
        if ga_rows is not None and ga_sc < best_scope:
            best_scope = ga_sc
            best_rows = ga_rows

    if _dead():
        return best_rows

    # Phase 4: tighten after GA
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 10.0
        if remaining > 3.0:
            best_rows = _tighten_polish(best_rows, n_rows, k, min(remaining, 25.0), max_mark)
            best_scope = _scope(best_rows)

    if _dead():
        return best_rows

    # Phase 5: row-replacement SA to escape local minima
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 8.0
        if remaining > 3.0:
            best_rows, best_scope = _sa_row_replacement(best_rows, n_rows, k, max_mark,
                                                        min(remaining, 30.0), rnd)

    if _dead():
        return best_rows

    # Phase 6: final tightening
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 4.0
        if remaining > 1.0:
            best_rows = _tighten_polish(best_rows, n_rows, k, min(remaining, 15.0), max_mark)
            best_scope = _scope(best_rows)

    # Final validation
    if best_rows is not None and not _valid(best_rows, n_rows, k):
        # fallback
        rows, _ = _interleaved_greedy(n_rows, k, max_mark, rnd)
        if rows is not None:
            best_rows = rows
            best_scope = _scope(rows)
        else:
            # last resort: pure greedy single seed
            tracker = DiffTracker(max_mark + 1)
            rows = []
            for _ in range(n_rows):
                r = _greedy_row_fast(tracker, k, max_mark)
                if r is None:
                    break
                rows.append(r)
            if len(rows) == n_rows:
                best_rows = rows
                best_scope = _scope(rows)
            else:
                best_rows = None

    return best_rows
