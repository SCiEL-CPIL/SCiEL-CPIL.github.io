import time
import random
import math


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _row_mask(row):
    m = 0
    for i in range(len(row)):
        ri = row[i]
        for j in range(i + 1, len(row)):
            m |= (1 << (row[j] - ri))
    return m


def _row_diffs_set(row):
    s = set()
    for i in range(len(row)):
        ri = row[i]
        for j in range(i + 1, len(row)):
            s.add(row[j] - ri)
    return s


def _construct(n_rows, k):
    start = time.time()
    DEADLINE = start + 140.0
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2.0 * total_needed))

    best_rows = None
    best_scope = float('inf')

    def _try_backtrack(seed, target_scope, max_nodes):
        nonlocal best_rows, best_scope
        rng = random.Random(seed)
        used_mask = 0
        rows = []

        # scarcity heuristic: weight a difference d by how "scarce" it is.
        # small differences are scarcer -> higher cost.
        def scarcity(d):
            # prefer to avoid small d; cost grows as d shrinks
            return 1.0 / (d + 1.0)

        def build_row():
            row = [0]
            # iterative deepening over the max value for this row
            # (bounded by target_scope)
            nodes = [0]

            def rec():
                nonlocal used_mask
                if len(row) == num_elements:
                    return True
                if nodes[0] >= max_nodes:
                    return False
                nodes[0] += 1
                last = row[-1]
                # candidate values: increasing, up to target_scope
                # order by scarcity cost of the new differences (lowest first)
                cands = []
                for v in range(last + 1, target_scope + 1):
                    # quick in-row duplicate check
                    nd = set()
                    ok = True
                    for prev in row:
                        d = v - prev
                        if d in nd:
                            ok = False
                            break
                        nd.add(d)
                    if not ok:
                        continue
                    nm = 0
                    for d in nd:
                        nm |= (1 << d)
                    if nm & used_mask:
                        continue
                    # cost: sum of scarcity of new diffs
                    cost = sum(scarcity(d) for d in nd)
                    cands.append((cost, v, nd, nm))
                cands.sort(key=lambda t: (t[0], t[1]))
                # limit branching for speed
                for cost, v, nd, nm in cands[:60]:
                    row.append(v)
                    old = used_mask
                    used_mask |= nm
                    if rec():
                        return True
                    used_mask = old
                    row.pop()
                return False

            if rec():
                return row
            return None

        for r in range(n_rows):
            if time.time() > DEADLINE - 10:
                return False
            row = build_row()
            if row is None:
                return False
            rows.append(row)
            used_mask |= _row_mask(row)

        # validate
        if _is_valid(rows):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [x[:] for x in rows]
                return True
        return False

    # Phase 1: backtracking construction at progressively tighter scopes
    # start a bit above counting bound and go down
    max_scope_try = min(220, counting_bound + 60)
    low_scope = max(counting_bound - 2, 1)
    seed_i = 0
    # try a handful of target scopes, each with a couple of seeds
    for target in range(max_scope_try, low_scope - 1, -1):
        if time.time() > DEADLINE - 60:
            break
        for seed in range(3):
            if time.time() > DEADLINE - 60:
                break
            # more nodes allowed for tighter (harder) targets
            max_nodes = 200000
            if _try_backtrack(seed * 31 + 5, target, max_nodes):
                if best_scope <= counting_bound + 5:
                    break
        if best_scope <= counting_bound + 5:
            break

    # Fallback: simple sequential construction if nothing found
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                nd = {x - a for a in row}
                if not (nd & used):
                    row.append(x)
                    used |= nd
            rows.append(row)
        best_rows = rows
        best_scope = _scope(best_rows)

    # Phase 2: fast local-search polish (bitmask-based) to shrink scope
    def _polish(rows, time_limit):
        nonlocal best_rows, best_scope
        cur = [r[:] for r in rows]
        cur_scope = _scope(cur)
        bm, rmask = [], []
        for r in cur:
            m = _row_mask(r)
            rmask.append(m)
            bm.append(m)
        gmask = 0
        for m in rmask:
            gmask |= m

        rng = random.Random(1234)
        n_rows_l = len(cur)
        n_elem = num_elements

        def try_move(ri, pos, nv):
            row = cur[ri]
            if pos > 0 and nv <= row[pos - 1]:
                return False, None
            if pos < n_elem - 1 and nv >= row[pos + 1]:
                return False, None
            if nv == row[pos]:
                return False, None
            dset = set()
            for i in range(n_elem):
                for j in range(i + 1, n_elem):
                    d = row[j] - row[i] if not (i == pos or j == pos) else None
            # recompute diffs with the proposed value
            new_row = row[:]
            new_row[pos] = nv
            dset = set()
            for i in range(n_elem):
                for j in range(i + 1, n_elem):
                    d = new_row[j] - new_row[i]
                    if d in dset:
                        return False, None
                    dset.add(d)
            nm = 0
            for d in dset:
                nm |= (1 << d)
            other = gmask & ~rmask[ri]
            if nm & other:
                return False, None
            return True, nm

        # greedy downward polish: repeatedly try to pull elements down
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for ri in range(n_rows_l):
                for pos in range(n_elem - 1, 0, -1):
                    row = cur[ri]
                    old = row[pos]
                    low = row[pos - 1] + 1
                    # try values below old
                    for nv in range(old - 1, low - 1, -1):
                        ok, nm = try_move(ri, pos, nv)
                        if ok:
                            row[pos] = nv
                            rmask[ri] = nm
                            gmask = (gmask & ~rmask[ri]) | nm
                            improved = True
                            break

        # random local search for further reductions
        t_end = time_limit
        temp = 5.0
        while time.time() < t_end:
            ri = rng.randint(0, n_rows_l - 1)
            pos = rng.randint(1, n_elem - 1)
            row = cur[ri]
            old = row[pos]
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < n_elem - 1 else cur_scope + 5
            if low >= high:
                continue
            if old > low + 1 and rng.random() < 0.75:
                hi = min(old - 1, high)
                if low > hi:
                    continue
                nv = rng.randint(low, hi)
            else:
                nv = rng.randint(low, high)
            if nv == old:
                continue
            ok, nm = try_move(ri, pos, nv)
            if not ok:
                continue
            old_rm = rmask[ri]
            row[pos] = nv
            rmask[ri] = nm
            gmask = (gmask & ~old_rm) | nm
            new_scope = _scope(cur)
            delta = new_scope - cur_scope
            if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 0.1)):
                cur_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [x[:] for x in cur]
            else:
                row[pos] = old
                rmask[ri] = old_rm
                gmask = (gmask & ~nm) | old_rm
            temp = max(0.5, temp * 0.9998)

        return best_rows, best_scope

    if time.time() < DEADLINE - 20:
        _polish(best_rows, DEADLINE - 15)

    # Final validation
    if best_rows is not None and _is_valid(best_rows):
        return best_rows

    # ultimate fallback
    rows = []
    used = set()
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_elements:
            x += 1
            nd = {x - a for a in row}
            if not (nd & used):
                row.append(x)
                used |= nd
        rows.append(row)
    return rows


def construct(n_rows, k):
    return _construct(n_rows, k)
