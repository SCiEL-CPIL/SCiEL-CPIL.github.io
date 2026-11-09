"""XOR-circuit synthesis: global cost-lookahead greedy + witness cleanup + restarts."""
import time


def _row_bitsets(matrix, m, n):
    rows = []
    for i in range(m):
        bits = 0
        row = matrix[i]
        for j in range(n):
            if row[j]:
                bits |= (1 << j)
        rows.append(bits)
    return rows


def _baseline_build(rows, m, n):
    """Independent per-row balanced build (always valid, depth <= ceil(log2(popcount)))."""
    wires = [1 << j for j in range(n)] + [0]
    parents = [None] * (n + 1)
    available = {wires[w]: w for w in range(n + 1)}

    def add_gate(a, b):
        c = wires[a] ^ wires[b]
        w = len(wires)
        wires.append(c)
        parents.append((a, b))
        available[c] = w
        return w

    outputs = []
    for bits in rows:
        if bits in available:
            outputs.append(available[bits])
            continue
        idxs = []
        x = bits
        j = 0
        while x:
            if x & 1:
                idxs.append(j)
            x >>= 1
            j += 1
        if not idxs:
            outputs.append(n)
            continue
        level = idxs[:]
        while len(level) > 1:
            nxt = []
            for k in range(0, len(level), 2):
                if k + 1 < len(level):
                    nxt.append(add_gate(level[k], level[k + 1]))
                else:
                    nxt.append(level[k])
            level = nxt
        outputs.append(level[0])
    gates = []
    for w in range(n + 1, len(wires)):
        a, b = parents[w]
        gates.append([a, b])
    return gates, outputs


def _max_depth(gates, n):
    depth = [0] * (n + 1 + len(gates))
    for k in range(len(gates)):
        a, b = gates[k]
        w = n + 1 + k
        depth[w] = 1 + max(depth[a], depth[b])
    return max(depth) if depth else 0


def _build_once(rows, m, n, max_pc, rng=None):
    wires = [1 << j for j in range(n)] + [0]
    parents = [None] * (n + 1)
    available = {wires[w]: w for w in range(n + 1)}
    target_pc = [t.bit_count() for t in rows]

    def add_gate(a, b):
        c = wires[a] ^ wires[b]
        w = len(wires)
        wires.append(c)
        parents.append((a, b))
        available[c] = w
        return w

    # Greedy global cost-lookahead build
    while True:
        unsat = [i for i in range(m) if rows[i] not in available]
        if not unsat:
            break
        best_c = None
        best_score = -1.0
        best_pair = None
        keys = list(available.keys())
        L = len(keys)
        for ia in range(L):
            a = keys[ia]
            wa = available[a]
            for ib in range(ia, L):
                b = keys[ib]
                c = a ^ b
                if c == 0 or c in available:
                    continue
                pc_c = c.bit_count()
                if pc_c > max_pc:
                    continue
                wb = available[b]
                score = 0.0
                for i in unsat:
                    t = rows[i]
                    if t == c:
                        score += 1000.0 + target_pc[i]
                        continue
                    tc = t ^ c
                    if tc in available:
                        after = 1.0
                    else:
                        after = 1.0 + max(0, tc.bit_count() - 1)
                    before = max(0, target_pc[i] - 1)
                    d = before - after
                    if d > 0:
                        score += d
                if rng is not None:
                    score += rng.random() * 0.05
                if score > best_score:
                    best_score = score
                    best_c = c
                    best_pair = (wa, wb)
        if best_c is None or best_score <= 0.0:
            # fallback: build remaining rows directly
            for i in unsat:
                t = rows[i]
                if t in available:
                    continue
                idxs = []
                x = t
                j = 0
                while x:
                    if x & 1:
                        idxs.append(j)
                    x >>= 1
                    j += 1
                if not idxs:
                    continue
                level = idxs[:]
                while len(level) > 1:
                    nxt = []
                    for k in range(0, len(level), 2):
                        if k + 1 < len(level):
                            nxt.append(add_gate(level[k], level[k + 1]))
                        else:
                            nxt.append(level[k])
                    level = nxt
            break
        add_gate(best_pair[0], best_pair[1])

    # Witness assignment: pick per target the wire minimizing parent-closure size
    N = len(wires)
    closure_size = [0] * N
    closure_set = [None] * N

    def closure(w):
        if closure_set[w] is not None:
            return closure_set[w]
        seen = set()
        stack = [w]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            p = parents[u]
            if p is not None:
                stack.append(p[0])
                stack.append(p[1])
        closure_set[w] = seen
        closure_size[w] = len(seen)
        return seen

    # precompute closures for all wires (N small)
    for w in range(N):
        closure(w)

    witness = [0] * m
    for i in range(m):
        t = rows[i]
        cands = [w for w in range(N) if wires[w] == t]
        if not cands:
            # should not happen; build it
            cands = [available.get(t, n)]
        witness[i] = min(cands, key=lambda w: closure_size[w])

    needed = set()
    for i in range(m):
        needed |= closure(witness[i])

    kept = sorted(w for w in range(n + 1, N) if w in needed)
    old_to_new = {w: w for w in range(n + 1)}
    for new_idx, old in enumerate(kept):
        old_to_new[old] = n + 1 + new_idx
    gates = []
    for old in kept:
        a, b = parents[old]
        gates.append([old_to_new[a], old_to_new[b]])
    outputs = [old_to_new[witness[i]] for i in range(m)]
    return gates, outputs


def synthesize(matrix, depth_limit):
    t_start = time.time()
    deadline = t_start + 82.0  # safety margin under 90s

    try:
        import numpy as _np
        is_np = isinstance(matrix, _np.ndarray)
    except Exception:
        is_np = False

    if is_np:
        m = int(matrix.shape[0])
        n = int(matrix.shape[1])
    else:
        m = len(matrix)
        n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}
    if n == 0:
        return {'gates': [], 'outputs': [0] * m}

    rows = _row_bitsets(matrix, m, n)
    max_pc = max((t.bit_count() for t in rows), default=0)
    cap = min(depth_limit if depth_limit else 16, 16)

    # Deterministic first build
    best = None
    try:
        g, o = _build_once(rows, m, n, max_pc, rng=None)
        if _max_depth(g, n) <= cap:
            best = (len(g), g, o)
    except Exception:
        best = None

    # Randomized restarts to try to improve
    import random
    restart = 0
    while time.time() < deadline - 3.0:
        restart += 1
        try:
            rng = random.Random(1000 + restart)
            g, o = _build_once(rows, m, n, max_pc, rng=rng)
            if _max_depth(g, n) <= cap and (best is None or len(g) < best[0]):
                best = (len(g), g, o)
        except Exception:
            continue
        if time.time() > deadline - 6.0:
            break

    if best is None:
        g, o = _baseline_build(rows, m, n)
        return {'gates': g, 'outputs': o}

    return {'gates': best[1], 'outputs': best[2]}
