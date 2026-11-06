"""XOR-circuit synthesis with iterative shared-subexpression extraction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

_DEADLINE = 42.0  # safety margin under 60s per-eval


def _rows_to_masks(matrix):
    M = np.asarray(matrix, dtype=np.uint8)
    m, n = M.shape
    rows = []
    for i in range(m):
        mask = 0
        r = M[i]
        for j in range(n):
            if r[j]:
                mask |= (1 << j)
        rows.append(mask)
    return rows, m, n


def _min_need_count(A_masks, targets, n):
    """For each target, min number of elements of A_masks that XOR to it."""
    size = 1 << n
    A = np.asarray(A_masks, dtype=np.uint16)
    reached = np.zeros(size, dtype=bool)
    result = {}
    remaining = set()
    for t in targets:
        if t == 0:
            result[0] = 0
        elif t not in result:
            remaining.add(t)
    if not remaining:
        return result
    reached[0] = True
    level = np.array([0], dtype=np.uint16)
    d = 0
    while level.size and remaining:
        d += 1
        nxt = (level[:, None] ^ A[None, :]).ravel()
        nxt = np.unique(nxt)
        new = nxt[~reached[nxt]]
        if new.size:
            reached[nxt] = True
            for mm in new:
                if mm in remaining:
                    result[mm] = d
                    remaining.discard(mm)
            level = new
        else:
            break
    for t in remaining:
        result[t] = 10 ** 9
    return result


def _decompose(A_masks, t, n):
    """BFS: list of elements of A_masks that XOR to t (minimal count)."""
    A = list(A_masks)
    parent = {0: None}
    q = deque([0])
    while q:
        m = q.popleft()
        if m == t:
            break
        for a in A:
            nm = m ^ a
            if nm not in parent:
                parent[nm] = (m, a)
                q.append(nm)
    if t not in parent:
        return None
    elems = []
    cur = t
    while parent[cur] is not None:
        prev, a = parent[cur]
        elems.append(a)
        cur = prev
    return elems


def _verify(matrix, result):
    """Exact GF(2) verification."""
    rows, m, n = _rows_to_masks(matrix)
    val = {}
    for j in range(n):
        val[j] = 1 << j
    val[n] = 0
    depth = {j: 0 for j in range(n + 1)}
    wid = n + 1
    for (a, b) in result['gates']:
        if a not in val or b not in val:
            return False
        val[wid] = val[a] ^ val[b]
        depth[wid] = max(depth[a], depth[b]) + 1
        if depth[wid] > 16:
            return False
        wid += 1
    outs = result['outputs']
    if len(outs) != m:
        return False
    for i in range(m):
        if outs[i] not in val or val[outs[i]] != rows[i]:
            return False
    return True


def _baseline(matrix):
    """Simple baseline: build each row from input bits."""
    rows, m, n = _rows_to_masks(matrix)
    gates = []
    outputs = []
    next_id = n + 1
    for t in rows:
        bits = [j for j in range(n) if (t >> j) & 1]
        if not bits:
            outputs.append(n)
            continue
        cur = bits[0]
        for j in bits[1:]:
            gates.append([cur, j])
            cur = next_id
            next_id += 1
        outputs.append(cur)
    return {'gates': gates, 'outputs': outputs}


def synthesize(matrix, depth_limit):
    t0 = time.time()
    rows, m, n = _rows_to_masks(matrix)
    if m == 0:
        return {'gates': [], 'outputs': []}

    # Initialize wires: inputs 0..n-1, zero wire = n
    wire_mask = {}
    mask_id = {}
    depth = {}
    for j in range(n):
        wire_mask[j] = 1 << j
        mask_id[1 << j] = j
        depth[j] = 0
    wire_mask[n] = 0
    mask_id[0] = n
    depth[n] = 0

    gates = []
    next_id = [n + 1]

    def add_gate(a, b):
        c = wire_mask[a] ^ wire_mask[b]
        wid = next_id[0]
        next_id[0] += 1
        wire_mask[wid] = c
        depth[wid] = max(depth[a], depth[b]) + 1
        if c not in mask_id:
            mask_id[c] = wid
        gates.append([a, b])
        return wid

    # Available wires (indices into wire_mask)
    A_ids = list(range(n + 1))
    A_masks = [wire_mask[i] for i in A_ids]
    targets = rows

    def cur_future(need):
        """Total future gate count if we decompose each target independently."""
        return sum(max(0, need.get(t, 10**9) - 1) for t in targets)

    # Iteratively introduce intermediate XORs that maximize sharing
    max_gates = 200
    K = 40 if n >= 14 else 60

    while time.time() - t0 < _DEADLINE and (next_id[0] - (n + 1)) < max_gates:
        need = _min_need_count(A_masks, targets, n)
        if all(need.get(t, 10**9) <= 1 for t in targets):
            break
        cur_fut = cur_future(need)

        # Build candidate pool: all pairwise XORs of available wires
        pool = {}
        L = len(A_ids)
        for i in range(L):
            mi = A_masks[i]
            di = depth[A_ids[i]]
            for j in range(i + 1, L):
                c = mi ^ A_masks[j]
                if c in mask_id:
                    continue
                d = max(di, depth[A_ids[j]]) + 1
                if d <= min(16, depth_limit):
                    pool[c] = (A_ids[i], A_ids[j])

        # Score candidates by how much they reduce total future cost
        cand = []
        for c, (a, b) in pool.items():
            # Quick score: count how many targets have reduced need
            score = 0
            for t in targets:
                wt = bin(t).count('1')
                wt2 = bin(t ^ c).count('1')
                if wt2 < wt:
                    score += (wt - wt2)
            cand.append((score, c, a, b))

        if not cand:
            break
        cand.sort(key=lambda x: -x[0])
        cand = cand[:K]

        best = None
        best_cost = None
        for (sc, c, a, b) in cand:
            if time.time() - t0 > _DEADLINE:
                break
            need2 = _min_need_count(A_masks + [c], targets, n)
            cost = 1 + cur_future(need2)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = (c, a, b)

        if best is None or best_cost >= cur_fut:
            break
        c, a, b = best
        add_gate(a, b)
        A_ids.append(next_id[0] - 1)
        A_masks.append(c)

    # Produce any target not yet an available wire
    need = _min_need_count(A_masks, targets, n)
    for t in targets:
        if need.get(t, 10**9) <= 1:
            continue
        done = False
        elems = _decompose(A_masks, t, n)
        if elems is not None:
            maxd = max(depth[mask_id[e]] for e in elems)
            if (len(elems) - 1) + maxd <= 16:
                cur = None
                for e in elems:
                    eid = mask_id[e]
                    if cur is None:
                        cur = eid
                    else:
                        cur = add_gate(cur, eid)
                if t not in mask_id:
                    mask_id[t] = cur
                done = True
        if not done:
            # Fallback: build from scratch
            bits = [j for j in range(n) if (t >> j) & 1]
            if not bits:
                continue
            cur = bits[0]
            for j in bits[1:]:
                cur = add_gate(cur, j)
            if t not in mask_id:
                mask_id[t] = cur

    # Map outputs
    outputs = []
    for i in range(m):
        target = targets[i]
        if target in mask_id:
            outputs.append(mask_id[target])
        else:
            outputs.append(n)

    result = {'gates': gates, 'outputs': outputs}

    # Verify and fallback if invalid
    if not _verify(matrix, result):
        return _baseline(matrix)

    return result
# EVOLVE-BLOCK-END
