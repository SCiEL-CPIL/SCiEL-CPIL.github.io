"""XOR-circuit synthesis: greedy subexpression sharing with exact GF(2) verification and gate elimination."""
import time
import numpy as np
from collections import deque

_DEADLINE = 45.0  # safety margin under the 60s per-eval / 90s wall-clock limit


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


def _min_need(A, targets, n):
    """For each target, min number of elements of A that XOR to it (0 for target 0)."""
    size = 1 << n
    A = np.asarray(A, dtype=np.uint16)
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


def _decompose(A, t, n):
    """BFS: list of elements of A that XOR to t (minimal count)."""
    A = list(A)
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


def _eliminate_gates(gates, outputs, n):
    """
    Eliminate redundant gates by checking if a gate's output is already available.
    Returns new_gates, new_outputs.
    """
    if not gates:
        return gates, outputs

    # Compute wire values
    vals = [1 << i for i in range(n)]
    vals.append(0)
    for g in gates:
        a, b = g
        vals.append(vals[a] ^ vals[b])

    new_gates = []
    wire_remap = list(range(n + 1))  # wire_remap[i] = new wire number for old wire i
    new_vals = vals[:n + 1]  # values for inputs and zero

    for gi in range(len(gates)):
        a, b = gates[gi]
        old_wire = n + 1 + gi
        val = vals[old_wire]

        # Check if this value is already available in new_vals
        found_existing = -1
        for wi in range(len(new_vals)):
            if new_vals[wi] == val:
                found_existing = wi
                break

        if found_existing != -1:
            # This gate is redundant; redirect old_wire to found_existing
            wire_remap[old_wire] = found_existing
        else:
            # Keep this gate
            new_a = wire_remap[a]
            new_b = wire_remap[b]
            new_wire = n + 1 + len(new_gates)
            new_gates.append([new_a, new_b])
            new_vals.append(val)
            wire_remap[old_wire] = new_wire

    # Remap outputs
    new_outputs = []
    for o in outputs:
        if o < len(wire_remap):
            new_outputs.append(wire_remap[o])
        else:
            new_outputs.append(n)  # fallback to zero

    return new_gates, new_outputs


def synthesize(matrix, depth_limit):
    t0 = time.time()
    rows, m, n = _rows_to_masks(matrix)
    if m == 0:
        return {'gates': [], 'outputs': []}

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
        # Reuse existing wire if available to minimize gate count
        if c in mask_id:
            return mask_id[c]
        
        wid = next_id[0]
        next_id[0] += 1
        wire_mask[wid] = c
        depth[wid] = max(depth[a], depth[b]) + 1
        mask_id[c] = wid
        gates.append((a, b))
        return wid

    A_ids = list(range(n + 1))
    A_masks = [wire_mask[i] for i in A_ids]
    targets = rows

    def cur_future(need):
        return sum(max(0, need[t] - 1) for t in targets)

    K = 50 if n >= 14 else 80
    max_gates = 250

    # ---- greedy subexpression sharing ----
    while time.time() - t0 < _DEADLINE and (next_id[0] - (n + 1)) < max_gates:
        need = _min_need(A_masks, targets, n)
        if all(need[t] <= 1 for t in targets):
            break
        cur_fut = cur_future(need)

        # build candidate pool
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
                if d <= depth_limit:
                    pool[c] = (A_ids[i], A_ids[j])
        # target-adjacent candidates (very promising for sharing)
        for t in targets:
            for i in range(L):
                c = t ^ A_masks[i]
                if c in mask_id:
                    continue
                d = depth[A_ids[i]] + 1
                if d <= depth_limit:
                    pool[c] = (A_ids[i], c)  # placeholder pair, refined below
        # fix placeholder pairs for target-adjacent: need a real second wire
        # (we only need *some* valid producing pair; recompute properly)
        cand = []
        for c, (a, b) in pool.items():
            # ensure (a,b) is a valid producing pair for c
            if wire_mask.get(a, -1) ^ wire_mask.get(b, -1) != c:
                # find a valid pair
                found = False
                for i in range(L):
                    if A_masks[i] ^ c in mask_id:
                        jid = mask_id[A_masks[i] ^ c]
                        if jid in depth:
                            a, b = A_ids[i], jid
                            found = True
                            break
                if not found:
                    continue
            d = max(depth[a], depth[b]) + 1
            if d > depth_limit:
                continue
            # cheap popcount-proxy score
            sc = 0
            for t in targets:
                wt = bin(t).count('1')
                wt2 = bin(t ^ c).count('1')
                if wt2 < wt:
                    sc += (wt - wt2)
            cand.append((sc, c, a, b))
        if not cand:
            break
        cand.sort(key=lambda x: -x[0])
        cand = cand[:K]

        best = None
        best_cost = None
        for (sc, c, a, b) in cand:
            if time.time() - t0 > _DEADLINE:
                break
            need2 = _min_need(A_masks + [c], targets, n)
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

    # ---- produce any target not yet an available wire (depth-safe) ----
    need = _min_need(A_masks, targets, n)
    for t in targets:
        if need.get(t, 10 ** 9) <= 1:
            continue
        done = False
        elems = _decompose(A_masks, t, n)
        if elems is not None:
            # check resulting depth
            maxd = max(depth[mask_id[e]] for e in elems)
            if (len(elems) - 1) + maxd <= 16:
                cur = None
                for e in elems:
                    eid = mask_id[e]
                    if cur is None:
                        cur = eid
                    else:
                        cur = add_gate(cur, eid)
                mask_id[t] = cur
                done = True
        if not done:
            # fallback: produce from inputs directly (depth <= 15)
            bits = [j for j in range(n) if (t >> j) & 1]
            if not bits:
                mask_id[t] = n
            else:
                cur = bits[0]
                for j in bits[1:]:
                    cur = add_gate(cur, j)
                mask_id[t] = cur

    outputs = [mask_id[t] for t in targets]
    
    result = {'gates': [[a, b] for (a, b) in gates], 'outputs': outputs}
    
    if not _verify(matrix, result):
        result = _baseline(matrix)
    return result
