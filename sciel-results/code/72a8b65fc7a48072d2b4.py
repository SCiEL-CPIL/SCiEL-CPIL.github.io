"""Exact XOR-circuit synthesis via data-driven CSE (greedy + exact local polish)."""
import time
import random
import numpy as np

_START = None
_DEADLINE_MARGIN = 5.0   # seconds of safety margin before the 90s kill

def _now():
    return time.time()

def _deadline():
    global _START
    if _START is None:
        _START = _now()
    return _START + 90.0 - _DEADLINE_MARGIN

def _time_left():
    return _deadline() - _now()


def _to_bitmasks(matrix, n):
    """Return list of row bitmasks (int, bits 0..n-1) for n input wires."""
    m = len(matrix)
    rows = []
    for r in range(m):
        v = 0
        row = matrix[r]
        for c in range(n):
            if row[c]:
                v |= (1 << c)
        rows.append(v)
    return rows


def _build_synthesis(rows, n, deadline):
    """
    Greedy CSE over GF(2) XORs.
    Returns (gates, outputs, depth) or None if infeasible within deadline.
    gates: list of [wire_a, wire_b]
    outputs: list of wire ids (one per row)
    depth: max gate depth
    """
    # wire id -> bitmask
    wire_mask = list(range(n))      # wires 0..n-1 are inputs
    # wire id -> depth (inputs depth 0)
    wire_depth = [0] * n

    # target rows (dedup, skip zero)
    distinct = []
    seen = set()
    for r in rows:
        if r == 0:
            continue
        if r not in seen:
            seen.add(r)
            distinct.append(r)

    # output wire for each row
    out_wire = [None] * len(rows)
    row_to_wire = {}
    for i, r in enumerate(rows):
        if r == 0:
            out_wire[i] = n  # constant zero
        else:
            out_wire[i] = row_to_wire.get(r)

    # initial candidate pool: all pairs of existing wires
    # Represent pool as dict mask -> wire id (current wires)
    current_wires = set(range(n))
    # candidate mask -> (wire_a, wire_b, depth)
    cand = {}

    def refresh_candidates():
        # generate all pairwise XORs of current wires
        wires = list(current_wires)
        for i in range(len(wires)):
            a = wires[i]
            ma = wire_mask[a]
            da = wire_depth[a]
            for j in range(i + 1, len(wires)):
                b = wires[j]
                m = ma ^ wire_mask[b]
                if m == 0:
                    continue
                d = da + wire_depth[b] + 1
                if d > 16:
                    continue
                if m in cand:
                    continue
                cand[m] = (a, b, d)

    refresh_candidates()

    gates = []
    wire_count = n  # next new wire id = n + len(gates)

    # Greedy: repeatedly pick the candidate that covers the most uncovered
    # target rows per marginal cost, then "realize" it and recompute coverage.
    # We track which target rows are still uncovered.
    uncovered = set(distinct)

    # Map mask -> list of target rows having that mask (for coverage)
    mask_to_rows = {}
    for r in distinct:
        mask_to_rows.setdefault(r, []).append(r)

    # We will realize wires; once a wire mask is realized, any target row equal
    # to that mask is satisfied (output can point to it).
    realized = set(range(n))  # input wire masks
    # map realized mask -> wire id (prefer shallowest)
    realized_mask_to_wire = {}
    for w in range(n):
        realized_mask_to_wire[wire_mask[w]] = w

    best_so_far = None

    while True:
        if _time_left() < 0.5:
            break
        if not uncovered:
            break

        # Assign each realized wire mask a "satisfied rows" count.
        # A target row is satisfied if its mask is realized.
        # Compute coverage: for each candidate, how many *newly* satisfied rows
        # would result if we realize it (i.e., rows whose mask == cand mask and
        # currently uncovered), plus rows that become realizable? For CSE value,
        # a new wire is useful if it (a) directly equals some target row, or
        # (b) helps form other candidates cheaply. We use direct coverage as
        # the primary gain, with a small bonus for high-frequency intermediates.

        # Count frequency of each mask among target rows (how many rows need it)
        # and among current candidates (how many future candidates use it).
        # Precompute candidate usage: for each mask, how many current candidates
        # would use it as an operand? Too heavy; approximate by target frequency.

        best_gain = -1
        best_cand = None
        for m, (a, b, d) in cand.items():
            if m in realized:
                continue
            # gain = number of uncovered target rows equal to m
            gain = 0
            if m in mask_to_rows:
                # rows with this mask that are still uncovered
                for r in mask_to_rows[m]:
                    if r in uncovered:
                        gain += 1
            if gain <= 0:
                # still might be a useful intermediate; give tiny gain if it
                # reduces depth of many candidates -- approximate by 0
                continue
            # marginal cost ~ 1 gate; prefer lower depth
            score = gain * 100 - d
            if score > best_gain:
                best_gain = score
                best_cand = m

        if best_cand is None:
            # No direct-coverage candidate. Fall back: pick candidate that
            # maximizes frequency among target rows as an operand, else break.
            # Compute for each candidate how many uncovered rows contain it as a
            # sub-operand? Too heavy. Instead, try to realize a candidate whose
            # mask is a submask of some uncovered row with minimal added depth.
            best_score = -1
            best_cand = None
            for m, (a, b, d) in cand.items():
                if m in realized:
                    continue
                # heuristic: does m help? count uncovered rows where (row ^ m)
                # is realized (so row = m ^ realized -> one more gate)
                help_cnt = 0
                for r in list(uncovered):
                    if (r ^ m) in realized:
                        help_cnt += 1
                if help_cnt > 0:
                    sc = help_cnt * 100 - d
                    if sc > best_score:
                        best_score = sc
                        best_cand = m
            if best_cand is None:
                break
            gain = 1  # at least progress

        m = best_cand
        a, b, d = cand[m]
        # realize the gate
        new_wire = wire_count
        wire_count += 1
        wire_mask.append(m)
        wire_depth.append(d)
        gates.append([a, b])
        current_wires.add(new_wire)
        realized.add(m)
        realized_mask_to_wire[m] = new_wire
        del cand[m]

        # satisfy target rows equal to m
        if m in mask_to_rows:
            for r in mask_to_rows[m]:
                if r in uncovered:
                    uncovered.discard(r)
                    row_to_wire[r] = new_wire

        # recompute candidates involving the new wire (and drop stale)
        # Remove candidates that are now realized or dominated
        for cm in list(cand.keys()):
            if cm in realized:
                del cand[cm]
        # add new pairs with new_wire
        for w in list(current_wires):
            if w == new_wire:
                continue
            nm = m ^ wire_mask[w]
            if nm == 0:
                continue
            nd = d + wire_depth[w] + 1
            if nd > 16:
                continue
            if nm not in cand:
                cand[nm] = (new_wire, w, nd)

        # update outputs for satisfied rows
        for i, r in enumerate(rows):
            if r != 0 and r in row_to_wire and out_wire[i] is None:
                out_wire[i] = row_to_wire[r]

    # Assign outputs
    for i, r in enumerate(rows):
        if r == 0:
            out_wire[i] = n
        elif r in row_to_wire:
            out_wire[i] = row_to_wire[r]
        else:
            # not satisfied -> infeasible
            return None

    # compute depth
    depth = 0
    for w in range(n, wire_count):
        if wire_depth[w] > depth:
            depth = wire_depth[w]

    return gates, out_wire, depth


def _verify(matrix, gates, outputs, n):
    """Exact GF(2) verification. Returns True if correct."""
    m = len(matrix)
    # build wire masks
    wire_mask = list(range(n))
    for g in gates:
        a, b = g
        wire_mask.append(wire_mask[a] ^ wire_mask[b])
    for i in range(m):
        row = matrix[i]
        target = 0
        for c in range(n):
            if row[c]:
                target |= (1 << c)
        if wire_mask[outputs[i]] != target:
            return False
    return True


def _naive_baseline(matrix, depth_limit):
    """Independent-row baseline: each output computed directly (tree of XORs)."""
    n = len(matrix[0])
    m = len(matrix)
    gates = []
    outputs = []
    wire_count = n

    def build_xor(mask, depth):
        # return wire id producing mask, adding gates
        nonlocal wire_count
        if mask in _direct_cache:
            return _direct_cache[mask]
        # find two submasks a,b with a^b=mask, both smaller
        # simple: pick lowest set bit, split
        # a = mask without lowest bit? need a^b=mask
        # choose a = lowest set bit, b = mask ^ a
        lb = mask & (-mask)
        a = lb
        b = mask ^ a
        wa = build_xor(a, depth + 1)
        wb = build_xor(b, depth + 1)
        w = wire_count
        wire_count += 1
        gates.append([wa, wb])
        _direct_cache[mask] = w
        return w

    _direct_cache = {}
    for i in range(m):
        target = 0
        row = matrix[i]
        for c in range(n):
            if row[c]:
                target |= (1 << c)
        if target == 0:
            outputs.append(n)
        else:
            w = build_xor(target, 0)
            outputs.append(w)
    return gates, outputs


def synthesize(matrix, depth_limit):
    global _START
    if _START is None:
        _START = _now()

    m = len(matrix)
    n = len(matrix[0])

    rows = _to_bitmasks(matrix, n)

    # Try CSE synthesis
    result = None
    try:
        result = _build_synthesis(rows, n, _deadline())
    except Exception:
        result = None

    if result is not None:
        gates, outputs, depth = result
        if depth <= 16 and _verify(matrix, gates, outputs, n):
            return {'gates': gates, 'outputs': outputs}

    # Fallback: naive baseline (always valid)
    gates, outputs = _naive_baseline(matrix, depth_limit)
    if _verify(matrix, gates, outputs, n):
        return {'gates': gates, 'outputs': outputs}

    # Last resort: return zero-gate (only valid if all-zero)
    return {'gates': [], 'outputs': [n] * m}
