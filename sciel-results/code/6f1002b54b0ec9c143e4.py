"""Rewritten program: global shared-subexpression XOR-circuit synthesis.

Strategy (structural change from the seed baseline):
  - Represent every wire / intermediate as an integer bitmask over the n input
    wires (GF(2) vectors). The constant-zero wire is the empty mask.
  - The targets are the row-vectors of the matrix (each output column).
  - We grow a shared set of available vectors starting from the inputs (and 0).
    At each step we consider, for every pair of already-available vectors, the
    XOR combination they produce, and we pick the combination that is (a) new
    and (b) maximally useful to the remaining targets per unit of cost.
    Cost of adding a new vector v is 1 gate; its benefit is the reduction in
    total remaining "distance" to the targets. This is a greedy arborescence /
    Steiner-tree heuristic on the hypergraph of linear combinations.
  - We stop once every target is available. The set of added vectors is the
    circuit; depth is tracked by the number of XOR levels each wire needs,
    and we keep it within the limit by preferring low-depth parents.
  - A deadline guard returns the best feasible result found before the wall
    clock budget is exhausted.
"""

import time
import heapq
import numpy as np

_DEADLINE_SECONDS = 80.0  # safety margin below the 90s kill
_START = None


def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _mat_to_masks(matrix):
    """Convert a binary matrix (m x n) into a list of integer masks, one per row.
    Row i mask has bit j set iff matrix[i, j] == 1.
    Returns (masks, n)."""
    mat = np.asarray(matrix)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    m, n = mat.shape
    masks = []
    for i in range(m):
        v = 0
        row = mat[i]
        for j in range(n):
            if row[j]:
                v |= (1 << j)
        masks.append(v)
    return masks, n


def _baseline_row_gates(masks):
    """Independent-row baseline: sum of popcounts of each row mask."""
    return sum(bin(v).count('1') for v in masks)


def _build_shared_circuit(masks, n, depth_limit):
    """Global shared-subexpression synthesis.

    Returns (gates, outputs) where gates is a list of [wire_a, wire_b] and
    outputs is a list of wire indices, or None on failure.
    Wire indices: inputs 0..n-1, zero = n, gate k -> n+1+k.
    """
    targets = list(masks)
    zero_wire = n
    # available: mask -> wire index
    available = {}
    for j in range(n):
        available[1 << j] = j
    available[0] = zero_wire

    # depth[mask] = minimum XOR-depth (levels) to produce this mask
    depth = {}
    for j in range(n):
        depth[1 << j] = 0
    depth[0] = 0

    gates = []          # list of (wire_a, wire_b)
    mask_to_wire = dict(available)

    # Remaining targets that are not yet available
    remaining = [v for v in targets if v not in available]

    # If any target is zero, it maps to zero_wire.
    # We'll resolve outputs at the end.

    max_depth = depth_limit if depth_limit is not None else 16
    max_depth = max(1, int(max_depth))

    # We will greedily add vectors. To measure benefit, precompute for each
    # remaining target a "cost-to-build" estimate = popcount (number of inputs
    # it directly needs). Adding a vector v reduces a target's remaining cost
    # by 1 if v is a sub-vector (v & target == v) ... actually in XOR terms the
    # useful relation is: if we have v, target can be built as v XOR (target^v),
    # and popcount(target^v) <= popcount(target) typically helps.
    #
    # Benefit heuristic: for a candidate new vector v, its benefit is the number
    # of remaining targets t for which popcount(t ^ v) < popcount(t) AND v is
    # "aligned" (v & t) != 0. We add the candidate with max benefit; ties broken
    # by lower depth and lower popcount.

    # Candidate generation: for efficiency, we only combine each new vector with
    # the inputs and with recently added vectors. We maintain a frontier.
    # Keep list of currently available masks (excluding 0).
    avail_masks = list(available.keys())

    added_count = 0
    # To avoid O(m^2) blowup, cap the number of gate additions.
    total_targets = len(remaining)
    gate_budget = max(total_targets * 4, 64)

    while remaining and added_count < gate_budget:
        if _now() > _DEADLINE_SECONDS - 5.0:
            break
        best_candidate = None
        best_benefit = -1
        best_depth = None
        best_pop = None
        best_pair = None

        # Combine each available mask with each input to form candidates.
        # This is O(available * n). We also combine the most recently added
        # vectors pairwise in a limited way.
        # For each candidate v = a ^ b (a,b available, a!=b, v not available):
        #   depth_v = max(depth[a], depth[b]) + 1
        #   if depth_v > max_depth: skip
        #   benefit = count of remaining targets t with popcount(t^v) < popcount(t)
        #             and (v & t) != 0

        # Precompute popcount for remaining targets
        # We'll compute benefit incrementally.

        # To keep this tractable, iterate over available masks (cap the list).
        am = avail_masks
        # Limit am size for speed: use inputs + recently added (last ~64)
        if len(am) > 256:
            am = am[-256:]

        # Build candidate set
        seen_cand = set()
        # Combine within am
        L = len(am)
        for i in range(L):
            a = am[i]
            da = depth.get(a, 0)
            if da + 1 > max_depth:
                continue
            for j in range(i + 1, L):
                b = am[j]
                db = depth.get(b, 0)
                dv = da if da >= db else db
                dv += 1
                if dv > max_depth:
                    continue
                v = a ^ b
                if v == 0:
                    continue
                if v in available:
                    continue
                if v in seen_cand:
                    continue
                seen_cand.add(v)
                # benefit
                benefit = 0
                for t in remaining:
                    if (v & t) == 0:
                        continue
                    if bin(t ^ v).count('1') < bin(t).count('1'):
                        benefit += 1
                if benefit <= 0:
                    continue
                # pick max benefit, then min depth, then min popcount
                pop = bin(v).count('1')
                if (benefit > best_benefit or
                    (benefit == best_benefit and (best_depth is None or dv < best_depth)) or
                    (benefit == best_benefit and best_depth == dv and (best_pop is None or pop < best_pop))):
                    best_benefit = benefit
                    best_depth = dv
                    best_pop = pop
                    best_candidate = v
                    best_pair = (a, b)

        if best_candidate is None:
            # No beneficial combination found; fall back to building each
            # remaining target directly from inputs (independent synthesis).
            break

        a, b = best_pair
        wa = mask_to_wire[a]
        wb = mask_to_wire[b]
        new_wire = n + 1 + added_count
        gates.append([wa, wb])
        available[best_candidate] = new_wire
        mask_to_wire[best_candidate] = new_wire
        depth[best_candidate] = best_depth
        avail_masks.append(best_candidate)
        added_count += 1

        # Remove any remaining targets that are now available
        still = []
        for t in remaining:
            if t in available:
                continue
            still.append(t)
        remaining = still

    # Resolve any still-remaining targets by independent row synthesis.
    # For each remaining target, build it from inputs (or from available vectors).
    # Use a simple method: express target as XOR of a subset of available vectors.
    # Greedy: repeatedly pick an available vector that reduces popcount.
    extra_gates = []
    for t in remaining:
        cur = t
        used = []
        # limit iterations
        for _ in range(64):
            if cur == 0:
                break
            if cur in available:
                used.append(available[cur])
                break
            # find available v maximizing popcount reduction
            best_v = None
            best_red = 0
            cur_pop = bin(cur).count('1')
            for v, w in available.items():
                if v == 0:
                    continue
                if (v & cur) == 0:
                    continue
                red = cur_pop - bin(cur ^ v).count('1')
                if red > best_red:
                    best_red = red
                    best_v = (v, w)
            if best_v is None:
                # fall back to direct inputs
                break
            v, w = best_v
            used.append(w)
            cur = cur ^ v
        # Now 'cur' may be nonzero; build from inputs directly
        # Combine used wires plus input wires for cur
        wires = list(used)
        # add inputs for cur
        for j in range(n):
            if (cur >> j) & 1:
                wires.append(j)
        # XOR all wires together sequentially
        if len(wires) == 0:
            out_wire = zero_wire
        elif len(wires) == 1:
            out_wire = wires[0]
        else:
            acc = wires[0]
            for w in wires[1:]:
                if acc == w:
                    # XOR of same wire = 0
                    acc = zero_wire
                else:
                    new_wire = n + 1 + added_count
                    gates.append([acc, w])
                    added_count += 1
                    acc = new_wire
            out_wire = acc
        extra_gates.append(out_wire)

    # Build outputs list in original target order
    outputs = []
    for t in targets:
        if t in available:
            outputs.append(available[t])
        else:
            # shouldn't happen; map to zero
            outputs.append(zero_wire)

    # Convert gates to list of lists [a, b]
    gates_out = [[g[0], g[1]] for g in gates]
    return gates_out, outputs


def _verify(masks, n, gates, outputs, depth_limit):
    """Symbolic GF(2) verification. Returns True if circuit is correct and depth ok."""
    # wire -> mask
    wire_mask = {}
    for j in range(n):
        wire_mask[j] = 1 << j
    wire_mask[n] = 0
    for k, g in enumerate(gates):
        w = n + 1 + k
        a = g[0]
        b = g[1]
        if a not in wire_mask or b not in wire_mask:
            return False
        wire_mask[w] = wire_mask[a] ^ wire_mask[b]
    for t, ow in zip(masks, outputs):
        if ow not in wire_mask:
            return False
        if wire_mask[ow] != t:
            return False
    # depth check
    if depth_limit is not None:
        dep = {}
        for j in range(n):
            dep[j] = 0
        dep[n] = 0
        for k, g in enumerate(gates):
            w = n + 1 + k
            a = g[0]
            b = g[1]
            da = dep.get(a, 0)
            db = dep.get(b, 0)
            dep[w] = (da if da >= db else db) + 1
            if dep[w] > depth_limit:
                return False
    return True


def _independent_synthesis(masks, n, depth_limit):
    """Fallback: build each row independently from inputs. Always valid."""
    zero_wire = n
    gates = []
    outputs = []
    added = 0
    for t in masks:
        wires = []
        for j in range(n):
            if (t >> j) & 1:
                wires.append(j)
        if len(wires) == 0:
            outputs.append(zero_wire)
        elif len(wires) == 1:
            outputs.append(wires[0])
        else:
            acc = wires[0]
            for w in wires[1:]:
                new_wire = n + 1 + added
                gates.append([acc, w])
                added += 1
                acc = new_wire
            outputs.append(acc)
    return gates, outputs


def synthesize(matrix, depth_limit):
    global _START
    _START = None  # reset clock per call
    masks, n = _mat_to_masks(matrix)

    # Trivial: no targets
    if len(masks) == 0:
        return {'gates': [], 'outputs': []}

    # Independent baseline (always valid, depth = max popcount - 1)
    base_gates, base_outputs = _independent_synthesis(masks, n, depth_limit)
    best = (base_gates, base_outputs)

    # Try the shared synthesis if time permits
    if _now() < _DEADLINE_SECONDS - 10.0:
        try:
            g, o = _build_shared_circuit(masks, n, depth_limit)
            if _verify(masks, n, g, o, depth_limit):
                # Compare gate counts; keep the better (fewer gates)
                if len(g) <= len(base_gates):
                    best = (g, o)
        except Exception:
            pass

    # If best is not valid (shouldn't be), fall back
    if not _verify(masks, n, best[0], best[1], depth_limit):
        best = (base_gates, base_outputs)

    # Deadline guard: ensure we return before the limit
    return {'gates': best[0], 'outputs': best[1]}
