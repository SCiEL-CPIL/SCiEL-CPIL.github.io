"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

import numpy as np


def synthesize(matrix, depth_limit):
    """
    Exact XOR-circuit synthesis.

    Strategy:
      1. Build the row masks (GF(2) columns).
      2. Try several construction strategies, each using a shared CSE gate
         factory (every unique XOR pair computed once):
         a) Direct balanced XOR trees.
         b) Incremental nearest-neighbour row ordering.
         c) Common-factor extraction (precompute frequent pairs).
         d) Cost-based split with full submask enumeration for small masks
            (globally optimal splits) + balanced-tree fallback for large k.
      3. Refine the best candidate with a local-search gate-merge pass.
      4. Return the candidate with the fewest gates.
    """
    deadline = time.time() + 82  # safety margin under the 90s wall-clock limit

    def _elapsed():
        return time.time() > deadline

    # --- Parse matrix -------------------------------------------------------
    A = np.asarray(matrix)
    if A.size == 0:
        return {'gates': [], 'outputs': []}
    A = A.astype(bool)
    m, n = A.shape  # m outputs, n inputs

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Row masks
    row_masks = []
    for i in range(m):
        mask = 0
        row = A[i]
        for j in range(n):
            if row[j]:
                mask |= (1 << j)
        row_masks.append(mask)

    zero_wire = n
    # Precompute popcount for masks up to 2^n (n <= 16)
    size = 1 << n
    popcount = [0] * size
    for i in range(1, size):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # --- Gate factory with CSE --------------------------------------------
    def _make_factory():
        gates = []
        registry = {}
        next_gate = [0]

        def get_xor_wire(a, b):
            if a == b:
                return zero_wire
            if a > b:
                a, b = b, a
            key = (a, b)
            if key in registry:
                return registry[key]
            wire = n + 1 + next_gate[0]
            gates.append([a, b])
            registry[key] = wire
            next_gate[0] += 1
            return wire

        return get_xor_wire, gates, registry

    # --- Strategy a: direct balanced trees ---------------------------------
    def _xor_tree(indices, get_xor_wire):
        if len(indices) == 1:
            return indices[0]
        mid = len(indices) // 2
        left = _xor_tree(indices[:mid], get_xor_wire)
        right = _xor_tree(indices[mid:], get_xor_wire)
        return get_xor_wire(left, right)

    def _build_direct():
        get_xor_wire, gates, _ = _make_factory()
        outputs = []
        for i in range(m):
            ones = [j for j in range(n) if row_masks[i] & (1 << j)]
            if len(ones) == 0:
                outputs.append(zero_wire)
            elif len(ones) == 1:
                outputs.append(ones[0])
            else:
                outputs.append(_xor_tree(ones, get_xor_wire))
        return {'gates': gates, 'outputs': outputs}

    # --- Strategy b: incremental nearest-neighbour -------------------------
    def _build_incremental():
        remaining = list(range(m))
        ordered = []
        start = min(remaining, key=lambda i: popcount[row_masks[i]])
        ordered.append(start)
        remaining.remove(start)
        while remaining and not _elapsed():
            last = ordered[-1]
            lm = row_masks[last]
            best = min(remaining, key=lambda i: popcount[lm ^ row_masks[i]])
            ordered.append(best)
            remaining.remove(best)

        get_xor_wire, gates, _ = _make_factory()
        outputs = [None] * m
        current_wire = None
        current_set = set()

        for idx in ordered:
            row_mask = row_masks[idx]
            row_set = set(j for j in range(n) if (row_mask >> j) & 1)
            if current_wire is None:
                if len(row_set) == 0:
                    current_wire = zero_wire
                elif len(row_set) == 1:
                    current_wire = next(iter(row_set))
                else:
                    current_wire = _xor_tree(sorted(row_set), get_xor_wire)
                current_set = row_set
            else:
                to_remove = current_set - row_set
                to_add = row_set - current_set
                if to_remove or to_add:
                    change_count = len(to_remove) + len(to_add)
                    if change_count <= 4:
                        new_wire = current_wire
                        for bit in to_remove:
                            new_wire = get_xor_wire(new_wire, bit)
                        for bit in to_add:
                            new_wire = get_xor_wire(new_wire, bit)
                        current_wire = new_wire
                    else:
                        if len(row_set) == 0:
                            current_wire = zero_wire
                        elif len(row_set) == 1:
                            current_wire = next(iter(row_set))
                        else:
                            current_wire = _xor_tree(sorted(row_set), get_xor_wire)
                current_set = row_set
            outputs[idx] = current_wire
        return {'gates': gates, 'outputs': outputs}

    # --- Strategy c: common factors ----------------------------------------
    def _build_common_factors():
        pair_counts = {}
        row_ones = []
        for i in range(m):
            ones = [j for j in range(n) if row_masks[i] & (1 << j)]
            row_ones.append(ones)
            for a in range(len(ones)):
                for b in range(a + 1, len(ones)):
                    x, y = ones[a], ones[b]
                    key = (x, y) if x < y else (y, x)
                    pair_counts[key] = pair_counts.get(key, 0) + 1

        get_xor_wire, gates, registry = _make_factory()
        frequent = sorted(pair_counts.items(), key=lambda x: -x[1])
        for pair, count in frequent:
            if count >= 3:
                get_xor_wire(pair[0], pair[1])

        outputs = []
        for i in range(m):
            ones = row_ones[i]
            if len(ones) == 0:
                outputs.append(zero_wire)
                continue
            if len(ones) == 1:
                outputs.append(ones[0])
                continue
            remaining = list(ones)
            wire = None
            changed = True
            while remaining and changed and not _elapsed():
                changed = False
                for a_idx in range(len(remaining)):
                    for b_idx in range(a_idx + 1, len(remaining)):
                        a, b = remaining[a_idx], remaining[b_idx]
                        key = (a, b) if a < b else (b, a)
                        if key in registry:
                            pw = registry[key]
                            wire = pw if wire is None else get_xor_wire(wire, pw)
                            remaining.pop(b_idx)
                            remaining.pop(a_idx)
                            changed = True
                            break
                    if changed:
                        break
            for elem in remaining:
                wire = elem if wire is None else get_xor_wire(wire, elem)
            if wire is None:
                wire = zero_wire
            outputs.append(wire)
        return {'gates': gates, 'outputs': outputs}

    # --- Strategy d: cost-based split with submask enumeration --------------
    def _build_split():
        get_xor_wire, gates, registry = _make_factory()
        # available masks -> wire
        mask_to_wire = {0: zero_wire}
        for j in range(n):
            mask_to_wire[1 << j] = j

        def build(mask):
            if mask in mask_to_wire:
                return mask_to_wire[mask]
            if mask == 0:
                return zero_wire
            k = popcount[mask]
            if k == 1:
                w = mask.bit_length() - 1
                mask_to_wire[mask] = w
                return w

            # Try full submask enumeration for small k (optimal split)
            best_a = None
            best_cost = float('inf')
            if k <= 12:
                sub = (mask - 1) & mask
                while sub:
                    a = sub
                    b = mask ^ sub
                    if a == b:
                        sub = (sub - 1) & mask
                        continue
                    ca = 0 if a in mask_to_wire else (popcount[a] - 1)
                    cb = 0 if b in mask_to_wire else (popcount[b] - 1)
                    tc = ca + cb
                    if tc < best_cost:
                        best_cost = tc
                        best_a = a
                    sub = (sub - 1) & mask

            if best_a is not None:
                a = best_a
                b = mask ^ best_a
                w_a = build(a)
                w_b = build(b)
                w = get_xor_wire(w_a, w_b)
                mask_to_wire[mask] = w
                return w

            # Fallback: balanced tree
            bits = [j for j in range(n) if mask & (1 << j)]
            w = _xor_tree(bits, get_xor_wire)
            mask_to_wire[mask] = w
            return w

        # Process outputs in increasing complexity
        order = sorted(range(m), key=lambda i: popcount[row_masks[i]])
        outputs = [0] * m
        for i in order:
            if _elapsed():
                # emergency: finish remaining with direct trees
                for r in order[order.index(i):]:
                    ones = [j for j in range(n) if row_masks[r] & (1 << j)]
                    if not ones:
                        outputs[r] = zero_wire
                    elif len(ones) == 1:
                        outputs[r] = ones[0]
                    else:
                        outputs[r] = _xor_tree(ones, get_xor_wire)
                break
            outputs[i] = build(row_masks[i])
        return {'gates': gates, 'outputs': outputs}

    # --- Local-search refinement: merge redundant gates --------------------
    def _refine(circuit):
        """
        Try to reduce gate count by replacing a gate g = a ^ b with an
        existing wire w such that mask(w) == mask(a) ^ mask(b). We do this by
        recomputing each gate's mask and checking if an earlier wire already
        has that mask. If so, redirect the output (and any gate that used g).
        This is a safe, exact simplification.
        """
        gates = circuit['gates']
        outputs = circuit['outputs']
        if not gates:
            return circuit

        # Compute mask of every wire
        wire_mask = {}
        for i in range(n):
            wire_mask[i] = 1 << i
        wire_mask[n] = 0
        for k, (a, b) in enumerate(gates):
            wire_mask[n + 1 + k] = wire_mask.get(a, 0) ^ wire_mask.get(b, 0)

        # For each mask, remember the lowest wire that produces it
        mask_to_lowest = {}
        for w, mk in wire_mask.items():
            if mk not in mask_to_lowest or w < mask_to_lowest[mk]:
                mask_to_lowest[mk] = w

        # Build a map from wire -> list of consumers (gate index, operand pos)
        consumers = {}
        for k, (a, b) in enumerate(gates):
            consumers.setdefault(a, []).append((k, 0))
            consumers.setdefault(b, []).append((k, 1))
        for i, w in enumerate(outputs):
            consumers.setdefault(w, []).append((i, -1))  # -1 marks output

        # Iteratively drop redundant gates
        changed = True
        max_iters = 4
        while changed and not _elapsed():
            changed = False
            new_gates = []
            # wire remap: old wire -> new wire
            remap = {i: i for i in range(n + 1)}
            # We process gates in order; if a gate's mask equals an earlier
            # wire's mask, we skip it and remap its wire to that earlier wire.
            for k, (a, b) in enumerate(gates):
                mk = wire_mask[n + 1 + k]
                # find an earlier wire (index < n+1+k) with same mask
                alt = mask_to_lowest.get(mk)
                if alt is not None and alt != n + 1 + k:
                    # redirect: this gate is redundant
                    remap[n + 1 + k] = alt
                    changed = True
                    continue
                # keep gate, but remap operands
                na = remap.get(a, a)
                nb = remap.get(b, b)
                new_gates.append([na, nb])
            # rebuild outputs with remap
            new_outputs = [remap.get(w, w) for w in outputs]
            # rebuild wire_mask for new gates
            wire_mask = {}
            for i in range(n):
                wire_mask[i] = 1 << i
            wire_mask[n] = 0
            for k, (a, b) in enumerate(new_gates):
                wire_mask[n + 1 + k] = wire_mask.get(a, 0) ^ wire_mask.get(b, 0)
            mask_to_lowest = {}
            for w, mk in wire_mask.items():
                if mk not in mask_to_lowest or w < mask_to_lowest[mk]:
                    mask_to_lowest[mk] = w
            gates = new_gates
            outputs = new_outputs
            if max_iters <= 0:
                break
            max_iters -= 1

        return {'gates': gates, 'outputs': outputs}

    def _valid(circuit):
        # exact verification
        gates = circuit['gates']
        outputs = circuit['outputs']
        wire_mask = {}
        for i in range(n):
            wire_mask[i] = 1 << i
        wire_mask[n] = 0
        for k, (a, b) in enumerate(gates):
            wire_mask[n + 1 + k] = wire_mask.get(a, 0) ^ wire_mask.get(b, 0)
        for i in range(m):
            if wire_mask.get(outputs[i], 0) != row_masks[i]:
                return False
        if depth_limit >= 0:
            depth = {i: 0 for i in range(n + 1)}
            md = 0
            for k, (a, b) in enumerate(gates):
                d = 1 + max(depth.get(a, 0), depth.get(b, 0))
                depth[n + 1 + k] = d
                if d > md:
                    md = d
            for w in outputs:
                if depth.get(w, 0) > md:
                    md = depth.get(w, 0)
            if md > depth_limit:
                return False
        return True

    # --- Run strategies ----------------------------------------------------
    candidates = []
    for builder in (_build_direct, _build_incremental, _build_common_factors, _build_split):
        if _elapsed():
            break
        try:
            c = builder()
            if _valid(c):
                candidates.append(c)
        except Exception:
            pass

    if not candidates:
        # ultimate fallback
        c = _build_direct()
        candidates.append(c)

    best = min(candidates, key=lambda c: len(c['gates']))

    # Refine the best candidate
    if not _elapsed():
        try:
            refined = _refine(best)
            if _valid(refined) and len(refined['gates']) <= len(best['gates']):
                best = refined
        except Exception:
            pass

    return best
# EVOLVE-BLOCK-END
