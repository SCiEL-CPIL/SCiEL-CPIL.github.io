"""Rewritten program: exact XOR-circuit synthesis via greedy common-subexpression elimination."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (m x n) where each row is a linear combination of inputs.
        depth_limit: maximum circuit depth (<=16).
    
    Returns:
        dict with 'gates' (list of [wire_a, wire_b]) and 'outputs' (list of wire indices).
    """
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    # Convert to numpy for fast ops
    mat = np.array(matrix, dtype=np.uint8)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    m, n = mat.shape

    # Wires:
    #   0 .. n-1 : inputs
    #   n        : constant zero
    #   n+1 ..    : gate outputs
    # Each wire is represented by a bitmask (Python int) of length n (input bits)
    # We track which wire index corresponds to which mask.

    # Start with all input wires and zero wire
    wire_masks = []       # wire index -> mask (int)
    mask_to_wires = {}    # mask -> list of wire indices (for CSE)
    wire_depth = []       # wire index -> depth

    # Initialize
    for i in range(n):
        mask = 1 << i
        w = i
        wire_masks.append(mask)
        wire_depth.append(0)
        mask_to_wires.setdefault(mask, []).append(w)

    # Constant zero wire
    zero_mask = 0
    zero_w = n
    wire_masks.append(zero_mask)
    wire_depth.append(0)
    mask_to_wires.setdefault(zero_mask, []).append(zero_w)

    # Target masks for each output row
    target_masks = []
    for r in range(m):
        mask = 0
        for c in range(n):
            if mat[r, c] != 0:
                mask |= (1 << c)
        target_masks.append(mask)

    # Greedy CSE: iteratively pick the "best" XOR to compute that maximizes reuse
    # Strategy: repeatedly find a pair of existing wires whose XOR either:
    #   (a) matches a target mask not yet computed, OR
    #   (b) is an intermediate that will help reduce total gate count
    # We use a heuristic: prioritize XORs that directly produce a target,
    # then XORs that produce masks appearing in multiple remaining targets.

    # Track which targets are already satisfied
    target_wires = [None] * m  # target row -> wire index (None if not yet computed)
    for r in range(m):
        tm = target_masks[r]
        if tm in mask_to_wires:
            target_wires[r] = mask_to_wires[tm][0]

    # Count remaining targets
    remaining = [r for r in range(m) if target_wires[r] is None]

    gates = []  # list of [wire_a, wire_b]
    gate_count = 0

    # Precompute: for each remaining target, what's the minimum number of XORs needed
    # (just popcount - 1 as a lower bound, but we'll use greedy)

    # Greedy loop: at each step, find the best XOR to add
    # Score a candidate XOR (a XOR b) by:
    #   - How many remaining targets does it directly satisfy? (high priority)
    #   - How many remaining targets contain this mask as a sub-expression?
    #   - Depth constraint

    max_depth = min(depth_limit, 16)
    current_max_depth = 0

    # To make this efficient, we'll use a simpler but effective strategy:
    # For each remaining target, greedily build it using existing wires,
    # adding new gates as needed, and share intermediates.

    # Better approach: iterative refinement
    # 1. For each target, find the set of input bits needed
    # 2. Group targets by shared subsets
    # 3. Build a tree of shared XORs

    # Let's use a different, more effective greedy:
    # Repeatedly find the XOR of two existing wires that appears in the most
    # remaining target expressions (counting multiplicity).

    while remaining and gate_count < 500:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            break

        if not remaining:
            break

        # Find best candidate XOR
        best_score = -1
        best_pair = None
        best_mask = None

        # Try XOR of each pair of existing wires (limit to recent + inputs for speed)
        num_wires = len(wire_masks)
        if num_wires > 200:
            # Only consider wires involved in remaining targets + recent gates
            candidate_indices = set()
            for r in remaining:
                tm = target_masks[r]
                # Find wires whose mask is a subset of tm
                for w in range(num_wires):
                    wm = wire_masks[w]
                    if (wm & tm) == wm and wm != 0:
                        candidate_indices.add(w)
            # Also add the zero wire
            candidate_indices.add(zero_w)
            # Limit candidates
            candidates = list(candidate_indices)[:150]
        else:
            candidates = list(range(num_wires))

        # For each pair, compute XOR mask and score it
        # Score = number of remaining targets that contain this mask as a submask
        # Plus bonus if it directly equals a target

        for i_idx in range(len(candidates)):
            wi = candidates[i_idx]
            mi = wire_masks[wi]
            if mi == 0:
                continue
            for j_idx in range(i_idx + 1, len(candidates)):
                wj = candidates[j_idx]
                mj = wire_masks[wj]
                if mj == 0:
                    continue
                xm = mi ^ mj
                if xm == 0:
                    continue

                # Check if this mask already exists
                if xm in mask_to_wires:
                    continue

                # Score: how many remaining targets contain xm as a submask?
                score = 0
                direct_hit = 0
                for r in remaining:
                    tm = target_masks[r]
                    if xm == tm:
                        direct_hit += 1
                    elif (xm & tm) == xm:
                        score += 1

                if direct_hit > 0:
                    score += 100 * direct_hit  # big bonus for direct hits

                # Depth check
                d = max(wire_depth[wi], wire_depth[wj]) + 1
                if d > max_depth:
                    continue

                if score > best_score:
                    best_score = score
                    best_pair = (wi, wj)
                    best_mask = xm

        if best_pair is None:
            # No good XOR found; fall back to direct construction for each target
            # This shouldn't happen often, but handle it
            for r in remaining[:]:
                tm = target_masks[r]
                # Build target from inputs directly
                bits = []
                for c in range(n):
                    if tm & (1 << c):
                        bits.append(c)
                if not bits:
                    target_wires[r] = zero_w
                    remaining.remove(r)
                    continue
                if len(bits) == 1:
                    target_wires[r] = bits[0]
                    remaining.remove(r)
                    continue
                # XOR them together
                acc = bits[0]
                for b in bits[1:]:
                    if (wire_masks[acc] ^ wire_masks[b]) in mask_to_wires:
                        acc = mask_to_wires[wire_masks[acc] ^ wire_masks[b]][0]
                    else:
                        new_w = len(wire_masks)
                        wire_masks.append(wire_masks[acc] ^ wire_masks[b])
                        wire_depth.append(max(wire_depth[acc], wire_depth[b]) + 1)
                        mask_to_wires.setdefault(wire_masks[acc] ^ wire_masks[b], []).append(new_w)
                        gates.append([acc, b])
                        acc = new_w
                target_wires[r] = acc
                remaining.remove(r)
            break

        wi, wj = best_pair
        xm = best_mask
        new_w = len(wire_masks)
        wire_masks.append(xm)
        wire_depth.append(max(wire_depth[wi], wire_depth[wj]) + 1)
        mask_to_wires.setdefault(xm, []).append(new_w)
        gates.append([wi, wj])
        gate_count += 1

        # Update targets that are now satisfied
        still_remaining = []
        for r in remaining:
            if target_masks[r] == xm:
                target_wires[r] = new_w
            else:
                still_remaining.append(r)
        remaining = still_remaining

    # If we ran out of time or gates, handle remaining targets directly
    for r in remaining:
        if time.time() - start_time > DEADLINE:
            # Fallback: use baseline (direct XOR of inputs)
            tm = target_masks[r]
            bits = [c for c in range(n) if tm & (1 << c)]
            if not bits:
                target_wires[r] = zero_w
            elif len(bits) == 1:
                target_wires[r] = bits[0]
            else:
                # Build directly, adding gates
                acc = bits[0]
                for b in bits[1:]:
                    xor_mask = wire_masks[acc] ^ wire_masks[b]
                    if xor_mask in mask_to_wires:
                        acc = mask_to_wires[xor_mask][0]
                    else:
                        new_w = len(wire_masks)
                        wire_masks.append(xor_mask)
                        wire_depth.append(max(wire_depth[acc], wire_depth[b]) + 1)
                        mask_to_wires.setdefault(xor_mask, []).append(new_w)
                        gates.append([acc, b])
                        acc = new_w
                target_wires[r] = acc
            remaining.remove(r)

    outputs = target_wires

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
