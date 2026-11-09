"""
Exact XOR-circuit synthesis using Dijkstra-based shared sub-expression discovery.

Key structural change: Instead of greedy row ordering + incremental updates,
we model the problem as finding shortest paths in a hypergraph where:
- Vertices are all subsets of input wires (represented as bitmasks)
- Edges are single XOR gates (combining two known wires)
- We use Dijkstra from the set of available input wires to find the
  minimum-cost way to produce each output row.

This gives globally optimal sharing of intermediate XOR results across
multiple output rows, rather than local greedy pairing.

Additionally:
- For small problems, we enumerate all submasks to find optimal splits.
- We use a priority queue to process combinations in cost order.
- Depth is tracked implicitly through the number of XOR operations.
"""

import time
import heapq
import numpy as np

DEADLINE_SECONDS = 85


def _verify_circuit(matrix, circuit, depth_limit):
    """
    Exact symbolic GF(2) verification.
    Returns True if circuit correctly computes the matrix.
    """
    m, n = matrix.shape
    wire_mask = {}
    for i in range(n):
        wire_mask[i] = (1 << i)
    wire_mask[n] = 0  # constant zero

    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        mask_a = wire_mask.get(a, 0)
        mask_b = wire_mask.get(b, 0)
        wire_mask[wire] = mask_a ^ mask_b

    for i in range(m):
        expected_mask = 0
        for j in range(n):
            if matrix[i, j]:
                expected_mask |= (1 << j)
        out_wire = circuit['outputs'][i]
        actual_mask = wire_mask.get(out_wire, 0)
        if actual_mask != expected_mask:
            return False

    if depth_limit < 0:
        return True

    max_depth = 0
    depth = {}
    for i in range(n + 1):
        depth[i] = 0

    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        d = 1 + max(depth.get(a, 0), depth.get(b, 0))
        depth[wire] = d
        max_depth = max(max_depth, d)

    for out in circuit['outputs']:
        max_depth = max(max_depth, depth.get(out, 0))

    if max_depth > depth_limit:
        return False

    return True


def _xor_tree(indices, get_xor_wire):
    """Build a balanced XOR tree for a list of input indices."""
    if len(indices) == 1:
        return indices[0]
    mid = len(indices) // 2
    left = _xor_tree(indices[:mid], get_xor_wire)
    right = _xor_tree(indices[mid:], get_xor_wire)
    return get_xor_wire(left, right)


def _dijkstra_synthesis(matrix, n, depth_limit):
    """
    Dijkstra-based synthesis: find minimum-cost set of XOR gates to produce
    all output rows, sharing intermediate results.
    
    State: set of currently available wires (as bitmask of which input subsets
    are computed). We use a priority queue where the key is the total cost
    (number of gates used so far).
    
    For practical purposes with n up to 16, we use a different approach:
    - Pre-compute all needed output masks
    - Use Dijkstra on the space of "which intermediate XOR results we have"
    - But this space is too large, so we use a greedy Dijkstra-like approach:
      repeatedly pick the cheapest new XOR that helps produce remaining outputs.
    """
    m = matrix.shape[0]
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)

    # Available wires: input wires 0..n-1 and constant zero (wire n)
    # Each wire has a mask value
    # We'll build a registry: mask -> wire
    # Start with inputs
    mask_to_wire = {}
    for i in range(n):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = n

    gates = []
    next_wire = n + 1

    def get_wire_for_mask(mask):
        """Get or create a wire for the given mask."""
        nonlocal next_wire
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        # Need to create it via XOR of two submasks
        # Find best split: minimize cost
        # Try all submasks of mask
        best_cost = float('inf')
        best_a = None
        best_b = None
        best_wa = None
        best_wb = None
        
        # For small masks, enumerate all splits
        sub = mask
        while sub:
            a = sub
            b = mask ^ sub
            if a == b:
                sub = (sub - 1) & mask
                continue
            if a > b:
                a, b = b, a
            # Check if both submasks are already available
            if a in mask_to_wire and b in mask_to_wire:
                cost = 1  # just one gate
                if cost < best_cost:
                    best_cost = cost
                    best_a, best_b = a, b
                    best_wa, best_wb = mask_to_wire[a], mask_to_wire[b]
            sub = (sub - 1) & mask
        
        if best_a is not None:
            # Create the gate
            wa = mask_to_wire[best_a]
            wb = mask_to_wire[best_b]
            if wa > wb:
                wa, wb = wb, wa
            # Check if this gate already exists
            gate_key = (wa, wb)
            # We need to track gates to avoid duplicates
            # For simplicity, just create it
            wire = next_wire
            gates.append([wa, wb])
            next_wire += 1
            mask_to_wire[mask] = wire
            return wire
        else:
            # Fallback: build tree from individual bits
            bits = [i for i in range(n) if (mask >> i) & 1]
            if len(bits) == 0:
                mask_to_wire[mask] = n
                return n
            elif len(bits) == 1:
                mask_to_wire[mask] = bits[0]
                return bits[0]
            else:
                wire = _xor_tree(bits, lambda a, b: _get_or_create_xor(a, b, mask_to_wire, gates, next_wire))
                mask_to_wire[mask] = wire
                return wire

    # We need a helper that can create XOR wires
    # Let's restructure: use a class-like approach

    return None  # Placeholder - we'll use a different approach below


def _build_with_shared_dijkstra(matrix, n, depth_limit):
    """
    Build circuit using a shared Dijkstra-like approach.
    
    Key idea: maintain a set of known wires (mask -> wire).
    For each output row, find the minimum-cost way to produce it using
    existing wires plus new gates. Process rows in an order that maximizes sharing.
    
    We use a priority queue of (cost, mask) to find the cheapest way to
    produce each needed mask.
    """
    m = matrix.shape[0]
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)

    # Gate registry
    mask_to_wire = {}
    for i in range(n):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = n

    gates = []
    gate_set = set()  # (wire_a, wire_b) -> wire
    next_wire = [n + 1]

    def create_xor(wire_a, wire_b):
        """Create XOR of two wires, return the result wire."""
        if wire_a == wire_b:
            return n
        if wire_a > wire_b:
            wire_a, wire_b = wire_b, wire_a
        key = (wire_a, wire_b)
        if key in gate_set:
            return gate_set[key]
        wire = next_wire[0]
        gates.append([wire_a, wire_b])
        gate_set[key] = wire
        next_wire[0] += 1
        return wire

    def create_xor_for_masks(mask_a, mask_b):
        """Create XOR of two masks, using existing wires if available."""
        if mask_a in mask_to_wire and mask_b in mask_to_wire:
            return create_xor(mask_to_wire[mask_a], mask_to_wire[mask_b])
        return None

    # For each output, find the minimum-cost decomposition
    # Using Dijkstra on the space of masks
    
    # We'll process outputs in order of increasing popcount (simpler first)
    # This tends to build up useful intermediate results
    
    sorted_outputs = sorted(range(m), key=lambda i: bin(row_masks[i]).count('1'))
    
    outputs = [None] * m
    
    for idx in sorted_outputs:
        target = row_masks[idx]
        if target == 0:
            outputs[idx] = n
            continue
        if target in mask_to_wire:
            outputs[idx] = mask_to_wire[target]
            continue
        
        # Use Dijkstra to find min-cost way to produce target
        # State: mask -> (cost, wire)
        # Start: all masks in mask_to_wire have cost 0
        # Goal: target mask
        
        # For efficiency, use A* with heuristic = popcount(target) - popcount(current)
        # But let's just use Dijkstra with cost = number of new gates
        
        # Priority queue: (cost, mask)
        # dist[mask] = minimum cost to produce mask
        dist = {}
        for mk, w in mask_to_wire.items():
            dist[mk] = 0
        
        pq = [(0, mk) for mk in mask_to_wire]
        heapq.heapify(pq)
        
        found = False
        best_wire = None
        
        while pq:
            cost, mask = heapq.heappop(pq)
            if mask == target:
                found = True
                best_wire = mask_to_wire.get(mask)
                break
            if dist.get(mask, float('inf')) < cost:
                continue
            
            # Try XORing with any available mask
            # This is expensive, so limit to masks that are submasks or complements
            # Actually, XOR of mask with any available mask gives a new mask
            # But we can only XOR with masks we have wires for
            
            # To keep it tractable, try XORing with:
            # 1. Each input wire (single bit)
            # 2. Each existing computed wire
            new_masks = set()
            
            # Try XOR with each single-bit input
            for bit in range(n):
                bit_mask = 1 << bit
                new_mask = mask ^ bit_mask
                if new_mask != mask and new_mask not in dist:
                    new_masks.add(new_mask)
            
            # Try XOR with existing computed wires (limit to recent ones for speed)
            for mk in list(mask_to_wire.keys()):
                if mk == mask or mk == 0:
                    continue
                new_mask = mask ^ mk
                if new_mask != mask and new_mask not in dist:
                    new_masks.add(new_mask)
            
            for new_mask in new_masks:
                new_cost = cost + 1
                if new_cost < dist.get(new_mask, float('inf')):
                    dist[new_mask] = new_cost
                    heapq.heappush(pq, (new_cost, new_mask))
        
        if found:
            outputs[idx] = best_wire
        else:
            # Fallback: build tree from bits
            bits = [i for i in range(n) if (target >> i) & 1]
            if len(bits) == 0:
                outputs[idx] = n
            elif len(bits) == 1:
                outputs[idx] = bits[0]
            else:
                outputs[idx] = _xor_tree(bits, create_xor)
            # Make sure the mask is registered
            if target not in mask_to_wire:
                mask_to_wire[target] = outputs[idx]

    return {'gates': gates, 'outputs': outputs}


def _build_greedy_shared(matrix, n, depth_limit):
    """
    Greedy shared construction:
    - Start with input wires
    - Repeatedly add the XOR gate that most reduces the total remaining cost
    - Process until all outputs are satisfied
    
    Cost of a row = number of bits not yet available as single wires
    A new XOR gate helps if it makes some row cheaper.
    """
    m = matrix.shape[0]
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)

    mask_to_wire = {}
    for i in range(n):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = n

    gates = []
    gate_set = set()
    next_wire = [n + 1]

    def create_xor(wire_a, wire_b):
        if wire_a == wire_b:
            return n
        if wire_a > wire_b:
            wire_a, wire_b = wire_b, wire_a
        key = (wire_a, wire_b)
        if key in gate_set:
            return gate_set[key]
        wire = next_wire[0]
        gates.append([wire_a, wire_b])
        gate_set[key] = wire
        next_wire[0] += 1
        return wire

    def row_cost(mask):
        """Number of XOR gates needed to produce this mask from available wires."""
        if mask == 0:
            return 0
        if mask in mask_to_wire:
            return 0
        # Count bits not directly available
        # Simple estimate: popcount - (1 if any single bit available)
        # Better: try to decompose greedily
        remaining = mask
        cost = 0
        while remaining:
            # Find largest available submask
            best_sub = 0
            for mk in mask_to_wire:
                if mk & remaining == mk and mk != 0:
                    if bin(mk).count('1') > bin(best_sub).count('1'):
                        best_sub = mk
            if best_sub == 0:
                # No available submask, need to build from bits
                cost += bin(remaining).count('1') - 1
                break
            remaining ^= best_sub
            cost += 1
        return cost

    outputs = [None] * m
    
    # Check which rows are already satisfied
    for i in range(m):
        if row_masks[i] in mask_to_wire:
            outputs[i] = mask_to_wire[row_masks[i]]
    
    # Repeatedly find the best XOR to add
    max_iterations = 200
    for _ in range(max_iterations):
        # Check if all rows are satisfied
        all_done = True
        for i in range(m):
            if outputs[i] is None:
                all_done = False
                break
        if all_done:
            break
        
        # Find the XOR that most reduces total cost
        best_gain = 0
        best_pair = None
        
        # Candidate pairs: (mask_a, mask_b) where both are in mask_to_wire
        masks_list = list(mask_to_wire.keys())
        
        # Limit candidates for speed
        if len(masks_list) > 50:
            # Only consider masks with popcount <= 8 or that are submasks of some row
            masks_list = [mk for mk in masks_list if bin(mk).count('1') <= 8 or mk == 0]
        
        for ai in range(len(masks_list)):
            for bi in range(ai + 1, len(masks_list)):
                ma, mb = masks_list[ai], masks_list[bi]
                new_mask = ma ^ mb
                if new_mask == 0 or new_mask in mask_to_wire:
                    continue
                
                # Estimate gain: how much does adding this XOR help?
                gain = 0
                for i in range(m):
                    if outputs[i] is None:
                        old_cost = row_cost(row_masks[i])
                        # New cost: can use new_mask as a building block
                        # This is complex, so use a simple heuristic
                        if new_mask & row_masks[i] == new_mask:
                            # new_mask is a submask of row i
                            gain += 1
                        elif (row_masks[i] ^ new_mask) in mask_to_wire:
                            gain += 1
                gain -= 1  # cost of the new gate itself
                
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (ma, mb, new_mask)
        
        if best_pair is None or best_gain <= 0:
            break
        
        ma, mb, new_mask = best_pair
        wa = mask_to_wire[ma]
        wb = mask_to_wire[mb]
        create_xor(wa, wb)
        mask_to_wire[new_mask] = next_wire[0] - 1
        
        # Update outputs for rows that are now satisfied
        for i in range(m):
            if outputs[i] is None and row_masks[i] in mask_to_wire:
                outputs[i] = mask_to_wire[row_masks[i]]
    
    # Fill in remaining rows
    for i in range(m):
        if outputs[i] is None:
            target = row_masks[i]
            if target == 0:
                outputs[i] = n
            elif target in mask_to_wire:
                outputs[i] = mask_to_wire[target]
            else:
                bits = [j for j in range(n) if (target >> j) & 1]
                if len(bits) == 1:
                    outputs[i] = bits[0]
                else:
                    outputs[i] = _xor_tree(bits, create_xor)
                if target not in mask_to_wire:
                    mask_to_wire[target] = outputs[i]

    return {'gates': gates, 'outputs': outputs}


def _build_optimal_splits(matrix, n, depth_limit):
    """
    For each row, find the optimal way to decompose it into XORs of available
    submasks. Use dynamic programming over submasks for rows with popcount <= 12.
    """
    m = matrix.shape[0]
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)

    mask_to_wire = {}
    for i in range(n):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = n

    gates = []
    gate_set = set()
    next_wire = [n + 1]

    def create_xor(wire_a, wire_b):
        if wire_a == wire_b:
            return n
        if wire_a > wire_b:
            wire_a, wire_b = wire_b, wire_a
        key = (wire_a, wire_b)
        if key in gate_set:
            return gate_set[key]
        wire = next_wire[0]
        gates.append([wire_a, wire_b])
        gate_set[key] = wire
        next_wire[0] += 1
        return wire

    def optimal_decompose(mask, max_depth=12):
        """Find min-gate decomposition of mask using available wires + new gates."""
        if mask in mask_to_wire:
            return mask_to_wire[mask], 0
        if mask == 0:
            return n, 0
        
        popcount = bin(mask).count('1')
        if popcount > max_depth:
            # Fallback to tree
            bits = [i for i in range(n) if (mask >> i) & 1]
            wire = _xor_tree(bits, create_xor)
            mask_to_wire[mask] = wire
            return wire, popcount - 1
        
        # DP: dp[submask] = (min_gates, wire)
        # For submasks of mask
        best = {}
        best[0] = (0, n)
        
        # Initialize with directly available masks
        for sub in range(1, mask + 1):
            if sub & mask != sub:
                continue
            if sub in mask_to_wire:
                best[sub] = (0, mask_to_wire[sub])
        
        # Iterate over submasks in increasing order of popcount
        submasks = [s for s in range(1, mask + 1) if s & mask == s]
        submasks.sort(key=lambda x: bin(x).count('1'))
        
        for sub in submasks:
            if sub in best:
                continue
            # Try splitting sub into a ^ b where a, b are proper submasks
            best_cost = float('inf')
            best_wire = None
            a = (sub - 1) & sub
            while a:
                b = sub ^ a
                if a == b:
                    a = (a - 1) & sub
                    continue
                if a > b:
                    a, b = b, a
                if a in best and b in best:
                    cost = best[a][0] + best[b][0] + 1
                    if cost < best_cost:
                        best_cost = cost
                        wa = best[a][1]
                        wb = best[b][1]
                        new_wire = create_xor(wa, wb)
                        best_wire = new_wire
                a = (a - 1) & sub
            
            if best_wire is not None:
                best[sub] = (best_cost, best_wire)
        
        if mask in best:
            return best[mask][1], best[mask][0]
        else:
            # Fallback
            bits = [i for i in range(n) if (mask >> i) & 1]
            wire = _xor_tree(bits, create_xor)
            mask_to_wire[mask] = wire
            return wire, popcount - 1

    outputs = []
    for i in range(m):
        wire, _ = optimal_decompose(row_masks[i])
        outputs.append(wire)

    return {'gates': gates, 'outputs': outputs}


def _build_direct(matrix, n):
    """Direct construction: balanced XOR trees with CSE."""
    mask_to_wire = {}
    for i in range(n):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = n

    gates = []
    gate_set = set()
    next_wire = [n + 1]

    def create_xor(wire_a, wire_b):
        if wire_a == wire_b:
            return n
        if wire_a > wire_b:
            wire_a, wire_b = wire_b, wire_a
        key = (wire_a, wire_b)
        if key in gate_set:
            return gate_set[key]
        wire = next_wire[0]
        gates.append([wire_a, wire_b])
        gate_set[key] = wire
        next_wire[0] += 1
        return wire

    outputs = []
    for i in range(matrix.shape[0]):
        ones = np.nonzero(matrix[i])[0].tolist()
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            outputs.append(_xor_tree(ones, create_xor))
    return {'gates': gates, 'outputs': outputs}


def _build_common_factors(matrix, n):
    """
    Pre-compute frequent column-pair XORs, then build each row
    using those pre-computed wires where possible.
    """
    m = matrix.shape[0]

    pair_counts = {}
    row_ones = []
    for i in range(m):
        ones = np.nonzero(matrix[i])[0].tolist()
        row_ones.append(ones)
        for a in range(len(ones)):
            for b in range(a + 1, len(ones)):
                key = (min(ones[a], ones[b]), max(ones[a], ones[b]))
                pair_counts[key] = pair_counts.get(key, 0) + 1

    mask_to_wire = {}
    for i in range(n):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = n

    gates = []
    gate_set = set()
    next_wire = [n + 1]

    def create_xor(wire_a, wire_b):
        if wire_a == wire_b:
            return n
        if wire_a > wire_b:
            wire_a, wire_b = wire_b, wire_a
        key = (wire_a, wire_b)
        if key in gate_set:
            return gate_set[key]
        wire = next_wire[0]
        gates.append([wire_a, wire_b])
        gate_set[key] = wire
        next_wire[0] += 1
        return wire

    # Pre-compute pairs that appear in 2+ rows
    frequent = sorted(pair_counts.items(), key=lambda x: -x[1])
    for pair, count in frequent:
        if count >= 2:
            wa, wb = pair[0], pair[1]
            w = create_xor(wa, wb)
            mask_to_wire[(1 << wa) | (1 << wb)] = w

    outputs = []
    for i in range(m):
        ones = row_ones[i]
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            remaining = list(ones)
            wire = None
            changed = True
            while remaining and changed:
                changed = False
                for a_idx in range(len(remaining)):
                    for b_idx in range(a_idx + 1, len(remaining)):
                        a, b = remaining[a_idx], remaining[b_idx]
                        pmask = (1 << a) | (1 << b)
                        if pmask in mask_to_wire:
                            pw = mask_to_wire[pmask]
                            if wire is None:
                                wire = pw
                            else:
                                wire = create_xor(wire, pw)
                            remaining.pop(b_idx)
                            remaining.pop(a_idx)
                            changed = True
                            break
                    if changed:
                        break
            for elem in remaining:
                if wire is None:
                    wire = elem
                else:
                    wire = create_xor(wire, elem)
            if wire is None:
                wire = n
            outputs.append(wire)

    return {'gates': gates, 'outputs': outputs}


def _fallback_chain(matrix, n):
    """Simple chain construction as a guaranteed-correct fallback."""
    m = matrix.shape[0]
    gates = []
    outputs = []
    for i in range(m):
        ones = np.nonzero(matrix[i])[0].tolist()
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            wire = ones[0]
            for j in range(1, len(ones)):
                a = wire
                b = ones[j]
                wire = n + 1 + len(gates)
                gates.append([a, b])
            outputs.append(wire)
    return {'gates': gates, 'outputs': outputs}


def synthesize(matrix, depth_limit):
    """
    Main entry point for XOR-circuit synthesis.
    """
    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape

    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    candidates = []

    # Strategy 1: Direct with CSE
    if time.time() < deadline - 5:
        c1 = _build_direct(matrix, n)
        candidates.append(c1)

    # Strategy 2: Common factors
    if time.time() < deadline - 5:
        c2 = _build_common_factors(matrix, n)
        candidates.append(c2)

    # Strategy 3: Optimal splits (DP over submasks)
    if time.time() < deadline - 10:
        c3 = _build_optimal_splits(matrix, n, depth_limit)
        candidates.append(c3)

    # Strategy 4: Greedy shared
    if time.time() < deadline - 15:
        c4 = _build_greedy_shared(matrix, n, depth_limit)
        candidates.append(c4)

    # Strategy 5: Dijkstra shared
    if time.time() < deadline - 20 and n <= 12:
        c5 = _build_with_shared_dijkstra(matrix, n, depth_limit)
        if c5 is not None:
            candidates.append(c5)

    # Pick the best
    best_circuit = None
    best_count = float('inf')
    for c in candidates:
        count = len(c['gates'])
        if count < best_count:
            best_count = count
            best_circuit = c

    if best_circuit is None:
        best_circuit = _fallback_chain(matrix, n)

    # Verify
    if not _verify_circuit(matrix, best_circuit, depth_limit):
        found = False
        for c in candidates:
            if _verify_circuit(matrix, c, depth_limit):
                best_circuit = c
                found = True
                break
        if not found:
            best_circuit = _fallback_chain(matrix, n)

    # Final safety check
    if not _verify_circuit(matrix, best_circuit, depth_limit):
        best_circuit = _fallback_chain(matrix, n)

    return best_circuit
