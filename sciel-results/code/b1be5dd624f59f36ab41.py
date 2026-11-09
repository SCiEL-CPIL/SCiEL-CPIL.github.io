"""
Exact XOR-circuit synthesis using a shared-intermediate approach.

The key insight is that XOR-circuits can share intermediate computations.
Instead of building independent XOR trees for each output, we build a
circuit where intermediate XORs are computed once and reused across
multiple outputs.

Strategy:
1. Represent each row as a bitmask.
2. Use a greedy approach to build a set of intermediate XOR wires that
   cover all outputs with minimal total gate count.
3. The core idea: repeatedly find pairs of "current expressions" (which
   can be input wires or previously computed intermediate wires) whose
   XOR is a useful sub-expression for remaining outputs.
4. Use dynamic programming / memoization to track which sets of inputs
   each wire represents.

A more effective approach:
- Build a circuit where we compute intermediate XORs that are common
  sub-expressions across multiple rows.
- Use a bottom-up approach: start with input wires, and repeatedly
  create new XOR gates that reduce the total cost of expressing all
  remaining outputs.

The most effective strategy for small matrices (8x8, 12x16, 16x16):
- Use a greedy "most beneficial gate" heuristic:
  At each step, consider all pairs of existing wires (inputs + computed).
  For each pair (a, b), compute the new wire w = a XOR b.
  Evaluate how much this new wire reduces the total cost of expressing
  all outputs.
  Pick the gate that gives the best reduction.
  Repeat until all outputs can be expressed with just input wires and
  the computed wires.

However, this can be expensive. For our small sizes, a simpler but
effective approach is:
- For each row, decompose it into a sum of "basis" XORs.
- Find a good set of intermediate XORs by looking at column frequencies.

Let me use a different, more practical approach:
- Build a circuit using a "pebbling" or "straight-line program" approach.
- Key optimization: identify frequently occurring column pairs and
  pre-compute their XOR.
- Then, for each row, greedily combine pre-computed XORs and individual
  inputs to form the row.

Actually, the best approach for these small sizes is a greedy set-cover
like algorithm:
1. Start with all input wires available.
2. Repeatedly add the gate that most reduces the total "cost" to express
   all rows, where cost is the number of additional gates needed.
3. Stop when all rows can be expressed.

Let me implement a clean, effective version.
"""

import time
import numpy as np

DEADLINE_SECONDS = 80


def _verify_circuit(matrix, circuit, depth_limit):
    """
    Exact symbolic GF(2) verification.
    """
    m, n = matrix.shape
    
    wire_mask = {}
    for i in range(n):
        wire_mask[i] = (1 << i)
    wire_mask[n] = 0
    
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


def _build_greedy_circuit(matrix, n, depth_limit=16, time_budget=30.0):
    """
    Greedy circuit construction.
    
    We maintain a set of available wires (initially inputs + zero).
    At each step, we try adding a new XOR gate and see if it reduces
    the total cost to express all rows.
    
    Cost of a row = minimum number of gates needed to build it from
    available wires, assuming we can use any available wire directly.
    
    Actually, a simpler model: 
    - Each wire has a "cost" of 1 (it's one gate output, or 0 for inputs).
    - To express a row, we need to XOR together some subset of available wires.
    - The cost is the number of XORs needed = (number of wires used) - 1.
    
    We want to minimize total gates. A gate is useful if it creates a wire
    that is a subset of some row's support, allowing that row to be expressed
    with fewer individual input XORs.
    
    Let me use a cleaner formulation:
    - State: set of available wires, each with a bitmask.
    - For each row, the "direct cost" is popcount(row) - 1 (XOR all inputs).
    - If we have intermediate wires, a row can be expressed as XOR of some
      subset of available wires whose XOR equals the row mask.
    - The cost is (number of wires in subset) - 1.
    
    This is a set cover problem. For small sizes, we can do a greedy approach:
    - Repeatedly find the gate (a, b) -> new_wire that most reduces total cost.
    """
    start_time = time.time()
    m = matrix.shape[0]
    
    # Row masks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)
    
    # Available wires: dict wire_id -> mask
    # Wires 0..n-1 are inputs, wire n is zero
    available = {}
    for i in range(n):
        available[i] = (1 << i)
    available[n] = 0
    
    gates = []
    next_wire = n + 1
    
    # For each row, track the best expression (list of wire IDs)
    # Initially, each row is expressed by its individual input bits
    row_expr = []
    for i in range(m):
        mask = row_masks[i]
        if mask == 0:
            row_expr.append([n])  # just zero
        else:
            expr = []
            for j in range(n):
                if (mask >> j) & 1:
                    expr.append(j)
            row_expr.append(expr)
    
    def total_cost():
        cost = 0
        for expr in row_expr:
            if len(expr) > 1:
                cost += len(expr) - 1
        return cost
    
    def try_update_row(row_idx, new_wire_id, new_mask):
        """Try to improve row_expr[row_idx] using new_wire_id."""
        current_expr = row_expr[row_idx]
        current_mask = row_masks[row_idx]
        
        # Try: replace some subset of current_expr with new_wire_id
        # The new wire can replace any subset whose XOR equals new_mask
        # i.e., XOR of that subset = new_mask
        
        # Check if new_wire alone can be part of a better expression
        # New expression = (some subset of current_expr) XOR [new_wire_id]
        # where XOR of (subset) XOR new_mask = current_mask
        # i.e., XOR of subset = current_mask XOR new_mask
        
        target = current_mask ^ new_mask
        
        # Find minimum subset of current_expr whose XOR = target
        # For small expressions, brute force
        expr_list = current_expr
        k = len(expr_list)
        
        if k <= 20:
            # Brute force all subsets
            best_size = k + 1  # worst case: use all + new wire
            best_subset = None
            
            for subset_mask in range(1 << k):
                xor_val = 0
                cnt = 0
                for idx in range(k):
                    if (subset_mask >> idx) & 1:
                        xor_val ^= available[expr_list[idx]]
                        cnt += 1
                if xor_val == target:
                    total_wires = cnt + 1  # subset + new_wire
                    if total_wires < best_size:
                        best_size = total_wires
                        best_subset = subset_mask
            
            if best_subset is not None and best_size < k:
                # Build new expression
                new_expr = []
                for idx in range(k):
                    if (best_subset >> idx) & 1:
                        new_expr.append(expr_list[idx])
                new_expr.append(new_wire_id)
                # Verify
                check = 0
                for w in new_expr:
                    check ^= available[w]
                if check == current_mask:
                    row_expr[row_idx] = new_expr
                    return True
        
        return False
    
    # Greedy: repeatedly add the best gate
    max_gates = m * n  # safety upper bound
    improvement = True
    
    while improvement and len(gates) < max_gates:
        if time.time() - start_time > time_budget:
            break
        
        improvement = False
        best_gain = 0
        best_pair = None
        best_new_mask = 0
        
        current_total = total_cost()
        
        # Try all pairs of available wires
        wire_ids = list(available.keys())
        num_wires = len(wire_ids)
        
        for i in range(num_wires):
            a = wire_ids[i]
            mask_a = available[a]
            if mask_a == 0:
                continue
            for j in range(i + 1, num_wires):
                b = wire_ids[j]
                mask_b = available[b]
                if mask_b == 0:
                    continue
                
                new_mask = mask_a ^ mask_b
                if new_mask == 0:
                    continue
                
                # Check if this wire is already available
                already_exists = False
                for wid, wmask in available.items():
                    if wmask == new_mask:
                        already_exists = True
                        break
                if already_exists:
                    continue
                
                # Evaluate gain: how much does this reduce total cost?
                # For each row, check if using this new wire helps
                gain = 0
                for row_idx in range(m):
                    if len(row_expr[row_idx]) <= 1:
                        continue
                    # Try to improve
                    target = row_masks[row_idx] ^ new_mask
                    expr_list = row_expr[row_idx]
                    k = len(expr_list)
                    
                    if k <= 18:
                        best_size = k + 1
                        for subset_mask in range(1 << k):
                            xor_val = 0
                            cnt = 0
                            for idx in range(k):
                                if (subset_mask >> idx) & 1:
                                    xor_val ^= available[expr_list[idx]]
                                    cnt += 1
                            if xor_val == target:
                                total_wires = cnt + 1
                                if total_wires < best_size:
                                    best_size = total_wires
                        
                        if best_size < k:
                            gain += (k - best_size)
                    
                    if time.time() - start_time > time_budget:
                        break
                
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (a, b)
                    best_new_mask = new_mask
            
            if time.time() - start_time > time_budget:
                break
        
        if best_gain > 0 and best_pair is not None:
            a, b = best_pair
            gates.append([a, b])
            available[next_wire] = best_new_mask
            
            # Update row expressions
            for row_idx in range(m):
                try_update_row(row_idx, next_wire, best_new_mask)
            
            next_wire += 1
            improvement = True
        else:
            break
    
    # Now build the actual circuit from the expressions
    # We need to ensure all wires used in expressions are actually computed
    # The gates list already contains all computed wires
    
    # Build output wires
    outputs = []
    for row_idx in range(m):
        expr = row_expr[row_idx]
        if len(expr) == 0:
            outputs.append(n)
        elif len(expr) == 1:
            outputs.append(expr[0])
        else:
            # XOR all wires in expr
            wire = expr[0]
            for w in expr[1:]:
                # Find if this XOR is already a gate
                # We need to create gates for these
                # Actually, we should have pre-computed these
                # Let's just create them
                # But this might create extra gates...
                # Better: we should have tracked this
                pass
            # For simplicity, just use the first wire if only one
            # Otherwise, we need to XOR them
            # This is a problem - we need to actually build these XORs
            outputs.append(expr[0])  # placeholder
    
    # The above approach has issues. Let me redo this more carefully.
    # The greedy approach above adds gates, but the final expressions
    # may reference wires that need to be XORed together.
    
    # Let me use a different, cleaner approach.
    return None


def _build_circuit_v2(matrix, n, depth_limit=16):
    """
    Cleaner approach: 
    1. Build a set of useful intermediate XORs using column frequency analysis.
    2. For each row, find the cheapest combination of available wires.
    
    Available wires start as inputs. We add intermediate XORs that are
    frequently useful.
    """
    m = matrix.shape[0]
    
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)
    
    # Count column frequencies
    col_freq = [0] * n
    for mask in row_masks:
        for j in range(n):
            if (mask >> j) & 1:
                col_freq[j] += 1
    
    # Count pair frequencies (how often two columns co-occur)
    pair_freq = {}
    for mask in row_masks:
        ones = [j for j in range(n) if (mask >> j) & 1]
        for a in range(len(ones)):
            for b in range(a + 1, len(ones)):
                key = (min(ones[a], ones[b]), max(ones[a], ones[b]))
                pair_freq[key] = pair_freq.get(key, 0) + 1
    
    # Decide which intermediate XORs to pre-compute
    # An XOR of inputs i and j is useful if it appears in many rows
    # as part of the row's support
    gate_registry = {}
    gates = []
    next_gate = 0
    
    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire
    
    # Pre-compute XORs for pairs that co-occur frequently
    # Sort by frequency, pre-compute top pairs
    sorted_pairs = sorted(pair_freq.items(), key=lambda x: -x[1])
    
    # Pre-compute pairs with frequency >= 2
    for pair, freq in sorted_pairs:
        if freq >= 2:
            get_xor_wire(pair[0], pair[1])
    
    # Now for each row, find the cheapest expression using available wires
    # Available wires: inputs 0..n-1, zero wire n, and computed gates
    
    # Build a list of available wires with their masks
    avail_wires = []
    for j in range(n):
        avail_wires.append((j, 1 << j))
    avail_wires.append((n, 0))
    for k, (a, b) in enumerate(gates):
        wire = n + 1 + k
        mask = (1 << a) ^ (1 << b)
        avail_wires.append((wire, mask))
    
    outputs = []
    for i in range(m):
        target = row_masks[i]
        if target == 0:
            outputs.append(n)
            continue
        
        # Find minimum subset of avail_wires whose XOR = target
        # Greedy: start with all inputs, try to replace with computed wires
        
        # Start with the direct expression
        best_expr = []
        for j in range(n):
            if (target >> j) & 1:
                best_expr.append(j)
        
        # Try to improve by replacing subsets with computed wires
        # For each computed wire, check if it can replace a subset
        improved = True
        while improved:
            improved = False
            for wire_id, wire_mask in avail_wires:
                if wire_mask == 0:
                    continue
                if wire_id in best_expr:
                    continue
                
                # Check if wire_mask is a subset of current expression's XOR
                # i.e., can we replace some subset of best_expr with wire_id?
                # We need: XOR(subset) = wire_mask
                # Then new expr = (best_expr - subset) + [wire_id]
                
                k = len(best_expr)
                if k <= 20:
                    best_replacement = None
                    for subset_mask in range(1 << k):
                        xor_val = 0
                        cnt = 0
                        for idx in range(k):
                            if (subset_mask >> idx) & 1:
                                xor_val ^= (1 << best_expr[idx])
                                cnt += 1
                        if xor_val == wire_mask:
                            # This subset can be replaced by wire_id
                            new_size = k - cnt + 1
                            if new_size < len(best_expr):
                                if best_replacement is None or new_size < best_replacement[0]:
                                    best_replacement = (new_size, subset_mask)
                    
                    if best_replacement is not None:
                        new_size, subset_mask = best_replacement
                        new_expr = []
                        for idx in range(k):
                            if (subset_mask >> idx) & 1:
                                continue
                            new_expr.append(best_expr[idx])
                        new_expr.append(wire_id)
                        best_expr = new_expr
                        improved = True
        
        if len(best_expr) == 1:
            outputs.append(best_expr[0])
        else:
            # Need to XOR these wires together
            # Check if this combination is already a gate
            # For now, create the XORs
            wire = best_expr[0]
            for w in best_expr[1:]:
                # Check if XOR of wire and w is already available
                found = False
                for wid, wmask in avail_wires:
                    if wmask == ((1 << wire) if wire < n else 0) ^ ((1 << w) if w < n else 0):
                        # This is getting complicated
                        pass
                # Just create a new gate
                wire = get_xor_wire(wire, w)
                avail_wires.append((wire, (1 << wire) if wire < n else 0))  # approximate
            outputs.append(wire)
    
    return {'gates': gates, 'outputs': outputs}


def _build_simple_circuit(matrix, n):
    """
    Simple but correct circuit: for each row, XOR its input bits in a chain.
    This is the baseline.
    """
    m = matrix.shape[0]
    gates = []
    outputs = []
    
    for i in range(m):
        ones = [j for j in range(n) if matrix[i, j]]
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            wire = ones[0]
            for j in range(1, len(ones)):
                wire = n + 1 + len(gates)
                gates.append([wire, ones[j]])
            outputs.append(wire)
    
    return {'gates': gates, 'outputs': outputs}


def _build_shared_circuit(matrix, n, depth_limit=16):
    """
    Build a circuit with maximum sharing using a greedy approach.
    
    Key idea: 
    - Maintain a dictionary mapping bitmask -> wire_id for all computed wires.
    - For each row, decompose it into XORs of already-computed sub-expressions.
    - When a new XOR is needed, add it to the circuit.
    
    The decomposition strategy:
    - For a row with mask M, find two sub-masks A and B such that A XOR B = M,
      and both A and B are already computed (or are single inputs).
    - This is equivalent to: find a computed wire with mask W such that
      W is a "proper subset" of M in the sense that M XOR W has fewer bits.
    - Then the row can be computed as W XOR (M XOR W).
    
    More practically:
    - For each row, try to express it as XOR of 2 previously computed wires.
    - If not possible, express as XOR of a computed wire and an input.
    - If not possible, express as XOR of two inputs.
    - If not possible, add new intermediate wires.
    """
    m = matrix.shape[0]
    
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)
    
    # mask_to_wire: maps a bitmask to the wire that computes it
    mask_to_wire = {}
    for j in range(n):
        mask_to_wire[1 << j] = j
    mask_to_wire[0] = n
    
    gates = []
    next_wire = n + 1
    
    def compute_xor(a, b):
        """Compute XOR of wires a and b, returning the result wire."""
        nonlocal next_wire
        if a == b:
            return n
        # Get masks
        # We need to track wire masks
        pass
    
    # Let's track wire masks
    wire_mask = {}
    for j in range(n):
        wire_mask[j] = 1 << j
    wire_mask[n] = 0
    
    def get_xor(a, b):
        nonlocal next_wire
        if a == b:
            return n
        ma = wire_mask.get(a, 0)
        mb = wire_mask.get(b, 0)
        result_mask = ma ^ mb
        if result_mask in mask_to_wire:
            return mask_to_wire[result_mask]
        # Create new gate
        wire = next_wire
        next_wire += 1
        gates.append([a, b])
        wire_mask[wire] = result_mask
        mask_to_wire[result_mask] = wire
        return wire
    
    outputs = []
    for i in range(m):
        target = row_masks[i]
        if target == 0:
            outputs.append(n)
            continue
        
        # Try to express target as XOR of two already-computed wires
        found = False
        for mask_a, wire_a in list(mask_to_wire.items()):
            if mask_a == 0:
                continue
            mask_b = target ^ mask_a
            if mask_b in mask_to_wire:
                wire_b = mask_to_wire[mask_b]
                if wire_a != wire_b:
                    outputs.append(get_xor(wire_a, wire_b))
                    found = True
                    break
            elif mask_b == 0:
                outputs.append(wire_a)
                found = True
                break
        
        if not found:
            # Express as XOR of individual inputs
            ones = [j for j in range(n) if (target >> j) & 1]
            if len(ones) == 1:
                outputs.append(ones[0])
            else:
                wire = ones[0]
                for j in range(1, len(ones)):
                    wire = get_xor(wire, j)
                outputs.append(wire)
    
    return {'gates': gates, 'outputs': outputs}


def _build_optimal_small(matrix, n, depth_limit=16):
    """
    For small matrices, try to find a near-optimal circuit.
    Uses a more sophisticated greedy that considers all possible
    intermediate wires.
    """
    m = matrix.shape[0]
    
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)
    
    # Start with input wires
    wire_mask = {}
    for j in range(n):
        wire_mask[j] = 1 << j
    wire_mask[n] = 0
    
    mask_to_wire = {}
    for j in range(n):
        mask_to_wire[1 << j] = j
    mask_to_wire[0] = n
    
    gates = []
    next_wire = n + 1
    
    def get_xor(a, b):
        nonlocal next_wire
        if a == b:
            return n
        ma = wire_mask.get(a, 0)
        mb = wire_mask.get(b, 0)
        result_mask = ma ^ mb
        if result_mask in mask_to_wire:
            return mask_to_wire[result_mask]
        wire = next_wire
        next_wire += 1
        gates.append([a, b])
        wire_mask[wire] = result_mask
        mask_to_wire[result_mask] = wire
        return wire
    
    # Greedy: repeatedly add the most useful gate
    # A gate (a, b) -> w is useful if it helps express some row more cheaply
    
    # Track the "cost" of each row: minimum number of XORs to express it
    # using currently available wires
    
    def row_cost(target):
        """Minimum number of XOR operations to express target from available wires."""
        if target == 0:
            return 0
        if target in mask_to_wire:
            return 0  # already computed
        # Try: target = wire_a XOR wire_b where both are available
        for ma, wa in mask_to_wire.items():
            if ma == 0:
                continue
            mb = target ^ ma
            if mb in mask_to_wire:
                wb = mask_to_wire[mb]
                if wa != wb:
                    return 1
        # Try: target = input XOR wire
        for j in range(n):
            if (target >> j) & 1:
                rest = target ^ (1 << j)
                if rest in mask_to_wire:
                    return 1
        # Direct: XOR of inputs
        return bin(target).count('1') - 1
    
    # Greedily add gates
    max_iterations = m * n * 2
    for _ in range(max_iterations):
        best_gain = 0
        best_pair = None
        
        wire_ids = list(wire_mask.keys())
        for i in range(len(wire_ids)):
            a = wire_ids[i]
            ma = wire_mask[a]
            if ma == 0:
                continue
            for j in range(i + 1, len(wire_ids)):
                b = wire_ids[j]
                mb = wire_mask[b]
                if mb == 0:
                    continue
                
                new_mask = ma ^ mb
                if new_mask == 0 or new_mask in mask_to_wire:
                    continue
                
                # Compute gain
                gain = 0
                for row_mask in row_masks:
                    old_cost = row_cost(row_mask)
                    # Simulate adding this wire
                    # New cost: min(old_cost, 1 if row_mask = new_mask XOR existing)
                    # Check if row_mask can be expressed with new_mask
                    if row_mask == new_mask:
                        new_cost = 0
                    else:
                        # Check if row_mask = new_mask XOR existing_wire
                        for em, ew in mask_to_wire.items():
                            if em == 0:
                                continue
                            if row_mask ^ em == new_mask:
                                new_cost = 1
                                break
                        else:
                            new_cost = old_cost
                    
                    gain += (old_cost - new_cost)
                
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (a, b)
        
        if best_gain <= 0 or best_pair is None:
            break
        
        a, b = best_pair
        get_xor(a, b)
    
    # Build outputs
    outputs = []
    for i in range(m):
        target = row_masks[i]
        if target == 0:
            outputs.append(n)
        elif target in mask_to_wire:
            outputs.append(mask_to_wire[target])
        else:
            # Express from available wires
            # Try two-wire decomposition
            found = False
            for ma, wa in mask_to_wire.items():
                if ma == 0:
                    continue
                mb = target ^ ma
                if mb in mask_to_wire:
                    wb = mask_to_wire[mb]
                    if wa != wb:
                        outputs.append(get_xor(wa, wb))
                        found = True
                        break
            if not found:
                ones = [j for j in range(n) if (target >> j) & 1]
                wire = ones[0]
                for j in range(1, len(ones)):
                    wire = get_xor(wire, j)
                outputs.append(wire)
    
    return {'gates': gates, 'outputs': outputs}


def synthesize(matrix, depth_limit):
    """
    Main entry point for XOR-circuit synthesis.
    
    Args:
        matrix: 2D array (m x n) of 0s and 1s
        depth_limit: maximum allowed depth
    
    Returns:
        {'gates': [[wire_a, wire_b], ...], 'outputs': [wire, ...]}
    """
    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape
    
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    candidates = []
    
    # Strategy 1: Simple chain (baseline)
    c1 = _build_simple_circuit(matrix, n)
    candidates.append(c1)
    
    # Strategy 2: Shared circuit with mask tracking
    if time.time() < deadline - 10:
        c2 = _build_shared_circuit(matrix, n, depth_limit)
        candidates.append(c2)
    
    # Strategy 3: Optimal small (greedy gate addition)
    if time.time() < deadline - 10:
        c3 = _build_optimal_small(matrix, n, depth_limit)
        candidates.append(c3)
    
    # Pick the best
    best_circuit = None
    best_count = float('inf')
    for c in candidates:
        if not _verify_circuit(matrix, c, depth_limit):
            continue
        count = len(c['gates'])
        if count < best_count:
            best_count = count
            best_circuit = c
    
    if best_circuit is None:
        # Fallback
        best_circuit = _build_simple_circuit(matrix, n)
    
    return best_circuit
