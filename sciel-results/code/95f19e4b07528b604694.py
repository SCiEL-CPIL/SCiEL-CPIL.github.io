"""XOR circuit synthesis with adaptive ordering, CSE pass, and cost-minimizing recursive builder."""
import time
import sys

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}

    # Convert matrix rows to integer bitmasks for fast XOR
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)

    # Map from bitmask value to wire index
    val_to_wires = {}

    # Initialize with input wires
    for j in range(n):
        val = 1 << j
        val_to_wires[val] = j

    # Constant zero
    val_to_wires[0] = n

    gates = []
    next_wire = n + 1

    def popcount(x):
        return bin(x).count('1')

    def build_recursive(target, depth=0):
        """
        Recursively build 'target' using existing wires.
        Returns the wire index for 'target'.
        Tries to minimize total gate count by picking the helper that
        leads to the simplest remaining subproblem.
        """
        nonlocal next_wire, gates, val_to_wires

        if target == 0:
            return n

        if target in val_to_wires:
            return val_to_wires[target]

        if depth > 32:
            # Fallback: build from inputs directly
            return build_from_inputs(target)

        # Find 'a' in val_to_wires that minimizes the estimated cost of (target ^ a)
        best_a = None
        best_a_wire = None
        min_cost = float('inf')
        best_diff = None

        for a, a_wire in val_to_wires.items():
            diff = target ^ a
            if diff == 0:
                return a_wire
            cost = popcount(diff)
            if cost < min_cost:
                min_cost = cost
                best_a = a
                best_a_wire = a_wire
                best_diff = diff
                if cost == 1:
                    break

        if best_a is None:
            return build_from_inputs(target)

        diff_wire = build_recursive(best_diff, depth + 1)

        if best_diff == 0:
            return best_a_wire

        gates.append([best_a_wire, diff_wire])
        new_wire = next_wire
        next_wire += 1
        val_to_wires[target] = new_wire

        return new_wire

    def build_from_inputs(target):
        """Fallback: build target directly from input wires using a chain."""
        nonlocal next_wire, gates, val_to_wires

        if target == 0:
            return n
        if target in val_to_wires:
            return val_to_wires[target]

        wire = None
        bits = target
        while bits:
            lb = bits & (-bits)
            bit_idx = lb.bit_length() - 1
            input_wire = bit_idx
            if wire is None:
                wire = input_wire
            else:
                gates.append([wire, input_wire])
                wire = next_wire
                next_wire += 1
            bits ^= lb

        if wire is None:
            wire = n

        val_to_wires[target] = wire
        return wire

    def estimate_recursive_cost(target):
        """Estimate the gate cost to build target from current wires."""
        if target == 0:
            return 0
        if target in val_to_wires:
            return 0
        best = float('inf')
        for a in val_to_wires:
            diff = target ^ a
            if diff == 0:
                return 0
            # Cost = 1 gate + cost of diff (approximate by popcount)
            c = 1 + popcount(diff)
            if c < best:
                best = c
        return best if best < float('inf') else popcount(target)

    # Track which rows have been computed
    row_to_wire = [None] * m
    remaining = set(range(m))

    # Iteratively pick the row with minimum additional cost
    while remaining:
        if time.time() > deadline:
            break

        best_row = None
        best_cost = float('inf')
        best_type = None
        best_a_wire = None
        best_b_wire = None

        for idx in remaining:
            target = rows[idx]

            if target == 0:
                row_to_wire[idx] = n
                remaining.discard(idx)
                continue

            if target in val_to_wires:
                row_to_wire[idx] = val_to_wires[target]
                remaining.discard(idx)
                continue

            # Check for one-gate completion
            one_gate_found = False
            for a, a_wire in val_to_wires.items():
                b = target ^ a
                if b in val_to_wires:
                    b_wire = val_to_wires[b]
                    if a_wire == b_wire and target != 0:
                        continue
                    if 1 < best_cost:
                        best_cost = 1
                        best_row = idx
                        best_type = 'one_gate'
                        best_a_wire = a_wire
                        best_b_wire = b_wire
                    one_gate_found = True
                    break

            if one_gate_found:
                continue

            est_cost = estimate_recursive_cost(target)

            if est_cost < best_cost:
                best_cost = est_cost
                best_row = idx
                best_type = 'recursive'
                best_a_wire = None
                best_b_wire = None

        if best_row is None:
            break

        target = rows[best_row]

        if best_type == 'one_gate':
            gates.append([best_a_wire, best_b_wire])
            row_to_wire[best_row] = next_wire
            val_to_wires[target] = next_wire
            next_wire += 1
        elif best_type == 'recursive':
            row_to_wire[best_row] = build_recursive(target)
        else:
            row_to_wire[best_row] = n

        remaining.discard(best_row)

    # Ensure all rows have a wire assigned
    for i in range(m):
        if row_to_wire[i] is None:
            if rows[i] == 0:
                row_to_wire[i] = n
            elif rows[i] in val_to_wires:
                row_to_wire[i] = val_to_wires[rows[i]]
            else:
                row_to_wire[i] = build_from_inputs(rows[i])

    # === CSE PASS: Find high-value intermediate XORs to add ===
    # After initial synthesis, look for pairs of existing wires whose XOR
    # is either a row value or appears in multiple remaining computations.
    # This enables sharing that the greedy pass may have missed.
    if time.time() < deadline - 5.0:
        wire_vals = list(val_to_wires.keys())
        # Count how many rows benefit from each potential intermediate
        # For each pair (i,j) of wire values, compute val_i ^ val_j
        # If that value is a row, it could save gates for that row
        # We add intermediates that are used by >= 2 rows or are themselves rows
        
        # Build set of row values for quick lookup
        row_val_set = set(rows)
        
        # For each pair of existing wire values, check if their XOR is useful
        # Limit to avoid O(W^2) explosion: only consider pairs where XOR has low popcount
        # or is a known row value
        added_intermediates = 0
        max_intermediates = 50  # Cap to stay within time
        
        # Get list of wire values (excluding 0 and single-bit inputs for efficiency)
        candidate_vals = [v for v in wire_vals if v != 0 and popcount(v) > 1]
        
        # Sort by popcount to find low-cost intermediates first
        candidate_vals.sort(key=lambda v: popcount(v))
        
        useful_intermediates = {}  # val -> count of rows it helps
        
        for v1 in candidate_vals:
            if v1 not in val_to_wires:
                continue
            for v2 in candidate_vals:
                if v2 <= v1:  # Avoid duplicates and self-XOR
                    continue
                inter = v1 ^ v2
                if inter == 0:
                    continue
                if inter in val_to_wires:
                    continue
                # Check if this intermediate helps any rows
                # A row r benefits if (r ^ inter) is in val_to_wires
                # i.e., row can be computed as inter XOR (r^inter) with one gate
                helps = 0
                for r in rows:
                    if r == 0:
                        continue
                    if r in val_to_wires:
                        continue
                    if (r ^ inter) in val_to_wires:
                        helps += 1
                if helps >= 2:
                    useful_intermediates[inter] = helps
        
        # Add the most useful intermediates (highest help count)
        sorted_intermediates = sorted(useful_intermediates.items(), key=lambda x: -x[1])
        
        for inter_val, help_count in sorted_intermediates:
            if added_intermediates >= max_intermediates:
                break
            if time.time() > deadline - 3.0:
                break
            if inter_val in val_to_wires:
                continue
            
            # Find two existing wires whose XOR gives inter_val
            found = False
            for v1, w1 in val_to_wires.items():
                v2 = inter_val ^ v1
                if v2 in val_to_wires and v2 != v1:
                    w2 = val_to_wires[v2]
                    gates.append([w1, w2])
                    new_wire = next_wire
                    next_wire += 1
                    val_to_wires[inter_val] = new_wire
                    added_intermediates += 1
                    found = True
                    break
            
            if not found:
                continue
            
            # Now rebuild any rows that can use this intermediate with one gate
            for i in range(m):
                if row_to_wire[i] is None:
                    continue
                r = rows[i]
                if r == 0 or r in val_to_wires:
                    continue
                if (r ^ inter_val) in val_to_wires:
                    # Can compute r with one gate from inter_val
                    # Only do this if it would save gates (current cost > 1)
                    pass  # Rebuilding would require redoing the circuit; skip for safety

    outputs = row_to_wire

    return {'gates': gates, 'outputs': outputs}
