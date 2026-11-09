"""XOR circuit synthesis with adaptive ordering, CSE, and pairwise-lookup acceleration."""
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 52.0  # Safety margin before 90s kill

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

    def build_from_inputs(target):
        """Fallback: build target directly from input wires using a chain."""
        nonlocal next_wire, gates, val_to_wires

        if target == 0:
            return n
        if target in val_to_wires:
            return val_to_wires[target]

        # Build from set bits
        wire = None
        bits = target
        while bits:
            lb = bits & (-bits)
            bit_idx = lb.bit_length() - 1
            input_wire = bit_idx  # input wire for bit bit_idx
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

    def find_one_gate_pair(target):
        """
        Find a pair (a_wire, b_wire) such that a_wire XOR b_wire = target,
        both already in val_to_wires, and a_wire != b_wire.
        Returns (a_wire, b_wire) or None.
        Uses O(|val_to_wires|) but with early exit on first valid pair.
        """
        for a, a_wire in val_to_wires.items():
            b = target ^ a
            if b in val_to_wires:
                b_wire = val_to_wires[b]
                if a_wire != b_wire:
                    return (a_wire, b_wire)
        return None

    # Track which rows have been computed
    row_to_wire = [None] * m
    remaining = set(range(m))

    # Pre-compute popcount for each row for cost estimation
    row_popcount = [popcount(rows[i]) for i in range(m)]

    # Iteratively pick the row with minimum additional cost
    while remaining:
        if time.time() > deadline:
            break

        # First pass: check if any remaining row is already available or one-gate completable
        one_gate_rows = []
        for idx in list(remaining):
            target = rows[idx]

            if target == 0:
                row_to_wire[idx] = n
                remaining.discard(idx)
                continue

            if target in val_to_wires:
                row_to_wire[idx] = val_to_wires[target]
                remaining.discard(idx)
                continue

            # Check for one-gate completion with fast lookup
            pair = find_one_gate_pair(target)
            if pair is not None:
                one_gate_rows.append((idx, pair))

        if one_gate_rows:
            # Pick the one-gate row with lowest popcount (most "valuable" to complete early)
            # Actually, any one-gate row is equally cheap (1 gate), so pick first
            # But completing one may enable more one-gate completions, so just pick one
            best_idx, (a_wire, b_wire) = one_gate_rows[0]
            target = rows[best_idx]
            gates.append([a_wire, b_wire])
            row_to_wire[best_idx] = next_wire
            val_to_wires[target] = next_wire
            next_wire += 1
            remaining.discard(best_idx)
            continue

        # No one-gate rows available; pick the row with minimum estimated recursive cost
        best_row = None
        best_est_cost = float('inf')

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

            # Estimate recursive cost: find best helper
            est_cost = float('inf')
            for a in val_to_wires.keys():
                diff = target ^ a
                if diff == 0:
                    est_cost = 0
                    break
                diff_pc = popcount(diff)
                est = 1 + diff_pc
                if est < est_cost:
                    est_cost = est

            # Add small tiebreaker: prefer lower popcount targets
            total_est = est_cost + 0.1 * row_popcount[idx]

            if total_est < best_est_cost:
                best_est_cost = total_est
                best_row = idx

        if best_row is None:
            break

        target = rows[best_row]
        if target in val_to_wires:
            row_to_wire[best_row] = val_to_wires[target]
        else:
            row_to_wire[best_row] = build_recursive(target)
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

    # Polish pass: try to reduce gate count by checking if any gate's output
    # can be replaced by an existing wire (CSE cleanup)
    # This is a best-effort pass that may not always help, but try it
    if len(gates) > 0 and time.time() < deadline - 2.0:
        # Build wire-to-value map for gate outputs
        gate_output_wires = {}
        for k in range(len(gates)):
            wire = n + 1 + k
            gate_output_wires[wire] = True

        # For each gate, check if its output value is already available via another wire
        # This is complex to do safely, so skip for now to avoid correctness issues

    outputs = row_to_wire

    return {'gates': gates, 'outputs': outputs}
