"""XOR-circuit synthesis: shared-basis (row-echelon) method with exact GF(2) verification."""
import time

def synthesize(matrix, depth_limit):
    start = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill

    # Parse matrix: list of rows (or 2D array), each row is list of 0/1
    # Normalize to list of bit-vectors as Python ints for fast GF(2) ops.
    try:
        if isinstance(matrix, (list, tuple)):
            rows_in = [list(r) for r in matrix]
        else:
            import numpy as np
            a = np.asarray(matrix)
            rows_in = [list(map(int, r)) for r in a.tolist()]
    except Exception:
        rows_in = [list(r) for r in matrix]

    if not rows_in:
        return {'gates': [], 'outputs': []}

    n = len(rows_in[0])
    m = len(rows_in)

    # Represent each linear form as an int bitmask over n input bits.
    def to_int(row):
        v = 0
        for i, b in enumerate(row):
            if b:
                v |= (1 << i)
        return v

    target = [to_int(r) for r in rows_in]

    # ---------- Build a row-echelon basis, emitting gates ----------
    # We process columns left to right; maintain basis rows as ints.
    # Each basis row is a linear combo of inputs; we emit gates to realize it.
    # Strategy: elimination on the target rows to get echelon; each elimination
    # step (row_j ^= row_i) corresponds to one XOR gate producing a new wire.

    # Work on a copy of target rows as ints, tracking wire id per row.
    # Wire ids: inputs are 0..n-1, constant zero = n. New gates start at n+1.
    wire_id = {}  # int value -> wire id (for available linear forms)
    for i in range(n):
        wire_id[1 << i] = i
    wire_id[0] = n  # constant zero

    gates = []
    next_wire = n + 1

    def emit_gate(a, b):
        nonlocal next_wire
        w = next_wire
        next_wire += 1
        gates.append([a, b])
        return w

    # We'll do Gaussian elimination to produce an echelon basis AND keep track
    # of each original output as a combination of basis vectors.
    #
    # Approach: build basis incrementally. For each target row, reduce it against
    # existing basis; the reduced remainder (if nonzero) becomes a new basis vector.
    # Each XOR performed is a gate. We record each output as the list of basis
    # vectors that compose it.

    basis = []          # list of (pivot_col, int_value, wire_id)
    basis_by_pivot = {} # pivot_col -> index into basis

    # For each original output, track the set of basis indices that XOR to it.
    # We do this by processing outputs one by one with the same elimination,
    # recording which basis vectors were used.

    output_basis_combo = []  # for each output: list of basis indices

    # First pass: build the basis from all rows (union of row space).
    all_rows = list(target)
    for r in all_rows:
        v = r
        # reduce
        while v != 0:
            pc = v.bit_length() - 1
            if pc in basis_by_pivot:
                bi = basis_by_pivot[pc]
                bv = basis[bi][1]
                # v ^= bv  -> gate
                # but we need wire ids; basis[bi] has a wire id already.
                # We'll recompute wire ids at emission; for basis building we
                # just track int values.
                v ^= bv
            else:
                # new basis vector
                idx = len(basis)
                basis.append([pc, v, None])  # wire assigned later
                basis_by_pivot[pc] = idx
                break
    r_rank = len(basis)

    # ---------- Assign wire ids to basis vectors via echelon elimination gates ----------
    # Build basis vectors in increasing pivot order using previously built basis
    # vectors as inputs where possible to reuse and keep depth low.
    basis_wire = [None] * r_rank
    # Sort by pivot column
    order = sorted(range(r_rank), key=lambda i: basis[i][0])
    for idx in order:
        pc, v, _ = basis[idx]
        # express v as XOR of inputs and lower-pivot basis vectors
        rem = v
        used = []
        # try to subtract lower basis vectors
        for j in order:
            if j >= idx:
                break
            pcj, vj, _ = basis[j]
            if rem & (1 << pcj):
                rem ^= vj
                used.append(j)
        # rem should now be a single bit (the pivot) if basis is proper echelon
        if rem and (rem & (rem - 1)) == 0:
            pivot_bit = rem
            # start from the input wire of that bit
            cur = wire_id[pivot_bit]
            for j in used:
                cur = emit_gate(cur, basis_wire[j])
            basis_wire[idx] = cur
        else:
            # fallback: build directly from input bits
            bits = [i for i in range(n) if (v >> i) & 1]
            if not bits:
                basis_wire[idx] = n  # zero
            else:
                cur = wire_id[1 << bits[0]]
                for b in bits[1:]:
                    cur = emit_gate(cur, wire_id[1 << b])
                basis_wire[idx] = cur

    # ---------- Build each output from basis vectors ----------
    outputs = []
    for r in target:
        v = r
        combo = []
        while v != 0:
            pc = v.bit_length() - 1
            if pc in basis_by_pivot:
                bi = basis_by_pivot[pc]
                v ^= basis[bi][1]
                combo.append(bi)
            else:
                # shouldn't happen since basis spans row space
                break
        if not combo:
            outputs.append(n)  # zero output
        elif len(combo) == 1:
            outputs.append(basis_wire[combo[0]])
        else:
            cur = basis_wire[combo[0]]
            for j in combo[1:]:
                cur = emit_gate(cur, basis_wire[j])
            outputs.append(cur)

    # ---------- Exact verification ----------
    def verify():
        # simulate: wire -> int value
        val = [0] * next_wire
        for i in range(n):
            val[i] = 1 << i
        val[n] = 0
        for a, b in gates:
            val[a + 0]  # ensure in range
            if a >= next_wire or b >= next_wire:
                return False
            val[a + 1]  # placeholder
        # recompute properly
        val = [0] * next_wire
        for i in range(n):
            val[i] = 1 << i
        val[n] = 0
        for a, b in gates:
            if a >= next_wire or b >= next_wire:
                return False
            val[a + 1] = val[a] ^ val[b]  # WRONG indexing
        return True

    # Simpler verification: recompute wire values by index.
    def verify2():
        if next_wire > len(val) if False else False:
            pass
        vals = [0] * next_wire
        for i in range(n):
            vals[i] = 1 << i
        vals[n] = 0
        for a, b in gates:
            if a >= next_wire or b >= next_wire:
                return False
            vals[a + 1] = vals[a] ^ vals[b]
        for k, o in enumerate(outputs):
            if o >= next_wire:
                return False
            if vals[o] != target[k]:
                return False
        return True

    # The gate wire id for gate k is n+1+k. Fix indexing in verify2.
    def verify3():
        vals = [0] * next_wire
        for i in range(n):
            vals[i] = 1 << i
        vals[n] = 0
        for k, (a, b) in enumerate(gates):
            w = n + 1 + k
            if a >= next_wire or b >= next_wire:
                return False
            vals[w] = vals[a] ^ vals[b]
        for k, o in enumerate(outputs):
            if o >= next_wire:
                return False
            if vals[o] != target[k]:
                return False
        return True

    if not verify3():
        # Fallback to trivial per-output circuit (always correct)
        gates = []
        next_wire = n + 1
        outputs = []
        for r in target:
            bits = [i for i in range(n) if (r >> i) & 1]
            if not bits:
                outputs.append(n)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                cur = bits[0]
                for b in bits[1:]:
                    gates.append([cur, b])
                    cur = n + 1 + (len(gates) - 1)
                outputs.append(cur)
        next_wire = n + 1 + len(gates)

    # Depth check: compute depth of each output; if exceeds depth_limit, fall back.
    def compute_depth():
        # depth[input]=0, depth[zero]=0; gate k depth = 1+max(depth[a],depth[b])
        dep = [0] * next_wire
        for k, (a, b) in enumerate(gates):
            w = n + 1 + k
            dep[w] = 1 + max(dep[a], dep[b])
        return max((dep[o] for o in outputs), default=0)

    if compute_depth() > depth_limit:
        # fall back to direct per-output (depth = weight-1 <= n-1 <= 15 for n<=16)
        gates = []
        next_wire = n + 1
        outputs = []
        for r in target:
            bits = [i for i in range(n) if (r >> i) & 1]
            if not bits:
                outputs.append(n)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                cur = bits[0]
                for b in bits[1:]:
                    gates.append([cur, b])
                    cur = n + 1 + (len(gates) - 1)
                outputs.append(cur)
        next_wire = n + 1 + len(gates)

    return {'gates': gates, 'outputs': outputs}
