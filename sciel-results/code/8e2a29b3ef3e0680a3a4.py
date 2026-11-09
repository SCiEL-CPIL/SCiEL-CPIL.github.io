"""XOR-circuit synthesis: greedy CSE with global shared-subexpression extraction."""
import time
import numpy as np
from collections import defaultdict

_TIME_LIMIT = 82.0
_DEADLINE = None


def _now():
    return time.time()


def _remaining():
    global _DEADLINE
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - _now())


def _bit_vector_to_int(vec):
    """Convert a 1-D numpy bool/int array (row of matrix) to an int bit-vector."""
    n = vec.shape[0]
    x = 0
    for i, v in enumerate(vec):
        if v:
            x |= (1 << i)
    return x


def _popcount(x):
    return bin(x).count("1")


def _verify(matrix, gates, outputs, n):
    """Exact GF(2) verification. matrix: (m, n) target. gates: [[wire, wire], ...].
    outputs: list of wire indices. Returns True if outputs reproduce matrix rows."""
    m = matrix.shape[0]
    wire_val = [0] * (n + 1 + len(gates))
    for i in range(n):
        wire_val[i] = (1 << i)
    wire_val[n] = 0
    for k, (a, b) in enumerate(gates):
        wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
    for r in range(m):
        target = _bit_vector_to_int(matrix[r])
        if outputs[r] >= len(wire_val):
            return False
        if wire_val[outputs[r]] != target:
            return False
    return True


def _depth_ok(gates, outputs, n):
    """Compute depth (longest path) of the DAG; inputs depth 0, gate depth = max+1."""
    m = len(gates)
    depth = [0] * (n + 1 + m)
    for k in range(m):
        a, b = gates[k]
        depth[n + 1 + k] = max(depth[a], depth[b]) + 1
    maxd = 0
    for o in outputs:
        if o < len(depth):
            maxd = max(maxd, depth[o])
    return maxd


def _baseline_gate_count(matrix, n):
    """Independent-row baseline: each output row built independently by XOR of its set bits."""
    total = 0
    for r in range(matrix.shape[0]):
        total += _popcount(_bit_vector_to_int(matrix[r])) - 1
    return max(total, 0)


def _xor_tree(indices, get_xor_wire):
    """Build a balanced XOR tree for the given wire indices."""
    if len(indices) == 1:
        return indices[0]
    mid = len(indices) // 2
    left = _xor_tree(indices[:mid], get_xor_wire)
    right = _xor_tree(indices[mid:], get_xor_wire)
    return get_xor_wire(left, right)


def _build_greedy_cse(matrix, n, row_order=None, max_iters=200):
    """
    Build circuit via greedy CSE with shared subexpression extraction.
    Maintains a set of wires that XOR to give each row value.
    Repeatedly finds the pair of wires appearing together in the most rows,
    and replaces that pair with a new XOR wire.
    """
    m = matrix.shape[0]
    if row_order is None:
        row_order = list(range(m))

    row_masks = [_bit_vector_to_int(matrix[i]) for i in range(m)]

    row_sets = []
    for idx in row_order:
        mask = row_masks[idx]
        s = set(j for j in range(n) if (mask >> j) & 1)
        if not s:
            s = {n}
        row_sets.append(s)

    gate_registry = {}
    gates = []
    next_gate = [0]

    def get_wire(a, b):
        if a == b:
            return n
        key = frozenset((a, b))
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate[0]
        gates.append([min(a, b), max(a, b)])
        gate_registry[key] = wire
        next_gate[0] += 1
        return wire

    def compute_pair_counts():
        pc = defaultdict(int)
        for s in row_sets:
            lst = sorted(s)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    pc[frozenset((lst[i], lst[j]))] += 1
        return pc

    pair_count = compute_pair_counts()

    changed = True
    iter_count = 0
    while changed and iter_count < max_iters:
        if _remaining() < 2.0:
            break
        changed = False
        iter_count += 1

        best_pair = None
        best_benefit = 0
        for key, cnt in pair_count.items():
            a, b = tuple(key)
            if key in gate_registry:
                continue
            benefit = cnt - 1
            if benefit > best_benefit:
                best_benefit = benefit
                best_pair = key

        if best_pair is None or best_benefit <= 0:
            break

        a, b = tuple(best_pair)
        new_wire = get_wire(a, b)
        changed = True

        for i in range(len(row_sets)):
            s = row_sets[i]
            if a in s and b in s:
                s.discard(a)
                s.discard(b)
                s.add(new_wire)

        pair_count = compute_pair_counts()

    outputs = []
    for i in range(len(row_sets)):
        s = row_sets[i]
        if len(s) == 0:
            outputs.append(n)
        elif len(s) == 1:
            outputs.append(next(iter(s)))
        else:
            lst = sorted(s)
            outputs.append(_xor_tree(lst, get_wire))

    if row_order != list(range(m)):
        final_outputs = [0] * m
        for new_idx, orig_idx in enumerate(row_order):
            final_outputs[orig_idx] = outputs[new_idx]
        outputs = final_outputs

    return {'gates': gates, 'outputs': outputs}


def _build_direct(matrix, n):
    """Direct construction: balanced XOR trees with CSE."""
    gate_registry = {}
    gates = []
    next_gate = [0]

    def get_xor_wire(a, b):
        if a == b:
            return n
        key = frozenset((a, b))
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate[0]
        gates.append([min(a, b), max(a, b)])
        gate_registry[key] = wire
        next_gate[0] += 1
        return wire

    outputs = []
    for i in range(matrix.shape[0]):
        ones = [j for j in range(n) if (matrix[i, j])]
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            outputs.append(_xor_tree(ones, get_xor_wire))

    return {'gates': gates, 'outputs': outputs}


def _local_search(matrix, circuit, depth_limit, time_budget=5.0):
    """Local search: remove unused gates, collapse trivial gates."""
    gates = [g[:] for g in circuit['gates']]
    outputs = circuit['outputs'][:]
    n = matrix.shape[1]
    m = matrix.shape[0]

    def compute_masks():
        wm = {}
        for i in range(n):
            wm[i] = (1 << i)
        wm[n] = 0
        for k, (a, b) in enumerate(gates):
            wm[n + 1 + k] = wm.get(a, 0) ^ wm.get(b, 0)
        return wm

    # Remove unused gates iteratively
    changed = True
    while changed:
        if _remaining() < 1.0:
            break
        changed = False
        if not gates:
            break
        usage = defaultdict(int)
        for k, (a, b) in enumerate(gates):
            usage[a] += 1
            usage[b] += 1
        for out in outputs:
            usage[out] += 1

        new_gates = []
        removed_wires = set()
        for k, (a, b) in enumerate(gates):
            wire = n + 1 + k
            if usage[wire] == 0:
                removed_wires.add(wire)
            else:
                new_gates.append([a, b])

        if removed_wires:
            # Remap wire indices
            remap = {}
            new_next = n + 1
            for k in range(len(gates)):
                wire = n + 1 + k
                if wire not in removed_wires:
                    remap[wire] = new_next
                    new_next += 1

            gates = []
            for g in new_gates:
                gates.append([remap.get(g[0], g[0]), remap.get(g[1], g[1])])

            outputs = [remap.get(o, o) for o in outputs]
            changed = True

    return {'gates': gates, 'outputs': outputs}


def synthesize(matrix, depth_limit):
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + _TIME_LIMIT

    mat = np.array(matrix, dtype=np.int32)
    n_rows, n_cols = mat.shape
    n = n_cols

    # Try multiple strategies and pick the best
    best_circuit = None
    best_gate_count = float('inf')

    # Strategy 1: Direct construction
    if _remaining() > 5.0:
        circuit1 = _build_direct(mat, n)
        if _verify(mat, circuit1['gates'], circuit1['outputs'], n):
            gc = len(circuit1['gates'])
            if gc < best_gate_count:
                best_gate_count = gc
                best_circuit = circuit1

    # Strategy 2: Greedy CSE
    if _remaining() > 5.0:
        circuit2 = _build_greedy_cse(mat, n)
        if _verify(mat, circuit2['gates'], circuit2['outputs'], n):
            gc = len(circuit2['gates'])
            if gc < best_gate_count:
                best_gate_count = gc
                best_circuit = circuit2

    # Strategy 3: Greedy CSE with local search
    if _remaining() > 10.0 and best_circuit is not None:
        circuit3 = _local_search(mat, best_circuit, depth_limit)
        if _verify(mat, circuit3['gates'], circuit3['outputs'], n):
            gc = len(circuit3['gates'])
            if gc < best_gate_count:
                best_gate_count = gc
                best_circuit = circuit3

    # Check depth constraint
    if best_circuit is not None:
        d = _depth_ok(best_circuit['gates'], best_circuit['outputs'], n)
        if d > depth_limit:
            # Fall back to direct construction which tends to have lower depth
            circuit_direct = _build_direct(mat, n)
            if _verify(mat, circuit_direct['gates'], circuit_direct['outputs'], n):
                best_circuit = circuit_direct

    if best_circuit is None:
        # Fallback: just use direct construction
        best_circuit = _build_direct(mat, n)

    return {'gates': best_circuit['gates'], 'outputs': best_circuit['outputs']}
