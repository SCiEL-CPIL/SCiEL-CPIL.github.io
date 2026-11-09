import numpy as np
import time
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
    # wire value: int bit-vector over n input bits
    wire_val = [0] * (n + 1 + len(gates))
    # inputs 0..n-1 are unit vectors
    for i in range(n):
        wire_val[i] = (1 << i)
    wire_val[n] = 0  # constant zero
    for k, (a, b) in enumerate(gates):
        wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
    # outputs must equal the m target rows
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
    # inputs and zero are depth 0
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
    
    The key insight: we maintain a set of wires that XOR to give each row value.
    We repeatedly find the pair of wires that appears together in the most rows,
    and replace that pair with a new XOR wire. This reduces total wire count.
    """
    m = matrix.shape[0]
    if row_order is None:
        row_order = list(range(m))
    
    # Convert rows to bit masks
    row_masks = [_bit_vector_to_int(matrix[i]) for i in range(m)]
    
    # For each row in the given order, maintain a set of wire indices that XOR to give the row value.
    # Initially, each row is its set of 1-bits (or {n} if zero).
    row_sets = []
    for idx in row_order:
        mask = row_masks[idx]
        s = set(j for j in range(n) if (mask >> j) & 1)
        if not s:
            s = {n}
        row_sets.append(s)
    
    # Gate registry: frozenset({a,b}) -> wire index
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
    
    # Compute pair counts: for each pair of wires, how many rows contain both
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
    
    # Build final outputs
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
    
    # Map outputs back to original row order
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
        
        if len(new_gates) < len(gates):
            gates = new_gates
            changed = True
            # Remap outputs and gate inputs
            remap = {}
            old_wires = [n + 1 + k for k in range(len(gates) + len(removed_wires))]
            # Rebuild mapping: old wire index -> new wire index
            # This is complex, so let's just rebuild from scratch if needed
            # For simplicity, if we removed gates, we need to remap
            # Let's do a proper remap
            old_to_new = {}
            new_idx = 0
            for k in range(len(gates)):
                old_wire = n + 1 + k
                if old_wire not in removed_wires:
                    old_to_new[old_wire] = n + 1 + new_idx
                    new_idx += 1
            
            new_outputs = []
            for o in outputs:
                if o in old_to_new:
                    new_outputs.append(old_to_new[o])
                else:
                    new_outputs.append(o)
            outputs = new_outputs
            
            new_gates2 = []
            for a, b in gates:
                na = old_to_new.get(a, a)
                nb = old_to_new.get(b, b)
                new_gates2.append([na, nb])
            gates = new_gates2
    
    return {'gates': gates, 'outputs': outputs}

def synthesize(matrix, depth_limit):
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + _TIME_LIMIT
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}
    
    # Try greedy CSE
    circuit = _build_greedy_cse(matrix, n)
    
    # Verify
    if not _verify(matrix, circuit['gates'], circuit['outputs'], n):
        # Fallback to direct
        circuit = _build_direct(matrix, n)
    
    # Local search to remove redundant gates
    if _remaining() > 5.0:
        circuit = _local_search(matrix, circuit, depth_limit, time_budget=5.0)
    
    # Final verification
    if not _verify(matrix, circuit['gates'], circuit['outputs'], n):
        circuit = _build_direct(matrix, n)
    
    return circuit
