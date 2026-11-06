"""Rewritten program for exact XOR-circuit synthesis with stronger algorithms."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def _xor_count(vec):
    """Count number of 1s in a binary vector."""
    return int(np.sum(vec))

def _row_space_gf2(matrix):
    """Find a basis for the row space of a binary matrix over GF(2)."""
    A = matrix.copy().astype(np.int64)
    m, n = A.shape
    rank = 0
    for col in range(n):
        pivot = -1
        for row in range(rank, m):
            if A[row, col] % 2 == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        if pivot != rank:
            A[rank], A[pivot] = A[pivot].copy(), A[rank].copy()
        for row in range(m):
            if row != rank and A[row, col] % 2 == 1:
                A[row] = (A[row] + A[rank]) % 2
        rank += 1
    return A[0:rank], rank

def _verify_solution(matrix, gates, outputs, depth_limit=16):
    """
    Verify that the given gates and outputs correctly compute matrix @ x over GF(2).
    matrix: shape (m, n) binary. Input wires 0..n-1. Constant-zero wire n.
    Gate k produces wire n + 1 + k. gates is list of [wire_a, wire_b].
    outputs: list of wire indices, one per output row.
    
    Returns True if valid and depth <= depth_limit.
    """
    m, n = matrix.shape
    num_gates = len(gates)
    total_wires = n + 1 + num_gates
    
    # Check depth
    if depth_limit is not None:
        depth = [0] * total_wires
        for k in range(num_gates):
            a, b = gates[k]
            if a < 0 or a >= total_wires or b < 0 or b >= total_wires:
                return False
            if a >= n + 1 + k or b >= n + 1 + k:
                return False
            depth[n + 1 + k] = 1 + max(depth[a], depth[b])
        max_depth = max(depth[n + 1 + k] for k in range(num_gates)) if num_gates > 0 else 0
        for wire in outputs:
            if wire < 0 or wire >= total_wires:
                return False
            max_depth = max(max_depth, depth[wire])
        if max_depth > depth_limit:
            return False
    
    # Symbolic verification
    wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
    for i in range(n):
        wire_vec[i][i] = 1
    
    for k in range(num_gates):
        a, b = gates[k]
        wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
    
    for i in range(m):
        wire = outputs[i]
        if wire < 0 or wire >= total_wires:
            return False
        if not np.array_equal(matrix[i], wire_vec[wire]):
            return False
    
    return True

def _baseline_synthesis(matrix, depth_limit=16):
    """
    Baseline synthesis: independent XOR chains for each row.
    """
    m, n = matrix.shape
    gates = []
    outputs = []
    
    for i in range(m):
        row = matrix[i]
        positions = np.where(row)[0].tolist()
        if len(positions) == 0:
            outputs.append(n)
        elif len(positions) == 1:
            outputs.append(positions[0])
        else:
            current = positions[0]
            for j in range(1, len(positions)):
                a = current
                b = positions[j]
                gate_idx = len(gates)
                new_wire = n + 1 + gate_idx
                gates.append([a, b])
                current = new_wire
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}

def _greedy_shared_synthesis(matrix, depth_limit=16, time_budget=2.0):
    """
    Greedy synthesis with shared subexpressions.
    Process rows in an order that maximizes sharing.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # Sort rows by popcount (ascending) to potentially share more
    rows_sorted = sorted(range(m), key=lambda i: _xor_count(matrix[i]))
    
    best_gates = None
    best_outputs = None
    best_count = float('inf')
    
    # Try multiple orderings
    for restart in range(30):
        if time.time() > deadline:
            break
        
        if restart == 0:
            order = rows_sorted
        else:
            order = list(range(m))
            random.shuffle(order)
        
        gates = []
        outputs = [None] * m
        wire_vec = {}
        vec_to_wire = {}
        
        for i in range(n):
            v = np.zeros(n, dtype=np.int64)
            v[i] = 1
            wire_vec[i] = v
            vec_to_wire[v.tobytes()] = i
        wire_vec[n] = np.zeros(n, dtype=np.int64)
        vec_to_wire[wire_vec[n].tobytes()] = n
        
        success = True
        for i in order:
            target = matrix[i]
            target_bytes = target.tobytes()
            
            if target_bytes in vec_to_wire:
                outputs[i] = vec_to_wire[target_bytes]
                continue
            
            # Try to find a, b in existing wires such that a XOR b = target
            found = False
            target_list = target.tolist()
            
            # Try each existing wire a, check if target XOR a is also existing
            for a_wire, a_vec in list(wire_vec.items()):
                candidate = (target + a_vec) % 2
                candidate_bytes = candidate.tobytes()
                if candidate_bytes in vec_to_wire:
                    b_wire = vec_to_wire[candidate_bytes]
                    if a_wire != b_wire:  # Avoid XOR of same wire
                        gate_idx = len(gates)
                        new_wire = n + 1 + gate_idx
                        gates.append([a_wire, b_wire])
                        wire_vec[new_wire] = candidate
                        vec_to_wire[candidate_bytes] = new_wire
                        outputs[i] = new_wire
                        found = True
                        break
            
            if found:
                continue
            
            # Fallback: build chain
            positions = np.where(target)[0].tolist()
            if len(positions) == 0:
                outputs[i] = n
            elif len(positions) == 1:
                outputs[i] = positions[0]
            else:
                current = positions[0]
                for j in range(1, len(positions)):
                    a = current
                    b = positions[j]
                    gate_idx = len(gates)
                    new_wire = n + 1 + gate_idx
                    gates.append([a, b])
                    new_vec = (wire_vec[a] + wire_vec[b]) % 2
                    wire_vec[new_wire] = new_vec
                    vec_to_wire[new_vec.tobytes()] = new_wire
                    current = new_wire
                outputs[i] = current
        
        if all(o is not None for o in outputs):
            if _verify_solution(matrix, gates, outputs, depth_limit):
                if len(gates) < best_count:
                    best_count = len(gates)
                    best_gates = gates[:]
                    best_outputs = outputs[:]
    
    if best_gates is not None:
        return {'gates': best_gates, 'outputs': best_outputs}
    
    return _baseline_synthesis(matrix, depth_limit)

def _local_search_synthesis(matrix, depth_limit=16, time_budget=5.0):
    """
    Local search: start from a good solution and try to reduce gate count
    by replacing gates with equivalent shorter sequences.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # Start from greedy shared
    result = _greedy_shared_synthesis(matrix, depth_limit, time_budget=1.0)
    gates = result['gates'][:]
    outputs = result['outputs'][:]
    best_count = len(gates)
    
    if best_count == 0:
        return result
    
    # Build wire_vec for current circuit
    def build_wire_vec(gates, outputs):
        num_gates = len(gates)
        total_wires = n + 1 + num_gates
        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
        return wire_vec
    
    wire_vec = build_wire_vec(gates, outputs)
    
    # Try to remove redundant gates
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        for k in range(len(gates) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Check if gate k is used by any output or other gate
            wire_k = n + 1 + k
            used = False
            for out_wire in outputs:
                if out_wire == wire_k:
                    used = True
                    break
            if not used:
                for other_gate in gates:
                    if other_gate[0] == wire_k or other_gate[1] == wire_k:
                        used = True
                        break
            
            if not used:
                # Remove this gate
                gates.pop(k)
                best_count -= 1
                improved = True
                # Rebuild wire_vec
                wire_vec = build_wire_vec(gates, outputs)
                break
        
        if not improved:
            break
    
    result = {'gates': gates, 'outputs': outputs}
    
    if _verify_solution(matrix, gates, outputs, depth_limit):
        return result
    
    return _greedy_shared_synthesis(matrix, depth_limit, time_budget=1.0)

def _advanced_synthesis(matrix, depth_limit=16, time_budget=20.0):
    """
    Advanced synthesis using basis reduction and smarter sharing.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # First, get a good initial solution
    initial = _greedy_shared_synthesis(matrix, depth_limit, time_budget=2.0)
    gates = initial['gates'][:]
    outputs = initial['outputs'][:]
    best_count = len(gates)
    
    if best_count == 0:
        return initial
    
    # Build wire_vec
    def build_wire_vec(gates):
        num_gates = len(gates)
        total_wires = n + 1 + num_gates
        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
        return wire_vec
    
    # Try to improve by finding better decompositions
    # For each output, try to find a shorter path using existing wires
    wire_vec = build_wire_vec(gates)
    vec_to_wire = {}
    for w in range(n + 1 + len(gates)):
        vec_to_wire[wire_vec[w].tobytes()] = w
    
    # For each output, try to find a 2-gate decomposition
    for i in range(m):
        if time.time() > deadline:
            break
        target = matrix[i]
        target_bytes = target.tobytes()
        
        if target_bytes in vec_to_wire:
            continue
        
        # Try all pairs of existing wires
        found_better = False
        for a in range(n + 1 + len(gates)):
            if time.time() > deadline:
                break
            for b in range(a + 1, n + 1 + len(gates)):
                candidate = (wire_vec[a] + wire_vec[b]) % 2
                if np.array_equal(candidate, target):
                    # This pair produces the target directly
                    # But we need to add a gate for this
                    # Only useful if it replaces multiple gates
                    # For now, just note it
                    pass
        # This approach is too slow for large cases
        # Instead, use a heuristic: try to split target into two parts
        # that are already computed
    
    # Try local search: replace pairs of gates
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        # Try to find a gate that can be eliminated
        for k in range(len(gates) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            wire_k = n + 1 + k
            vec_k = wire_vec[wire_k]
            
            # Check if vec_k can be expressed as XOR of two other wires
            # that are not wire_k itself
            for a in range(n + 1 + len(gates)):
                if a == wire_k:
                    continue
                for b in range(a + 1, n + 1 + len(gates)):
                    if b == wire_k:
                        continue
                    if (wire_vec[a] + wire_vec[b]) % 2 == vec_k:
                        # Check if a and b are "earlier" wires (lower index)
                        # to maintain topological order
                        if max(a, b) < wire_k:
                            # Replace gate k with a new gate using a, b
                            # But this doesn't reduce gate count
                            # Instead, if wire_k is only used once, we can inline it
                            pass
            break
        
        if not improved:
            break
    
    result = {'gates': gates, 'outputs': outputs}
    if _verify_solution(matrix, gates, outputs, depth_limit):
        return result
    
    return initial

def synthesize(matrix, depth_limit):
    """
    Entry point. Returns a dictionary with 'gates' and 'outputs'.
    """
    matrix = np.array(matrix, dtype=np.int64)
    m, n = matrix.shape
    
    # Start with baseline
    baseline = _baseline_synthesis(matrix, depth_limit)
    baseline_gates = len(baseline['gates'])
    
    if baseline_gates == 0:
        return baseline
    
    # Time budget management
    start_time = time.time()
    total_budget = 60.0  # Must finish within 60s per evaluation
    
    best_result = baseline
    best_count = baseline_gates
    
    # Try greedy shared synthesis
    try:
        greedy = _greedy_shared_synthesis(matrix, depth_limit, time_budget=3.0)
        if len(greedy['gates']) < best_count:
            best_result = greedy
            best_count = len(greedy['gates'])
    except Exception:
        pass
    
    # Try local search
    try:
        local = _local_search_synthesis(matrix, depth_limit, time_budget=10.0)
        if len(local['gates']) < best_count:
            best_result = local
            best_count = len(local['gates'])
    except Exception:
        pass
    
    # Safety margin: return if we've used too much time
    elapsed = time.time() - start_time
    if elapsed > 45.0:
        return best_result
    
    return best_result

# EVOLVE-BLOCK-END
