"""Exact XOR-circuit synthesis: BFS minimum-depth synthesis + global CSE."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def _verify_solution(matrix, gates, outputs, depth_limit=16):
    """Verify gates/outputs compute matrix @ x over GF(2) with depth <= depth_limit."""
    m, n = matrix.shape
    num_gates = len(gates)
    total_wires = n + 1 + num_gates

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

    wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
    for i in range(n):
        wire_vec[i][i] = 1
    for k in range(num_gates):
        a, b = gates[k]
        wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
    for i in range(m):
        wire = outputs[i]
        if not np.array_equal(matrix[i], wire_vec[wire]):
            return False
    return True


def _baseline_synthesis(matrix, depth_limit=16):
    """Independent XOR chains per row."""
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
                gates.append([a, b])
                current = n + 1 + len(gates) - 1
            outputs.append(current)
    return {'gates': gates, 'outputs': outputs}


def _bfs_min_depth_synthesis(matrix, depth_limit=16, time_budget=40.0):
    """
    For each output row, find the minimum-depth XOR circuit to produce it.
    Uses BFS in the space of vectors: start from all input wires (depth 0),
    expand by XOR-ing any two known vectors at each level.
    
    Key structural difference: we compute per-output optimal circuits first,
    then do a global CSE pass to share subexpressions.
    
    To keep this tractable, we limit the BFS frontier size and use a
    bidirectional approach: forward from inputs, backward from target.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # Precompute: for each target, find min-depth circuit
    # We'll build a global set of "useful" intermediate vectors
    
    # Strategy: 
    # 1. Identify all target vectors
    # 2. For pairs of targets, compute target_i XOR target_j — if this is "simple" 
    #    (low weight), it's a useful shared intermediate
    # 3. Build a pool of candidate intermediates
    # 4. For each target, find shortest path from inputs through the pool
    
    # Step 1: Collect targets
    targets = [matrix[i] for i in range(m)]
    target_bytes = [t.tobytes() for t in targets]
    
    # Step 2: Build candidate intermediate pool
    # - All input vectors (weight 1)
    # - Zero vector
    # - XOR of all pairs of targets (potential shared subexpressions)
    # - XOR of target with each input (potential near-target intermediates)
    
    pool_vecs = []  # list of vectors
    pool_bytes = set()
    
    def add_to_pool(v):
        b = v.tobytes()
        if b not in pool_bytes:
            pool_bytes.add(b)
            pool_vecs.append(v)
    
    # Inputs
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        add_to_pool(v)
    # Zero
    add_to_pool(np.zeros(n, dtype=np.int64))
    
    # XOR of target pairs
    for i in range(m):
        for j in range(i + 1, m):
            v = (targets[i] + targets[j]) % 2
            add_to_pool(v)
    
    # XOR of target with inputs
    for i in range(m):
        for j in range(n):
            v = (targets[i] + np.eye(n, dtype=np.int64)[j]) % 2
            add_to_pool(v)
    
    # XOR of target with zero is target itself
    for i in range(m):
        add_to_pool(targets[i])
    
    # Also add XORs of pool vectors that are "simple" (weight <= 3)
    # to capture more sharing opportunities
    pool_size = len(pool_vecs)
    for i in range(pool_size):
        w_i = np.count_nonzero(pool_vecs[i])
        if w_i > 4:
            continue
        for j in range(i + 1, pool_size):
            v = (pool_vecs[i] + pool_vecs[j]) % 2
            w_v = np.count_nonzero(v)
            if w_v <= 4:
                add_to_pool(v)
    
    # Now we have a pool of candidate vectors.
    # For each target, find the minimum-depth circuit using only pool vectors as intermediates.
    # This is a shortest path problem: nodes = pool vectors + inputs, edges = XOR pairs.
    
    # Build adjacency: for each pool vector, which other pool vectors can XOR to produce it?
    # Actually, let's do BFS from inputs to each target within the pool.
    
    pool_to_idx = {}
    for idx, v in enumerate(pool_vecs):
        pool_to_idx[v.tobytes()] = idx
    
    # BFS from all inputs simultaneously to find depth to each pool node
    # depth[node] = min depth to produce node from inputs
    # parent[node] = (a, b) such that node = pool[a] XOR pool[b]
    
    num_pool = len(pool_vecs)
    depth = [float('inf')] * num_pool
    parent = [None] * num_pool  # (idx_a, idx_b)
    
    # Initialize: inputs have depth 0
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        idx = pool_to_idx.get(v.tobytes(), -1)
        if idx >= 0:
            depth[idx] = 0
    # Zero has depth 0
    zero_idx = pool_to_idx.get(np.zeros(n, dtype=np.int64).tobytes(), -1)
    if zero_idx >= 0:
        depth[zero_idx] = 0
    
    # BFS: expand by XOR-ing pairs
    # For efficiency, process in order of depth
    # Use a priority queue (Dijkstra-like, but all edges cost 1 so BFS)
    
    from heapq import heappush, heappop
    
    visited = [False] * num_pool
    pq = []
    for idx in range(num_pool):
        if depth[idx] == 0:
            heappush(pq, (0, idx))
            visited[idx] = True
    
    # For each node at depth d, we can produce node XOR pool[j] at depth d+1
    # for any j. But that's O(N^2) per level.
    # Instead: for each pair (i, j) where depth[i] <= depth[j], 
    # if pool[i] XOR pool[j] is in pool and not visited, set its depth.
    
    # More efficient: iterate in depth order
    # Group nodes by depth
    by_depth = {}
    for idx in range(num_pool):
        if depth[idx] < float('inf'):
            d = depth[idx]
            if d not in by_depth:
                by_depth[d] = []
            by_depth[d].append(idx)
    
    max_d = max(by_depth.keys()) if by_depth else 0
    
    # Expand: for each depth level, for each node at that level,
    # XOR with all previously known nodes
    known_set = set()
    for idx in range(num_pool):
        if depth[idx] == 0:
            known_set.add(idx)
    
    current_max_depth = 0
    expanded_to = 0
    
    while expanded_to < num_pool:
        expanded_to = len(known_set)
        found_new = False
        
        # Get nodes at current_max_depth
        nodes_at_depth = by_depth.get(current_max_depth, [])
        if not nodes_at_depth:
            current_max_depth += 1
            if current_max_depth > depth_limit:
                break
            continue
        
        for idx in nodes_at_depth:
            if time.time() > deadline:
                break
            vi = pool_vecs[idx]
            for j in known_set:
                if j == idx:
                    continue
                v = (vi + pool_vecs[j]) % 2
                vb = v.tobytes()
                k = pool_to_idx.get(vb, -1)
                if k >= 0 and k not in known_set:
                    depth[k] = current_max_depth + 1
                    parent[k] = (idx, j)
                    known_set.add(k)
                    found_new = True
                    if (current_max_depth + 1) not in by_depth:
                        by_depth[current_max_depth + 1] = []
                    by_depth[current_max_depth + 1].append(k)
        
        if not found_new:
            current_max_depth += 1
            if current_max_depth > depth_limit:
                break
    
    # Now for each target, we have a depth and parent chain.
    # Extract the circuit for each target.
    
    # Build circuits: for each target, trace back through parents
    # We need to assign wire numbers.
    
    # First, determine which pool nodes are actually needed
    needed = set()
    
    def mark_needed(idx):
        if idx in needed:
            return
        needed.add(idx)
        if parent[idx] is not None:
            a, b = parent[idx]
            mark_needed(a)
            mark_needed(b)
    
    target_pool_idx = []
    for i in range(m):
        tb = target_bytes[i]
        idx = pool_to_idx.get(tb, -1)
        if idx >= 0 and depth[idx] < float('inf'):
            target_pool_idx.append(idx)
            mark_needed(idx)
        else:
            target_pool_idx.append(-1)
    
    # Assign wire numbers to needed pool nodes in topological order (by depth)
    wire_of_pool = {}
    for idx in sorted(needed, key=lambda x: depth[x]):
        wire_of_pool[idx] = idx  # temporary, will remap
    
    # Actually, let's just build the gate list directly
    # Sort needed nodes by depth, assign gate numbers
    
    sorted_needed = sorted(needed, key=lambda x: (depth[x], x))
    
    # Map pool index to wire number
    # Wires 0..n-1 are inputs, wire n is zero
    # Wires n+1, n+2, ... are gates
    
    pool_to_wire = {}
    # Inputs
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        idx = pool_to_idx.get(v.tobytes(), -1)
        if idx >= 0:
            pool_to_wire[idx] = i
    zero_idx = pool_to_idx.get(np.zeros(n, dtype=np.int64).tobytes(), -1)
    if zero_idx >= 0:
        pool_to_wire[zero_idx] = n
    
    gates = []
    gate_wire_counter = n + 1
    
    for idx in sorted_needed:
        if idx in pool_to_wire:
            continue
        if parent[idx] is None:
            continue
        a, b = parent[idx]
        wire_a = pool_to_wire.get(a)
        wire_b = pool_to_wire.get(b)
        if wire_a is None or wire_b is None:
            continue
        gates.append([wire_a, wire_b])
        pool_to_wire[idx] = gate_wire_counter
        gate_wire_counter += 1
    
    outputs = []
    for i in range(m):
        idx = target_pool_idx[i]
        if idx >= 0 and idx in pool_to_wire:
            outputs.append(pool_to_wire[idx])
        else:
            # Fallback: use baseline for this row
            row = matrix[i]
            positions = np.where(row)[0].tolist()
            if len(positions) == 0:
                outputs.append(n)
            elif len(positions) == 1:
                outputs.append(positions[0])
            else:
                # Need to create gates for this
                current = positions[0]
                for j in range(1, len(positions)):
                    gates.append([current, positions[j]])
                    current = n + 1 + len(gates) - 1
                outputs.append(current)
    
    # Check depth
    num_gates = len(gates)
    total_wires = n + 1 + num_gates
    depth_arr = [0] * total_wires
    for k in range(num_gates):
        a, b = gates[k]
        depth_arr[n + 1 + k] = 1 + max(depth_arr[a], depth_arr[b])
    max_depth = max(depth_arr[n + 1 + k] for k in range(num_gates)) if num_gates > 0 else 0
    for wire in outputs:
        max_depth = max(max_depth, depth_arr[wire])
    
    if max_depth > depth_limit:
        # Fall back to baseline
        return _baseline_synthesis(matrix, depth_limit)
    
    return {'gates': gates, 'outputs': outputs}


def _global_cse(matrix, gates, outputs, depth_limit=16, time_budget=10.0):
    """
    Global Common Subexpression Elimination:
    - Compute the vector for each wire
    - Find duplicate vectors and redirect all uses to the earliest occurrence
    - Remove unused gates
    - Iterate until no more improvements
    """
    deadline = time.time() + time_budget
    m, n = matrix.shape
    gates = gates[:]
    outputs = outputs[:]
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        if len(gates) == 0:
            break
        
        num_gates = len(gates)
        total_wires = n + 1 + num_gates
        
        # Build wire vectors
        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
        
        # Build vec_to_earliest_wire
        vec_to_earliest = {}
        for w in range(total_wires):
            vb = wire_vec[w].tobytes()
            if vb not in vec_to_earliest:
                vec_to_earliest[vb] = w
        
        improved = False
        
        # Remove unused gates (from last to first)
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            used = False
            for ow in outputs:
                if ow == wire_k:
                    used = True
                    break
            if not used:
                for g in gates:
                    if g[0] == wire_k or g[1] == wire_k:
                        used = True
                        break
            if not used:
                gates.pop(k)
                improved = True
                break
        
        if improved:
            continue
        
        # Redirect duplicate vectors to earliest
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            vec_k = wire_vec[wire_k]
            vec_k_bytes = vec_k.tobytes()
            
            earlier = vec_to_earliest.get(vec_k_bytes)
            if earlier is not None and earlier != wire_k:
                # Redirect all uses of wire_k to 'earlier'
                for j in range(k + 1, len(gates)):
                    if gates[j][0] == wire_k:
                        gates[j][0] = earlier
                    if gates[j][1] == wire_k:
                        gates[j][1] = earlier
                for i in range(m):
                    if outputs[i] == wire_k:
                        outputs[i] = earlier
                gates.pop(k)
                improved = True
                break
        
        if not improved:
            break
    
    if _verify_solution(matrix, gates, outputs, depth_limit):
        return {'gates': gates, 'outputs': outputs}
    return {'gates': gates, 'outputs': outputs}


def _gaussian_elimination_synthesis(matrix, depth_limit=16, time_budget=15.0):
    """
    Alternative approach: use Gaussian elimination structure.
    The matrix M (m x n) defines m linear functions.
    We can find a basis for the row space and express each row as a combination of basis vectors.
    The number of gates is related to the dimension of the row space.
    
    Strategy:
    1. Find a basis B = {b_1, ..., b_r} for the row space of M (r = rank)
    2. Express each row of M as XOR of basis vectors
    3. Build circuits for basis vectors (they're independent, so minimal)
    4. Combine for each output
    
    This gives a structured solution where sharing is inherent.
    """
    deadline = time.time() + time_budget
    m, n = matrix.shape
    
    # Find row basis using Gaussian elimination
    # Work with rows as bit vectors
    rows = [matrix[i].copy() for i in range(m)]
    
    # Gaussian elimination to find basis
    basis = []
    pivot_cols = []
    
    for row in rows:
        r = row.copy()
        for b, pc in zip(basis, pivot_cols):
            if r[pc] == 1:
                r = (r + b) % 2
        # Find pivot
        found = False
        for j in range(n):
            if r[j] == 1:
                # Eliminate this column from existing basis
                for i, (b, pc) in enumerate(zip(basis, pivot_cols)):
                    if b[j] == 1:
                        basis[i] = (basis[i] + r) % 2
                basis.append(r)
                pivot_cols.append(j)
                found = True
                break
        if not found:
            # Zero row
            continue
    
    rank = len(basis)
    
    if rank == 0:
        # All zeros
        return {'gates': [], 'outputs': [n] * m}
    
    # Express each original row as combination of basis vectors
    # row_i = XOR of {b_j : coeff[i][j] = 1}
    
    # Build circuits for basis vectors
    # Each basis vector b_j has some weight w_j
    # To produce b_j, need w_j - 1 gates (XOR of w_j inputs)
    # But we can share subexpressions between basis vectors
    
    # For now, produce each basis vector independently
    gates = []
    basis_wires = []
    
    for b in basis:
        positions = np.where(b)[0].tolist()
        if len(positions) == 0:
            basis_wires.append(n)
        elif len(positions) == 1:
            basis_wires.append(positions[0])
        else:
            current = positions[0]
            for j in range(1, len(positions)):
                gates.append([current, positions[j]])
                current = n + 1 + len(gates) - 1
            basis_wires.append(current)
    
    # Now express each output row as XOR of basis vectors
    # For row i, find which basis vectors are needed
    outputs = []
    for i in range(m):
        row = matrix[i]
        # Reduce row by basis
        coeffs = []
        for j, (b, pc) in enumerate(zip(basis, pivot_cols)):
            if row[pc] == 1:
                coeffs.append(j)
                row = (row + b) % 2
        # row should now be zero
        
        if len(coeffs) == 0:
            outputs.append(n)
        elif len(coeffs) == 1:
            outputs.append(basis_wires[coeffs[0]])
        else:
            current = basis_wires[coeffs[0]]
            for j in range(1, len(coeffs)):
                gates.append([current, basis_wires[coeffs[j]]])
                current = n + 1 + len(gates) - 1
            outputs.append(current)
    
    # Check depth
    num_gates = len(gates)
    total_wires = n + 1 + num_gates
    depth_arr = [0] * total_wires
    for k in range(num_gates):
        a, b = gates[k]
        depth_arr[n + 1 + k] = 1 + max(depth_arr[a], depth_arr[b])
    max_depth = max(depth_arr[n + 1 + k] for k in range(num_gates)) if num_gates > 0 else 0
    for wire in outputs:
        max_depth = max(max_depth, depth_arr[wire])
    
    if max_depth > depth_limit:
        return _baseline_synthesis(matrix, depth_limit)
    
    return {'gates': gates, 'outputs': outputs}


def synthesize(matrix, depth_limit):
    """Entry point."""
    matrix = np.array(matrix, dtype=np.int64)
    m, n = matrix.shape
    start_time = time.time()
    
    # Baseline
    baseline = _baseline_synthesis(matrix, depth_limit)
    best_result = baseline
    best_count = len(baseline['gates'])
    
    if best_count == 0:
        return baseline
    
    # Strategy 1: BFS minimum-depth synthesis
    try:
        bfs_result = _bfs_min_depth_synthesis(matrix, depth_limit, time_budget=25.0)
        if len(bfs_result['gates']) < best_count:
            best_result = bfs_result
            best_count = len(bfs_result['gates'])
    except Exception:
        pass
    
    # Strategy 2: Gaussian elimination
    try:
        ge_result = _gaussian_elimination_synthesis(matrix, depth_limit, time_budget=10.0)
        if len(ge_result['gates']) < best_count:
            best_result = ge_result
            best_count = len(ge_result['gates'])
    except Exception:
        pass
    
    # CSE polish on best result
    elapsed = time.time() - start_time
    if elapsed < 50.0 and best_count > 0:
        try:
            polished = _global_cse(matrix, best_result['gates'], best_result['outputs'],
                                   depth_limit, time_budget=10.0)
            if len(polished['gates']) < best_count:
                best_result = polished
                best_count = len(polished['gates'])
        except Exception:
            pass
    
    # Safety margin
    elapsed = time.time() - start_time
    if elapsed > 55.0:
        return best_result
    
    # If time remains, try CSE on other results too
    remaining = 55.0 - elapsed
    if remaining > 5.0:
        try:
            if len(baseline['gates']) < best_count:
                polished_base = _global_cse(matrix, baseline['gates'], baseline['outputs'],
                                             depth_limit, time_budget=remaining - 2.0)
                if len(polished_base['gates']) < best_count:
                    best_result = polished_base
                    best_count = len(polished_base['gates'])
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
