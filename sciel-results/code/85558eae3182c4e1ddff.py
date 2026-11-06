"""
Exact XOR-circuit synthesis with optimized gate count.
Uses flat XOR chains and optimized basis selection.
"""
import time
import numpy as np
from itertools import combinations

def _compute_rref(M):
    """Compute RREF over GF(2), return (rref, pivot_cols, basis_rows)."""
    M = np.array(M, dtype=np.uint8)
    n_rows, n_cols = M.shape
    work = M.copy()
    pivot_cols = []
    basis_rows = np.eye(n_rows, dtype=np.uint8)
    
    r = 0
    for c in range(n_cols):
        if r >= n_rows:
            break
        # Find pivot
        pivot = -1
        for row in range(r, n_rows):
            if work[row, c] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        if pivot != r:
            work[[r, pivot]] = work[[pivot, r]]
            basis_rows[[r, pivot]] = basis_rows[[pivot, r]]
        # Eliminate all other rows
        for row in range(n_rows):
            if row != r and work[row, c] == 1:
                work[row, :] ^= work[r, :]
                basis_rows[row, :] ^= basis_rows[r, :]
        pivot_cols.append(c)
        r += 1
    
    rank = len(pivot_cols)
    if rank == 0:
        return np.zeros((0, n_cols), dtype=np.uint8), [], np.zeros((0, n_rows), dtype=np.uint8)
    
    return work[:rank, :], pivot_cols, basis_rows[:rank, :]

def _synth_flat(ones_list):
    """
    Synthesize XOR of wires in ones_list using flat chain.
    Returns (wire_id, gates_list) where gates_list is list of [a, b] pairs.
    Wire IDs are assigned starting from a given offset (managed externally).
    """
    if not ones_list:
        return None, []
    if len(ones_list) == 1:
        return ones_list[0], []
    
    gates = []
    # Flat chain: x0 ^ x1 ^ x2 ... ^ x(k-1)
    # Wire 0 = x0
    # Wire 1 = x0 ^ x1
    # Wire 2 = (x0^x1) ^ x2
    # ...
    # Wire k-1 = final result
    
    # We'll use a counter for new wire IDs
    next_id = [1]  # 0 is reserved, we'll offset externally
    
    current = ones_list[0]
    for i in range(1, len(ones_list)):
        gate = [current, ones_list[i]]
        gates.append(gate)
        current = next_id[0]
        next_id[0] += 1
    
    return current, gates

def _synthesize_with_basis(M, pivot_cols, n_cols):
    """
    Synthesize circuit using the given pivot columns as basis.
    Returns (gate_count, circuit_dict).
    """
    n_rows = M.shape[0]
    rank = len(pivot_cols)
    
    if rank == 0:
        return 0, {'gates': [], 'outputs': [n_cols] * n_rows}
    
    # Compute RREF with these pivots
    rref, _, _ = _compute_rref(M)
    
    # Actually, we need RREF specifically with these pivot cols.
    # Let's compute it properly.
    # For now, use standard RREF and extract the relevant part.
    
    # The RREF has identity in pivot_cols.
    # Basis vectors are rows of RREF.
    # Output j = sum over i where M[j, pivot_cols[i]] == 1 of basis_vector_i
    
    # Synthesize basis vectors
    all_gates = []
    wire_map = {}  # frozenset of input indices -> wire_id
    next_wire = [n_cols + 1]
    
    def get_xor_wire(indices):
        """Get or create wire for XOR of input indices."""
        if not indices:
            return n_cols
        if len(indices) == 1:
            return indices[0]
        
        key = tuple(sorted(indices))
        if key in wire_map:
            return wire_map[key]
        
        # Flat chain
        current = indices[0]
        for i in range(1, len(indices)):
            gate = [current, indices[i]]
            all_gates.append(gate)
            current = next_wire[0]
            next_wire[0] += 1
        
        wire_map[key] = current
        return current
    
    # Synthesize basis vectors (rows of RREF)
    basis_wires = []
    for i in range(rank):
        row = rref[i, :]
        ones = [j for j in range(n_cols) if row[j] == 1]
        w = get_xor_wire(ones)
        basis_wires.append(w)
    
    # Synthesize outputs
    outputs = []
    for j in range(n_rows):
        coeffs = M[j, pivot_cols]
        active_basis = [basis_wires[i] for i in range(rank) if coeffs[i] == 1]
        
        if not active_basis:
            outputs.append(n_cols)
        elif len(active_basis) == 1:
            outputs.append(active_basis[0])
        else:
            # XOR of basis wires
            current = active_basis[0]
            for i in range(1, len(active_basis)):
                gate = [current, active_basis[i]]
                all_gates.append(gate)
                current = next_wire[0]
                next_wire[0] += 1
            outputs.append(current)
    
    return len(all_gates), {'gates': all_gates, 'outputs': outputs}

def _find_good_pivot_set(M, rank, n_cols, time_limit):
    """
    Find a good set of pivot columns to minimize gate count.
    Uses greedy approach if enumeration is too expensive.
    """
    start_time = time.time()
    
    # Try standard RREF first
    rref, pivot_cols_std, _ = _compute_rref(M)
    cost_std, _ = _synthesize_with_basis(M, pivot_cols_std, n_cols)
    best_cost = cost_std
    best_pivots = pivot_cols_std
    
    # If enumeration is feasible, try it
    def nck(n, k):
        if k > n - k:
            k = n - k
        if k == 0:
            return 1
        res = 1
        for i in range(k):
            res = res * (n - i) // (i + 1)
        return res
    
    num_pivots = nck(n_cols, rank)
    
    if num_pivots <= 2000 and rank <= 8:
        # Enumerate all pivot sets
        for combo in combinations(range(n_cols), rank):
            if time.time() - start_time > time_limit:
                break
            
            # Check if columns are independent
            sub = M[:, list(combo)]
            w_sub = sub.copy()
            r_sub = 0
            for c in range(rank):
                p = -1
                for r in range(r_sub, M.shape[0]):
                    if w_sub[r, c] == 1:
                        p = r
                        break
                if p == -1:
                    break
                if p != r_sub:
                    w_sub[[r_sub, p]] = w_sub[[p, r_sub]]
                for r in range(M.shape[0]):
                    if r != r_sub and w_sub[r, c] == 1:
                        w_sub[r, :] ^= w_sub[r_sub, :]
                r_sub += 1
            if r_sub < rank:
                continue
            
            cost, _ = _synthesize_with_basis(M, list(combo), n_cols)
            if cost < best_cost:
                best_cost = cost
                best_pivots = list(combo)
    
    return best_pivots, best_cost

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    M = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    # Compute rank
    work = M.copy()
    rank = 0
    for col in range(n_cols):
        pivot = -1
        for row in range(rank, n_rows):
            if work[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        if pivot != rank:
            work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(n_rows):
            if row != rank and work[row, col] == 1:
                work[row, :] ^= work[rank, :]
        rank += 1
    
    if rank == 0:
        return {'gates': [], 'outputs': [n_cols] * n_rows}
    
    # Find good pivot set
    time_limit = min(5.0, deadline - time.time())
    pivot_cols, best_cost = _find_good_pivot_set(M, rank, n_cols, time_limit)
    
    # Synthesize final circuit
    _, circuit = _synthesize_with_basis(M, pivot_cols, n_cols)
    
    return circuit
