import numpy as np
import time

def _popcount(x):
    return int(bin(x).count('1'))

def synthesize(matrix, depth_limit):
    start = time.time()
    deadline = start + 85.0
    
    matrix = np.array(matrix, dtype=np.uint8) % 2
    m, n = matrix.shape
    
    # Convert rows to ints
    row_ints = []
    for i in range(m):
        v = 0
        for j in range(n):
            if matrix[i, j] & 1:
                v |= (1 << j)
        row_ints.append(v)
        
    # 1. Greedy Minimum Weight Basis Construction
    # We want a basis for the row space.
    # We process columns 0..n-1.
    # For each column, find a row with bit set that has minimum popcount.
    
    candidates = list(row_ints)
    basis = []
    pivot_cols = []
    
    # We need to track which original rows are which? 
    # Actually, we just need the basis vectors.
    # But we need to eliminate them from the candidate pool to ensure independence.
    
    # Standard Gaussian Elimination structure, but with greedy pivot selection.
    # We maintain a list of active vectors.
    active = [v for v in row_ints if v != 0]
    
    for col in range(n):
        bit = 1 << col
        # Find candidate with min popcount that has this bit set
        min_idx = -1
        min_weight = float('inf')
        for i, v in enumerate(active):
            if v & bit:
                w = _popcount(v)
                if w < min_weight:
                    min_weight = w
                    min_idx = i
        
        if min_idx == -1:
            continue
            
        pivot_vec = active[min_idx]
        basis.append(pivot_vec)
        pivot_cols.append(col)
        
        # Eliminate this bit from all other active vectors
        new_active = []
        for i, v in enumerate(active):
            if i == min_idx:
                continue
            if v & bit:
                v = v ^ pivot_vec
            if v != 0:
                new_active.append(v)
        active = new_active
        
    r = len(basis)
    
    if r == 0:
        return {'gates': [], 'outputs': [n] * m} # All zero rows, output zero wire (n)

    # 2. Local Search to Improve Basis
    # Objective: Minimize Sum(popcount(basis_vecs)) + Sum(popcount(combo_vecs))
    # combo_vecs are the coefficients of original rows in terms of basis.
    
    def compute_combos(basis):
        # Solve M = C * B
        # B is r x n (list of ints)
        # M is m x n (row_ints)
        # C is m x r
        combos = []
        for j in range(m):
            v = row_ints[j]
            c = 0
            # Reduce v using basis
            # Basis is in row echelon form? 
            # The greedy construction maintains echelon form (pivot col increases).
            # So we can reduce from last basis vector to first.
            for i in range(r - 1, -1, -1):
                pv = basis[i]
                # Pivot col of basis[i] is the lowest set bit? 
                # In our construction, we picked a vector with bit `col` set, and eliminated it from others.
                # So `col` is a pivot. Is it the lowest set bit?
                # Not necessarily. The vector might have lower bits set if they were not pivots.
                # However, since we process cols 0..n-1, if a lower bit was a pivot, it would have been eliminated.
                # So the lowest set bit of basis[i] IS the pivot col.
                pcol = (pv & -pv).bit_length() - 1
                if v & (1 << pcol):
                    v ^= pv
                    c |= (1 << i)
            # v should be 0
            combos.append(c)
        return combos
        
    def total_cost(basis):
        combos = compute_combos(basis)
        cost = 0
        for b in basis:
            cost += _popcount(b)
        for c in combos:
            cost += _popcount(c)
        return cost

    current_basis = basis[:]
    current_cost = total_cost(current_basis)
    
    # Local Search Loop
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try XORing each basis vector with every other
        for i in range(r):
            for j in range(r):
                if i == j:
                    continue
                new_bi = current_basis[i] ^ current_basis[j]
                if new_bi == 0:
                    continue
                # Tentatively apply
                old_bi = current_basis[i]
                current_basis[i] = new_bi
                
                new_cost = total_cost(current_basis)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    current_basis[i] = old_bi

    # 3. Build Circuit
    zero_wire = n
    gates = []
    wire_count = n + 1
    
    def xor_gate(a, b):
        nonlocal wire_count
        w = wire_count
        wire_count += 1
        gates.append([a, b])
        return w
        
    def compute_vector(vec_int):
        bits = []
        for j in range(n):
            if vec_int & (1 << j):
                bits.append(j)
        if not bits:
            return zero_wire
        if len(bits) == 1:
            return bits[0]
        w = bits[0]
        for b in bits[1:]:
            w = xor_gate(w, b)
        return w

    # Compute basis wires
    basis_wires = []
    for i in range(r):
        basis_wires.append(compute_vector(current_basis[i]))
        
    # Compute output wires
    combos = compute_combos(current_basis)
    outputs = []
    for j in range(m):
        c = combos[j]
        idxs = [i for i in range(r) if c & (1 << i)]
        if not idxs:
            outputs.append(zero_wire)
        elif len(idxs) == 1:
            outputs.append(basis_wires[idxs[0]])
        else:
            w = basis_wires[idxs[0]]
            for i in idxs[1:]:
                w = xor_gate(w, basis_wires[i])
            outputs.append(w)
            
    # Verification
    wire_val = [0] * wire_count
    for j in range(n):
        wire_val[j] = 1 << j
    for k in range(len(gates)):
        a, b = gates[k]
        wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
        
    ok = True
    for j in range(m):
        if wire_val[outputs[j]] != row_ints[j]:
            ok = False
            break
            
    if not ok:
        # Fallback: Direct synthesis (no sharing)
        gates = []
        wire_count = n + 1
        def xor_gate2(a, b):
            nonlocal wire_count
            w = wire_count
            wire_count += 1
            gates.append([a, b])
            return w
            
        outputs = []
        for j in range(m):
            bits = [k for k in range(n) if row_ints[j] & (1 << k)]
            if not bits:
                outputs.append(zero_wire)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                w = bits[0]
                for b in bits[1:]:
                    w = xor_gate2(w, b)
                outputs.append(w)
                
    return {'gates': gates, 'outputs': outputs}
