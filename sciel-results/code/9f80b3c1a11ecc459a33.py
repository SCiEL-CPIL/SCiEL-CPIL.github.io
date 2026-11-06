# EVOLVE-BLOCK-START
import time
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0  # safety margin before 60s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse matrix to row masks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Column weights
    col_weights = [0] * n
    for mask in row_masks:
        mm = mask
        while mm:
            b = mm & (-mm)
            col_weights[b.bit_length() - 1] += 1
            mm ^= b

    def elapsed():
        return time.time() - start_time

    def try_synthesize(col_order):
        """
        Try to synthesize using a specific column ordering for pivot selection.
        The column order determines which bit we prefer to pivot on.
        """
        # Map each column to its priority (lower = preferred)
        col_priority = {}
        for priority, col in enumerate(col_order):
            col_priority[col] = priority
        
        # Gaussian elimination with custom pivot selection
        basis_dict = {}  # pivot_bit -> basis_mask
        basis_list = []
        basis_mask_to_idx = {}
        row_reps = []
        
        for i in range(m):
            mask = row_masks[i]
            if mask == 0:
                row_reps.append([])
                continue
            
            cm = mask
            rep = []
            
            # Reduce using existing basis
            # Find the pivot with lowest priority (most preferred)
            while cm:
                # Find the set bit with lowest priority
                best_bit = -1
                best_prio = float('inf')
                mm = cm
                while mm:
                    b = mm & (-mm)
                    bit = b.bit_length() - 1
                    prio = col_priority.get(bit, float('inf'))
                    if prio < best_prio:
                        best_prio = prio
                        best_bit = bit
                    mm ^= b
                
                if best_bit in basis_dict:
                    b_mask = basis_dict[best_bit]
                    idx = basis_mask_to_idx[b_mask]
                    rep.append(idx)
                    cm ^= b_mask
                else:
                    break
            
            if cm:
                # Find pivot for new basis vector (lowest priority set bit)
                best_bit = -1
                best_prio = float('inf')
                mm = cm
                while mm:
                    b = mm & (-mm)
                    bit = b.bit_length() - 1
                    prio = col_priority.get(bit, float('inf'))
                    if prio < best_prio:
                        best_prio = prio
                        best_bit = bit
                    mm ^= b
                
                b_idx = len(basis_list)
                basis_dict[best_bit] = cm
                basis_list.append(cm)
                basis_mask_to_idx[cm] = b_idx
                rep.append(b_idx)
            
            row_reps.append(rep)
        
        if not basis_list:
            return {'gates': [], 'outputs': [n] * m}
        
        # Build circuit with global memoization
        next_wire = n + 1
        gates = []
        memo = {}
        
        for j in range(n):
            memo[1 << j] = j
        memo[0] = n
        
        def add_gate(a, b):
            nonlocal next_wire
            gates.append([a, b])
            w = next_wire
            next_wire += 1
            return w
        
        def compute_xor(mask):
            if mask in memo:
                return memo[mask]
            if mask == 0:
                return n
            if mask & (mask - 1) == 0:
                return mask.bit_length() - 1
            
            # Balanced split
            bits = []
            mm = mask
            while mm:
                b = mm & (-mm)
                bits.append(b.bit_length() - 1)
                mm ^= b
            
            mid = len(bits) // 2
            mask1 = 0
            for b in bits[:mid]:
                mask1 |= (1 << b)
            mask2 = mask ^ mask1
            
            w1 = compute_xor(mask1)
            w2 = compute_xor(mask2)
            
            if w1 == w2:
                result = n
            else:
                result = add_gate(w1, w2)
            
            memo[mask] = result
            return result
        
        # Build basis wires
        basis_wires = []
        for b_mask in basis_list:
            w = compute_xor(b_mask)
            basis_wires.append(w)
        
        # Build outputs
        outputs = []
        for i in range(m):
            rep = row_reps[i]
            if not rep:
                outputs.append(n)
            elif len(rep) == 1:
                outputs.append(basis_wires[rep[0]])
            else:
                target_mask = row_masks[i]
                if target_mask in memo:
                    outputs.append(memo[target_mask])
                else:
                    wires = [basis_wires[idx] for idx in rep]
                    while len(wires) > 1:
                        new_wires = []
                        for j in range(0, len(wires), 2):
                            if j + 1 < len(wires):
                                new_wires.append(add_gate(wires[j], wires[j+1]))
                            else:
                                new_wires.append(wires[j])
                        wires = new_wires
                    outputs.append(wires[0])
                    memo[target_mask] = wires[0]
        
        return {'gates': gates, 'outputs': outputs}
    
    # Generate candidate column orderings
    candidate_orders = []
    
    # Natural order
    candidate_orders.append(list(range(n)))
    
    # Sorted by column weight ascending (prefer low-weight columns as pivots)
    candidate_orders.append(sorted(range(n), key=lambda j: col_weights[j]))
    
    # Sorted by column weight descending
    candidate_orders.append(sorted(range(n), key=lambda j: -col_weights[j]))
    
    # Random permutations
    rng = random.Random(42)
    num_random = min(15, max(0, int((DEADLINE - elapsed()) * 8)))
    for _ in range(num_random):
        perm = list(range(n))
        rng.shuffle(perm)
        candidate_orders.append(perm)
    
    best_result = None
    best_gate_count = float('inf')
    
    for order in candidate_orders:
        if elapsed() > DEADLINE - 5:
            break
        
        result = try_synthesize(order)
        if result is None:
            continue
        
        gate_count = len(result['gates'])
        if gate_count < best_gate_count:
            best_gate_count = gate_count
            best_result = result
        
        if best_gate_count == 0:
            break
    
    if best_result is None:
        best_result = try_synthesize(list(range(n)))
    
    if best_result is None:
        return {'gates': [], 'outputs': [n] * m}
    
    return best_result
# EVOLVE-BLOCK-END
