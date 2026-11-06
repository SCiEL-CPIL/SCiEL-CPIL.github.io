# EVOLVE-BLOCK-START
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    # 90s wall clock limit. Use a generous internal deadline to allow more search.
    DEADLINE = start_time + 80.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse rows as bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Wire assignment: inputs 0..n-1, const-zero n, gates n+1...
    next_wire = n + 1
    gates = []
    # memo: bitmask -> wire_id
    memo = {}
    # Initialize with inputs
    for j in range(n):
        memo[1 << j] = j
    # Const zero
    memo[0] = n

    def elapsed():
        return time.time() - start_time

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    def build_from_bits(bits_list):
        """
        Takes a list of input wire indices (0..n-1).
        Builds a balanced XOR tree, memoizing the mask for each new gate.
        Returns the final wire.
        """
        if not bits_list:
            return n
        if len(bits_list) == 1:
            return bits_list[0]

        wires = []
        masks = []
        for b in bits_list:
            wires.append(b)
            masks.append(1 << b)

        while len(wires) > 1:
            new_wires = []
            new_masks = []
            for i in range(0, len(wires), 2):
                if i + 1 < len(wires):
                    a_w = wires[i]
                    a_m = masks[i]
                    b_w = wires[i + 1]
                    b_m = masks[i + 1]

                    new_m = a_m ^ b_m
                    # Check if new_m is already in memo
                    if new_m in memo:
                        # Reuse existing wire
                        new_wires.append(memo[new_m])
                        new_masks.append(new_m)
                    else:
                        new_w = add_gate(a_w, b_w)
                        memo[new_m] = new_w
                        new_wires.append(new_w)
                        new_masks.append(new_m)
                else:
                    new_wires.append(wires[i])
                    new_masks.append(masks[i])
            wires = new_wires
            masks = new_masks

        return wires[0]

    def compute_mask(mask):
        """
        Compute wire for mask using memoization and greedy nearest-neighbor search.
        """
        if mask in memo:
            return memo[mask]

        # If we hit the deadline, fallback to direct tree construction
        if elapsed() > DEADLINE - 5.0:
            bits = [j for j in range(n) if (mask >> j) & 1]
            w = build_from_bits(bits)
            memo[mask] = w
            return w

        # Find best M in memo to minimize popcount(mask ^ M)
        # We iterate through all keys in memo.
        best_M = 0
        best_rem = mask
        min_pop = mask.bit_count()

        # Optimization: If mask has small popcount, the best M is likely 0 or a subset.
        # Scanning all keys is O(|memo|).
        for M, w_m in memo.items():
            rem = mask ^ M
            pc = rem.bit_count()
            if pc < min_pop:
                min_pop = pc
                best_M = M
                best_rem = rem
                if pc == 0:
                    break
                # If pc == 1, rem is a single bit, which is in memo.
                # So we can break early as this is the best possible non-zero remainder
                # that allows a 1-gate solution (XOR with input).
                if pc == 1:
                    break

        # Check if best_rem is in memo
        if best_rem in memo:
            w_rem = memo[best_rem]
        else:
            # If best_rem is not in memo, build a tree from its bits
            bits = [j for j in range(n) if (best_rem >> j) & 1]
            w_rem = build_from_bits(bits)

        w_m = memo[best_M]
        w_new = add_gate(w_m, w_rem)
        memo[mask] = w_new
        return w_new

    # 1. Parse matrix to masks (already done)
    # 2. Compute Basis for Row Space using Gaussian Elimination
    # We want a basis that minimizes the cost of synthesizing the basis vectors themselves.
    # A standard basis (RREF) has pivot rows with specific structure.
    # However, for XOR circuits, it is often better to have a basis of low-weight vectors.
    # We can try to find a "minimum weight" basis or at least a basis that leads to efficient synthesis.
    # Given the constraints, we will stick to a standard basis but order them to maximize sharing.
    
    # Let's find a basis for the row space.
    # basis_masks: list of masks that form a basis.
    # We also need to express every row as a combination of these basis masks.
    
    basis_masks = []
    # pivot_map: pivot_bit -> index in basis_masks
    pivot_map = {}
    
    # To get a good basis, we can sort the input rows by popcount (ascending) and pick the first independent ones.
    # This tends to produce a basis with lower weights.
    sorted_indices_by_weight = sorted(range(m), key=lambda i: row_masks[i].bit_count())
    
    # We need to track the representation of each original row in terms of the basis.
    # rep[i] will be a bitmask of size len(basis_masks) (conceptually, but we don't know size ahead).
    # Actually, since basis grows, we can store the representation as a list of indices or a dynamic mask.
    # Let's use a list of indices for each row.
    
    # However, for the synthesis, we just need to compute the wire for each row.
    # If we have a basis, we can compute the wire for each basis vector, then combine them for each row.
    
    # Let's build the basis and the representations.
    
    # We will maintain a list of basis vectors.
    # For each vector, we also store its "representation" in terms of the *previous* basis? No, that's circular.
    
    # Standard approach:
    # 1. Find basis vectors B_1, ..., B_k.
    # 2. For each row R_i, find coefficients c_{i,j} such that R_i = XOR(c_{i,j} * B_j).
    # 3. Synthesize each B_j into a wire W_j.
    # 4. Synthesize each R_i by XORing the W_j where c_{i,j}=1.
    
    # To find the basis and representations efficiently:
    # We can process rows in some order.
    # Let's process rows in order of increasing popcount.
    
    basis_list = [] # List of masks
    # rep_list: for each row processed, store the list of indices of basis vectors used.
    # But we need to be careful: when we add a new basis vector, it might change the representation of previous rows?
    # No, if we maintain a reduced basis (RREF), the representation is unique and stable.
    
    # Let's implement a simple Gaussian elimination that maintains the basis in RREF form.
    # This ensures that each basis vector has a unique pivot, and no other basis vector has that bit set.
    
    # pivot_to_basis: pivot_bit -> mask
    pivot_to_basis = {}
    
    # row_reps: list of lists of pivot bits? Or indices?
    # Let's store the representation as a set of pivot bits.
    row_reps = [None] * m
    
    for i in range(m):
        mask = row_masks[i]
        if mask == 0:
            row_reps[i] = set()
            continue
        
        current_mask = mask
        current_rep = set()
        
        # Reduce current_mask by existing basis
        # Iterate through pivots in descending order (or any consistent order)
        # For efficiency, we can just check all set bits in current_mask.
        
        changed = True
        while changed:
            changed = False
            # Find the highest set bit in current_mask
            if current_mask == 0:
                break
            highest_bit = current_mask.bit_length() - 1
            if highest_bit in pivot_to_basis:
                b_mask = pivot_to_basis[highest_bit]
                current_mask ^= b_mask
                current_rep.add(highest_bit)
                changed = True
        
        if current_mask != 0:
            # This is a new basis vector
            # Add it to basis
            pivot = current_mask.bit_length() - 1
            pivot_to_basis[pivot] = current_mask
            current_rep.add(pivot)
            
            # We need to update the representations of previous rows?
            # If we are building RREF, we should eliminate this new pivot from previous basis vectors.
            # But that would change their masks, which complicates things if we've already synthesized them.
            # Alternative: Don't maintain RREF. Just maintain a triangular basis.
            # If we use a triangular basis, the representation is unique but not necessarily minimal weight.
            
            # Let's stick to a simpler strategy:
            # We don't strictly need a basis for the synthesis if we use the greedy memoization approach.
            # The greedy approach (compute_mask) already finds good combinations.
            # The inspiration suggests using RREF to minimize total Hamming weight.
            
            # However, implementing full RREF with dynamic synthesis is complex.
            # Let's try a hybrid:
            # 1. Identify independent rows.
            # 2. Synthesize independent rows first (using greedy).
            # 3. Synthesize dependent rows using the greedy approach, which will likely find the combination of independent rows.
            
            # Actually, the current greedy approach already does this implicitly.
            # The memo stores all computed masks.
            # When a dependent row is processed, compute_mask finds the best M in memo.
            # If the independent rows are in memo, it will find the combination.
            
            # So, the key is to process independent rows first?
            # Or process rows in an order that maximizes sharing?
            
            # The current code sorts by popcount descending.
            # Maybe sorting by popcount ascending is better for basis?
            # But for greedy synthesis, high popcount rows might benefit more from sharing?
            
            # Let's try to identify the basis rows and process them first.
            
            # We'll mark which rows are in the basis.
            # But we need to do this before the main synthesis loop.
            
            # Let's restructure:
            # 1. Find a basis for the row space.
            # 2. Get the list of basis row indices.
            # 3. Process these rows first (sorted by popcount descending, as before).
            # 4. Process the remaining rows.
            
            # This is a change in the main loop logic.
            
            # Let's implement the basis finding first.
            
            pass # We'll do this before the main loop.

    # --- Revised Plan ---
    # 1. Find basis rows.
    # 2. Sort basis rows by popcount descending.
    # 3. Sort non-basis rows by popcount descending.
    # 4. Process basis rows, then non-basis rows.
    
    # Find basis
    pivot_to_row = {}
    basis_row_indices = []
    
    # We need to reduce rows to find which are independent.
    # We'll do this in the order of popcount descending? Or ascending?
    # For basis finding, order doesn't change the span, but it changes which rows are chosen as basis.
    # Choosing low-weight rows as basis might be better for synthesis.
    # So let's sort by popcount ascending for basis selection.
    
    sorted_indices_for_basis = sorted(range(m), key=lambda i: row_masks[i].bit_count())
    
    for idx in sorted_indices_for_basis:
        mask = row_masks[idx]
        if mask == 0:
            continue
            
        current_mask = mask
        # Reduce by existing basis
        for pivot, row_idx in pivot_to_row.items():
            b_mask = row_masks[row_idx]
            # Check if pivot bit is set in current_mask
            if (current_mask >> pivot) & 1:
                current_mask ^= b_mask
        
        if current_mask != 0:
            # New basis vector
            pivot = current_mask.bit_length() - 1
            pivot_to_row[pivot] = idx
            basis_row_indices.append(idx)
    
    # Now we have basis_row_indices.
    # We want to process these first.
    # Sort them by popcount descending (as in the original code) to maximize sharing among them?
    # Or ascending?
    # The original code used descending. Let's stick to descending for the basis rows.
    # Then process non-basis rows, also sorted by popcount descending.
    
    basis_set = set(basis_row_indices)
    non_basis_indices = [i for i in range(m) if i not in basis_set]
    
    # Sort basis indices by popcount descending
    basis_indices_sorted = sorted(basis_row_indices, key=lambda i: row_masks[i].bit_count(), reverse=True)
    # Sort non-basis indices by popcount descending
    non_basis_indices_sorted = sorted(non_basis_indices, key=lambda i: row_masks[i].bit_count(), reverse=True)
    
    # Combined processing order
    processing_order = basis_indices_sorted + non_basis_indices_sorted
    
    outputs = [None] * m

    for idx in processing_order:
        if elapsed() > DEADLINE - 10.0:
            # Fast path: direct tree construction without search
            mask = row_masks[idx]
            if mask in memo:
                outputs[idx] = memo[mask]
            else:
                bits = [j for j in range(n) if (mask >> j) & 1]
                w = build_from_bits(bits)
                memo[mask] = w
                outputs[idx] = w
        else:
            outputs[idx] = compute_mask(row_masks[idx])

    # Fill any None (should be none)
    for i in range(m):
        if outputs[i] is None:
            # Fallback to zero (invalid if row != 0, but prevents crash)
            outputs[i] = n

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
