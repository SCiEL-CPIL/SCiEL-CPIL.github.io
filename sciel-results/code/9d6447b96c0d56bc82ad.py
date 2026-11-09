"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Synthesize XOR circuit using iterative matching with global sub-expression sharing.
    
    Structural change: Instead of greedy per-output synthesis, we model the problem
    as building a shared circuit where at each step we choose the XOR of two existing
    wires that maximizes the number of "useful" new intermediates (wires that help
    complete remaining outputs). This is a form of iterative greedy matching that
    prioritizes global sharing over local per-output optimality.
    
    Key insight: The optimal XOR circuit for multiple outputs is a Steiner tree in
    the XOR space. We approximate this by iteratively adding the most "useful" gate,
    where usefulness is measured by how many remaining output masks can be expressed
    as XOR of the new wire with an existing wire (or as the new wire itself).
    """
    deadline = time.time() + 50  # Safety margin for 60s limit
    
    n = len(matrix)
    m = len(matrix[0]) if n > 0 else 0
    
    if n == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert matrix rows to bitmasks
    rows = []
    for i in range(n):
        mask = 0
        for j in range(m):
            if matrix[i][j]:
                mask |= (1 << j)
        rows.append(mask)
    
    # Zero wire is m, inputs are 0..m-1
    zero_wire = m
    next_wire = m + 1
    gates = []
    
    # Wire management
    wire_to_mask = {}  # wire_id -> bitmask
    mask_to_wire = {}  # bitmask -> wire_id
    for j in range(m):
        wire_to_mask[j] = 1 << j
        mask_to_wire[1 << j] = j
    wire_to_mask[zero_wire] = 0
    mask_to_wire[0] = zero_wire
    
    # Precompute popcount
    popcount = [0] * (1 << m)
    for i in range(1, 1 << m):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Track which outputs are already satisfied
    outputs = [-1] * n
    remaining_outputs = set(range(n))
    
    # For each remaining output, track the "best decomposition" using existing wires
    # best_decomp[i] = (w1, w2) such that rows[i] = wire_to_mask[w1] ^ wire_to_mask[w2]
    # or (w, -1) if rows[i] == wire_to_mask[w]
    
    def check_output_satisfied(i):
        """Check if output i is already satisfied by an existing wire."""
        if rows[i] in mask_to_wire:
            return mask_to_wire[rows[i]]
        return -1
    
    # Initially, check if any outputs are directly satisfied
    for i in range(n):
        w = check_output_satisfied(i)
        if w != -1:
            outputs[i] = w
            remaining_outputs.discard(i)
    
    # If all outputs are satisfied, return immediately
    if not remaining_outputs:
        return {'gates': gates, 'outputs': outputs}
    
    # Main loop: iteratively add the most useful gate
    # Usefulness of a new wire W = A ^ B is measured by:
    # 1. How many remaining outputs i have rows[i] == W (direct satisfaction)
    # 2. How many remaining outputs i can now be expressed as W ^ X where X is existing
    #    (i.e., rows[i] ^ W is in mask_to_wire)
    # 3. How many remaining outputs i can be expressed as X ^ Y where one of X,Y is new
    #    and the other is existing, and the result helps further
    
    # For efficiency, we maintain a count of how many remaining outputs
    # are "one step away" from being satisfied by each existing wire
    # one_step_away[w] = number of remaining outputs i where rows[i] ^ wire_to_mask[w] is in mask_to_wire
    
    existing_wires = list(wire_to_mask.keys())
    
    # Precompute for each existing wire, which remaining outputs it helps
    # help_count[w] = number of remaining outputs i where rows[i] == wire_to_mask[w] ^ wire_to_mask[w2] for some existing w2
    # This is expensive to maintain, so we recompute periodically or use a simpler heuristic
    
    # Simpler approach: at each step, for each pair of existing wires (a, b),
    # compute W = wire_to_mask[a] ^ wire_to_mask[b], and score it by:
    # - 100 * (1 if W == rows[i] for some remaining i)  [direct satisfaction]
    # - 1 * (number of remaining i where rows[i] ^ W is in mask_to_wire)  [one-step help]
    # - 0.1 * (number of remaining i where rows[i] ^ W has popcount < popcount[rows[i]])  [progress]
    
    max_iterations = 200  # Safety limit on number of gates to add
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        if not remaining_outputs:
            break
        
        best_score = -1
        best_a = -1
        best_b = -1
        best_new_mask = -1
        
        # Try all pairs of existing wires
        ew = list(wire_to_mask.keys())
        num_wires = len(ew)
        
        # For large number of wires, sample pairs to keep it fast
        if num_wires > 30:
            # Use a subset of pairs
            import random
            random.seed(42)  # Deterministic
            indices = list(range(num_wires))
            # Sample pairs
            candidates = []
            for _ in range(500):
                i1 = random.randint(0, num_wires - 1)
                i2 = random.randint(0, num_wires - 1)
                if i1 != i2:
                    candidates.append((i1, i2))
            # Also include some structured pairs
            for i in range(0, num_wires, max(1, num_wires // 50)):
                for j in range(i + 1, num_wires, max(1, num_wires // 50)):
                    candidates.append((i, j))
        else:
            candidates = [(i, j) for i in range(num_wires) for j in range(num_wires) if i != j]
        
        for i1, i2 in candidates:
            w1 = ew[i1]
            w2 = ew[i2]
            m1 = wire_to_mask[w1]
            m2 = wire_to_mask[w2]
            new_mask = m1 ^ m2
            
            if new_mask == 0:
                continue
            if new_mask in mask_to_wire:
                continue  # Already have this wire
            
            score = 0.0
            
            # Direct satisfaction
            for i in remaining_outputs:
                if rows[i] == new_mask:
                    score += 100.0
            
            # One-step help: rows[i] ^ new_mask is in mask_to_wire
            for i in remaining_outputs:
                if rows[i] == new_mask:
                    continue
                comp = rows[i] ^ new_mask
                if comp in mask_to_wire:
                    score += 1.0
            
            # Progress: new_mask reduces popcount of some remaining output
            for i in remaining_outputs:
                if rows[i] == new_mask:
                    continue
                comp = rows[i] ^ new_mask
                if popcount[comp] < popcount[rows[i]]:
                    score += 0.1
            
            if score > best_score:
                best_score = score
                best_a = w1
                best_b = w2
                best_new_mask = new_mask
        
        if best_score <= 0:
            # No useful gate found, fall back to building remaining outputs independently
            break
        
        # Add the best gate
        gates.append((best_a, best_b))
        w = next_wire
        next_wire += 1
        wire_to_mask[w] = best_new_mask
        mask_to_wire[best_new_mask] = w
        
        # Check if any remaining outputs are now satisfied
        for i in list(remaining_outputs):
            if rows[i] == best_new_mask:
                outputs[i] = w
                remaining_outputs.discard(i)
            else:
                comp = rows[i] ^ best_new_mask
                if comp in mask_to_wire:
                    # Output i can now be satisfied with one more gate
                    # But we don't add that gate yet; we wait for the next iteration
                    # Actually, let's add it immediately if it's cheap
                    pass
        
        # Also check if any remaining output can now be completed with one gate
        for i in list(remaining_outputs):
            comp = rows[i] ^ best_new_mask
            if comp in mask_to_wire:
                w_comp = mask_to_wire[comp]
                # Add gate to complete this output
                gates.append((w, w_comp))
                w_out = next_wire
                next_wire += 1
                wire_to_mask[w_out] = rows[i]
                mask_to_wire[rows[i]] = w_out
                outputs[i] = w_out
                remaining_outputs.discard(i)
    
    # Handle any remaining outputs with fallback
    for i in remaining_outputs:
        if time.time() > deadline:
            # Emergency: just use a chain
            bits = [j for j in range(m) if rows[i] & (1 << j)]
            if not bits:
                outputs[i] = zero_wire
            elif len(bits) == 1:
                outputs[i] = bits[0]
            else:
                w = bits[0]
                for j in bits[1:]:
                    gates.append((w, j))
                    w = next_wire
                    next_wire += 1
            continue
        
        # Try to build this output using existing wires
        if rows[i] in mask_to_wire:
            outputs[i] = mask_to_wire[rows[i]]
            continue
        
        # Try one-step: find w such that rows[i] ^ wire_to_mask[w] is in mask_to_wire
        found = False
        for w in wire_to_mask.keys():
            comp = rows[i] ^ wire_to_mask[w]
            if comp in mask_to_wire:
                w_comp = mask_to_wire[comp]
                gates.append((w, w_comp))
                w_out = next_wire
                next_wire += 1
                wire_to_mask[w_out] = rows[i]
                mask_to_wire[rows[i]] = w_out
                outputs[i] = w_out
                found = True
                break
        
        if found:
            continue
        
        # Fallback: build from scratch
        bits = [j for j in range(m) if rows[i] & (1 << j)]
        if not bits:
            outputs[i] = zero_wire
        elif len(bits) == 1:
            outputs[i] = bits[0]
        else:
            w = bits[0]
            for j in bits[1:]:
                gates.append((w, j))
                w = next_wire
                next_wire += 1
            outputs[i] = w
    
    # Ensure all outputs are set
    for i in range(n):
        if outputs[i] == -1:
            bits = [j for j in range(m) if rows[i] & (1 << j)]
            if not bits:
                outputs[i] = zero_wire
            elif len(bits) == 1:
                outputs[i] = bits[0]
            else:
                w = bits[0]
                for j in bits[1:]:
                    gates.append((w, j))
                    w = next_wire
                    next_wire += 1
                outputs[i] = w
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
