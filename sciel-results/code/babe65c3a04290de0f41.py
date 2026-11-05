"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def synthesize(matrix, depth_limit):
    """Synthesize an XOR circuit using Global Common Subexpression Elimination (CSE).
    
    Iteratively identifies the most beneficial common sub-expression (mask) across
    the current set of row masks, creates a wire for it, and XORs it out of all
    rows that contain it. This minimizes total gate count by maximizing sharing.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit
    
    mat = np.array(matrix, dtype=np.int8)
    n_outputs, n_inputs = mat.shape
    
    def row_to_mask(row):
        m = 0
        for j, v in enumerate(row):
            if v:
                m |= (1 << j)
        return m
        
    output_masks = [row_to_mask(mat[i]) for i in range(n_outputs)]
    
    # If no outputs, return empty
    if n_outputs == 0:
        return {'gates': [], 'outputs': []}
        
    # Initial state:
    # current_masks: The masks we still need to compute for each output.
    # available_masks: The masks we already have wires for.
    # We start with input bits and zero.
    # Wire mapping:
    # 0..n_inputs-1 : input bits (masks 1<<i)
    # n_inputs      : zero (mask 0)
    # n_inputs+1... : generated wires
    
    available_masks = {0}
    for i in range(n_inputs):
        available_masks.add(1 << i)
        
    current_masks = list(output_masks)
    
    gates = []
    wire_map = {0: n_inputs}
    for i in range(n_inputs):
        wire_map[1 << i] = i
        
    next_wire = n_inputs + 1
    
    # Helper to get or create a wire for a mask, assuming it is in available_masks or can be built
    # But for CSE, we only add to available_masks when we explicitly decide to synthesize it.
    # However, for the final outputs, we might need to build the remaining masks.
    
    # CSE Loop
    # We look for a mask M such that:
    # 1. M is not in available_masks
    # 2. M can be formed by XORing two masks A, B in available_masks
    # 3. M reduces the total cost (sum of popcounts of current_masks)
    #    Cost reduction = (popcount(M) - 1) * count(M in current_masks)
    #    Actually, creating M costs 1 gate.
    #    Using M in k rows saves (popcount(M) - 1) gates per row? No.
    #    Let's think about the metric: Total Gates.
    #    If we don't use M, each row i containing M will be built independently.
    #    If we use M, we pay 1 gate for M, and then each row i containing M uses M.
    #    The cost of building a mask R is roughly popcount(R) - 1 if built from scratch (binary tree).
    #    If we extract M from R, R becomes R^M.
    #    Cost change for row i:
    #       Old: Cost(R_i)
    #       New: Cost(R_i ^ M) + 1 (for using M, assuming M is available)
    #    This is complex because Cost is not additive in a simple way for XOR trees.
    #    
    #    Simpler heuristic often used:
    #    Benefit = (number of times M appears in current_masks) * (popcount(M) - 1) - 1
    #    This assumes that replacing M in a row saves popcount(M)-1 gates.
    #    This is an upper bound, but a good heuristic.
    
    while time.time() < DEADLINE:
        best_mask = None
        best_benefit = 0
        
        # Generate candidate masks from available pairs
        # To keep it fast, we only consider pairs from available_masks
        # available_masks can grow large.
        # For n_inputs=16, size is 2^16 = 65536 max.
        # Pairs is O(N^2). 65536^2 is too big.
        # We should restrict candidates.
        
        # Strategy: Only consider masks that are actually present in current_masks.
        # Count frequencies of all masks in current_masks.
        freq = {}
        for m in current_masks:
            if m != 0:
                freq[m] = freq.get(m, 0) + 1
        
        if not freq:
            break
            
        # We want to find M in freq such that M = A ^ B for A, B in available_masks.
        # And Benefit(M) = freq[M] * (popcount(M) - 1) - 1 > 0
        
        # To find A, B efficiently:
        # For each candidate M in freq:
        #   For each A in available_masks:
        #     B = M ^ A
        #     if B in available_masks:
        #        Found a decomposition.
        
        # This is O(|freq| * |available|).
        # |freq| <= n_outputs (usually small, e.g., 16).
        # |available| can be large.
        
        # Optimization:
        # If |available| is large, iterate over smaller set?
        # Or just iterate over A in available_masks for each M.
        
        candidates = list(freq.keys())
        
        # Sort candidates by potential benefit descending to find good ones fast?
        # Or just check all.
        
        for M in candidates:
            if M in available_masks:
                continue
                
            # Check if M can be formed
            # We don't need to check all A if we just want to know IF it's possible.
            # But we also need to know the "cost" of forming it?
            # The gate count for M is 1 regardless of which A, B we use (as long as they are available).
            # So any valid decomposition is fine for the gate count of M itself.
            
            # However, we want to ensure we pick the "best" M.
            # Benefit depends only on M and its frequency.
            
            # So we just need to check existence of A, B.
            
            # Optimization:
            # If popcount(M) is small, it's less likely to be formed by XOR of two available masks?
            # No, XOR of two masks has popcount <= popcount(A)+popcount(B).
            
            # Let's just iterate A.
            found = False
            # To speed up, we can break early if found.
            # We don't need to store which A, B because we just need to know it's possible.
            # Wait, we DO need to record the gate [wA, wB] when we add it.
            # So we need to store one valid pair.
            
            valid_pair = None
            for A in available_masks:
                B = M ^ A
                if B in available_masks:
                    valid_pair = (A, B)
                    break
            
            if valid_pair:
                benefit = freq[M] * (bin(M).count('1') - 1) - 1
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_mask = M
                    best_pair = valid_pair
                    
        if best_mask is None:
            break
            
        # Add best_mask to available
        A, B = best_pair
        wA = wire_map[A]
        wB = wire_map[B]
        
        gates.append([wA, wB])
        wNew = next_wire
        next_wire += 1
        
        wire_map[best_mask] = wNew
        available_masks.add(best_mask)
        
        # Update current_masks
        # For each row containing best_mask, XOR it out
        for i in range(n_outputs):
            if current_masks[i] & best_mask: # This check is wrong. We need exact match of subset?
                # No, CSE works on masks.
                # If current_masks[i] == best_mask, it becomes 0.
                # If current_masks[i] contains best_mask as a subset?
                # XOR is not subset removal.
                # If current_masks[i] = M, then new mask is 0.
                # If current_masks[i] = M ^ R, then new mask is R.
                # So we just XOR best_mask into current_masks[i] if best_mask is a "factor"?
                # In GF(2), we can always XOR.
                # But we only want to XOR if it reduces complexity?
                # The heuristic assumed we replace M in the row.
                # So we should XOR M into any row that has M?
                # No, we should XOR M into rows where it helps.
                # Since M is a linear combination, we can XOR it into any row.
                # But to maintain the "extraction" logic, we typically XOR M into rows that are equal to M or contain M in a way that simplifies.
                # Actually, if we just XOR M into row i, the new mask is current_masks[i] ^ M.
                # We should do this for all rows where this reduces popcount?
                # Or just all rows where current_masks[i] == M?
                # The standard CSE for XOR circuits:
                # If we introduce a node for M, we can replace M in any expression.
                # So for each row i, if current_masks[i] == M, set to 0.
                # If current_masks[i] has M as a sub-term?
                # Since these are just masks, "sub-term" isn't well-defined unless we track structure.
                # But we are working with the final linear combination.
                # So we can XOR M into ANY row.
                # To minimize popcount, we should XOR M into row i if popcount(current_masks[i] ^ M) < popcount(current_masks[i]).
                
                if current_masks[i] == best_mask:
                    current_masks[i] = 0
                else:
                    new_mask = current_masks[i] ^ best_mask
                    if bin(new_mask).count('1') < bin(current_masks[i]).count('1'):
                        current_masks[i] = new_mask
                    # Else, don't change? Or change anyway?
                    # If we don't change, we didn't "use" the wire for this row.
                    # That's fine. We just created the wire.
                    # But if we don't use it, did we save gates?
                    # The benefit calculation assumed we used it.
                    # So we should only count benefit if we actually use it.
                    # This makes the heuristic adaptive.
                    
        # Re-calculate freq?
        # The loop recalculates freq at the start of each iteration.
        # So we just continue.

    # Finalize
    # Build remaining masks
    # We have a set of available_masks and wire_map.
    # For each remaining current_mask, we need to create a wire.
    # We can use a recursive builder that tries to reuse available masks.
    
    def build_wire(mask):
        nonlocal next_wire
        if mask in wire_map:
            return wire_map[mask]
            
        if mask == 0:
            return n_inputs
            
        # Try to find a split A, B in available_masks such that A^B is a sub-part?
        # No, we need to build mask from available_masks.
        # We can look for A in available_masks such that A is a subset of mask?
        # Or just find A, B in available_masks such that A^B = mask?
        # If so, we just need 1 gate.
        # But we might not have A^B=mask directly.
        
        # Greedy approach:
        # Find A in available_masks that reduces popcount most?
        # Or just pick the first available A that is non-zero and not mask?
        
        # Better: Find A in available_masks such that popcount(mask ^ A) < popcount(mask).
        # Then recurse on mask ^ A.
        
        best_A = None
        min_pop = bin(mask).count('1')
        
        for A in available_masks:
            if A == 0 or A == mask:
                continue
            new_mask = mask ^ A
            p = bin(new_mask).count('1')
            if p < min_pop:
                min_pop = p
                best_A = A
                if p == 0:
                    break
                    
        if best_A is None:
            # No available mask helps?
            # This shouldn't happen if we have input bits.
            # Fallback: split into two halves
            bits = []
            m = mask
            idx = 0
            while m:
                if m & 1:
                    bits.append(idx)
                m >>= 1
                idx += 1
                
            if not bits:
                return n_inputs
                
            if len(bits) == 1:
                w = wire_map[1 << bits[0]]
                wire_map[mask] = w
                return w
                
            mid = len(bits) // 2
            m1 = 0
            m2 = 0
            for b in bits[:mid]:
                m1 |= (1 << b)
            for b in bits[mid:]:
                m2 |= (1 << b)
                
            w1 = build_wire(m1)
            w2 = build_wire(m2)
            gates.append([w1, w2])
            w = next_wire
            next_wire += 1
            wire_map[mask] = w
            return w
        else:
            # Use best_A
            rest = mask ^ best_A
            wA = wire_map[best_A]
            wRest = build_wire(rest)
            gates.append([wA, wRest])
            w = next_wire
            next_wire += 1
            wire_map[mask] = w
            return w

    output_wires = []
    for m in current_masks:
        output_wires.append(build_wire(m))
        
    return {'gates': gates, 'outputs': output_wires}

# EVOLVE-BLOCK-END
