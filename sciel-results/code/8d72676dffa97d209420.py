def synthesize(matrix, depth_limit):
    import numpy as np
    import time
    import heapq
    from collections import defaultdict

    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to integer bitmasks for fast XOR operations
    # Bit i corresponds to input wire i
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j]:
                mask |= (1 << j)
        target_masks.append(mask)

    # Unique targets
    unique_targets = list(set(target_masks))
    
    # Initial available wires:
    # 0..n-1: inputs (bitmasks 1<<0, 1<<1, ...)
    # n: constant zero (0)
    # We represent the "state" of the circuit by the set of available masks.
    # However, the space of all subsets of GF(2)^n is huge.
    # Instead of a full Steiner tree search which is NP-hard, we use a 
    # greedy heuristic that mimics Dijkstra's expansion but prioritizes 
    # global utility: at each step, add the gate that reduces the "cost" 
    # to reach all targets the most, given the current available wires.
    
    # Actually, a simpler and effective structural change is to use 
    # "Iterative Deepening A*" or a greedy set cover approach on the 
    # linear combinations.
    
    # Let's try a different angle: 
    # The current program builds wires recursively. 
    # A better approach for XOR circuits is often to find a basis and 
    # express targets in that basis, but minimize the number of intermediate 
    # nodes by reusing partial sums.
    
    # Strategy: 
    # 1. Identify all unique target masks.
    # 2. Use a priority queue to generate new wires. 
    # 3. The "cost" of a wire is not just its depth, but its contribution 
    #    to satisfying the targets.
    
    # Let's implement a Greedy Set Cover over Linear Combinations.
    # We maintain a set of available wires (masks).
    # We want to cover all target_masks.
    # A "gate" creates a new wire w = a ^ b.
    # We can think of this as: we have a set of available vectors. We want 
    # to generate the target vectors.
    
    # Alternative Structural Change: 
    # Use Gaussian Elimination to find a basis, then optimize the circuit 
    # for that basis using a shortest-path approach on the dependency graph.
    
    # Let's try a "Reverse Search" or "BFS from Targets" approach?
    # No, forward generation is standard.
    
    # The prompt suggests: "Reformulate ... as a minimum-cost arborescence 
    # (Steiner tree) ... using Dijkstra's algorithm from the zero-vector".
    # This implies a state space search where the state is the set of 
    # generated wires. This is exponential. 
    # 
    # A practical approximation for "global optimal gate sharing" is to 
    # use a **Greedy Algorithm with Lookahead** or **Beam Search** on the 
    # set of available wires, where the score is the number of targets 
    # still missing.
    
    # Let's implement a Beam Search over the set of available wires.
    # State: frozenset of available masks.
    # Initial State: {0, 1, 2, ..., 1<<(n-1)} (as masks)
    # Goal: State contains all target_masks.
    # Action: Add a new wire w = a ^ b where a, b are in State.
    # Cost: 1 gate.
    
    # Since n can be up to 16, the number of possible masks is 2^16 = 65536.
    # The state space is too large for full BFS.
    
    # Instead, let's use a **Greedy Construction with Global Reuse**.
    # 1. Start with inputs and zero.
    # 2. While there are unmet targets:
    #    a. For each unmet target T, find the pair (a, b) in available 
    #       wires that minimizes the "effort" to create T.
    #       Effort is roughly the distance of T from the span of {a, b} 
    #       plus existing wires.
    #    b. Actually, simpler: For each unmet target T, check if it can 
    #       be formed by XORing two *existing* wires. If so, add the gate.
    #    c. If not, find the pair (a, b) such that (a ^ b) is "closest" 
    #       to T in terms of Hamming distance or linear dependency.
    
    # Let's refine the "closest" metric.
    # If we add w = a ^ b, we hope w helps form T.
    # T = w ^ r, where r is the remainder.
    # We want r to be easily formable.
    
    # This is essentially what the current program does, but locally.
    # The structural change is to make the choice **globally optimal** 
    # for the set of all remaining targets, not just the one being processed.
    
    # Algorithm:
    # 1. available = set of masks for inputs + 0.
    # 2. targets = set of target masks not in available.
    # 3. While targets is not empty:
    #    a. Generate all possible new wires w = a ^ b for a, b in available.
    #       (Prune duplicates).
    #    b. For each candidate w, calculate the "benefit":
    #       benefit(w) = number of targets T such that T can be formed 
    #       with fewer steps if w is added.
    #       Approximation: Count targets T where T ^ w is in available 
    #       (immediate completion) or T ^ w has lower complexity.
    #    c. Pick w with max benefit.
    #    d. Add gate for w. Add w to available.
    #    e. Update targets: remove any T that is now in available.
    
    # This is a heuristic for the Steiner Tree problem.
    
    # Optimization: 
    # For n=16, |available| grows. |available|^2 can be large.
    # We limit the search to the most promising candidates.
    
    available_masks = set()
    for i in range(n):
        available_masks.add(1 << i)
    available_masks.add(0)
    
    target_set = set(target_masks)
    # Remove any targets already available (inputs or zero)
    initial_targets = target_set - available_masks
    
    gates = []
    wire_to_gate_idx = {} # mask -> index of gate that created it? No, multiple gates can create same mask? 
    # We should avoid creating the same mask twice.
    # If a mask is already in available, we don't need a new gate for it.
    
    # Map: mask -> wire_index
    # Wire 0..n-1 are inputs. Wire n is 0.
    mask_to_wire = {}
    for i in range(n):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = n
    
    next_wire = n + 1
    wire_to_mask = {} # wire_index -> mask
    
    # Record the gate that produced each wire for verification/debugging
    # gate k produces wire n+1+k
    # gates[k] = [wire_a, wire_b]
    
    # We need to track which wires are "live" (available).
    
    # To make this efficient, we don't iterate ALL pairs of available masks 
    # at each step if the set gets large.
    # But for 16x16, the number of unique masks generated is likely small 
    # compared to 2^16.
    
    # Let's add a time check.
    
    while initial_targets:
        if time.time() > DEADLINE:
            # Fallback: just create gates for remaining targets using a simple chain
            break
            
        # Generate candidates
        # We look for a new wire w that maximizes the reduction in "total distance" 
        # to the remaining targets.
        
        # Heuristic: 
        # For each remaining target T, if we have w, then T = w ^ (T^w).
        # If T^w is in available, T is solved in 1 more step (0 if w==T).
        # If T^w is not in available, we need more steps.
        
        # Let's score candidates w by:
        # 1. If w is in target_set: score = 1000 (solves one directly)
        # 2. Count how many targets T have (T ^ w) in available. score += count * 10
        # 3. Count how many targets T have (T ^ w) with smaller popcount? 
        #    (Proxy for easier to build).
        
        best_w = None
        best_score = -1
        best_a = None
        best_b = None
        
        avail_list = list(available_masks)
        # To speed up, if avail_list is large, we might sample or limit.
        # But for now, try all pairs.
        
        # Optimization: Pre-compute a set of available masks for O(1) lookup.
        # We already have available_masks as a set.
        
        # Iterate pairs
        # Symmetry: a ^ b == b ^ a.
        for i in range(len(avail_list)):
            a = avail_list[i]
            for j in range(i + 1, len(avail_list)):
                b = avail_list[j]
                w = a ^ b
                
                # If w is already available, skip
                if w in available_masks:
                    continue
                
                # Score w
                score = 0
                if w in target_set:
                    score += 100000 # Huge bonus for direct hit
                
                # Check how many targets are "closer"
                # T is solved if T^w in available.
                # Also, if w is 0 or input, it's already there (handled by skip).
                
                # We can't iterate all targets for every candidate if 
                # candidates are many.
                # Let's limit candidates or use a faster metric.
                
                # Faster metric:
                # Popcount of w? No.
                # Let's just check direct hits and "one-step completions" 
                # for a subset of targets or all if small.
                
                # For n=16, max targets 16. Candidates ~ |A|^2/2.
                # If |A| is 100, candidates 5000. 5000 * 16 = 80k ops. OK.
                
                for t in initial_targets:
                    if t == w:
                        score += 1000
                    elif (t ^ w) in available_masks:
                        score += 100
                
                if score > best_score:
                    best_score = score
                    best_w = w
                    best_a = a
                    best_b = b
        
        if best_w is None:
            # No improvement found? This shouldn't happen if we have inputs.
            # Fallback: pick any new wire, or break.
            # If we can't find a beneficial wire, we might be stuck in a local 
            # minimum of the heuristic.
            # Fallback: Pick a random pair or the first pair.
            if len(avail_list) < 2:
                break
            best_a = avail_list[0]
            best_b = avail_list[1]
            best_w = best_a ^ best_b
            if best_w in available_masks:
                # Try to find any new wire
                found = False
                for i in range(len(avail_list)):
                    for j in range(i+1, len(avail_list)):
                        w = avail_list[i] ^ avail_list[j]
                        if w not in available_masks:
                            best_a = avail_list[i]
                            best_b = avail_list[j]
                            best_w = w
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break
        
        # Add the gate
        # We need wire indices for best_a and best_b
        # Note: best_a and best_b are masks.
        wire_a = mask_to_wire[best_a]
        wire_b = mask_to_wire[best_b]
        
        gates.append([wire_a, wire_b])
        new_wire_idx = next_wire
        next_wire += 1
        
        # Update tracking
        mask_to_wire[best_w] = new_wire_idx
        wire_to_mask[new_wire_idx] = best_w
        available_masks.add(best_w)
        
        # Update targets
        # Any target that is now in available is solved.
        new_initial_targets = set()
        for t in initial_targets:
            if t not in available_masks:
                new_initial_targets.add(t)
        initial_targets = new_initial_targets

    # If we hit the deadline or broke early, we need to ensure all outputs are valid.
    # The current loop stops when initial_targets is empty.
    # If it breaks due to deadline, some targets might be unsolved in `available_masks`.
    # We must generate valid wires for them, even if it takes more gates.
    # Simple fallback: Build each remaining target from inputs using a chain.
    
    # Identify unsolved targets
    unsolved = set(target_masks) - available_masks
    
    for t in unsolved:
        # Build t from inputs
        # t is a bitmask.
        # We can XOR all inputs i where bit i is set in t.
        # This uses popcount(t) - 1 gates.
        # We can optimize by reusing existing wires if possible, 
        # but for safety and speed, let's just do a simple chain 
        # or use the existing available wires to reduce steps.
        
        # Try to find a wire w in available such that t ^ w has smaller popcount?
        # Or just use inputs.
        
        # Let's try to be a bit smarter:
        # Find w in available that minimizes popcount(t ^ w).
        best_w = 0
        min_pop = t.bit_count()
        for w in available_masks:
            diff = t ^ w
            p = diff.bit_count()
            if p < min_pop:
                min_pop = p
                best_w = w
        
        # Now we need to build t ^ best_w from inputs (or other wires).
        # Let's just build from inputs for simplicity and correctness guarantee.
        # t = best_w ^ (t ^ best_w)
        # If best_w is 0, t = 0 ^ t.
        
        # We need to build remainder = t ^ best_w.
        # We can build remainder by XORing inputs.
        # But we can also check if remainder is already available.
        
        rem = t ^ best_w
        
        if rem in available_masks:
            # t = best_w ^ rem
            wire_bw = mask_to_wire[best_w]
            wire_rem = mask_to_wire[rem]
            gates.append([wire_bw, wire_rem])
            new_wire_idx = next_wire
            next_wire += 1
            mask_to_wire[t] = new_wire_idx
            wire_to_mask[new_wire_idx] = t
            available_masks.add(t)
        else:
            # Build rem from inputs
            # rem is a mask.
            # We XOR inputs corresponding to set bits in rem.
            # If rem has 0 bits, rem=0 (wire n).
            # If rem has 1 bit, it's an input.
            
            bits = [i for i in range(n) if (rem >> i) & 1]
            
            if len(bits) == 0:
                wire_rem = n # 0
            elif len(bits) == 1:
                wire_rem = bits[0]
            else:
                # Chain XOR
                # w0 = input[bits[0]]
                # w1 = w0 ^ input[bits[1]]
                # ...
                curr_wire = bits[0]
                for b in bits[1:]:
                    gates.append([curr_wire, b])
                    new_wire_idx = next_wire
                    next_wire += 1
                    # This new wire represents curr_wire_mask ^ (1<<b)
                    # We need to track the mask for this new wire to use it 
                    # in subsequent XORs? 
                    # Actually, we just need the wire index.
                    # But we need to know the mask to verify at the end?
                    # The interface returns gates and outputs. 
                    # The evaluator verifies.
                    # We don't strictly need to update mask_to_wire for intermediate 
                    # chain wires unless we reuse them. 
                    # We are not reusing them here.
                    curr_wire = new_wire_idx
                
                wire_rem = curr_wire
            
            # Now t = best_w ^ wire_rem
            wire_bw = mask_to_wire[best_w]
            gates.append([wire_bw, wire_rem])
            new_wire_idx = next_wire
            next_wire += 1
            mask_to_wire[t] = new_wire_idx
            wire_to_mask[new_wire_idx] = t
            available_masks.add(t)

    # Map original targets to their wire indices
    outputs = []
    for t in target_masks:
        # t must be in mask_to_wire
        # If not, something went wrong.
        if t in mask_to_wire:
            outputs.append(mask_to_wire[t])
        else:
            # Should not happen
            outputs.append(n) # Fallback to 0

    return {'gates': gates, 'outputs': outputs}
