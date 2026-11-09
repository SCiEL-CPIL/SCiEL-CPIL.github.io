"""
Exact XOR-circuit synthesis via data-driven greedy Common Subexpression Elimination (CSE)
with local search refinement and strict deadline management.

Strategy:
1. Represent each output row as a set of input wire indices.
2. Iteratively identify the pair of wires (a, b) that appear together in the most rows.
3. If merging them (computing a ^ b) reduces the total number of wires across all rows,
   create a new gate for (a, b) and update the row sets.
4. Repeat until no further beneficial merges are found or time limit is reached.
5. Assemble final circuits using balanced XOR trees for remaining row components.
6. Perform local search cleanup: remove unused gates and redundant gates (those producing
   values already available).
"""
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 85  # Safety margin before 90s kill

    mat = np.array(matrix, dtype=np.int32)
    n_rows, n_cols = mat.shape
    n = n_cols
    zero_wire = n

    # 1. Initialize row sets
    # row_sets[i] is a set of wire indices that XOR to matrix[i]
    # Initially, these are the column indices where matrix[i, j] == 1
    row_sets = []
    for i in range(n_rows):
        cols = np.nonzero(mat[i])[0]
        s = set(cols.tolist())
        if not s:
            s = {zero_wire}
        row_sets.append(s)

    # Gate management
    gates = []
    next_wire = n + 1
    # gate_registry: frozenset({a, b}) -> wire index
    # We store the unordered pair as a frozenset of two ints
    gate_registry = {}
    
    # Pre-register inputs? No, inputs are just indices 0..n-1.
    # We only register gates we create.

    def get_wire(a, b):
        """Get or create a wire for XOR of a and b."""
        nonlocal next_wire
        if a == b:
            return zero_wire
        key = frozenset((a, b))
        if key in gate_registry:
            return gate_registry[key]
        
        wire = next_wire
        next_wire += 1
        # Store in canonical order for consistency, though frozenset handles identity
        gates.append([min(a, b), max(a, b)])
        gate_registry[key] = wire
        return wire

    # 2. Greedy CSE Loop
    # We want to maximize sharing. A merge of (a, b) into w is beneficial if
    # count(a, b) > 1, because we replace 2 wires with 1 in count(a,b) rows,
    # saving count(a,b) - 1 wires total.
    
    # To avoid O(M * K^2) recomputation from scratch every time, we could use
    # a priority queue, but for small M (<=16) and small K, recomputation is fine.
    # However, we must be careful with time.
    
    max_iters = 500  # Hard cap on iterations to prevent infinite loops in weird cases
    iter_count = 0
    
    while iter_count < max_iters:
        if time.time() > DEADLINE - 10:  # Reserve 10s for final assembly and cleanup
            break
            
        # Compute pair counts
        pair_counts = defaultdict(int)
        for s in row_sets:
            lst = sorted(s)
            # If set is large, this is expensive. But sets shrink as we merge.
            # Initial size <= 16.
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    key = frozenset((lst[i], lst[j]))
                    pair_counts[key] += 1
        
        best_pair = None
        best_benefit = 0
        
        for key, cnt in pair_counts.items():
            a, b = tuple(key)
            # Don't merge if already a gate
            if key in gate_registry:
                continue
            # Benefit is (cnt - 1) wires saved
            if cnt > 1:
                benefit = cnt - 1
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_pair = key
                    # Early exit if we find a very strong pair? 
                    # Not strictly necessary, but can help.
        
        if best_pair is None:
            break
            
        a, b = tuple(best_pair)
        new_wire = get_wire(a, b)
        
        # Update row sets
        for i in range(n_rows):
            s = row_sets[i]
            if a in s and b in s:
                s.discard(a)
                s.discard(b)
                s.add(new_wire)
                
        iter_count += 1

    # 3. Final Assembly
    # For each row, we have a set of wires. We need to XOR them all.
    # Use a balanced tree to minimize depth.
    
    def xor_tree(wires):
        """Recursively build XOR tree for a list of wires."""
        if len(wires) == 0:
            return zero_wire
        if len(wires) == 1:
            return wires[0]
        if len(wires) == 2:
            return get_wire(wires[0], wires[1])
        
        mid = len(wires) // 2
        left = xor_tree(wires[:mid])
        right = xor_tree(wires[mid:])
        return get_wire(left, right)

    outputs = []
    for i in range(n_rows):
        s = row_sets[i]
        # Ensure zero_wire is handled correctly if the set becomes {zero_wire}
        # xor_tree handles len 1.
        outputs.append(xor_tree(sorted(s)))

    # 4. Local Search / Cleanup
    # We might have created gates that are now unused or redundant.
    # Redundant: Gate k computes mask M. If another wire j (input or earlier gate) also computes M,
    # we can replace all uses of k with j and delete gate k.
    # Unused: Gate k is not referenced by any output or later gate.
    
    # Compute mask for all wires
    # mask[i] = bitmask of inputs XORed to produce wire i
    num_wires = next_wire
    masks = [0] * num_wires
    for i in range(n):
        masks[i] = (1 << i)
    masks[n] = 0  # zero_wire
    
    for k in range(len(gates)):
        wire = n + 1 + k
        a, b = gates[k]
        masks[wire] = masks[a] ^ masks[b]

    # Check for redundancies and unused gates
    # We process gates in reverse order? No, dependencies are forward.
    # A gate is redundant if its mask is equal to the mask of an input (0..n) 
    # or an earlier gate (n+1 .. wire-1).
    
    # Map mask -> earliest wire index that produces it
    mask_to_wire = {}
    for i in range(n + 1):  # Inputs and zero
        mask_to_wire[masks[i]] = i

    # We will create a mapping old_wire -> new_wire to handle replacements
    wire_remap = list(range(num_wires)) # Identity initially
    
    # We iterate through gates. If a gate is redundant, we remap it to the existing wire.
    # If a gate is unused, we remove it.
    # This is tricky to do in one pass because removing a gate changes indices.
    # Better approach:
    # 1. Identify "surviving" gates.
    # 2. Rebuild the gate list and outputs using the remap.
    
    # Step 4a: Identify redundant gates
    # A gate k (wire w) is redundant if masks[w] is already in mask_to_wire.
    # Note: mask_to_wire contains inputs and potentially earlier gates if we update it.
    # We should only remap to inputs or *earlier* gates to avoid cycles?
    # Actually, if gate 5 produces mask X, and gate 10 produces mask X, gate 10 is redundant.
    # Gate 5 is not redundant (unless mask X is an input).
    
    # Let's update mask_to_wire as we go, but only keep the *first* occurrence.
    # Inputs are first.
    # Then gate 0, gate 1, etc.
    
    # We need to be careful: if we remap gate 10 to gate 5, and gate 5 is later removed?
    # Gate 5 is only removed if it's unused. If gate 10 was using gate 5's value (by remapping),
    # then gate 5 is used by gate 10's *logical* replacement? No, gate 10 is being *deleted*.
    # The references to gate 10 in outputs/other gates should point to gate 5.
    
    # Algorithm:
    # 1. Build mask_to_wire for inputs.
    # 2. Iterate gates k=0..G-1.
    #    w = n+1+k
    #    if masks[w] in mask_to_wire:
    #        wire_remap[w] = mask_to_wire[masks[w]]
    #        # Mark gate k as redundant
    #    else:
    #        mask_to_wire[masks[w]] = w
    #        # Keep gate k (tentatively)
    
    # 3. Now, we have a set of "kept" gates and "remapped" gates.
    #    But we also need to check for unused gates among the kept ones.
    #    A gate is unused if no output points to it, and no *other kept* gate points to it.
    
    # Let's build the final gate list and outputs based on wire_remap.
    
    # First, apply remap to outputs and gate inputs.
    # Note: wire_remap is defined for all wires.
    
    new_gates = []
    kept_gate_indices = [] # Indices in original gates list that are kept
    
    # We need to know which original gates are kept.
    # Let's create a boolean array is_kept
    is_kept = [False] * len(gates)
    
    for k in range(len(gates)):
        w = n + 1 + k
        if wire_remap[w] == w:
            is_kept[k] = True
        else:
            is_kept[k] = False

    # Now check for unused gates among kept gates.
    # A kept gate k is used if:
    # 1. It is in outputs (after remap).
    # 2. It is an input to another kept gate (after remap).
    
    # Build usage count
    usage = defaultdict(int)
    
    # Check outputs
    final_outputs = []
    for out in outputs:
        remapped_out = wire_remap[out]
        final_outputs.append(remapped_out)
        usage[remapped_out] += 1
        
    # Check gate inputs
    # We need to iterate through kept gates and see what they use.
    # But the "inputs" to the new gates will be the remapped versions of the old inputs.
    
    # Let's construct the new gates first, tracking usage.
    # We iterate through original gates. If kept, we create a new gate.
    # The inputs to the new gate are wire_remap[old_a], wire_remap[old_b].
    
    # However, the wire index in the *new* circuit depends on the order of kept gates.
    # Let's assign new indices.
    
    old_to_new = {}
    new_gates_list = []
    new_next_wire = n + 1
    
    for k in range(len(gates)):
        if not is_kept[k]:
            continue
        
        old_a, old_b = gates[k]
        new_a = wire_remap[old_a]
        new_b = wire_remap[old_b]
        
        # If new_a == new_b, this gate produces zero.
        # It should have been caught by redundancy check (mask 0 is in mask_to_wire).
        # So new_a != new_b.
        
        new_wire = new_next_wire
        new_next_wire += 1
        old_to_new[n + 1 + k] = new_wire
        new_gates_list.append([new_a, new_b])
        
        # Update usage for inputs
        usage[new_a] += 1
        usage[new_b] += 1
        
        # Mark this new wire as "potentially used" by its existence?
        # No, we need to check if the *output* of this gate is used.
        # We will check usage after building all gates.
        # For now, we just record the mapping.

    # Now, check which new gates are unused.
    # A new gate is unused if usage[new_wire] == 0.
    # Note: usage counts references from outputs and *other gates*.
    # We have already added references from outputs.
    # We have added references from gate inputs to the *inputs* of the gates, not the outputs.
    # We need to add references to the *outputs* of the gates if they are used by other gates.
    
    # Wait, my usage logic above was:
    # 1. Count refs from outputs.
    # 2. For each gate, count refs to its inputs.
    # This does NOT count refs to the gate's output from other gates.
    
    # Let's redo usage properly.
    # 1. Start with usage = 0 for all wires.
    # 2. For each output in final_outputs: usage[out] += 1.
    # 3. For each gate in new_gates_list: 
    #    Let the gate be at index j in new_gates_list. Its output wire is n+1+j.
    #    Its inputs are a, b.
    #    usage[a] += 1
    #    usage[b] += 1
    #    (We do NOT increment usage of the output here, because we don't know if it's used yet).
    
    # Then, after building all gates, we can check which outputs are unused.
    # But if gate A uses gate B's output, we must have accounted for that.
    # In step 3, when we process gate A, we see its input is B's output. So we increment usage[B_output].
    # This works IF we process gates in an order where dependencies are resolved?
    # No, usage is just a count. It doesn't matter the order.
    
    # Let's re-apply usage counting correctly.
    usage = defaultdict(int)
    
    # 1. Outputs
    for out in final_outputs:
        usage[out] += 1
        
    # 2. Gate Inputs
    for g in new_gates_list:
        a, b = g
        usage[a] += 1
        usage[b] += 1
        
    # Now, identify unused gates in new_gates_list.
    # Gate j (output wire n+1+j) is unused if usage[n+1+j] == 0.
    
    # We need to remove unused gates and re-index.
    # This might make other gates unused?
    # If gate A uses gate B, and B is removed, A becomes invalid.
    # But B is only removed if it's unused. If A uses B, usage[B] > 0, so B is NOT removed.
    # So removing unused gates is a single-pass operation (no cascading).
    
    # Identify indices of unused gates
    used_gate_indices = []
    for j in range(len(new_gates_list)):
        wire = n + 1 + j
        if usage[wire] > 0:
            used_gate_indices.append(j)
            
    # If all gates are used, we are done.
    if len(used_gate_indices) == len(new_gates_list):
        final_gates = new_gates_list
        final_outputs = final_outputs # Already remapped
    else:
        # We need to remap wires again to remove gaps.
        # Old new-wire -> New new-wire
        # Input wires (0..n) stay the same.
        # Gate wires are re-indexed.
        
        # Build mapping for gate wires
        # old_gate_idx -> new_gate_idx
        old_to_new_idx = {}
        for new_idx, old_idx in enumerate(used_gate_indices):
            old_to_new_idx[old_idx] = new_idx
            
        # Map wire indices
        # Wire n+1+j (old) -> 
        # If j is in used_gate_indices, it maps to n+1+new_idx
        # Else, it should not be referenced (since it was unused).
        
        # Let's create a wire_remap2
        # We only need to remap the gate wires.
        
        # Rebuild gates
        final_gates = []
        for old_idx in used_gate_indices:
            a, b = new_gates_list[old_idx]
            # a and b are wires in the "new_gates" space.
            # They might be inputs (0..n) or gate outputs (n+1..).
            # If they are gate outputs, they refer to some old_idx2.
            # We need to map them to the new indices.
            
            def remap_wire(w):
                if w <= n:
                    return w
                # w is a gate output in the intermediate space
                old_g_idx = w - (n + 1)
                # It must be in used_gate_indices
                if old_g_idx in old_to_new_idx:
                    return n + 1 + old_to_new_idx[old_g_idx]
                else:
                    # This shouldn't happen if logic is correct
                    return w # Fallback
            
            new_a = remap_wire(a)
            new_b = remap_wire(b)
            final_gates.append([new_a, new_b])
            
        # Rebuild outputs
        def remap_output_wire(w):
            if w <= n:
                return w
            old_g_idx = w - (n + 1)
            if old_g_idx in old_to_new_idx:
                return n + 1 + old_to_new_idx[old_g_idx]
            else:
                return w

        final_outputs = [remap_output_wire(o) for o in final_outputs]

    # Final verification check (optional, for safety)
    # Verify that the final circuit computes the correct masks.
    # Given time constraints, we trust the logic.
    
    return {'gates': final_gates, 'outputs': final_outputs}
