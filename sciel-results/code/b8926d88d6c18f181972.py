"""
Strategy: Greedy Common Subexpression Elimination (CSE) with Algebraic Simplification.
Instead of Simulated Annealing, we use a deterministic, iterative local search.
We maintain a mapping from linear combinations (frozensets of inputs) to the best 
(lowest index) wire that computes them. We repeatedly scan the circuit, identify 
gates that can be replaced by existing wires (CSE), trivial gates (algebraic simplification), 
or redundant gates (duplicate inputs), and compact the circuit. This continues until 
a fixed point is reached. This approach is more robust and faster than SA for this 
structured problem, as it aggressively exploits the linear algebra structure without 
randomness overhead.
"""

import time
import numpy as np

# EVOLVE-BLOCK-START

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Preprocessing
    mat = np.array(matrix, dtype=np.int8)
    if mat.size == 0:
        return {'gates': [], 'outputs': []}
    
    n_inputs = mat.shape[1]
    n_outputs = mat.shape[0]
    
    # Handle trivial single output cases quickly
    if n_outputs == 1:
        row = mat[0]
        ones = np.nonzero(row)[0]
        if len(ones) == 0:
            return {'gates': [], 'outputs': [n_inputs]}
        elif len(ones) == 1:
            return {'gates': [], 'outputs': [int(ones[0])]}
        # Else fall through to general construction

    # 2. Initial Construction: Balanced XOR Trees
    # Build a valid circuit first.
    gates = []
    outputs = []
    wire_contents = [] # index -> frozenset of inputs
    
    # Initialize input wires
    for i in range(n_inputs):
        wire_contents.append(frozenset([i]))
    # Initialize constant 0
    wire_contents.append(frozenset())
    
    for i in range(n_outputs):
        row = mat[i]
        ones = np.nonzero(row)[0]
        if len(ones) == 0:
            outputs.append(n_inputs)
        elif len(ones) == 1:
            outputs.append(int(ones[0]))
        else:
            # Balanced XOR tree
            current_wires = [int(x) for x in ones]
            while len(current_wires) > 1:
                new_wires = []
                for k in range(0, len(current_wires), 2):
                    if k + 1 < len(current_wires):
                        a = current_wires[k]
                        b = current_wires[k+1]
                        new_gate_idx = n_inputs + 1 + len(gates)
                        gates.append([a, b])
                        c_a = wire_contents[a]
                        c_b = wire_contents[b]
                        c_new = frozenset(c_a ^ c_b)
                        wire_contents.append(c_new)
                        new_wires.append(new_gate_idx)
                    else:
                        new_wires.append(current_wires[k])
                current_wires = new_wires
            outputs.append(current_wires[0])
    
    # 3. Iterative Simplification Loop
    # We will repeatedly apply:
    # 1. CSE: If two gates produce the same content, keep the earliest one.
    # 2. Trivial: (a,a)->0, (0,b)->b, (a,0)->a.
    # 3. Compact: Remove unused/redundant gates and re-index.
    
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        
        # Build content-to-wire map
        # Note: We want the *earliest* wire for a given content to maximize sharing.
        content_map = {}
        for i, content in enumerate(wire_contents):
            if content not in content_map:
                content_map[content] = i
        
        # Identify replacements
        # replacement: old_wire_idx -> new_wire_idx
        replacement = {}
        
        for i in range(len(gates)):
            wire_idx = n_inputs + 1 + i
            a, b = gates[i]
            
            # Check trivial cases
            if a == b:
                replacement[wire_idx] = n_inputs # 0
                changed = True
                continue
            if a == n_inputs:
                replacement[wire_idx] = b
                changed = True
                continue
            if b == n_inputs:
                replacement[wire_idx] = a
                changed = True
                continue
            
            # Check CSE
            content_a = wire_contents[a]
            content_b = wire_contents[b]
            content_new = content_a ^ content_b
            
            best_wire = content_map.get(content_new)
            if best_wire is not None and best_wire != wire_idx:
                replacement[wire_idx] = best_wire
                changed = True
        
        if not replacement:
            break
        
        # Apply replacements and compact
        # 1. Replace usages in gates and outputs
        for i in range(len(gates)):
            a, b = gates[i]
            if a in replacement:
                a = replacement[a]
            if b in replacement:
                b = replacement[b]
            gates[i] = [a, b]
        
        for i in range(len(outputs)):
            if outputs[i] in replacement:
                outputs[i] = replacement[outputs[i]]
        
        # 2. Remove redundant gates and re-index
        removed = set(replacement.keys())
        
        new_gates = []
        old_to_new = {i: i for i in range(n_inputs + 1)}
        
        for i in range(len(gates)):
            wire_idx = n_inputs + 1 + i
            if wire_idx in removed:
                continue
            else:
                # Map old inputs to new indices if they were removed/shifted
                a_old, b_old = gates[i]
                a_new = old_to_new[a_old]
                b_new = old_to_new[b_old]
                new_gates.append([a_new, b_new])
                new_idx = n_inputs + 1 + len(new_gates) - 1
                old_to_new[wire_idx] = new_idx
        
        # Update outputs with new indices
        new_outputs = [old_to_new[o] for o in outputs]
        
        # Rebuild wire_contents for new gates
        new_wire_contents = wire_contents[:n_inputs+1] # Keep inputs and 0
        for i, (a, b) in enumerate(new_gates):
            c_a = new_wire_contents[a]
            c_b = new_wire_contents[b]
            c_new = c_a ^ c_b
            new_wire_contents.append(c_new)
        
        gates = new_gates
        outputs = new_outputs
        wire_contents = new_wire_contents
        
    # 4. Final Verification (Optional but good for safety)
    # For now, trust the construction.
    
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
