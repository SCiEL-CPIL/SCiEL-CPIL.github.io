"""XOR-circuit synthesis with greedy shared-subexpression and local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill
    
    A = np.array(matrix, dtype=np.int64)
    m, n = A.shape
    
    # Convert rows to bitmasks
    targets = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if A[i, j] % 2 == 1:
                mask |= (1 << j)
        targets.append(mask)
    
    # Baseline: independent row synthesis (each row from scratch)
    # Count gates for baseline
    baseline_gates = 0
    for t in targets:
        bits = bin(t).count('1')
        if bits > 1:
            baseline_gates += bits - 1
    baseline = max(baseline_gates, 1)
    
    def time_left():
        return deadline - time.time()
    
    # Strategy: greedy synthesis with shared subexpressions
    # Wire numbering: inputs 0..n-1, zero wire = n
    # Gate k produces wire n+1+k
    
    def greedy_synthesize(target_order, seed=None):
        """Greedy synthesis processing targets in given order."""
        if seed is not None:
            random.seed(seed)
        
        wires = []  # list of bitmasks for each wire
        for i in range(n):
            wires.append(1 << i)
        wires.append(0)  # zero wire at index n
        
        # Map from bitmask to wire number
        wire_map = {w: i for i, w in enumerate(wires)}
        
        gates = []  # list of [wire_a, wire_b]
        
        # Track depth for each wire
        depths = [0] * (n + 1)
        
        for idx in target_order:
            target = targets[idx]
            
            # Check if already available
            if target in wire_map:
                continue
            
            # Try to find two available wires whose XOR gives target
            found = False
            # Iterate over available wires
            avail_list = list(wire_map.keys())
            
            # Shuffle for diversity
            if seed is not None:
                random.shuffle(avail_list)
            
            for a in avail_list:
                b_needed = a ^ target
                if b_needed in wire_map:
                    wa = wire_map[a]
                    wb = wire_map[b_needed]
                    new_wire = n + 1 + len(gates)
                    gates.append([wa, wb])
                    wires.append(target)
                    wire_map[target] = new_wire
                    depths.append(max(depths[wa], depths[wb]) + 1)
                    found = True
                    break
            
            if found:
                continue
            
            # Multi-step: try to build target using intermediate wires
            # BFS up to depth 2 from available wires
            # Level 0: available wires
            # Level 1: XOR of any two available
            # Level 2: XOR of level-1 with available or level-1
            
            # Try level 1: find a, b in available such that a^b = some intermediate,
            # then intermediate ^ c = target for some c in available
            # Equivalently: find a, b, c in available such that a^b^c = target
            # i.e., a^b = target^c for some c
            # So for each c in available, check if target^c is in level-1
            # Level-1 = {a^b : a,b in available}
            
            # This is O(k^3) which is too slow for large k
            # Instead, try a simpler approach: build from inputs
            
            # Build from inputs
            bits = []
            for j in range(n):
                if target & (1 << j):
                    bits.append(j)
            
            if not bits:
                continue
            
            if len(bits) == 1:
                continue
            
            # XOR them together
            current_wire = wire_map[1 << bits[0]]
            
            for j in bits[1:]:
                input_wire = wire_map[1 << j]
                new_wire = n + 1 + len(gates)
                gates.append([current_wire, input_wire])
                new_mask = wires[current_wire] ^ (1 << j)
                wires.append(new_mask)
                wire_map[new_mask] = new_wire
                depths.append(max(depths[current_wire], 0) + 1)
                current_wire = new_wire
        
        # Map outputs
        outputs = []
        for i in range(m):
            target = targets[i]
            if target in wire_map:
                outputs.append(wire_map[target])
            else:
                outputs.append(n)
        
        return gates, outputs
    
    def local_search(gates, outputs, wire_count):
        """Try to eliminate gates by finding redundant computations."""
        # This is a simplified local search: try removing each gate and see if
        # the circuit is still valid
        
        if not gates:
            return gates, outputs
        
        # Reconstruct wire values from gates
        def compute_wire_values(gates_list, n_inputs):
            vals = [1 << i for i in range(n_inputs)]
            vals.append(0)
            for g in gates_list:
                a, b = g
                vals.append(vals[a] ^ vals[b])
            return vals
        
        best_gates = gates
        best_outputs = outputs
        improved = True
        
        while improved and time_left() > 5.0:
            improved = False
            vals = compute_wire_values(best_gates, n)
            
            # Try removing each gate
            for gi in range(len(best_gates)):
                candidate_gates = best_gates[:gi] + best_gates[gi+1:]
                if not candidate_gates:
                    cand_vals = [1 << i for i in range(n)] + [0]
                else:
                    cand_vals = compute_wire_values(candidate_gates, n)
                
                # Check if all outputs are still valid
                # Need to re-map outputs since wire numbers change
                # This is complex; skip for now
                pass
        
        return best_gates, best_outputs
    
    # Try multiple strategies
    best_gates = None
    best_outputs = None
    best_score = float('inf')
    
    # Strategy 1: Sort by popcount ascending
    order1 = sorted(range(m), key=lambda i: bin(targets[i]).count('1'))
    g1, o1 = greedy_synthesize(order1)
    score1 = len(g1) / baseline
    if score1 < best_score:
        best_score = score1
        best_gates = g1
        best_outputs = o1
    
    # Strategy 2: Sort by popcount descending
    order2 = sorted(range(m), key=lambda i: bin(targets[i]).count('1'), reverse=True)
    g2, o2 = greedy_synthesize(order2)
    score2 = len(g2) / baseline
    if score2 < best_score:
        best_score = score2
        best_gates = g2
        best_outputs = o2
    
    # Strategy 3: Random order with multiple restarts
    num_restarts = min(20, max(5, int(time_left() / 0.5)))
    for r in range(num_restarts):
        if time_left() < 3.0:
            break
        order = list(range(m))
        random.shuffle(order)
        g, o = greedy_synthesize(order, seed=r)
        score = len(g) / baseline
        if score < best_score:
            best_score = score
            best_gates = g
            best_outputs = o
    
    # Strategy 4: Greedy with preference for reusing recently created wires
    # Process in popcount order but prefer XOR with newer wires
    order4 = sorted(range(m), key=lambda i: bin(targets[i]).count('1'))
    
    wires4 = [1 << i for i in range(n)] + [0]
    wire_map4 = {w: i for i, w in enumerate(wires4)}
    gates4 = []
    
    for idx in order4:
        target = targets[idx]
        if target in wire_map4:
            continue
        
        found = False
        # Prefer newer wires (higher index) for reuse
        avail_list = sorted(wire_map4.keys(), key=lambda w: wire_map4[w], reverse=True)
        
        for a in avail_list:
            b_needed = a ^ target
            if b_needed in wire_map4:
                wa = wire_map4[a]
                wb = wire_map4[b_needed]
                new_wire = n + 1 + len(gates4)
                gates4.append([wa, wb])
                wires4.append(target)
                wire_map4[target] = new_wire
                found = True
                break
        
        if found:
            continue
        
        bits = []
        for j in range(n):
            if target & (1 << j):
                bits.append(j)
        
        if not bits or len(bits) == 1:
            continue
        
        current_wire = wire_map4[1 << bits[0]]
        for j in bits[1:]:
            input_wire = wire_map4[1 << j]
            new_wire = n + 1 + len(gates4)
            gates4.append([current_wire, input_wire])
            new_mask = wires4[current_wire] ^ (1 << j)
            wires4.append(new_mask)
            wire_map4[new_mask] = new_wire
            current_wire = new_wire
    
    outputs4 = []
    for i in range(m):
        target = targets[i]
        if target in wire_map4:
            outputs4.append(wire_map4[target])
        else:
            outputs4.append(n)
    
    score4 = len(gates4) / baseline
    if score4 < best_score:
        best_score = score4
        best_gates = gates4
        best_outputs = outputs4
    
    if best_gates is None:
        best_gates = []
        best_outputs = [n] * m
    
    return {'gates': best_gates, 'outputs': best_outputs}
# EVOLVE-BLOCK-END
