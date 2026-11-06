"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 82.0  # safety margin
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Parse rows as frozensets of column indices
    rows = []
    for i in range(m):
        row_set = frozenset(j for j in range(n) if matrix[i][j] % 2 == 1)
        rows.append(row_set)
    
    # Wire assignment: inputs are 0..n-1, const-zero is n
    # New gates get wires n+1, n+2, ...
    next_wire = n + 1
    gates = []  # list of [wire_a, wire_b]
    
    # Memo: frozenset -> wire_id
    memo = {}
    
    def elapsed_ok():
        return time.time() < DEADLINE
    
    def compute_set(wire):
        """Return the frozenset of input indices that wire represents."""
        if wire < n:
            return frozenset([wire])
        if wire == n:
            return frozenset()
        idx = wire - (n + 1)
        if idx < 0 or idx >= len(gates):
            return frozenset()
        a, b = gates[idx]
        return compute_set(a) ^ compute_set(b)
    
    def add_gate(set_a, set_b):
        """Add an XOR gate combining set_a and set_b. Return wire id."""
        nonlocal next_wire
        result = set_a ^ set_b
        if result in memo:
            return memo[result]
        # Find wires for set_a and set_b
        wire_a = memo.get(set_a)
        wire_b = memo.get(set_b)
        if wire_a is None:
            # Need to synthesize set_a
            if not set_a:
                wire_a = n
            elif len(set_a) == 1:
                wire_a = next(iter(set_a))
            else:
                # Recursively synthesize
                # Split into two parts
                items = list(set_a)
                mid = len(items) // 2
                part1 = frozenset(items[:mid])
                part2 = frozenset(items[mid:])
                wire_a = add_gate(part1, part2)
        if wire_b is None:
            if not set_b:
                wire_b = n
            elif len(set_b) == 1:
                wire_b = next(iter(set_b))
            else:
                items = list(set_b)
                mid = len(items) // 2
                part1 = frozenset(items[:mid])
                part2 = frozenset(items[mid:])
                wire_b = add_gate(part1, part2)
        
        gates.append([wire_a, wire_b])
        wire = next_wire
        next_wire += 1
        memo[result] = wire
        return wire
    
    # Greedy synthesis: process rows, try to reuse existing wires
    # Strategy: for each row, try to express it as XOR of existing wires
    # This is like a linear algebra problem over GF(2)
    
    # First pass: simple greedy with memoization
    outputs = []
    for i in range(m):
        if not elapsed_ok():
            break
        target = rows[i]
        if target in memo:
            outputs.append(memo[target])
        elif not target:
            outputs.append(n)
        elif len(target) == 1:
            outputs.append(next(iter(target)))
        else:
            # Try to find a subset of existing wires that XOR to target
            # Simple approach: just synthesize directly
            wire = add_gate(target, frozenset())
            outputs.append(wire)
    
    # Second pass: try to optimize by finding better decompositions
    # For each output wire that was newly created, try to see if it can be
    # expressed as XOR of fewer existing wires
    
    # Build a lookup from wire to set
    wire_to_set = {}
    for w in range(n):
        wire_to_set[w] = frozenset([w])
    wire_to_set[n] = frozenset()
    for idx, g in enumerate(gates):
        w = n + 1 + idx
        wire_to_set[w] = compute_set(w)
    
    # Try to optimize: for each newly created gate, see if its output set
    # can be expressed as XOR of a smaller number of existing wires
    # This is a set cover / linear combination problem
    
    # Simple improvement: try pairwise combinations
    # For each output, if it was created as a gate, try to find 2 existing wires
    # that XOR to it
    
    optimized_outputs = []
    for i in range(m):
        if not elapsed_ok():
            break
        target = rows[i]
        if target in memo:
            optimized_outputs.append(memo[target])
        elif not target:
            optimized_outputs.append(n)
        elif len(target) == 1:
            optimized_outputs.append(next(iter(target)))
        else:
            # Try to find two wires whose XOR gives target
            found = False
            # Check against all existing wires
            wire_list = list(wire_to_set.keys())
            for wa in wire_list:
                if not elapsed_ok():
                    break
                set_a = wire_to_set[wa]
                needed = target ^ set_a
                if needed in memo:
                    # We can use wa and memo[needed]
                    # But we need to make sure this doesn't create a cycle
                    # Since we're just reading, it's fine
                    optimized_outputs.append(wa)
                    # We need to add a gate if needed, but for now just record
                    # Actually, we need the actual wire, so let's add the gate
                    wire_b = memo[needed]
                    gates.append([wa, wire_b])
                    wire = next_wire
                    next_wire += 1
                    memo[target] = wire
                    wire_to_set[wire] = target
                    optimized_outputs[-1] = wire
                    found = True
                    break
            if not found:
                # Synthesize directly
                wire = add_gate(target, frozenset())
                optimized_outputs.append(wire)
    
    # Ensure all outputs are valid
    final_outputs = []
    for i in range(m):
        if i < len(optimized_outputs):
            final_outputs.append(optimized_outputs[i])
        else:
            # Fallback
            target = rows[i]
            if target in memo:
                final_outputs.append(memo[target])
            elif not target:
                final_outputs.append(n)
            elif len(target) == 1:
                final_outputs.append(next(iter(target)))
            else:
                wire = add_gate(target, frozenset())
                final_outputs.append(wire)
    
    return {'gates': gates, 'outputs': final_outputs}
# EVOLVE-BLOCK-END
