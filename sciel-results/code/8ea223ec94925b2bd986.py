"""XOR-circuit synthesis with subexpression sharing."""
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    # Normalize input
    if isinstance(matrix, list):
        matrix = np.array(matrix, dtype=np.uint8)
    matrix = (matrix > 0).astype(np.uint8)
    
    m, n = matrix.shape
    
    # If no rows, return empty
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert rows to Python ints for fast XOR
    def row_to_int(row):
        v = 0
        for i in range(n):
            if row[i]:
                v |= (1 << i)
        return v
    
    rows = [row_to_int(r) for r in matrix]
    
    # Trivial case: single row
    if m == 1:
        if rows[0] == 0:
            return {'gates': [], 'outputs': [n]}  # constant zero
        # Need to build this row from inputs
        # Direct: XOR all set bits
        bits = [i for i in range(n) if (rows[0] >> i) & 1]
        if len(bits) == 1:
            return {'gates': [], 'outputs': [bits[0]]}
        gates = []
        # XOR all bits
        if len(bits) == 2:
            gates.append([bits[0], bits[1]])
            return {'gates': gates, 'outputs': [n + 1]}
        # Chain
        acc = bits[0]
        for b in bits[1:]:
            gates.append([acc, b])
            acc = n + len(gates)
        return {'gates': gates, 'outputs': [acc]}
    
    # Greedy synthesis with subexpression sharing
    # Available vectors: input wires 0..n-1, zero wire n
    # We'll build up a dictionary: vector_int -> wire_id
    
    # Start with input wires
    vec_to_wire = {}
    for i in range(n):
        vec_to_wire[1 << i] = i
    vec_to_wire[0] = n  # zero
    
    gates = []  # list of [wire_a, wire_b]
    
    def get_wire_for_vector(vec):
        """Get wire for vector, or create one via decomposition."""
        if vec in vec_to_wire:
            return vec_to_wire[vec]
        if vec == 0:
            return n
        # Decompose: find two available vectors whose XOR = vec
        # Try: for each available vector a, check if vec^a is available
        # This is O(|available|) per lookup
        # Better: iterate over set bits of vec, try to build up
        # Simplest: pick highest set bit, XOR with that input
        # But that's just the naive approach
        
        # Try to find a decomposition using existing wires
        # For each wire w in vec_to_wire (excluding 0 and vec itself):
        #   if vec ^ w in vec_to_wire: use that pair
        # This is expensive. Instead, use a recursive approach:
        # Decompose vec by picking a set bit b, XOR with input b, recurse on remainder
        
        # Actually, let's do: find any two vectors in vec_to_wire that XOR to vec
        # If none, create new intermediate
        
        # Efficient approach: iterate over vectors in vec_to_wire
        # For small n this is fine
        candidates = list(vec_to_wire.keys())
        for a in candidates:
            if a == 0 or a == vec:
                continue
            b = vec ^ a
            if b in vec_to_wire:
                wa = vec_to_wire[a]
                wb = vec_to_wire[b]
                gates.append([wa, wb])
                new_wire = n + len(gates)
                vec_to_wire[vec] = new_wire
                return new_wire
            # Also check: can we create b first?
            # Recursive: get wire for b, then XOR with a
            # But this could be expensive. Let's just do one level of recursion
            if b != 0 and b not in vec_to_wire:
                # Try to get b
                wb = _get_wire_recursive(b, vec_to_wire, gates, n, depth=0)
                if wb is not None:
                    wa = vec_to_wire[a]
                    gates.append([wa, wb])
                    new_wire = n + len(gates)
                    vec_to_wire[vec] = new_wire
                    return new_wire
        # Fallback: build from scratch using input bits
        bits = [i for i in range(n) if (vec >> i) & 1]
        if not bits:
            return n
        if len(bits) == 1:
            vec_to_wire[vec] = bits[0]
            return bits[0]
        acc = bits[0]
        for b in bits[1:]:
            gates.append([acc, b])
            acc = n + len(gates)
        vec_to_wire[vec] = acc
        return acc
    
    def _get_wire_recursive(vec, vec_to_wire, gates, n, depth):
        """Recursively get wire for vector, limited depth."""
        if depth > 4:
            return None
        if vec in vec_to_wire:
            return vec_to_wire[vec]
        if vec == 0:
            return n
        # Try decomposition
        candidates = list(vec_to_wire.keys())
        for a in candidates:
            if a == 0 or a == vec:
                continue
            b = vec ^ a
            if b in vec_to_wire:
                wa = vec_to_wire[a]
                wb = vec_to_wire[b]
                gates.append([wa, wb])
                new_wire = n + len(gates)
                vec_to_wire[vec] = new_wire
                return new_wire
        # Fallback: build from inputs
        bits = [i for i in range(n) if (vec >> i) & 1]
        if not bits:
            return n
        if len(bits) == 1:
            vec_to_wire[vec] = bits[0]
            return bits[0]
        acc = bits[0]
        for b in bits[1:]:
            gates.append([acc, b])
            acc = n + len(gates)
        vec_to_wire[vec] = acc
        return acc
    
    # Process rows in order of decreasing weight (hardest first) to maximize sharing
    row_indices = list(range(m))
    row_indices.sort(key=lambda i: bin(rows[i]).count('1'), reverse=True)
    
    outputs = [0] * m
    for idx in row_indices:
        vec = rows[idx]
        if vec == 0:
            outputs[idx] = n
        else:
            outputs[idx] = get_wire_for_vector(vec)
    
    # Check deadline
    if time.time() > DEADLINE:
        return {'gates': gates, 'outputs': outputs}
    
    # Verify depth constraint (best effort)
    # Build dependency graph
    # wire -> list of (parent_wire1, parent_wire2)
    # Compute depth via topological sort
    
    # For now, return the result
    return {'gates': gates, 'outputs': outputs}
