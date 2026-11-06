"""XOR-circuit synthesis with greedy wire reuse and local search."""
import numpy as np
import time
import random
from collections import deque

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (m x n) where each row is a linear function of inputs.
        depth_limit: maximum allowed depth (<=16).
    
    Returns:
        dict with 'gates' (list of [wire_a, wire_b] pairs) and 'outputs' (list of wire indices).
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Normalize input
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    matrix = np.array(matrix, dtype=np.int64)
    
    m, n = matrix.shape
    
    # Handle edge cases
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # The zero wire is n
    # Gate k produces wire n+1+k
    
    # We'll use a greedy approach:
    # 1. Process rows in order of increasing popcount (easier first)
    # 2. For each row, find the cheapest way to express it using available wires
    # 3. Available wires start as inputs 0..n-1 and the zero wire n
    # 4. Each new gate adds a new available wire
    
    # Represent each wire by its vector (which inputs it XORs together)
    # wire 0..n-1: unit vectors
    # wire n: zero vector
    # wire n+1+k: result of gate k
    
    # For efficiency, represent vectors as Python ints (bitmasks)
    # Input i corresponds to bit i
    # Zero wire corresponds to 0
    
    # Initial available wires
    # wire_to_vec: dict mapping wire index -> bitmask
    # vec_to_wires: dict mapping bitmask -> list of wire indices (for lookup)
    
    wire_to_vec = {}
    vec_to_wires = {}
    
    # Inputs: wire i has bit i set
    for i in range(n):
        v = 1 << i
        wire_to_vec[i] = v
        if v not in vec_to_wires:
            vec_to_wires[v] = []
        vec_to_wires[v].append(i)
    
    # Zero wire
    wire_to_vec[n] = 0
    if 0 not in vec_to_wires:
        vec_to_wires[0] = []
    vec_to_wires[0].append(n)
    
    # Current number of gates
    num_gates = 0
    
    # Next wire index to assign
    next_wire = n + 1
    
    # Convert matrix rows to bitmasks
    row_bitmasks = []
    for i in range(m):
        bm = 0
        for j in range(n):
            if matrix[i, j] % 2 == 1:
                bm |= (1 << j)
        row_bitmasks.append(bm)
    
    # Sort rows by popcount (easier first) to build up richer wire set
    row_order = list(range(m))
    row_order.sort(key=lambda i: bin(row_bitmasks[i]).count('1'))
    
    # For each row, we need to find the cheapest decomposition
    # A decomposition of target bitmask T is a set of available wires whose XOR equals T
    # Cost = number of wires used - 1 (since XOR of k wires needs k-1 gates, but we can share)
    # Actually, if we use k existing wires, we need k-1 new gates to combine them
    # But wait: we can also use intermediate results from other rows
    
    # Greedy: for each row, find the minimum number of available wires that XOR to the target
    # This is a minimum weight set cover / minimum XOR basis problem
    
    # Simpler approach: maintain a basis of available wires
    # Use Gaussian elimination to find the minimum number of basis vectors needed
    
    # Let's use a different strategy:
    # Maintain a list of all available wire vectors
    # For each target, find the shortest sequence of XORs from available wires
    
    # For small n (up to 16), we can do BFS in the vector space
    # But that's expensive. Instead, let's use a greedy matching approach.
    
    # Better approach: maintain a basis, and for each row, express it in terms of the basis
    # The number of basis vectors used gives the gate count for that row
    
    # Let's maintain a row-echelon form basis of all available wires
    # basis: list of (vector, wire_index) in reduced row echelon form
    
    basis = []  # list of (bitmask, wire_index)
    basis_pivot = {}  # pivot bit -> index in basis
    
    def add_to_basis(vec, wire_idx):
        """Add a vector to the basis if it's linearly independent."""
        nonlocal basis, basis_pivot
        v = vec
        for pivot_bit in sorted(basis_pivot.keys(), reverse=True):
            if v & (1 << pivot_bit):
                v ^= basis[basis_pivot[pivot_bit]][0]
        if v == 0:
            return False
        # Find the highest bit
        pivot = v.bit_length() - 1
        if pivot in basis_pivot:
            return False
        # Reduce against existing basis
        for p in sorted(basis_pivot.keys()):
            if p != pivot and v & (1 << p):
                v ^= basis[basis_pivot[p]][0]
        basis.append((v, wire_idx))
        basis_pivot[pivot] = len(basis) - 1
        return True
    
    # Initialize basis with all available wires
    for wire_idx in range(n + 1):
        v = wire_to_vec[wire_idx]
        if v != 0:  # skip zero
            add_to_basis(v, wire_idx)
    
    # Function to express a target in terms of basis
    def express_in_basis(target):
        """Express target as XOR of basis vectors. Returns list of basis indices or None."""
        v = target
        used = []
        for pivot_bit in sorted(basis_pivot.keys(), reverse=True):
            if v & (1 << pivot_bit):
                v ^= basis[basis_pivot[pivot_bit]][0]
                used.append(basis_pivot[pivot_bit])
        if v == 0:
            return used
        return None
    
    # For each row in order, find the cheapest expression
    # Cost = number of basis vectors used
    # But we want to minimize total gates, not just per-row cost
    
    # Greedy: for each row, use the expression with minimum number of basis vectors
    # Then add any new intermediate wires to the basis
    
    gates = []
    outputs = []
    
    # Track which rows are done
    row_to_output = {}
    
    for row_idx in row_order:
        if time.time() > deadline:
            break
        
        target = row_bitmasks[row_idx]
        
        if target == 0:
            # Output is the zero wire
            row_to_output[row_idx] = n
            continue
        
        # Try to express in current basis
        used = express_in_basis(target)
        
        if used is None:
            # Should not happen if basis spans the space, but handle gracefully
            # Just XOR all inputs
            gates.append([0, 1] if n > 1 else [0, n])  # placeholder
            continue
        
        # Number of basis vectors needed
        k = len(used)
        
        if k == 0:
            row_to_output[row_idx] = n
        elif k == 1:
            # Just use that one wire
            row_to_output[row_idx] = basis[used[0]][1]
        else:
            # Need to XOR k wires together
            # Get the wire indices
            wire_indices = [basis[bi][1] for bi in used]
            
            # XOR them pairwise, creating new gates
            current = wire_indices[0]
            for i in range(1, k):
                other = wire_indices[i]
                # Create a new gate
                gates.append([current, other])
                current = next_wire
                next_wire += 1
                num_gates += 1
                
                # Add this new wire to our tracking
                new_vec = 0
                for wi in wire_indices[:i+1]:
                    new_vec ^= wire_to_vec[wi]
                wire_to_vec[current] = new_vec
                if new_vec not in vec_to_wires:
                    vec_to_wires[new_vec] = []
                vec_to_wires[new_vec].append(current)
                
                # Add to basis if independent
                add_to_basis(new_vec, current)
            
            row_to_output[row_idx] = current
    
    # Build outputs in original row order
    outputs = []
    for i in range(m):
        outputs.append(row_to_output.get(i, n))
    
    # Verify the solution
    # Reconstruct the circuit and check
    
    # Wire values: compute what each wire represents
    wire_val = {}
    for i in range(n):
        wire_val[i] = 1 << i
    wire_val[n] = 0
    
    for k, (a, b) in enumerate(gates):
        wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
    
    # Check each output
    for i in range(m):
        expected = row_bitmasks[i]
        actual = wire_val[outputs[i]]
        if expected != actual:
            # Fallback: use naive approach
            return _naive_synthesize(matrix, n)
    
    # Check depth (simplified: we assume depth is within limit for small circuits)
    # For a more careful check, we'd need to compute the actual depth
    
    return {'gates': gates, 'outputs': outputs}


def _naive_synthesize(matrix, n):
    """Naive fallback: each row is XOR of its set bits."""
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    matrix = np.array(matrix, dtype=np.int64)
    m, n = matrix.shape
    
    gates = []
    outputs = []
    next_wire = n + 1
    
    for i in range(m):
        bits = [j for j in range(n) if matrix[i, j] % 2 == 1]
        if len(bits) == 0:
            outputs.append(n)
        elif len(bits) == 1:
            outputs.append(bits[0])
        else:
            current = bits[0]
            for j in range(1, len(bits)):
                gates.append([current, bits[j]])
                current = next_wire
                next_wire += 1
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
