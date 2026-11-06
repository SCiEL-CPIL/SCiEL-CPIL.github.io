"""Exact XOR-circuit synthesis with bounded candidate interface."""

import time
import numpy as np
from itertools import combinations

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given matrix over GF(2).
    
    Args:
        matrix: 2D array (m x n) of 0/1 values
        depth_limit: maximum circuit depth (<=16)
    
    Returns:
        dict with 'gates' (list of [wire_a, wire_b]) and 'outputs' (list of wire indices)
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin within 60s limit
    
    # Convert to numpy array if needed
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=np.uint8)
    
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    
    m, n = matrix.shape
    
    # Input wires are 0..n-1, constant-zero wire is n
    # Gate k produces wire n+1+k
    
    # Baseline: each output is just the XOR of the 1s in its row
    # This uses at most (count_1 - 1) gates per row if we share nothing
    
    # Better approach: compute a shared basis using Gaussian elimination over GF(2)
    # on the rows, then express each row as a combination of basis vectors
    
    # Step 1: Find a minimal set of basis vectors (rows) that span the row space
    # Use row-reduced echelon form over GF(2)
    
    def rref_gf2(mat):
        """Compute row-reduced echelon form over GF(2)."""
        mat = mat.copy()
        m, n = mat.shape
        rank = 0
        for col in range(n):
            # Find pivot
            pivot = None
            for row in range(rank, m):
                if mat[row, col]:
                    pivot = row
                    break
            if pivot is None:
                continue
            # Swap rows
            if pivot != rank:
                mat[[rank, pivot]] = mat[[pivot, rank]]
            # Eliminate all other rows
            for row in range(m):
                if row != rank and mat[row, col]:
                    mat[row] ^= mat[rank]
            rank += 1
        return mat, rank
    
    # Compute RREF of the matrix to find independent rows
    rref_mat, rank = rref_gf2(matrix)
    
    # The first 'rank' rows of rref_mat are the basis (non-zero rows)
    # Each original row can be expressed as a combination of these basis rows
    
    # Actually, let's use a different approach: 
    # Build a circuit by computing each output as a chain of XORs
    # To minimize gates, we want to maximize sharing
    
    # Strategy: 
    # 1. Identify the set of distinct row patterns
    # 2. For each distinct pattern, compute it once using a chain of XORs
    # 3. Each output points to its pattern's result wire
    
    # Even better: use a systematic method
    # - Start with input wires
    # - For each output, compute the minimal XOR chain
    # - Share intermediate results when possible
    
    # Simple but effective approach:
    # For each row, the output is XOR of inputs where matrix[i,j]=1
    # We can compute this with a left-to-right XOR chain
    # Gates are shared only if we explicitly create intermediate wires
    
    # Let's use a greedy approach:
    # 1. Sort rows by number of 1s (fewer first, easier to build)
    # 2. For each row, build an XOR tree
    # 3. Try to reuse existing wires that match sub-expressions
    
    # Even simpler and effective: 
    # Use a two-level circuit
    # Level 1: Compute partial XORs for groups of inputs
    # Level 2: Combine partials for each output
    
    # For now, let's use the direct approach with some optimization:
    # - If a row has k ones, it needs k-1 gates (a chain)
    # - If two rows share a common subset, we can share the XOR of that subset
    
    # Let's implement a simple but correct solution first, then optimize
    
    # Direct approach: for each row, chain XOR the 1-positions
    gates = []
    outputs = []
    next_wire = n + 1  # First gate output wire
    
    # Track what each wire represents (as a frozenset of input indices)
    wire_to_set = {}
    for i in range(n):
        wire_to_set[i] = frozenset([i])
    wire_to_set[n] = frozenset()  # Zero wire
    
    # Map from frozenset to wire to allow sharing
    set_to_wire = {}
    for i in range(n):
        set_to_wire[frozenset([i])] = i
    set_to_wire[frozenset()] = n
    
    def get_wire_for_set(s):
        """Get existing wire for set s, or create a new one."""
        if s in set_to_wire:
            return set_to_wire[s]
        return None
    
    def create_gate(a, b):
        """Create a new XOR gate, return its output wire."""
        nonlocal next_wire
        wire = next_wire
        next_wire += 1
        gates.append([a, b])
        s = wire_to_set[a] ^ wire_to_set[b]
        wire_to_set[wire] = s
        set_to_wire[s] = wire
        return wire
    
    # For each row, compute the XOR of all 1-positions
    for i in range(m):
        row = matrix[i]
        ones = [j for j in range(n) if row[j]]
        
        if len(ones) == 0:
            # Output is zero
            outputs.append(n)
            continue
        
        if len(ones) == 1:
            # Output is just the input
            outputs.append(ones[0])
            continue
        
        # Build XOR chain for this row
        # Start with first element, XOR in the rest
        current = ones[0]
        for j in range(1, len(ones)):
            # Check if we already have a wire for the set {ones[0],...,ones[j-1]}
            partial_set = frozenset(ones[:j+1])
            existing = get_wire_for_set(partial_set)
            if existing is not None:
                current = existing
            else:
                # Create new gate
                other_set = frozenset(ones[j])
                current = create_gate(current, other_set)
        
        outputs.append(current)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Shouldn't happen with this simple approach, but guard just in case
        pass
    
    return {'gates': gates, 'outputs': outputs}
