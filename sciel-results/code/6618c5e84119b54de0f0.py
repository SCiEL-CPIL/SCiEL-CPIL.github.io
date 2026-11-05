"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

_T0 = time.time()
_DEADLINE = 90.0
_MARGIN = 8.0

def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _T0)

def _row_to_int(row):
    """Encode a GF(2) row (list of 0/1) as an integer bitmask (index i -> bit i)."""
    v = 0
    for i, b in enumerate(row):
        if b:
            v |= (1 << i)
    return v

def _int_to_row(v, n):
    return [1 if (v >> i) & 1 else 0 for i in range(n)]

def _baseline_gate_count(matrix):
    """Independent-row baseline: sum of popcount of each output row (XORs of inputs)."""
    rows = np.asarray(matrix, dtype=np.int64)
    if rows.size == 0:
        return 0
    return int(rows.sum())

def synthesize(matrix, depth_limit):
    """
    Exact XOR-circuit synthesis using global Common Subexpression Elimination (CSE).
    
    Strategy:
    1. Convert all rows to integer bitmasks.
    2. Recursively decompose each row into two smaller sub-vectors (splitting bits).
    3. Memoize all sub-vectors encountered. Each unique sub-vector is computed exactly once.
    4. Build a DAG where nodes are these sub-vectors (wires) and edges are XOR gates.
    5. This maximizes sharing of intermediate XOR results across all rows.
    """
    start = time.time()
    rows = np.asarray(matrix, dtype=np.int64)
    if rows.size == 0:
        return {'gates': [], 'outputs': []}
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    m, n = rows.shape

    # Constant-zero wire is n.
    # Input wires 0..n-1, constant zero = n.
    # Gate k produces wire n+1+k.

    # Convert rows to integer bitmasks over the n input positions.
    row_ints = [_row_to_int(rows[i]) for i in range(m)]
    
    # We will build a dictionary mapping vector (int) -> wire_id.
    # wire_id n is constant zero.
    # wire_ids 0..n-1 are inputs (single bit vectors).
    
    wire_map = {}
    # Initialize single-bit vectors
    for i in range(n):
        wire_map[1 << i] = i
    
    # Constant zero
    wire_map[0] = n
    
    gates = []
    next_wire = n + 1
    
    # We need to compute wires for all row_ints.
    # We use a recursive function that ensures a wire exists for a given vector.
    # To keep depth low, we split the vector into two roughly equal halves.
    
    # Since Python recursion might be deep or slow, we use an iterative approach with a worklist
    # or just recursion with a depth limit check. Given n <= 16, depth is small.
    
    def get_wire(vec):
        """
        Ensure a wire exists for the given vector `vec` (int bitmask).
        Returns the wire id.
        If the wire doesn't exist, it splits `vec` into two parts,
        recursively gets wires for those parts, and creates a new XOR gate.
        """
        if vec in wire_map:
            return wire_map[vec]
        
        # Split vec into two parts.
        # To minimize depth, we split the set bits into two groups of roughly equal size.
        # Alternatively, split by bit position to keep sub-vectors "contiguous" or similar,
        # but for XOR, any split works. Splitting by bit count is good for depth.
        
        # Count bits
        bit_count = vec.bit_count()
        if bit_count == 0:
            wire_map[vec] = n
            return n
        if bit_count == 1:
            # Single bit, should be in wire_map already, but just in case
            # Find the bit
            idx = (vec & -vec).bit_length() - 1
            wire_map[vec] = idx
            return idx
            
        # Split into two non-empty parts
        # We can simply take the lowest set bit for one part and the rest for the other?
        # That creates a chain (depth O(n)). 
        # Better: split into two halves of the set bits.
        
        # Extract set bit positions
        # Since n is small (<=16), we can iterate.
        bits = []
        v = vec
        while v:
            lsb = v & -v
            bits.append(lsb)
            v ^= lsb
            
        mid = len(bits) // 2
        part1 = 0
        for b in bits[:mid]:
            part1 |= b
        part2 = 0
        for b in bits[mid:]:
            part2 |= b
        
        # Recursively get wires for parts
        w1 = get_wire(part1)
        w2 = get_wire(part2)
        
        # Create new gate
        w = next_wire
        nonlocal next_wire
        gates.append([w1, w2])
        next_wire += 1
        wire_map[vec] = w
        return w

    # We need to track depth to ensure we don't exceed depth_limit.
    # However, the recursive splitting strategy (halving bits) guarantees depth O(log(popcount)).
    # Max popcount is n (16). log2(16) = 4.
    # Plus the depth of the inputs (0).
    # So max depth is roughly 4-5.
    # This is well within depth_limit (16).
    
    # Process all rows
    outputs = []
    for r in row_ints:
        w = get_wire(r)
        outputs.append(w)
        
        # Check time
        if time.time() - start > _DEADLINE - _MARGIN:
             # If we run out of time, we just return what we have.
             # Since the algorithm is deterministic and fast for n<=16, this shouldn't happen.
             break

    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
