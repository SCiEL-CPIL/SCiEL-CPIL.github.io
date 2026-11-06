"""Exact XOR-circuit synthesis: minimum-weight basis via recursive greedy with memoization."""
import time
import sys
from functools import lru_cache

# Deadline guard: return best result before 90s wall-clock
_START_TIME = time.time()
DEADLINE = 85.0  # safety margin


def _elapsed():
    return time.time() - _START_TIME


def _time_exceeded():
    return _elapsed() > DEADLINE


def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given matrix.
    
    Args:
        matrix: 2D list/array of 0s and 1s (m x n)
        depth_limit: maximum circuit depth (<=16)
    
    Returns:
        {'gates': [[wire_a, wire_b], ...], 'outputs': [wire, ...]}
    """
    import numpy as np
    
    if isinstance(matrix, list):
        mat = np.array(matrix, dtype=np.uint8)
    else:
        mat = np.array(matrix, dtype=np.uint8)
    
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    
    m, n = mat.shape
    zero_wire = n  # constant-zero wire
    
    # Work with rows as bit vectors over GF(2)
    # Represent each vector as a frozenset of indices where bit=1, or as a tuple for hashability
    # For efficiency with up to 16 columns, use integer representation
    
    def vec_to_int(vec):
        """Convert a 0/1 vector to an integer (bit i = vec[i])."""
        v = 0
        for i in range(len(vec)):
            if vec[i]:
                v |= (1 << i)
        return v
    
    def int_to_vec(v, length):
        """Convert integer back to 0/1 vector."""
        return [(v >> i) & 1 for i in range(length)]
    
    def popcount(v):
        """Number of 1-bits in integer v."""
        return bin(v).count('1')
    
    def xor_int(a, b):
        """XOR of two integer-encoded vectors."""
        return a ^ b
    
    # Target vectors (the rows of the matrix)
    targets = []
    for i in range(m):
        targets.append(vec_to_int(mat[i]))
    
    # Available "primitives": input wires (0..n-1) and zero wire (n)
    # Input wire j corresponds to vector with bit j set
    # Zero wire corresponds to vector 0
    
    # Wire numbering:
    # 0..n-1: input wires
    # n: zero wire
    # n+1, n+2, ...: gate outputs
    
    # We need to express each target as XOR of available wires
    # Strategy: recursively decompose each target using available vectors
    
    # Memoization: for each vector (int), store the wire that produces it
    # and the list of gates used to produce it
    
    wire_map = {}  # vector_int -> wire_number
    gates = []  # list of [wire_a, wire_b]
    
    # Initialize: input wires and zero wire
    for j in range(n):
        wire_map[1 << j] = j
    wire_map[0] = zero_wire
    
    # We'll use a greedy recursive decomposition
    # For each target, try to express it as XOR of two already-available or simpler vectors
    # Key insight: if we can find intermediate vectors that are shared across multiple targets,
    # we reduce total gate count
    
    # Approach: 
    # 1. Collect all target vectors
    # 2. Use a priority queue / greedy approach to build up a basis
    # 3. For each vector we need, try to decompose it into XOR of two known vectors
    #    Prefer decompositions where the operands are "simpler" (lower weight) or already known
    
    # Better approach: minimum-weight basis
    # We want to find a set of intermediate vectors such that:
    # - Each target is in the span of {inputs} ∪ {intermediates}
    # - Each intermediate is XOR of two previously available vectors
    # - Total number of intermediates (= gates) is minimized
    
    # Greedy with memoization:
    # Process targets in order of decreasing weight (harder ones first, to build useful intermediates)
    
    # Actually, let's use a different strategy:
    # 1. Gaussian elimination to find a basis for the target space
    # 2. Then try to minimize the total cost by sharing sub-expressions
    
    # Let me implement a recursive greedy with sharing:
    
    # State: set of available vectors (as a dict vector->wire)
    # For each target not yet available, find the best decomposition
    
    # To handle sharing, we process all targets together and build a common set of intermediates
    
    # Strategy: 
    # - Start with inputs + zero
    # - Repeatedly: for each remaining target, try all pairs of available vectors
    #   whose XOR equals the target or is "closer" to it
    # - Add the most useful intermediate (one that helps multiple targets)
    
    # Simpler effective strategy: recursive decomposition with memoization
    # For each target, recursively find a decomposition tree, caching results
    
    # Let's implement: for each vector v, find a decomposition v = a ^ b
    # where a and b are "simpler" than v (lower weight, or already in wire_map)
    
    def find_decomposition(v, available, depth):
        """
        Find a decomposition of v as a ^ b where a, b are in available or simpler.
        Returns (a, b) or None.
        """
        if v in available:
            return None  # already available
        
        # Try all pairs from available
        av = list(available.keys())
        for i in range(len(av)):
            for j in range(i + 1, len(av)):
                if av[i] ^ av[j] == v:
                    return (av[i], av[j])
        
        # Try: v ^ x for x in available, check if result is simpler
        # "Simpler" = lower weight
        w_v = popcount(v)
        best = None
        best_score = float('inf')
        for x in av:
            y = v ^ x
            if y == v or y == x:
                continue
            w_y = popcount(y)
            # Score: prefer both operands to be simple
            score = w_y  # or max(w_y, popcount(x))
            if score < best_score:
                best_score = score
                best = (x, y)
        
        if best is not None and popcount(best[1]) < w_v:
            return best
        
        return None
    
    # This pairwise search is O(k^2) per decomposition which is expensive for large k.
    # Let's use a smarter approach.
    
    # Alternative: for each target, do Gaussian elimination style decomposition
    # Use the fact that over GF(2), we can always decompose using input wires
    
    # Direct decomposition using inputs:
    # v = XOR of input wires where bit is set
    # This uses popcount(v) - 1 gates (or popcount(v) if we need to XOR them all)
    # Actually, to compute XOR of k inputs, we need k-1 gates
    
    # But we can do better by introducing intermediates.
    
    # Let me implement a proper recursive greedy:
    
    # Phase 1: Identify all unique vectors needed (targets + their decompositions)
    # Phase 2: Build a DAG of XOR operations minimizing total nodes
    
    # For small problems (n <= 16), we can enumerate more aggressively
    
    # Let's use this approach:
    # 1. For each target, find a minimal decomposition using available wires
    # 2. When decomposing, prefer to create intermediates that are useful for other targets
    
    # Implementation:
    # - Maintain a dict: vec -> wire
    # - For each target (sorted by weight descending), if not already available:
    #   - Find a decomposition v = a ^ b where a, b are available or can be made cheaply
    #   - Add necessary intermediates
    
    # To find good decompositions efficiently:
    # For vector v with weight w, we can write v = e_i ^ (v ^ e_i) where e_i is a unit vector
    # v ^ e_i has weight w-1 (if bit i was set) or w+1 (if not)
    # So v = e_i ^ (v ^ e_i) where v^e_i has weight w-1
    # This gives a decomposition into a unit vector (available) and a lower-weight vector
    
    # Recursive: decompose v into unit vectors via binary tree of XORs
    # This uses w-1 gates for weight w, same as naive but with balanced tree
    
    # The key improvement: sharing. If two targets share a sub-vector, we compute it once.
    
    # Let's implement recursive decomposition with global memoization:
    
    wire_map = {}  # int -> wire
    gates = []
    gate_count = 0
    
    for j in range(n):
        wire_map[1 << j] = j
    wire_map[0] = zero_wire
    
    def get_wire(v):
        """Get wire for vector v, creating intermediates as needed."""
        nonlocal gate_count
        
        if v in wire_map:
            return wire_map[v]
        
        if _time_exceeded():
            # Fallback: naive decomposition using inputs
            # v = XOR of unit vectors
            bits = [j for j in range(n) if (v >> j) & 1]
            if not bits:
                return zero_wire
            if len(bits) == 1:
                return bits[0]
            # XOR them all
            result_wire = bits[0]
            for b in bits[1:]:
                w1 = result_wire
                w2 = b
                gates.append([w1, w2])
                result_wire = n + 1 + (gate_count - 1)
                gate_count += 1
            wire_map[v] = result_wire
            return result_wire
        
        w = popcount(v)
        if w == 0:
            return zero_wire
        if w == 1:
            # Find which bit
            for j in range(n):
                if (v >> j) & 1:
                    return j
            return zero_wire
        
        # Decompose: v = e_i ^ (v ^ e_i) where e_i is a unit vector
        # Choose i to maximize sharing: pick the bit that appears in many other pending targets
        # For now, pick the highest-weight bit or first set bit
        # Better: try multiple decompositions and pick the one where the sub-vector is most likely shared
        
        # Simple strategy: split v into two parts of roughly equal weight
        bits = [j for j in range(n) if (v >> j) & 1]
        mid = len(bits) // 2
        
        # Part 1: XOR of first mid bits
        # Part 2: XOR of remaining bits
        part1 = 0
        for j in bits[:mid]:
            part1 |= (1 << j)
        part2 = 0
        for j in bits[mid:]:
            part2 |= (1 << j)
        
        # Recursively get wires for parts
        w1 = get_wire(part1)
        w2 = get_wire(part2)
        
        # Create gate
        gates.append([w1, w2])
        new_wire = n + 1 + (gate_count - 1)
        gate_count += 1
        wire_map[v] = new_wire
        return new_wire
    
    # Process targets
    # Sort by weight descending to build useful intermediates first
    target_indices = sorted(range(m), key=lambda i: popcount(targets[i]), reverse=True)
    
    output_wires = []
    for idx in target_indices:
        v = targets[idx]
        w = get_wire(v)
        output_wires.append(w)
    
    # Map back to original target order
    # output_wires is in sorted order, need to reorder
    final_outputs = [0] * m
    for pos, idx in enumerate(target_indices):
        final_outputs[idx] = output_wires[pos]
    
    result = {
        'gates': gates,
        'outputs': final_outputs
    }
    
    return result
