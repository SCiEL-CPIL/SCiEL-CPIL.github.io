import time
import numpy as np
import itertools

# EVOLVE-BLOCK-START
def _paleyn_orientation(p):
    """
    Construct Paley tournament on p vertices.
    Returns list of bits for pairs (i, j) i < j.
    0 means i -> j, 1 means j -> i.
    """
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)  # i -> j
            else:
                orientation.append(1)  # j -> i
    return orientation

def _build_in_bits(p, orientation):
    """
    Build in-neighbor bitsets from orientation.
    in_bits[v] is a bitmask where bit u is set if u -> v.
    """
    in_bits = [0] * p
    idx = 0
    for i in range(p):
        for j in range(i + 1, p):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                # i -> j
                in_bits[j] |= (1 << i)
            else:
                # j -> i
                in_bits[i] |= (1 << j)
    return in_bits

def _check_subset_valid(in_bits, subset, mask):
    """
    Check if every 4-subset of 'subset' has a common dominator within 'subset'.
    in_bits: global in-neighbor bitsets
    subset: list of vertex indices (subset of original vertices)
    mask: bitmask of vertices in subset
    """
    for a, b, c, d in itertools.combinations(subset, 4):
        if (in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d] & mask) == 0:
            return False
    return True

def _out_degree_in_subset(v, subset, in_bits):
    """
    Count out-degree of vertex v within subset.
    v -> u means bit v is set in in_bits[u].
    """
    count = 0
    for u in subset:
        if u != v and (in_bits[u] >> v) & 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with Paley(67) - known to satisfy the property
    p = 67
    orientation = _paleyn_orientation(p)
    in_bits = _build_in_bits(p, orientation)
    
    # Verify the full tournament is valid
    full_mask = (1 << p) - 1
    if not _check_subset_valid(in_bits, list(range(p)), full_mask):
        # Fallback: try larger primes
        found = False
        for p_try in [71, 79, 83]:
            if time.time() > deadline:
                break
            orientation = _paleyn_orientation(p_try)
            in_bits = _build_in_bits(p_try, orientation)
            if _check_subset_valid(in_bits, list(range(p_try)), (1 << p_try) - 1):
                p = p_try
                found = True
                break
        if not found:
            # If nothing works, return the original 67
            p = 67
            orientation = _paleyn_orientation(67)
            in_bits = _build_in_bits(67, orientation)
    
    current_vertices = list(range(p))
    current_mask = (1 << p) - 1
    
    # Greedy pruning: repeatedly try to remove vertices
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        
        improved = False
        
        # Sort vertices by out-degree (ascending) to try removing least dominant first
        sorted_vertices = sorted(current_vertices, key=lambda v: _out_degree_in_subset(v, current_vertices, in_bits))
        
        for v in sorted_vertices:
            if time.time() > deadline - 5:
                break
            
            if len(current_vertices) <= 4:
                break
            
            new_vertices = [x for x in current_vertices if x != v]
            new_mask = current_mask & ~(1 << v)
            
            # Check validity of new subset
            if _check_subset_valid(in_bits, new_vertices, new_mask):
                current_vertices = new_vertices
                current_mask = new_mask
                improved = True
                break  # Restart with smaller set
    
    # Re-index the pruned tournament to 0..n-1
    n = len(current_vertices)
    
    if n == p:
        # No pruning happened, return as-is
        return {'n': n, 'orientation': orientation}
    
    # Build new orientation with re-indexed vertices
    new_orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            old_i = current_vertices[i]
            old_j = current_vertices[j]
            
            # Find the bit in the original orientation
            if old_i < old_j:
                row_start = old_i * (p - 1) - (old_i * (old_i - 1)) // 2
                col_offset = old_j - old_i - 1
                idx_orig = row_start + col_offset
                bit = orientation[idx_orig]
            else:
                row_start = old_j * (p - 1) - (old_j * (old_j - 1)) // 2
                col_offset = old_i - old_j - 1
                idx_orig = row_start + col_offset
                bit = orientation[idx_orig]
                bit = 1 - bit
            
            new_orientation.append(bit)
    
    return {'n': n, 'orientation': new_orientation}
# EVOLVE-BLOCK-END
