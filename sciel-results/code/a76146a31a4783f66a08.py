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

def _verify_paley_exact(p, orientation):
    """
    Exact verification of the 4-dominator property.
    Uses Python integer bitsets for speed.
    """
    # Build bitsets of in-neighbors
    in_bits = [0] * p
    idx = 0
    for i in range(p):
        for j in range(i + 1, p):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                # i -> j, so i is in-neighbor of j
                in_bits[j] |= (1 << i)
            else:
                # j -> i, so j is in-neighbor of i
                in_bits[i] |= (1 << j)
    
    # Iterate all 4-combinations
    for a, b, c, d in itertools.combinations(range(p), 4):
        if (in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]) == 0:
            return False
            
    return True

def _prune_tournament(vertices, orientation):
    """
    Attempts to prune the tournament by removing vertices that are not essential.
    vertices: list of vertex indices in the current tournament (subset of original)
    orientation: original orientation list for the full Paley tournament
    Returns a new list of vertices if pruning was successful, else None.
    """
    n_orig = len(orientation) * 2 - 1 # This is wrong, orientation is pairs.
    # We need to map the sub-tournament to the original indices.
    # Actually, it's easier to work with the sub-tournament directly if we can extract it.
    # But extracting the orientation for a subset is O(k^2).
    
    # Let's implement a helper to check validity of a subset of vertices given the global orientation.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Primes p ≡ 3 mod 4, sorted ascending
    candidates = [31, 43, 47, 59, 67, 71, 79, 83]
    
    best_n = 83
    best_orientation = None
    best_vertices = None
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        orientation = _paleyn_orientation(p)
        
        # Verify
        is_valid = _verify_paley_exact(p, orientation)
        
        if is_valid:
            best_n = p
            best_orientation = orientation
            best_vertices = list(range(p))
            
            # Attempt pruning
            # We try to remove vertices one by one.
            # We need a function to check if a subset of vertices is valid.
            # To do this efficiently, we should probably precompute in_bits for the full graph,
            # and then for a subset, we just mask the in_bits?
            # No, the in_bits for a subset must only consider edges *within* the subset.
            # If we use the global in_bits, `in_bits[a] & in_bits[b]...` might include vertices outside the subset.
            # So we must mask the in_bits with the subset mask.
            
            full_in_bits = [0] * p
            idx = 0
            for i in range(p):
                for j in range(i + 1, p):
                    bit = orientation[idx]
                    idx += 1
                    if bit == 0:
                        full_in_bits[j] |= (1 << i)
                    else:
                        full_in_bits[i] |= (1 << j)
            
            current_vertices = list(range(p))
            
            while True:
                if time.time() > deadline - 10: # Leave 10s for return
                    break
                
                pruned = False
                for v in current_vertices:
                    new_vertices = [x for x in current_vertices if x != v]
                    if len(new_vertices) < 4:
                        break
                    
                    # Check if new_vertices is valid
                    mask = 0
                    for x in new_vertices:
                        mask |= (1 << x)
                    
                    valid = True
                    # Iterate 4-subsets of new_vertices
                    # To speed up, we can just iterate combinations of new_vertices
                    for a, b, c, d in itertools.combinations(new_vertices, 4):
                        # Check if intersection of in-neighbors (masked) is non-empty
                        # in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d] & mask
                        if (full_in_bits[a] & full_in_bits[b] & full_in_bits[c] & full_in_bits[d] & mask) == 0:
                            valid = False
                            break
                    
                    if valid:
                        current_vertices = new_vertices
                        pruned = True
                        break # Restart loop with smaller set
                
                if not pruned:
                    break
            
            best_n = len(current_vertices)
            # We need to return the orientation of the sub-tournament?
            # The task says: "construct() -> {'n': n, 'orientation': [...]}: one bit per pair (i,j), i<j, of vertices 0..n-1"
            # This implies we need to re-index the vertices to 0..n-1.
            # So we must construct the new orientation list.
            
            new_orientation = []
            # Map old indices to new indices
            old_to_new = {old: new for new, old in enumerate(current_vertices)}
            
            # Iterate pairs in lexicographic order of NEW indices
            for i in range(best_n):
                for j in range(i + 1, best_n):
                    old_i = current_vertices[i]
                    old_j = current_vertices[j]
                    # Find bit in original orientation
                    # Original orientation is for pairs (u, v) with u < v.
                    if old_i < old_j:
                        # Index in original list
                        # Number of pairs before (old_i, old_j)
                        # Pairs (0,1)...(0,old_i-1) -> old_i pairs? No.
                        # Pairs with first element < old_i:
                        # 0: old_i-1 pairs (0,1)..(0,old_i-1)
                        # 1: old_i-1 pairs
                        # ...
                        # old_i-1: 0 pairs? No.
                        # Sum_{k=0}^{old_i-1} (p - 1 - k)
                        # = old_i * (p - 1) - old_i * (old_i - 1) / 2
                        # Plus (old_j - old_i - 1) pairs with first element old_i and second < old_j
                        idx_orig = old_i * (p - 1) - (old_i * (old_i - 1)) // 2 + (old_j - old_i - 1)
                        bit = orientation[idx_orig]
                    else:
                        # old_j < old_i
                        # Swap
                        idx_orig = old_j * (p - 1) - (old_j * (old_j - 1)) // 2 + (old_i - old_j - 1)
                        bit = orientation[idx_orig]
                        # If bit is 0, old_j -> old_i. In new graph, i -> j?
                        # Wait. 0 means first -> second.
                        # If old_j -> old_i, then in new graph, j -> i.
                        # So new bit should be 1 (j -> i).
                        # If bit is 1, old_i -> old_j. Then i -> j. New bit 0.
                        # So new_bit = 1 - bit?
                        # Let's check.
                        # Original: 0 means A -> B.
                        # If we swap A and B, the direction is reversed.
                        # So if original bit is 0 (A->B), new bit (B,A) should be 1 (B->A is false, A->B is true, so B->A is 1? No.
                        # Bit 0: i -> j.
                        # Bit 1: j -> i.
                        # If we have pair (j, i) in new order.
                        # If original was A->B (0), and new is (B, A), then B->A is false. So bit should be 1.
                        # If original was B->A (1), and new is (B, A), then B->A is true. So bit should be 0.
                        # So yes, new_bit = 1 - bit.
                        bit = 1 - bit
                    
                    new_orientation.append(bit)
            
            best_orientation = new_orientation
            break # We want the smallest, so we stop at the first valid one
            
    # Fallback
    if best_orientation is None:
        best_n = 83
        best_orientation = _paleyn_orientation(83)
        
    return {'n': best_n, 'orientation': best_orientation}
# EVOLVE-BLOCK-END
