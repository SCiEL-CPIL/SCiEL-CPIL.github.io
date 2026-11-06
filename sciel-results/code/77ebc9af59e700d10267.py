"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _get_paleyn_orientation(n):
    """
    Constructs a Paley tournament orientation for prime n.
    Returns list of bits or None if n is not a valid prime for Paley (n % 4 == 3).
    """
    if n < 2:
        return None
    # Check prime
    is_prime = True
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            is_prime = False
            break
    if not is_prime:
        return None
    if n % 4 != 3:
        return None
        
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
        
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
    return orientation

def _orientation_to_in_bits(n, orientation):
    """Convert orientation list to in_bits array."""
    in_bits = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                # i -> j
                in_bits[j] |= (1 << i)
            else:
                # j -> i
                in_bits[i] |= (1 << j)
            idx += 1
    return in_bits

def _in_bits_to_orientation(n, in_bits):
    """Convert in_bits array to orientation list."""
    orientation = [0] * (n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (in_bits[j] >> i) & 1:
                orientation[idx] = 0 # i -> j
            else:
                orientation[idx] = 1 # j -> i
            idx += 1
    return orientation

def _validate_tournament(n, in_bits):
    """Check if every 4-set has a common dominator."""
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
                    if (common & ~mask) == 0:
                        return False
    return True

def _greedy_prune(n, in_bits, deadline):
    """
    Greedy set-cover pruning: iteratively remove vertices that are not
    uniquely required to dominate any 4-set.
    Returns (pruned_n, pruned_in_bits) or (n, in_bits) if no improvement.
    
    Key insight: if vertex v is not the unique dominator of any 4-set,
    then removing v preserves validity (no re-validation needed).
    """
    start_time = time.time()
    current_n = n
    current_in_bits = in_bits[:]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_n = len(current_in_bits)
        if current_n < 5:
            break
        
        # For each 4-set, find if it has a unique dominator
        # If a vertex v is the unique dominator of some 4-set, it cannot be removed
        unique_dominator = [False] * current_n
        
        # Iterate over all 4-sets
        for a in range(current_n):
            for b in range(a + 1, current_n):
                for c in range(b + 1, current_n):
                    for d in range(c + 1, current_n):
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        common = current_in_bits[a] & current_in_bits[b] & current_in_bits[c] & current_in_bits[d]
                        dominators = common & ~mask
                        if dominators != 0:
                            if dominators & (dominators - 1) == 0:
                                # Exactly one dominator
                                v = dominators.bit_length() - 1
                                if v < current_n:
                                    unique_dominator[v] = True
        
        # Find removable vertices (not unique dominator of any 4-set)
        removable = [v for v in range(current_n) if not unique_dominator[v]]
        
        if not removable:
            break
        
        # Remove the first removable vertex (try in order)
        for v in removable:
            new_n = current_n - 1
            if new_n < 5:
                break
            
            # Build new in_bits without v
            new_in_bits = [0] * new_n
            for old_i in range(current_n):
                if old_i == v:
                    continue
                new_i = old_i if old_i < v else old_i - 1
                new_bits = 0
                for old_j in range(current_n):
                    if old_j == v:
                        continue
                    if (current_in_bits[old_i] >> old_j) & 1:
                        new_j = old_j if old_j < v else old_j - 1
                        new_bits |= (1 << new_j)
                new_in_bits[new_i] = new_bits
            
            # No need to validate - correctness guaranteed by the unique dominator check
            current_n = new_n
            current_in_bits = new_in_bits
            improved = True
            break  # Restart the outer loop
        
    return current_n, current_in_bits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with Paley(67)
    orientation = _get_paleyn_orientation(67)
    if orientation is None:
        # Fallback
        n = 83
        orientation = _get_paleyn_orientation(83)
        if orientation is None:
            n = 90
            orientation = [0] * (n * (n - 1) // 2)
        return {'n': n, 'orientation': orientation}
    
    n = 67
    in_bits = _orientation_to_in_bits(n, orientation)
    
    # Validate Paley(67)
    if not _validate_tournament(n, in_bits):
        # Should not happen, but fallback
        return {'n': n, 'orientation': orientation}
    
    # Try greedy pruning
    n_pruned, in_bits_pruned = _greedy_prune(n, in_bits, deadline)
    
    if n_pruned < n:
        n = n_pruned
        in_bits = in_bits_pruned
    
    # Final validation to ensure correctness
    if not _validate_tournament(n, in_bits):
        # Fallback to original if pruning somehow broke validity
        n = 67
        in_bits = _orientation_to_in_bits(n, _get_paleyn_orientation(67))
    
    orientation = _in_bits_to_orientation(n, in_bits)
    return {'n': n, 'orientation': orientation}

# EVOLVE-BLOCK-END
