"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from math import gcd

def construct():
    N = 512
    numbers = list(range(1, 512))
    
    # Calculate weights
    weights = [0.0] * N
    for a in numbers:
        weights[a] = 1.0 / (N - a)
    
    # Sort nodes by weight descending
    # Index 0 is highest weight (511), Index 510 is lowest weight (1)
    sorted_nodes = sorted(numbers, key=lambda x: -weights[x])
    
    # Build conflict masks
    # mask[i] is a bitmask of nodes j > i that conflict with sorted_nodes[i]
    conflict_mask = [0] * N
    
    for i in range(N):
        a = sorted_nodes[i]
        mask = 0
        for j in range(i + 1, N):
            b = sorted_nodes[j]
            if gcd(a, b) > 1:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    best_score = 0.0
    best_set = []
    
    # Precompute suffix weights for a faster bound?
    # No, the mask is dynamic.
    
    def search(idx, current_score, available_mask, current_set):
        nonlocal best_score, best_set
        
        if idx == N:
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
            return
        
        # Upper Bound Calculation
        # Sum of weights of all available nodes from idx to N
        bound = current_score
        # Mask out bits < idx
        m = available_mask & ~((1 << idx) - 1)
        
        # Iterate through set bits in m
        temp = m
        while temp:
            lsb = temp & -temp
            bit_idx = lsb.bit_length() - 1
            bound += weights[sorted_nodes[bit_idx]]
            temp ^= lsb
            
        if bound <= best_score:
            return
            
        # Option 1: Include sorted_nodes[idx]
        if (available_mask >> idx) & 1:
            # Remove idx and its conflicts from available_mask
            new_mask = available_mask & ~((1 << idx) | conflict_mask[idx])
            
            current_set.append(sorted_nodes[idx])
            search(idx + 1, current_score + weights[sorted_nodes[idx]], new_mask, current_set)
            current_set.pop()
            
        # Option 2: Exclude sorted_nodes[idx]
        search(idx + 1, current_score, available_mask, current_set)
        
    # Initial mask: all bits 0..510 set
    initial_mask = (1 << N) - 1
    search(0, 0.0, initial_mask, [])
    
    return best_set

# EVOLVE-BLOCK-END
