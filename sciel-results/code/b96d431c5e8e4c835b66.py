"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from typing import List, Tuple

def _bit_vector_to_int(vec):
    """Convert a numpy bit vector to an integer for hashing."""
    return int(np.dot(vec, 1 << np.arange(len(vec))))

def _int_to_bit_vector(val, length):
    """Convert an integer back to a numpy bit vector."""
    return np.array([(val >> i) & 1 for i in range(length)], dtype=np.int8)

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Ensure matrix is numpy array
    if isinstance(matrix, list):
        matrix = np.array(matrix, dtype=np.int8)
    
    n_rows, n_cols = matrix.shape
    n_input = n_cols
    n_output = n_rows
    
    # Convert rows to bitmasks
    current_funcs = [0] * n_output
    for i in range(n_output):
        current_funcs[i] = 0
        for j in range(n_input):
            if matrix[i, j]:
                current_funcs[i] |= (1 << j)
                
    # We will build a circuit.
    # Available wires:
    # 0..n_input-1: inputs
    # n_input: zero
    # n_input+1..: intermediate gates
    
    # Representation:
    # available: dict mapping bitmask -> wire_index
    # targets: dict mapping bitmask -> list of output_indices
    
    available = {}
    for i in range(n_input):
        available[1 << i] = i
    available[0] = n_input  # zero wire
    
    targets = {}
    for i in range(n_output):
        mask = current_funcs[i]
        if mask not in targets:
            targets[mask] = []
        targets[mask].append(i)
        
    # Remove targets that are already available
    for mask in list(targets.keys()):
        if mask in available:
            del targets[mask]
            
    gates = []
    next_wire = n_input + 1
    
    # Greedy CSE with structural decomposition
    # Strategy:
    # 1. Try to find a target that can be formed by XOR of two availables.
    # 2. If not, create a new intermediate wire that is a "sub-part" of a target.
    #    Specifically, pick a target T and a bit b in T.
    #    Create W = T ^ (1<<b).
    #    This is valid if W can be formed by two availables.
    #    Then T can be formed by W ^ (1<<b).
    #    This effectively peels off one bit from the target.
    #    This allows us to make progress even when no direct match exists.
    
    while targets and time.time() < deadline:
        matched = False
        
        # Step 1: Check for direct matches
        # Iterate through targets
        for t_mask in list(targets.keys()):
            # Check if t_mask can be formed by XOR of two available masks
            # Optimization: Iterate through available items
            for a_mask, a_wire in available.items():
                b_mask = t_mask ^ a_mask
                if b_mask in available:
                    b_wire = available[b_mask]
                    # Found a match!
                    gates.append([a_wire, b_wire])
                    available[t_mask] = next_wire
                    next_wire += 1
                    del targets[t_mask]
                    matched = True
                    break
            if matched:
                break
        
        if matched:
            continue
            
        # Step 2: No direct match. Try to create an intermediate.
        # We want to create a wire W such that W ^ X is a target, where X is available.
        # This implies W = Target ^ X.
        # We want W to be formable by two availables.
        # Let's pick a target T.
        # Let's pick an available wire X.
        # Let W = T ^ X.
        # Check if W is formable by two availables.
        # If yes, create gate for W, add W to available.
        # Then in the next iteration (or immediately if we check again), T can be formed by W ^ X.
        
        # To be efficient, we should pick T and X wisely.
        # Heuristic:
        # Pick T with smallest weight? Or largest?
        # Smallest weight T means W has weight close to T.
        # Let's just iterate.
        
        candidate_created = False
        # Sort targets by weight to prioritize simpler ones?
        sorted_targets = sorted(targets.keys(), key=lambda m: bin(m).count('1'))
        
        for t_mask in sorted_targets:
            # Try different available wires X
            # We can limit the number of X we try to avoid O(N^2) blowup
            # But N is small (<= 16 inputs + gates)
            for x_mask, x_wire in available.items():
                w_mask = t_mask ^ x_mask
                if w_mask in available:
                    # w_mask is already available, so t_mask = w_mask ^ x_mask
                    # This would have been found in Step 1.
                    continue
                
                # Check if w_mask can be formed by two availables
                # We need to find a, b in available such that a ^ b = w_mask
                for a_mask, a_wire in available.items():
                    b_mask = w_mask ^ a_mask
                    if b_mask in available:
                        b_wire = available[b_mask]
                        # We can form w_mask
                        gates.append([a_wire, b_wire])
                        available[w_mask] = next_wire
                        next_wire += 1
                        candidate_created = True
                        break
                if candidate_created:
                    break
            if candidate_created:
                break
        
        if not candidate_created:
            # We are stuck.
            # This means for all targets T and all available X, (T^X) is not formable by two availables.
            # This implies that the set of available wires is "closed" in a way that prevents forming any sub-part of any target.
            # This can happen if the available set is a subspace that doesn't intersect the "cosets" of the targets in a useful way?
            # Or simply if we are out of simple moves.
            
            # Fallback: Independent synthesis for remaining targets.
            # For each remaining target, synthesize it from scratch using inputs.
            # This is the baseline.
            
            for t_mask in list(targets.keys()):
                # Synthesize t_mask
                # Use a simple chain.
                bits = [i for i in range(n_input) if (t_mask >> i) & 1]
                if not bits:
                    available[t_mask] = n_input
                    del targets[t_mask]
                    continue
                
                if len(bits) == 1:
                    available[t_mask] = bits[0]
                    del targets[t_mask]
                    continue
                
                # Chain
                current_wire = bits[0]
                for b in bits[1:]:
                    gates.append([current_wire, b])
                    current_wire = next_wire
                    next_wire += 1
                
                available[t_mask] = current_wire
                del targets[t_mask]
            
            break # Exit while loop
            
    # Construct output
    outputs = []
    for i in range(n_output):
        mask = current_funcs[i]
        outputs.append(available[mask])
        
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
