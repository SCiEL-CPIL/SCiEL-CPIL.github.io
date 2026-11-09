import numpy as np
import time

N = 18
NUM_INPUTS = 1 << N
BLOCK_SIZE = 64
NUM_BLOCKS = NUM_INPUTS // BLOCK_SIZE

def _pack_bits_to_u64(arr_2d):
    flat = arr_2d.flatten()
    packed = np.packbits(flat, axis=0, bitorder='little')
    packed = packed.reshape(NUM_BLOCKS, 8)
    return packed.view(np.uint64).flatten()

def _calc_popcount_u64(diff):
    try:
        return np.bitwise_count(diff).sum()
    except AttributeError:
        u8 = diff.view(np.uint8).reshape(NUM_BLOCKS, 8)
        unpacked = np.unpackbits(u8, axis=1, count=8, bitorder='little')
        return unpacked.sum()

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Use more of the 90s budget

    # Pre-compute target states for all 2^18 inputs
    # Target for input k: wire i is 1 if i >= N - popcount(k)
    # We represent state as 18 arrays of uint64, each of length NUM_BLOCKS
    # Each uint64 holds 64 inputs.
    
    # Generate indices for all inputs
    # indices[b, bit] = b * 64 + bit
    indices = np.arange(NUM_INPUTS, dtype=np.uint32).reshape(NUM_BLOCKS, BLOCK_SIZE)
    
    # Compute popcount for all inputs
    bits = (indices[:, :, None] >> np.arange(N)[None, None, :]) & 1
    popcounts = bits.sum(axis=2) # Shape (NUM_BLOCKS, BLOCK_SIZE)
    
    # Now compute targets for each wire i
    targets = []
    for i in range(N):
        mask = (popcounts >= (N - i))
        targets.append(_pack_bits_to_u64(mask))
        
    # Initialize wires (identity permutation)
    wires = []
    for i in range(N):
        wires.append(_pack_bits_to_u64(bits[:, :, i]))
        
    # Pre-compute error for each wire
    wire_errors = np.array([_calc_popcount_u64(wires[i] ^ targets[i]) for i in range(N)], dtype=np.int64)
    current_error = wire_errors.sum()
    
    # Pre-compute all pairs for vectorized evaluation
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def apply_comparator(wires, i, j):
        """Apply comparator (i, j) and return new wires, new errors, new total error"""
        a = wires[i]
        b = wires[j]
        new_a = a & b
        new_b = a | b
        new_err_i = _calc_popcount_u64(new_a ^ targets[i])
        new_err_j = _calc_popcount_u64(new_b ^ targets[j])
        new_wires = wires.copy()
        new_wires[i] = new_a
        new_wires[j] = new_b
        new_wire_errors = wire_errors.copy()
        new_wire_errors[i] = new_err_i
        new_wire_errors[j] = new_err_j
        new_error = current_error + (new_err_i + new_err_j - wire_errors[i] - wire_errors[j])
        return new_wires, new_wire_errors, new_error

    current_wires = wires.copy()
    network = []
    
    # Greedy Search with look-ahead
    while time.time() < deadline:
        best_error = current_error
        best_comp = None
        best_new_wires = None
        best_new_wire_errors = None
        
        # First pass: single comparator greedy
        for i, j in all_pairs:
            a = current_wires[i]
            b = current_wires[j]
            new_a = a & b
            new_b = a | b
            new_err_i = _calc_popcount_u64(new_a ^ targets[i])
            new_err_j = _calc_popcount_u64(new_b ^ targets[j])
            delta = (new_err_i + new_err_j) - (wire_errors[i] + wire_errors[j])
            new_error = current_error + delta
            if new_error < best_error:
                best_error = new_error
                best_comp = (i, j)
                best_new_wires = current_wires.copy()
                best_new_wires[i] = new_a
                best_new_wires[j] = new_b
                best_new_wire_errors = wire_errors.copy()
                best_new_wire_errors[i] = new_err_i
                best_new_wire_errors[j] = new_err_j
                
        # If no improvement found, try 2-step look-ahead
        if best_comp is None:
            # Try sequences of 2 comparators
            found_improvement = False
            for i1, j1 in all_pairs:
                if time.time() > deadline - 5:
                    break
                # Apply first comparator
                a1 = current_wires[i1]
                b1 = current_wires[j1]
                new_a1 = a1 & b1
                new_b1 = a1 | b1
                new_err_i1 = _calc_popcount_u64(new_a1 ^ targets[i1])
                new_err_j1 = _calc_popcount_u64(new_b1 ^ targets[j1])
                err_after_first = current_error + (new_err_i1 + new_err_j1 - wire_errors[i1] - wire_errors[j1])
                
                # Create temporary wires state
                temp_wires = current_wires.copy()
                temp_wires[i1] = new_a1
                temp_wires[j1] = new_b1
                temp_wire_errors = wire_errors.copy()
                temp_wire_errors[i1] = new_err_i1
                temp_wire_errors[j1] = new_err_j1
                
                # Try second comparator
                for i2, j2 in all_pairs:
                    if i2 == i1 and j2 == j1:
                        continue
                    a2 = temp_wires[i2]
                    b2 = temp_wires[j2]
                    new_a2 = a2 & b2
                    new_b2 = a2 | b2
                    new_err_i2 = _calc_popcount_u64(new_a2 ^ targets[i2])
                    new_err_j2 = _calc_popcount_u64(new_b2 ^ targets[j2])
                    err_after_second = err_after_first + (new_err_i2 + new_err_j2 - temp_wire_errors[i2] - temp_wire_errors[j2])
                    
                    if err_after_second < best_error:
                        best_error = err_after_second
                        best_comp = ((i1, j1), (i2, j2))  # Mark as 2-step
                        best_new_wires = temp_wires.copy()
                        best_new_wires[i2] = new_a2
                        best_new_wires[j2] = new_b2
                        best_new_wire_errors = temp_wire_errors.copy()
                        best_new_wire_errors[i2] = new_err_i2
                        best_new_wire_errors[j2] = new_err_j2
                        found_improvement = True
                        break
                
                if found_improvement:
                    break
            
            if best_comp is None:
                # Still stuck, break
                break
        
        # Apply the best move(s)
        if isinstance(best_comp, tuple) and len(best_comp) == 2 and isinstance(best_comp[0], tuple):
            # 2-step move
            network.extend(best_comp)
        else:
            network.append(best_comp)
            
        current_wires = best_new_wires
        current_error = best_error
        wire_errors = best_new_wire_errors
        
        if current_error == 0:
            break
            
    # Check validity
    if current_error == 0:
        return network
    else:
        # Fallback to seed
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
