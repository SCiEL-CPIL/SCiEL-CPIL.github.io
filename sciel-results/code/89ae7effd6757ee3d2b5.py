"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = 85  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START

def _make_all_inputs_packed():
    """Create array of all 2^N inputs as uint64 array (each element is a packed N-bit value)."""
    return np.arange(NUM_INPUTS, dtype=np.uint64)

def _apply_comparator_packed(inputs, i, j):
    """Apply comparator (i, j) to packed inputs. min to i, max to j."""
    mask_i = np.uint64(1) << i
    mask_j = np.uint64(1) << j
    
    i_val = inputs & mask_i
    j_val = inputs & mask_j
    
    diff = j - i
    # Shift j_val to position i to check if both are 1
    j_val_at_i = j_val >> diff
    # Shift i_val to position j to check if either is 1
    i_val_at_j = i_val << diff
    
    new_i_val = i_val & j_val_at_i
    new_j_val = i_val_at_j | j_val
    
    return (inputs & ~(mask_i | mask_j)) | new_i_val | new_j_val

def _is_sorting_network_packed(comparators, inputs):
    """Check if network sorts all inputs using zero-one principle, with packed representation."""
    wires = inputs.copy()
    for i, j in comparators:
        wires = _apply_comparator_packed(wires, i, j)
    
    # Check if each row is sorted (non-decreasing)
    # Unsorted if exists k < N-1 such that bit_k=1 and bit_{k+1}=0
    # This corresponds to bits set in (wires & ~(wires >> 1))
    mask = (np.uint64(1) << (N - 1)) - 1
    bad = wires & ~(wires >> 1)
    bad &= mask
    
    return not np.any(bad)

def construct():
    start_time = time.time()
    
    # Base: odd-even transposition
    comps = [(i, i + 1) for phase in range(N)
             for i in range(phase % 2, N - 1, 2)]
    
    input_matrix = _make_all_inputs_packed()
    
    # Verify base is correct
    if not _is_sorting_network_packed(comps, input_matrix):
        # Fallback: should not happen, but return the base if it's somehow broken
        return comps
    
    # Iteratively remove redundant comparators
    changed = True
    while changed and (time.time() - start_time) < DEADLINE - 5:
        changed = False
        for idx in range(len(comps)):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            # Try removing comparator at idx
            new_comps = comps[:idx] + comps[idx+1:]
            if _is_sorting_network_packed(new_comps, input_matrix):
                comps = new_comps
                changed = True
                break  # Restart loop since indices changed
    
    return comps

# EVOLVE-BLOCK-END
