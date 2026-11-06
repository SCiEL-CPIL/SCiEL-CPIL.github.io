import time
import numpy as np

N = 18
NUM_BOOLS = 1 << N
DEADLINE = 85.0  # Safety margin

def _verify_network(comparators):
    """
    Verify if the network sorts all 2^N boolean inputs.
    Uses in-place XOR for speed.
    """
    if not comparators:
        return False
    
    # Initialize all possible boolean inputs
    # np.arange creates 0, 1, ..., 2^N-1
    curr = np.arange(NUM_BOOLS, dtype=np.int32)
    
    for i, j in comparators:
        # Check if wire i is 1 and wire j is 0
        # bi = (curr >> i) & 1
        # bj = (curr >> j) & 1
        # mask = (bi == 1) & (bj == 0)
        
        # Optimized:
        # bi & ~bj is 1 only if bi is 1 and bj is 0.
        # Since bi and bj are 0 or 1, ~bj in integer logic needs care with numpy.
        # In numpy, ~ is bitwise NOT.
        # bi is int8/16/32? dtype is int32.
        # bi is 0 or 1. ~bj is -1 or -2.
        # (0 & -1) = 0. (0 & -2) = 0.
        # (1 & -1) = 1. (1 & -2) = 0.
        # So (bi & ~bj) works correctly for 0/1 masks.
        
        bi = (curr >> i) & 1
        bj = (curr >> j) & 1
        mask = (bi & ~bj).astype(bool) # Ensure bool for indexing
        
        if np.any(mask):
            xor_val = (1 << i) | (1 << j)
            curr[mask] ^= xor_val
            
    # A value v is a valid sorted output (0...01...1) iff (v & (v+1)) == 0
    # Note: for v = 2^N - 1 (all 1s), v+1 = 2^N.
    # (2^N-1) & 2^N = 0. Correct.
    return np.all((curr & (curr + 1)) == 0)

def _get_oet_network():
    """Generate Odd-Even Transposition Network"""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def construct():
    start_time = time.time()
    
    # 1. Seed with OET
    comparators = _get_oet_network()
    
    # 2. Verify baseline
    if not _verify_network(comparators):
        # This shouldn't happen for OET, but just in case
        return comparators
        
    # 3. Greedy Removal
    # We use a while loop to control the iteration index after modifications.
    # Strategy: Scan from end to start. If removal succeeds, stay at the same index
    # to check the neighbor that shifted into that position. If it fails, move back.
    
    idx = len(comparators) - 1
    improved = True
    
    while improved and time.time() < start_time + DEADLINE:
        improved = False
        idx = len(comparators) - 1
        
        while idx >= 0 and time.time() < start_time + DEADLINE:
            # Try removing comparators[idx]
            trial = comparators[:idx] + comparators[idx+1:]
            
            if _verify_network(trial):
                # Success
                comparators = trial
                improved = True
                # Do not decrement idx. The element at idx is now the one that was at idx-1.
                # We need to check if it is also redundant.
                # So we stay at the same idx.
                # Note: list length decreased by 1.
                if idx > len(comparators) - 1:
                    idx = len(comparators) - 1
            else:
                idx -= 1
                
    return comparators
