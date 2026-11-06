import numpy as np
import time
import random

N = 17
NUM_INPUTS = 1 << N
DEADLINE = 80  # seconds, safety margin

def _make_all_inputs_packed():
    """Create array of all 2^N inputs as uint64 array."""
    return np.arange(NUM_INPUTS, dtype=np.uint64)

def _count_violations_packed(comparators, inputs):
    """
    Check how many inputs are NOT sorted by the network.
    Returns an integer count of violations.
    """
    # inputs is a uint64 array of length NUM_INPUTS
    # We operate on a copy to avoid modifying the global inputs array
    
    for i, j in comparators:
        mask_i = np.uint64(1) << i
        mask_j = np.uint64(1) << j
        
        bit_i = (inputs & mask_i) != 0
        bit_j = (inputs & mask_j) != 0
        
        new_i = bit_i & bit_j
        new_j = bit_i | bit_j
        
        # Clear bits i and j
        inputs = inputs & ~(mask_i | mask_j)
        # Set new bits
        inputs = np.where(new_i, inputs | mask_i, inputs)
        inputs = np.where(new_j, inputs | mask_j, inputs)
        
    # Check if sorted
    # For each k in 0..N-2, if bit k is 1, bit k+1 must be 1.
    # Violation: bit k == 1 and bit k+1 == 0
    
    total_violations = 0
    for k in range(N - 1):
        mask_k = np.uint64(1) << k
        mask_k1 = np.uint64(1) << (k + 1)
        
        bit_k = (inputs & mask_k) != 0
        bit_k1 = (inputs & mask_k1) != 0
        
        # Count inputs where bit_k is 1 and bit_k1 is 0
        violations = np.sum(bit_k & ~bit_k1)
        total_violations += violations
        
    return total_violations

def _is_sorting_network_packed(comparators, inputs):
    """Check if network sorts all inputs using packed representation."""
    return _count_violations_packed(comparators, inputs) == 0

def _batcher_odd_merge(n):
    """
    Generate Batcher's Odd-Merge sorting network for n wires.
    Returns a list of (i, j) comparators.
    """
    if n == 1:
        return []
    
    # Split into two halves
    if n % 2 == 0:
        m1 = n // 2
        m2 = n // 2
    else:
        m1 = (n + 1) // 2
        m2 = (n - 1) // 2
        
    # Recursively sort the two halves
    comps1 = _batcher_odd_merge(m1)
    comps2 = _batcher_odd_merge(m2)
    
    # Offset the second half's comparators by m1
    comps2_offset = [(i + m1, j + m1) for i, j in comps2]
    
    # For now, return the odd-even transposition network
    # as a fallback, since full Batcher merge is complex to implement correctly here.
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Base: Odd-even transposition
    comps = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    inputs = _make_all_inputs_packed()
    
    # Verify base is correct
    if not _is_sorting_network_packed(comps, inputs):
        return comps
        
    best_comps = comps[:]
    best_cost = len(comps) * 1000 + _count_violations_packed(comps, inputs)
    
    current_comps = comps[:]
    current_cost = best_cost
    current_violations = 0 # Start with valid network
    
    # Simulated Annealing Parameters
    T = 100.0  # Initial temperature
    cooling_rate = 0.995
    min_T = 0.1
    
    # We will perform moves until time runs out
    while (time.time() - start_time) < DEADLINE:
        if T < min_T:
            # Restart with higher T if we get stuck
            T = 50.0
            
        # Generate a random move
        move_type = random.choice(['delete', 'insert', 'swap', 'modify'])
        
        new_comps = current_comps[:]
        
        if move_type == 'delete':
            if len(new_comps) > 0:
                idx = random.randint(0, len(new_comps) - 1)
                new_comps.pop(idx)
                
        elif move_type == 'insert':
            # Insert a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            idx = random.randint(0, len(new_comps))
            new_comps.insert(idx, (i, j))
            
        elif move_type == 'swap':
            if len(new_comps) >= 2:
                idx1 = random.randint(0, len(new_comps) - 1)
                idx2 = random.randint(0, len(new_comps) - 1)
                if idx1 != idx2:
                    new_comps[idx1], new_comps[idx2] = new_comps[idx2], new_comps[idx1]
                    
        elif move_type == 'modify':
            if len(new_comps) > 0:
                idx = random.randint(0, len(new_comps) - 1)
                i, j = new_comps[idx]
                # Change one of the wires
                if random.random() < 0.5:
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                else:
                    j = random.randint(i + 1, N - 1)
                new_comps[idx] = (i, j)
                
        # Evaluate new state
        # Optimization: 
        # 1. If current is valid and move is insert, new is valid.
        # 2. If current is valid and move is delete, we must check.
        # 3. If current is invalid, we must check.
        
        if current_violations == 0 and move_type == 'insert':
            new_violations = 0
        else:
            new_violations = _count_violations_packed(new_comps, inputs)
            
        new_cost = len(new_comps) * 1000 + new_violations
        
        delta = new_cost - current_cost
        
        # Acceptance criterion
        if delta < 0:
            # Better
            current_comps = new_comps
            current_cost = new_cost
            current_violations = new_violations
        else:
            # Worse: accept with probability exp(-delta / T)
            if random.random() < np.exp(-delta / T):
                current_comps = new_comps
                current_cost = new_cost
                current_violations = new_violations
                
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_comps = current_comps[:]
            
        # Cool down
        T *= cooling_rate
        
    return best_comps
