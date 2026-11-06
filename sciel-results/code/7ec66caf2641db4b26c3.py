import numpy as np
import time
import random

N = 17

def generate_bitonic(n):
    """
    Generate a bitonic sorting network for n wires (n must be a power of 2).
    Returns a list of (i, j) pairs.
    """
    if n == 1:
        return []
    
    def bitonic_merge(k):
        # k is the length of the sequence to merge (power of 2)
        # This generates comparators for merging two sorted sequences of length k/2
        # into a bitonic sequence of length k.
        # Standard bitonic merge:
        # for step in range(log2(k), 0, -1):
        #    for i in range(0, k, 2**step):
        #       for j in range(i, i + 2**(step-1)):
        #           compare(j, j + 2**(step-1))
        
        # Actually, the standard recursive definition is easier to implement directly.
        # Let's use the recursive structure of Bitonic Sort.
        
        pass

    # Recursive Bitonic Sort Network Generation
    def gen_bitonic_sort(m):
        if m == 1:
            return []
        
        # Split into two halves
        half = m // 2
        left = gen_bitonic_sort(half)
        right = [ (i + half, j + half) for i, j in gen_bitonic_sort(half) ]
        
        # Merge the two halves
        # The merge step for bitonic sort of size m
        # It consists of log2(m) stages.
        # Stage 1: compare (i, i + half) for i in 0..half-1 ? No, that's not quite right for the full merge.
        
        # Let's use the standard iterative construction for Bitonic Sort on m wires.
        # A bitonic sort network for m=2^k wires is constructed recursively.
        # B(m) = B(m/2) on [0..m/2-1], B(m/2) on [m/2..m-1], then M(m) merge.
        
        # Merge M(m):
        # For step in range(log2(m), 0, -1):
        #   for start in range(0, m, 2**step):
        #     for i in range(start, start + 2**(step-1)):
        #       compare(i, i + 2**(step-1))
        
        merge_net = []
        # log2(m) stages
        # m is power of 2
        log_m = int(np.log2(m))
        
        for step in range(log_m, 0, -1):
            step_size = 2 ** step
            half_step = 2 ** (step - 1)
            for start in range(0, m, step_size):
                for i in range(start, start + half_step):
                    merge_net.append((i, i + half_step))
        
        return left + right + merge_net

    return gen_bitonic_sort(n)

def construct():
    start_time = time.time()
    deadline = start_time + 55 # 55s safety margin to stay under 60s
    
    # 1. Generate initial network
    # Bitonic sort for 16 wires
    net_16 = generate_bitonic(16)
    
    # Linear merge for 17th wire
    # To sort 17 elements using a 16-element sorter:
    # 1. Sort first 16.
    # 2. Insert 17th element into the sorted 16.
    # The 17th element (index 16) needs to be placed in its correct position.
    # We can compare it with the 16 sorted elements in reverse order?
    # If we compare (15, 16), then (14, 16)... this effectively inserts 16 into the sorted array.
    # Let's verify:
    # Array A[0..15] is sorted.
    # Compare A[15] with A[16]. Min goes to 15, Max to 16.
    # Compare A[14] with A[16] (which now holds max(A[15], A[16])).
    # This is a standard insertion sort step.
    merge_17 = [(i, 16) for i in range(15, -1, -1)]
    
    network = net_16 + merge_17
    
    # 2. Pre-compute boolean inputs for validation
    # Zero-one principle: we only need to check 2^17 inputs.
    # We represent the state as a uint8 array of shape (2^17, 17).
    
    # Generate all 2^17 integers
    inputs_int = np.arange(2**N, dtype=np.int32)
    # Extract bits
    # state[input_idx, bit_idx] = (inputs_int[input_idx] >> bit_idx) & 1
    shifts = np.arange(N, dtype=np.int32)
    state = (inputs_int[:, None] >> shifts) & 1
    
    def check_network(net):
        """
        Check if the network sorts all boolean inputs.
        Returns True if valid, False otherwise.
        """
        # Copy state to avoid modifying the original
        curr_state = state.copy()
        
        for i, j in net:
            # Compare-exchange: min to i, max to j
            # curr_state[:, i] <= curr_state[:, j]
            min_val = np.minimum(curr_state[:, i], curr_state[:, j])
            max_val = np.maximum(curr_state[:, i], curr_state[:, j])
            
            # Optimization: only update if changed? 
            # np.minimum/max are vectorized and fast.
            curr_state[:, i] = min_val
            curr_state[:, j] = max_val
        
        # Check if sorted: curr_state[:, k] <= curr_state[:, k+1] for all k
        # Use diff
        diffs = np.diff(curr_state, axis=1)
        return np.all(diffs >= 0)
    
    # Verify initial network
    if not check_network(network):
        # Fallback: If generation is broken, return a known valid large network
        # But bitonic + merge should be valid.
        # Let's force a valid state if needed, though it shouldn't happen.
        pass

    current_net = network
    best_net = current_net[:]
    best_score = len(current_net)
    
    # 3. Greedy Removal
    # Iterate through comparators and try to remove them.
    # To be efficient, we scan from start to end.
    
    # We maintain a list of active comparator indices or just the list itself.
    # Since N is small (~72), list operations are cheap.
    
    i = 0
    while i < len(current_net) and time.time() < deadline - 5: # Leave 5s for SA
        # Try removing current_net[i]
        test_net = current_net[:i] + current_net[i+1:]
        
        if check_network(test_net):
            # Removal successful
            current_net = test_net
            best_net = current_net[:]
            best_score = len(current_net)
            # Do not increment i, because the next element shifted down
        else:
            i += 1
            
    # 4. Simulated Annealing
    # Goal: Find smaller networks by swapping, moving, or removing comparators.
    # Moves:
    # 1. Remove a random comparator (if valid, accept).
    # 2. Add a random comparator (if valid, accept? No, we want to minimize size. 
    #    Adding makes it larger. We only add if it allows subsequent removals? 
    #    Standard SA for sorting networks often involves:
    #    - Remove a comparator
    #    - Swap two comparators
    #    - Change a comparator (i,j) to (i',j')
    #    - Insert a new comparator
    
    # Given the time constraint, we focus on:
    # - Random removal (if valid)
    # - Random swap of two comparators (if valid)
    # - Random mutation of a comparator (i,j) -> (i',j') (if valid)
    
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    # We will run SA for the remaining time
    while time.time() < deadline - 2:
        if not current_net:
            break
            
        # Generate a neighbor
        move_type = random.randint(0, 2)
        neighbor = current_net[:]
        
        if move_type == 0:
            # Remove a random comparator
            if len(neighbor) > 1:
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
            else:
                continue # Can't remove if size 1? Or maybe we can check if size 0 is valid (trivially yes if N=1, but N=17 no)
        elif move_type == 1:
            # Swap two comparators
            if len(neighbor) > 1:
                i, j = random.sample(range(len(neighbor)), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Mutate a comparator
            idx = random.randint(0, len(neighbor) - 1)
            old_i, old_j = neighbor[idx]
            # Pick new i, j
            new_i = random.randint(0, N - 2)
            new_j = random.randint(new_i + 1, N - 1)
            neighbor[idx] = (new_i, new_j)
        
        # Evaluate neighbor
        if check_network(neighbor):
            # Valid neighbor
            new_size = len(neighbor)
            old_size = len(current_net)
            
            # Accept if better, or with probability exp(-delta/T) if worse
            # Here "cost" is size. Lower is better.
            delta = new_size - old_size
            
            if delta < 0:
                # Better
                current_net = neighbor
                if new_size < best_score:
                    best_score = new_size
                    best_net = neighbor[:]
            else:
                # Worse or equal
                if delta == 0:
                    # Neutral move, accept to explore
                    if random.random() < 0.5:
                        current_net = neighbor
                else:
                    # Worse
                    # Probability of accepting worse solution
                    # p = exp(-delta / T)
                    # delta is positive (increase in size)
                    p = np.exp(-delta / T)
                    if random.random() < p:
                        current_net = neighbor
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_net
