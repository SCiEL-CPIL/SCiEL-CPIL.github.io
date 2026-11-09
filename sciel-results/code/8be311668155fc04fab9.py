import time
import random
import numpy as np

N = 16

# Pre-compute all 2^16 binary inputs
# This is done once at module load or first call.
# 65536 * 16 integers is ~1MB, very fast to generate.
def _get_all_inputs():
    # Use np.indices or bit manipulation
    # np.arange(2**N) then bit extraction
    # Faster: 
    # bits = np.arange(2**N, dtype=np.uint8)
    # out = np.zeros((2**N, N), dtype=np.uint8)
    # for i in range(N):
    #     out[:, i] = (bits >> i) & 1
    # This loop is 16 iterations over 65536 elements. Very fast.
    
    bits = np.arange(2**N, dtype=np.uint8)
    out = np.zeros((2**N, N), dtype=np.uint8)
    for i in range(N):
        out[:, i] = (bits >> i) & 1
    return out

ALL_INPUTS = _get_all_inputs()

def _check_network(net):
    """
    Check if the network sorts all 2^16 inputs.
    Returns True if valid, False otherwise.
    """
    if not net:
        return N == 1
    
    # Copy inputs to avoid modifying the global array
    # Using .copy() is important.
    state = ALL_INPUTS.copy()
    
    for i, j in net:
        # Compare-exchange: min to i, max to j
        # Vectorized operation
        # We need to be careful with in-place updates if i and j overlap? 
        # No, i and j are distinct wires.
        # But we are updating columns of the same matrix.
        # We must compute new values before assigning?
        # min(i, j) and max(i, j) depend on old values.
        # Since i != j, and we update both, we can do:
        # col_i = state[:, i]
        # col_j = state[:, j]
        # new_i = np.minimum(col_i, col_j)
        # new_j = np.maximum(col_i, col_j)
        # state[:, i] = new_i
        # state[:, j] = new_j
        
        col_i = state[:, i]
        col_j = state[:, j]
        
        # np.minimum and np.maximum are fast
        new_i = np.minimum(col_i, col_j)
        new_j = np.maximum(col_i, col_j)
        
        state[:, i] = new_i
        state[:, j] = new_j

    # Check if all rows are sorted
    # A row is sorted if state[:, k] <= state[:, k+1] for all k
    # np.diff(state, axis=1) >= 0
    # If any diff is negative, it's not sorted.
    # Since inputs are 0/1, outputs are 0/1.
    # We can just check if the last column is all 1s where expected?
    # No, simpler: check diffs.
    diffs = np.diff(state, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _generate_batcher_oes(n):
    """
    Generate Batcher's Odd-Even Merge Sort network for n wires.
    Returns a list of (i, j) comparators.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Recursive step for OES
    # 1. Recursively sort the odd-indexed wires and even-indexed wires?
    # No, Batcher's OES:
    # 1. Recursively sort the first half and second half? No, that's Odd-Even Transposition?
    # Batcher's Odd-Even Merge Sort:
    # To merge two sorted lists of size n/2:
    #   1. Recursively merge the odd-indexed elements and even-indexed elements of the combined list?
    #   Actually, the standard definition:
    #   OES(n):
    #     If n == 1: return
    #     OES(n/2) on even indices (0, 2, 4...)
    #     OES(n/2) on odd indices (1, 3, 5...)
    #     Merge the results using OES-Merge(n)
    
    # Wait, the standard Batcher OES for sorting n elements from scratch:
    # 1. Recursively sort the first n/2 elements and the last n/2 elements? No.
    # It sorts the even-indexed and odd-indexed elements independently, then merges them.
    # This is correct for generating a sorting network.
    
    def oes_merge(n):
        """
        Merge two sorted lists of size n/2 located at indices 0..n-1.
        Assumes input is already sorted in the even and odd positions?
        No, OES-Merge assumes the input is formed by interleaving two sorted lists.
        """
        if n == 1:
            return []
        
        # 1. Recursively merge the even-indexed wires and odd-indexed wires
        # The even-indexed wires are at 0, 2, 4...
        # The odd-indexed wires are at 1, 3, 5...
        # We need to apply oes_merge(n/2) to the sub-network of even wires and odd wires.
        
        comps = []
        
        # Map local index k in sub-network of size n/2 to global index
        # Even sub-network: local k -> global 2*k
        # Odd sub-network: local k -> global 2*k + 1
        
        sub_comps_even = oes_merge(n // 2)
        for i, j in sub_comps_even:
            comps.append((2 * i, 2 * j))
            
        sub_comps_odd = oes_merge(n // 2)
        for i, j in sub_comps_odd:
            comps.append((2 * i + 1, 2 * j + 1))
            
        # 2. Apply comparators between adjacent wires
        # (0,1), (2,3), (4,5)...
        for i in range(0, n, 2):
            comps.append((i, i + 1))
            
        return comps

    def oes_sort(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
            
        comps = []
        
        # 1. Recursively sort even and odd indices
        sub_comps_even = oes_sort(n // 2)
        for i, j in sub_comps_even:
            comps.append((2 * i, 2 * j))
            
        sub_comps_odd = oes_sort(n // 2)
        for i, j in sub_comps_odd:
            comps.append((2 * i + 1, 2 * j + 1))
            
        # 2. Merge the two sorted halves
        merge_comps = oes_merge(n)
        comps.extend(merge_comps)
        
        return comps

    return oes_sort(n)

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # 1. Generate initial network
    # Batcher's OES for 16
    initial_net = _generate_batcher_oes(N)
    
    # Verify initial network
    if not _check_network(initial_net):
        # Fallback to a known correct but larger network if OES generation is buggy
        # Odd-Even Transposition
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    best_net = initial_net
    best_cost = len(best_net)
    
    # 2. Simulated Annealing
    current_net = initial_net
    current_cost = len(current_net)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.95
    
    # Pre-define all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Random seed for reproducibility
    random.seed(42)
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate a move
        move_type = random.random()
        
        new_net = current_net.copy()
        
        if move_type < 0.4:
            # Remove a comparator
            if len(new_net) > 0:
                idx = random.randint(0, len(new_net) - 1)
                new_net.pop(idx)
        elif move_type < 0.8:
            # Add a comparator
            # Pick a random comparator not already in the net?
            # Or just pick any, duplicates are allowed but useless?
            # Duplicates are useless. Let's pick from all_comps.
            # To avoid duplicates, we could check, but it's O(N^2).
            # For now, just pick randomly. If duplicate, it's a no-op effectively.
            comp = random.choice(all_comps)
            new_net.append(comp)
        else:
            # Swap two comparators
            if len(new_net) >= 2:
                i = random.randint(0, len(new_net) - 1)
                j = random.randint(0, len(new_net) - 1)
                if i != j:
                    new_net[i], new_net[j] = new_net[j], new_net[i]
        
        new_cost = len(new_net)
        
        # Check validity
        # Optimization: If new_cost > current_cost + 10, skip check? No, we might need to go up to escape.
        # But if it's invalid, we reject.
        
        # Fast check: If the network is identical to current_net, skip.
        # Comparing lists of tuples is O(N).
        if new_net == current_net:
            T *= cooling
            continue
            
        is_valid = _check_network(new_net)
        
        if is_valid:
            delta_E = new_cost - current_cost
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                current_net = new_net
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_net = current_net
                    best_cost = current_cost
        # else: reject (keep current_net)
        
        T *= cooling
        
    return best_net

# EVOLVE-BLOCK-END
