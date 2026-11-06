import numpy as np
import time
import random

N = 18
TIME_LIMIT = 90.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    
    # Generate all 2^N boolean inputs once
    # Shape (2^N, N)
    # Using bit operations
    all_vals = np.arange(1 << N, dtype=np.uint32)
    # Unpack bits
    # base_inputs[i, k] = (all_vals[i] >> k) & 1
    # This can be done vectorized
    shifts = np.arange(N, dtype=np.uint32)
    # (all_vals[:, None] >> shifts[None, :]) & 1
    base_inputs = ((all_vals[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    
    def verify_network(network):
        if not network:
            return N <= 1
        inputs = base_inputs.copy()
        for i, j in network:
            v_i = inputs[:, i]
            v_j = inputs[:, j]
            min_v = np.minimum(v_i, v_j)
            max_v = np.maximum(v_i, v_j)
            inputs[:, i] = min_v
            inputs[:, j] = max_v
            
        diffs = np.diff(inputs, axis=1)
        if np.any(diffs < 0):
            return False
        return True

    # Initial network: Odd-Even Transposition
    initial_network = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                initial_network.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                initial_network.append((i, i + 1))
                
    current_network = list(initial_network)
    
    if not verify_network(current_network):
        return current_network
        
    best_network = list(current_network)
    best_size = len(current_network)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
        improved = False
        
        # 1. Try to remove a comparator
        if current_network:
            idx = random.randint(0, len(current_network) - 1)
            new_network = current_network[:idx] + current_network[idx+1:]
            
            if verify_network(new_network):
                current_network = new_network
                improved = True
                if len(current_network) < best_size:
                    best_network = list(current_network)
                    best_size = len(current_network)
                    
        # 2. If no improvement, try to move a comparator (remove + add)
        # This helps escape local minima where no single removal works.
        if not improved and current_network:
            idx_remove = random.randint(0, len(current_network) - 1)
            new_network = current_network[:idx_remove] + current_network[idx_remove+1:]
            
            comp_add = random.choice(all_pairs)
            idx_add = random.randint(0, len(new_network))
            
            new_network.insert(idx_add, comp_add)
            
            if verify_network(new_network):
                current_network = new_network
                # Size is same, so best_size doesn't change, 
                # but we are in a new state.
        
        # 3. Optional: If stuck for a very long time, restart from best
        # We can track steps since last improvement.
        
    return best_network
