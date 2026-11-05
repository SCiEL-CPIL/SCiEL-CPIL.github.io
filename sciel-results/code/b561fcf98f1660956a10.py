import time
import numpy as np
import random

N = 17

def _generate_inputs():
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)
    return (indices[:, None] >> np.arange(N)[None, :]) & 1

ALL_INPUTS = _generate_inputs()

def _is_valid(network):
    if not network:
        return N <= 1
    
    state = ALL_INPUTS.copy()
    
    for i, j in network:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    return bool(np.all(state[:, :-1] <= state[:, 1:]))

def _construct_seed():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _greedy_remove(network):
    current = list(network)
    changed = True
    while changed:
        changed = False
        for idx in range(len(current)):
            test_net = current[:idx] + current[idx+1:]
            if _is_valid(test_net):
                current = test_net
                changed = True
                break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_network = _construct_seed()
    best_network = _greedy_remove(best_network)
    best_size = len(best_network)
    
    current_network = list(best_network)
    current_size = best_size
    no_improve_count = 0
    
    while time.time() < deadline:
        # 1. Try direct random removal
        if len(current_network) > 0:
            idx = random.randint(0, len(current_network) - 1)
            test_net = current_network[:idx] + current_network[idx+1:]
            if _is_valid(test_net):
                current_network = test_net
                current_size -= 1
                if current_size < best_size:
                    best_network = list(current_network)
                    best_size = current_size
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                continue
        
        # 2. Try move + greedy remove
        if len(current_network) > 1:
            idx = random.randint(0, len(current_network) - 1)
            comp = current_network[idx]
            moved_net = current_network[:idx] + current_network[idx+1:]
            new_idx = random.randint(0, len(moved_net))
            moved_net.insert(new_idx, comp)
            
            # Only proceed if moved_net is potentially valid? 
            # It's expensive to check. Let's just run greedy remove.
            # But if moved_net is invalid, greedy remove returns invalid.
            # So we check validity of the result.
            optimized_net = _greedy_remove(moved_net)
            
            if _is_valid(optimized_net) and len(optimized_net) < current_size:
                current_network = optimized_net
                current_size = len(optimized_net)
                if current_size < best_size:
                    best_network = list(current_network)
                    best_size = current_size
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                # Revert current_network to best? 
                # No, current_network is still valid. We just didn't find an improvement.
                # We can keep current_network as is.
                
        # 3. Swap two comparators
        if no_improve_count > 20 and len(current_network) > 1:
            i1 = random.randint(0, len(current_network) - 1)
            i2 = random.randint(0, len(current_network) - 1)
            if i1 != i2:
                # Create a copy to swap
                swapped_net = list(current_network)
                swapped_net[i1], swapped_net[i2] = swapped_net[i2], swapped_net[i1]
                
                optimized_net = _greedy_remove(swapped_net)
                
                if _is_valid(optimized_net) and len(optimized_net) < current_size:
                    current_network = optimized_net
                    current_size = len(optimized_net)
                    if current_size < best_size:
                        best_network = list(current_network)
                        best_size = current_size
                        no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1

        # Restart if stuck
        if no_improve_count > 100:
            current_network = list(best_network)
            current_size = best_size
            no_improve_count = 0

    return best_network
