"""Size-efficient sorting network on 18 wires via improved local search."""
import numpy as np
import time
import random

N = 18

def _verify_network(network):
    """
    Verify if the network sorts all 2^N Boolean inputs using the 0-1 principle.
    Uses NumPy vectorization for speed.
    """
    if len(network) == 0:
        return True
        
    n_comparators = len(network)
    comp_i = np.array([c[0] for c in network], dtype=np.int32)
    comp_j = np.array([c[1] for c in network], dtype=np.int32)
    
    chunk_size = 1 << 16 # 65536
    total_inputs = 1 << N
    
    for start in range(0, total_inputs, chunk_size):
        end = min(start + chunk_size, total_inputs)
        masks = np.arange(start, end, dtype=np.int64)
        
        # Generate inputs for this chunk
        inputs = (masks[:, None] >> np.arange(N)[None, :]) & 1
        inputs = inputs.astype(np.int8)
        
        # Apply network
        for k in range(n_comparators):
            i, j = comp_i[k], comp_j[k]
            # Compare
            swap_mask = inputs[:, i] > inputs[:, j]
            if np.any(swap_mask):
                # Swap
                temp = inputs[:, i][swap_mask].copy()
                inputs[:, i][swap_mask] = inputs[:, j][swap_mask]
                inputs[:, j][swap_mask] = temp
        
        # Check sorted
        if np.any(np.diff(inputs, axis=1) < 0):
            return False
            
    return True

def construct():
    """
    Construct a sorting network on N=18 wires using optimized local search.
    """
    start_time = time.time()
    time_limit = 55.0
    
    # 1. Generate Seed: Odd-Even Transposition
    def odd_even_transposition():
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        return net
    
    network = odd_even_transposition()
    best_network = network[:]
    best_size = len(network)
    
    # Verify seed
    if not _verify_network(network):
        raise ValueError("Seed network is invalid")
        
    # Pre-compute all possible comparator pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    
    # 2. Greedy Removal Phase
    while time.time() - start_time < time_limit * 0.1:
        removed_any = False
        indices = list(range(len(network)))
        random.shuffle(indices)
        
        for idx in indices:
            if idx >= len(network):
                continue
            new_network = network[:idx] + network[idx+1:]
            if _verify_network(new_network):
                network = new_network
                best_network = network[:]
                best_size = len(network)
                removed_any = True
                break
        
        if not removed_any:
            break
            
    # 3. Simulated Annealing Phase
    T = 10.0
    cooling_rate = 0.995
    current_size = len(network)
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        if no_improve_count > max_no_improve:
            network = best_network[:]
            current_size = best_size
            T = 5.0
            no_improve_count = 0
            if len(network) > 1:
                i, j = random.sample(range(len(network)), 2)
                network[i], network[j] = network[j], network[i]
        
        r = random.random()
        new_network = network[:]
        
        if r < 0.3:
            if len(network) > 0:
                idx = random.randint(0, len(network) - 1)
                new_network = network[:idx] + network[idx+1:]
        elif r < 0.5:
            i, j = random.choice(all_pairs)
            pos = random.randint(0, len(network))
            new_network = network[:pos] + [(i, j)] + network[pos:]
        elif r < 0.8:
            if len(network) > 0:
                idx = random.randint(0, len(network) - 1)
                comp = network[idx]
                new_network = network[:idx] + network[idx+1:]
                new_pos = random.randint(0, len(new_network))
                new_network = new_network[:new_pos] + [comp] + new_network[new_pos:]
        else:
            if len(network) > 1:
                i, j = random.sample(range(len(network)), 2)
                new_network = network[:]
                new_network[i], new_network[j] = new_network[j], new_network[i]
            else:
                i, j = random.choice(all_pairs)
                pos = random.randint(0, len(network))
                new_network = network[:pos] + [(i, j)] + network[pos:]
                
        new_size = len(new_network)
        
        if _verify_network(new_network):
            delta = new_size - current_size
            
            if delta < 0:
                network = new_network
                current_size = new_size
                if new_size < best_size:
                    best_network = new_network[:]
                    best_size = new_size
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            elif delta == 0:
                if random.random() < 0.5:
                    network = new_network
                    current_size = new_size
                no_improve_count += 1
            else:
                p = np.exp(-delta / T)
                if random.random() < p:
                    network = new_network
                    current_size = new_size
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        T = max(0.1, T * cooling_rate)
        
    return best_network

# Pre-compute and cache the result to avoid recomputation
_cached_network = None

def _get_network():
    global _cached_network
    if _cached_network is None:
        _cached_network = construct()
    return _cached_network

# Override construct to use caching
_original_construct = construct

def construct():
    global _cached_network
    if _cached_network is None:
        _cached_network = _original_construct()
    return _cached_network
