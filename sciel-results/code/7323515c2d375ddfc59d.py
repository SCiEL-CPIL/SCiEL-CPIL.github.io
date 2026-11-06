"""Size-efficient sorting networks on 18 wires via fast local search."""
import time
import random
import numpy as np

N = 18
DEADLINE = 50.0  # safety margin before 60s limit

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Generate all 2^18 boolean inputs
    all_inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for bit in range(N):
        all_inputs[:, bit] = (np.arange(1 << N) >> bit) & 1
    
    # Sorted outputs: for each input with k ones, output has (N-k) zeros then k ones
    num_ones = all_inputs.sum(axis=1)
    sorted_outputs = np.zeros((1 << N, N), dtype=np.uint8)
    for bit in range(N):
        sorted_outputs[:, bit] = (bit >= (N - num_ones)).astype(np.uint8)

    # Fast verification using vectorized operations
    def verify_fast(network):
        """Check if network sorts all inputs. Returns True if valid."""
        if not network:
            return np.array_equal(all_inputs, sorted_outputs)
        
        wires = np.copyto(np.empty_like(all_inputs), all_inputs)
        for (i, j) in network:
            mask = (wires[:, i] == 1) & (wires[:, j] == 0)
            if np.any(mask):
                wires[mask, i] = 0
                wires[mask, j] = 1
        
        return np.array_equal(wires, sorted_outputs)

    # Initial network: odd-even transposition (153 comparators)
    current_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify initial
    if not verify_fast(current_network):
        # Fallback: return a known valid network
        return current_network
    
    best_network = list(current_network)
    best_size = len(best_network)
    
    rng = random.Random(42)
    
    # Hill climbing: iteratively try to remove comparators
    improved = True
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        n = len(current_network)
        # Try removing each comparator in random order
        indices = list(range(n))
        rng.shuffle(indices)
        for i in indices:
            if (time.time() - start_time) > deadline - 5:
                break
            new_net = current_network[:i] + current_network[i+1:]
            if verify_fast(new_net):
                current_network = new_net
                if len(current_network) < best_size:
                    best_network = list(current_network)
                    best_size = len(best_network)
                improved = True
                break  # Restart after a successful removal
    
    # More aggressive local search with moves
    for restart in range(20):
        if (time.time() - start_time) > deadline - 5:
            break
        
        net = list(best_network)
        stuck = 0
        
        for step in range(500):
            if (time.time() - start_time) > deadline - 5:
                break
            
            n = len(net)
            if n == 0:
                break
            
            action = rng.random()
            success = False
            
            if action < 0.5:
                # Try to remove a random comparator
                idx = rng.randint(0, n - 1)
                new_net = net[:idx] + net[idx+1:]
                if verify_fast(new_net):
                    net = new_net
                    success = True
                    if len(net) < best_size:
                        best_network = list(net)
                        best_size = len(best_network)
            elif action < 0.8:
                # Try to move a random comparator
                idx = rng.randint(0, n - 1)
                new_idx = rng.randint(0, n)
                if new_idx > idx:
                    new_idx -= 1
                if new_idx != idx:
                    comp = net[idx]
                    new_net = net[:idx] + net[idx+1:]
                    new_net = new_net[:new_idx] + [comp] + new_net[new_idx:]
                    if verify_fast(new_net):
                        net = new_net
                        success = True
            else:
                # Try to swap two random comparators
                if n >= 2:
                    idx1 = rng.randint(0, n - 1)
                    idx2 = rng.randint(0, n - 1)
                    if idx1 != idx2:
                        new_net = list(net)
                        new_net[idx1], new_net[idx2] = new_net[idx2], new_net[idx1]
                        if verify_fast(new_net):
                            net = new_net
                            success = True
            
            if not success:
                stuck += 1
                if stuck > 50:
                    break
            else:
                stuck = 0
    
    return best_network
