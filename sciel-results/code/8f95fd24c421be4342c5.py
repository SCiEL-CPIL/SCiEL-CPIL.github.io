import time
import numpy as np
import random

N = 18
NUM_MASKS = 1 << N

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Vectorized verification using 0-1 principle
    def verify_network(net):
        """Check if network sorts all 0-1 inputs using vectorized numpy."""
        if not net:
            return True
        # Create array of all 2^N masks
        masks = np.arange(NUM_MASKS, dtype=np.uint32)
        # Extract bits: x[m, i] = (masks[m] >> i) & 1
        # Shape: (NUM_MASKS, N)
        bits = (masks[:, None] >> np.arange(N)[None, :]) & 1
        
        for i, j in net:
            # Compare bits at positions i and j
            # min goes to i, max goes to j
            # min(x[i], x[j]) = x[i] & x[j]
            # max(x[i], x[j]) = x[i] | x[j]
            min_val = bits[:, i] & bits[:, j]
            max_val = bits[:, i] | bits[:, j]
            bits[:, i] = min_val
            bits[:, j] = max_val
        
        # Check if sorted: bits[:, k] <= bits[:, k+1] for all k
        # i.e., no k where bits[:, k] = 1 and bits[:, k+1] = 0
        for k in range(N - 1):
            if np.any((bits[:, k] == 1) & (bits[:, k + 1] == 0)):
                return False
        return True
    
    # Verify initial network
    assert verify_network(network), "Initial network is invalid"
    
    # Local search: try removing comparators
    improved = True
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 5:
                break
            # Try removing comparator at idx
            test_network = network[:idx] + network[idx + 1:]
            if verify_network(test_network):
                network = test_network
                improved = True
    
    # Simulated annealing to escape local minima
    if (time.time() - start_time) < deadline - 10:
        current_net = network[:]
        best_net = network[:]
        best_len = len(network)
        temperature = 5.0
        cooling = 0.995
        max_steps = 50000
        
        rng = random.Random(42)
        
        for step in range(max_steps):
            if (time.time() - start_time) > deadline - 5:
                break
            
            # Choose a random move
            move_type = rng.random()
            
            if move_type < 0.6:
                # Try removing a random comparator
                if len(current_net) <= 1:
                    continue
                idx = rng.randint(0, len(current_net) - 1)
                new_net = current_net[:idx] + current_net[idx + 1:]
                delta = -1  # one fewer comparator
            elif move_type < 0.85:
                # Try adding a random comparator
                existing = set(current_net)
                all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
                candidates = [p for p in all_pairs if p not in existing]
                if not candidates:
                    continue
                new_comp = rng.choice(candidates)
                # Insert at a random position
                pos = rng.randint(0, len(current_net))
                new_net = current_net[:pos] + [new_comp] + current_net[pos:]
                delta = 1  # one more comparator
            else:
                # Try swapping two adjacent comparators
                if len(current_net) < 2:
                    continue
                idx = rng.randint(0, len(current_net) - 2)
                new_net = current_net[:]
                new_net[idx], new_net[idx + 1] = new_net[idx + 1], new_net[idx]
                delta = 0
            
            # Verify new network
            if not verify_network(new_net):
                # Reject invalid moves
                temperature *= cooling
                continue
            
            # Accept or reject based on simulated annealing
            if delta < 0:
                # Better: always accept
                current_net = new_net
                if len(current_net) < best_len:
                    best_net = current_net[:]
                    best_len = len(current_net)
            elif delta == 0:
                # Same length: accept with some probability
                if rng.random() < 0.5:
                    current_net = new_net
            else:
                # Worse: accept with probability exp(-delta/temperature)
                prob = np.exp(-delta / max(temperature, 1e-10))
                if rng.random() < prob:
                    current_net = new_net
            
            temperature *= cooling
            if temperature < 0.01:
                break
        
        if len(best_net) < len(network):
            network = best_net
    
    # Final greedy removal pass
    improved = True
    while improved and (time.time() - start_time) < deadline - 2:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 2:
                break
            test_network = network[:idx] + network[idx + 1:]
            if verify_network(test_network):
                network = test_network
                improved = True
    
    return network
