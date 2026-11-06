"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

N = 17
TOTAL_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    """Find a minimal sorting network for N=17 wires."""
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute all boolean inputs: shape (TOTAL_INPUTS, N)
    indices = np.arange(TOTAL_INPUTS, dtype=np.int32)
    inputs = np.zeros((TOTAL_INPUTS, N), dtype=np.uint8)
    for j in range(N):
        inputs[:, j] = (indices >> j) & 1
    
    def verify_fast(network):
        """Check if network sorts all inputs. Returns True if correct."""
        if not network:
            return N <= 1
        mat = inputs.copy()
        for i, j in network:
            mask = (mat[:, i] == 1) & (mat[:, j] == 0)
            if np.any(mask):
                mat[mask, i] = 0
                mat[mask, j] = 1
        # Check sorted: no 1 followed by 0 in any row
        diffs = np.diff(mat.astype(np.int8), axis=1)
        return not np.any(diffs == -1)
    
    # Build a good initial network using odd-even merge sort idea
    def make_initial_network():
        """Create a reasonable initial sorting network."""
        # Start with odd-even transposition as base
        net = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
        return net
    
    network = make_initial_network()
    best_network = list(network)
    best_count = len(network)
    
    # Verify seed is valid
    assert verify_fast(network), "Seed network should be valid"
    
    # Simulated annealing with multiple restarts
    rng = random.Random(42)
    
    def time_left():
        return deadline - time.time()
    
    temp = 2.5
    cooling = 0.996
    
    while time_left() > 3.0:
        # Choose move
        move_type = rng.random()
        new_network = list(network)
        
        if move_type < 0.28:
            # Insert comparator
            if len(new_network) >= 4096:
                continue
            r = rng.random()
            if r < 0.35:
                # Adjacent
                i = rng.randint(0, N - 2)
                j = i + 1
            elif r < 0.70:
                # Medium distance (2-5)
                i = rng.randint(0, N - 3)
                j = i + rng.randint(2, min(5, N - 1 - i))
            else:
                # Long distance
                i = rng.randint(0, N - 4)
                j = i + rng.randint(6, N - 1 - i)
            pos = rng.randint(0, len(new_network))
            new_network.insert(pos, (i, j))
            
        elif move_type < 0.52:
            # Delete comparator
            if len(new_network) <= 2:
                continue
            pos = rng.randint(0, len(new_network) - 1)
            new_network.pop(pos)
            
        elif move_type < 0.72:
            # Swap adjacent
            if len(new_network) < 2:
                continue
            pos = rng.randint(0, len(new_network) - 2)
            new_network[pos], new_network[pos + 1] = new_network[pos + 1], new_network[pos]
            
        elif move_type < 0.87:
            # Move comparator
            if len(new_network) < 2:
                continue
            pos = rng.randint(0, len(new_network) - 1)
            new_pos = rng.randint(0, len(new_network) - 1)
            if pos != new_pos:
                comp = new_network.pop(pos)
                new_network.insert(new_pos, comp)
        else:
            # Replace comparator
            if len(new_network) < 1:
                continue
            pos = rng.randint(0, len(new_network) - 1)
            r = rng.random()
            if r < 0.4:
                i = rng.randint(0, N - 2)
                j = i + 1
            elif r < 0.75:
                i = rng.randint(0, N - 3)
                j = i + rng.randint(2, min(5, N - 1 - i))
            else:
                i = rng.randint(0, N - 4)
                j = i + rng.randint(6, N - 1 - i)
            new_network[pos] = (i, j)
        
        new_count = len(new_network)
        
        # Quick reject for much larger networks
        if new_count > best_count + 8:
            continue
        
        # Verify
        if not verify_fast(new_network):
            if rng.random() > 0.03:
                continue
        
        # Acceptance criterion
        if new_count < best_count:
            network = new_network
            best_network = list(new_network)
            best_count = new_count
            # Reheat
            temp = min(temp * 1.15, 3.0)
        elif new_count == best_count:
            network = new_network
        else:
            delta = new_count - len(network)
            if delta <= 0:
                network = new_network
            else:
                prob = np.exp(-delta / max(temp, 0.05))
                if rng.random() < prob:
                    network = new_network
        
        temp *= cooling
        if temp < 0.05:
            temp = 0.05
    
    return best_network
# EVOLVE-BLOCK-END
