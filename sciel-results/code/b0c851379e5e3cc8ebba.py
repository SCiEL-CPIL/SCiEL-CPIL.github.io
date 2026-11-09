import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sorting network for 16 wires.
    Starts with Odd-Even Transposition (120 comparators) and applies hill climbing
    to minimize the number of comparators.
    """
    start_time = time.time()
    time_limit = 50.0  # Leave 10s margin for safety

    # 1. Generate Odd-Even Transposition Network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # 2. Verify correctness of initial network
    if not is_correct_full(network):
        # Should not happen, but return as is
        return network
    
    best_network = network
    best_size = len(network)
    current_network = network
    current_size = best_size
    
    # Pre-compute masks for fast simulation
    masks = [1 << i for i in range(N)]
    
    def simulate(x, net):
        for i, j in net:
            mi = masks[i]
            mj = masks[j]
            if (x & mi) and not (x & mj):
                x ^= (mi | mj)
        return x

    def is_correct_full(net):
        for x in range(1 << N):
            out = simulate(x, net)
            if out & (out + 1) != 0:
                return False
        return True

    # Hill Climbing
    while time.time() - start_time < time_limit:
        # Try to remove a random comparator
        if len(current_network) > 0:
            idx = random.randint(0, len(current_network) - 1)
            new_network = current_network[:idx] + current_network[idx+1:]
            
            if is_correct_full(new_network):
                current_network = new_network
                current_size -= 1
                if current_size < best_size:
                    best_network = current_network
                    best_size = current_size
            else:
                # If removal fails, we don't add back immediately to save time.
                # We just try another removal in the next iteration.
                pass

    return best_network

# EVOLVE-BLOCK-END
