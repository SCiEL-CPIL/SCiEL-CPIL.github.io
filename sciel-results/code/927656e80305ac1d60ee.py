import time
import random

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Verification Function using bit-vector simulation
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask
    
    # Precompute the sorted state pattern
    # After sorting, wire k should have 1s only in inputs where the k-th largest bit is 1
    # For 17 inputs, sorted means: wire 0 has 1s in top 1 input, wire 1 in top 2, etc.
    # Actually, for 0-1 principle: after sorting, wire i should have exactly (N-1-i) ones
    # and they should be in the highest-valued positions
    sorted_state = [0] * N
    for i in range(N):
        # Wire i should have 1s in the inputs that are among the top (N-i) by value
        # Input k has value k, so wire i gets 1 for k >= i
        for k in range(i, N):
            sorted_state[i] |= (1 << k)

    def is_valid(network):
        state = initial_state[:]
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        for k in range(N):
            if state[k] != sorted_state[k]:
                return False
        return True

    if not is_valid(best_network):
        return best_network

    random.seed(123)
    
    # 3. Local Search with aggressive add-remove
    current_network = best_network[:]
    
    def try_remove_all(current_net, max_iterations=100):
        """Aggressively try to remove any redundant comparators."""
        removed = True
        iterations = 0
        while removed and iterations < max_iterations:
            if time.time() > deadline:
                break
            iterations += 1
            removed = False
            indices = list(range(len(current_net)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                candidate = current_net[:idx] + current_net[idx+1:]
                if is_valid(candidate):
                    current_net = candidate
                    removed = True
                    break
        return current_net

    def add_and_optimize(current_net):
        """Add a random comparator, then aggressively remove redundancies."""
        new_i = random.randint(0, N - 1)
        new_j = random.randint(0, N - 1)
        if new_i > new_j:
            new_i, new_j = new_j, new_i
        if new_i == new_j:
            return current_net
        
        pos = random.randint(0, len(current_net))
        candidate = current_net[:]
        candidate.insert(pos, (new_i, new_j))
        
        if not is_valid(candidate):
            return current_net
        
        optimized = try_remove_all(candidate)
        return optimized

    # Multi-restart local search
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            current_network = best_network[:]
        else:
            # Start from best with some random perturbation
            current_network = best_network[:]
            # Add 2-4 random comparators
            for _ in range(random.randint(2, 4)):
                dist = random.choice([1, 2, 3, 4, 5])
                new_i = random.randint(0, N - 1 - dist)
                new_j = new_i + dist
                pos = random.randint(0, len(current_network))
                current_network.insert(pos, (new_i, new_j))
        
        # Run local search
        while time.time() < deadline - 5.0:
            # First, try to remove
            optimized = try_remove_all(current_network)
            if len(optimized) < len(current_network):
                current_network = optimized
                if len(current_network) < len(best_network):
                    best_network = current_network[:]
            
            if time.time() > deadline - 5.0:
                break
            
            strategy = random.random()
            
            if strategy < 0.3:
                # Add and optimize
                new_net = add_and_optimize(current_network)
                if len(new_net) <= len(current_network):
                    current_network = new_net
                    if len(current_network) < len(best_network):
                        best_network = current_network[:]
            
            elif strategy < 0.5:
                # Move a comparator
                if len(current_network) > 1:
                    i = random.randint(0, len(current_network) - 1)
                    j = random.randint(0, len(current_network) - 1)
                    if i != j:
                        candidate = current_network[:]
                        comp = candidate.pop(i)
                        candidate.insert(j, comp)
                        if is_valid(candidate):
                            optimized = try_remove_all(candidate)
                            if len(optimized) <= len(current_network):
                                current_network = optimized
                                if len(current_network) < len(best_network):
                                    best_network = current_network[:]
            
            elif strategy < 0.7:
                # Swap two comparators
                if len(current_network) > 1:
                    i = random.randint(0, len(current_network) - 1)
                    j = random.randint(0, len(current_network) - 1)
                    if i != j:
                        candidate = current_network[:]
                        candidate[i], candidate[j] = candidate[j], candidate[i]
                        if is_valid(candidate):
                            optimized = try_remove_all(candidate)
                            if len(optimized) <= len(current_network):
                                current_network = optimized
                                if len(current_network) < len(best_network):
                                    best_network = current_network[:]
            
            elif strategy < 0.85:
                # Replace a comparator
                if len(current_network) > 0:
                    i = random.randint(0, len(current_network) - 1)
                    new_i = random.randint(0, N - 1)
                    new_j = random.randint(0, N - 1)
                    if new_i > new_j:
                        new_i, new_j = new_j, new_i
                    if new_i != new_j:
                        candidate = current_network[:]
                        candidate[i] = (new_i, new_j)
                        if is_valid(candidate):
                            optimized = try_remove_all(candidate)
                            if len(optimized) <= len(current_network):
                                current_network = optimized
                                if len(current_network) < len(best_network):
                                    best_network = current_network[:]
            
            else:
                # Add 1-3 comparators with various distances
                additions = random.randint(1, 3)
                candidate = current_network[:]
                for _ in range(additions):
                    dist = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
                    new_i = random.randint(0, N - 1 - dist)
                    new_j = new_i + dist
                    pos = random.randint(0, len(candidate))
                    candidate.insert(pos, (new_i, new_j))
                
                if is_valid(candidate):
                    optimized = try_remove_all(candidate)
                    if len(optimized) <= len(current_network):
                        current_network = optimized
                        if len(current_network) < len(best_network):
                            best_network = current_network[:]
    
    return best_network
# EVOLVE-BLOCK-END
