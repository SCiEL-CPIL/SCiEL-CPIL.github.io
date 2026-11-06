import time
import random
import math

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

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
    
    def is_valid(network):
        state = [initial_state[k] for k in range(N)]
        for i, j in network:
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    if not is_valid(best_network):
        return best_network

    random.seed(42)
    
    # 3. Optimized Local Search: Greedy Removal
    def try_remove_all(current_net, max_rounds=5):
        removed = True
        rounds = 0
        while removed and rounds < max_rounds:
            if time.time() > deadline:
                break
            removed = False
            rounds += 1
            indices = list(range(len(current_net)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                # In-place removal check
                comp = current_net[idx]
                del current_net[idx]
                if is_valid(current_net):
                    removed = True
                    break
                else:
                    current_net.insert(idx, comp)
        return current_net

    # Initial aggressive cleanup
    best_network = try_remove_all(best_network, max_rounds=10)
    
    # Restart SA Loop
    while time.time() < deadline:
        # Perturb best solution
        current_network = best_network[:]
        n = len(current_network)
        
        if n > 2:
            # Perform 1-2 random swaps or moves
            for _ in range(random.randint(1, 2)):
                if time.time() > deadline:
                    break
                action = random.random()
                if action < 0.5:
                    # Swap
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j:
                        current_network[i], current_network[j] = current_network[j], current_network[i]
                else:
                    # Move
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j:
                        comp = current_network.pop(i)
                        if i < j:
                            j -= 1
                        current_network.insert(j, comp)
        else:
            # If very small, shuffle
            random.shuffle(current_network)

        # SA Parameters
        initial_temp = 5.0
        cooling_rate = 0.995
        min_temp = 0.1
        temperature = initial_temp
        
        max_steps = 5000 
        steps = 0
        
        while steps < max_steps and time.time() < deadline and temperature > min_temp:
            steps += 1
            neighbor = current_network[:]
            strategy = random.random()
            n = len(neighbor)
            
            if strategy < 0.30:
                # Remove
                if n > 1:
                    idx = random.randint(0, n - 1)
                    neighbor.pop(idx)
            elif strategy < 0.60:
                # Add
                new_i = random.randint(0, N - 2)
                new_j = random.randint(new_i + 1, N - 1)
                pos = random.randint(0, len(neighbor))
                neighbor.insert(pos, (new_i, new_j))
            elif strategy < 0.80:
                # Move
                if n > 2:
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j:
                        comp = neighbor.pop(i)
                        if i < j:
                            j -= 1
                        neighbor.insert(j, comp)
            else:
                # Swap
                if n > 1:
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j:
                        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
            if not is_valid(neighbor):
                temperature *= cooling_rate
                continue
            
            current_cost = len(current_network)
            neighbor_cost = len(neighbor)
            delta = neighbor_cost - current_cost
            
            if delta <= 0:
                # Improvement or equal: Clean up
                neighbor = try_remove_all(neighbor, max_rounds=2)
                current_network = neighbor
                if len(current_network) < len(best_network):
                    best_network = current_network[:]
            else:
                # Worse: Accept with probability
                if temperature > 0:
                    prob = math.exp(-delta / temperature)
                    if random.random() < prob:
                        current_network = neighbor
            
            temperature *= cooling_rate

    # Final cleanup
    if time.time() < deadline:
        best_network = try_remove_all(best_network, max_rounds=10)
        
    return best_network
# EVOLVE-BLOCK-END
