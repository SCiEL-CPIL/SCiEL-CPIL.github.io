import numpy as np
import time
import random

N = 18
NUM_INPUTS = 1 << N

# Precompute the initial state of all wires for all 2^N Boolean inputs
# wires[k] will be a numpy array of uint8 of length NUM_INPUTS
# We can generate these efficiently.
# Wire k has bit k of the input index.
# Index ranges from 0 to 2^N - 1.
# We can create an array of indices and extract bits.
# But 2^18 * 18 bytes is about 4.7 MB, which is fine.

def generate_initial_wires():
    # Create indices 0..2^N-1
    # To avoid creating a huge array of indices, we can use broadcasting or loops.
    # Looping 18 times is fast enough.
    wires = np.zeros((N, NUM_INPUTS), dtype=np.uint8)
    for k in range(N):
        # Bit k of index i is (i >> k) & 1
        # We can compute this for all i using vectorized ops or just rely on the fact that 
        # we can generate the pattern.
        # Pattern for bit k is: 0...0 1...1 0...0 1...1 ...
        # Period is 2^(k+1), first half 0, second half 1.
        period = 1 << (k + 1)
        half = 1 << k
        # Repeat pattern NUM_INPUTS // period times
        pattern = np.array([0]*half + [1]*half, dtype=np.uint8)
        repeats = NUM_INPUTS // period
        wires[k] = np.tile(pattern, repeats)
    return wires

INITIAL_WIRES = generate_initial_wires()

def validate_network(network):
    """
    Returns (is_valid, fail_count).
    is_valid: True if network sorts all inputs.
    fail_count: Number of inputs that are not sorted (for fitness).
    """
    if not network:
        # Check if initial wires are sorted (they are not, unless N=1)
        # Sorted means wires[0] <= wires[1] <= ... <= wires[N-1]
        # For Boolean, sorted means all 0s then all 1s.
        # This is equivalent to checking if the final state matches the sorted state.
        pass
    
    wires = INITIAL_WIRES.copy()
    
    for i, j in network:
        # min to i, max to j
        # min = AND, max = OR
        # We need to update wires[i] and wires[j]
        # temp_i = wires[i] & wires[j]
        # temp_j = wires[i] | wires[j]
        # We can do this in place if we save one value? 
        # wires[i] and wires[j] are distinct arrays.
        # new_i = wires[i] & wires[j]
        # new_j = wires[i] | wires[j]
        
        # Optimization: 
        # Only update if there is a difference? 
        # diff = wires[i] ^ wires[j]
        # If no diff, skip? 
        # But we still need to assign.
        
        new_i = wires[i] & wires[j]
        new_j = wires[i] | wires[j]
        wires[i] = new_i
        wires[j] = new_j
        
    # Check if sorted
    # Sorted means for all k < N-1, wires[k] <= wires[k+1]
    # Since they are 0/1, this means wires[k] & wires[k+1] == wires[k] is NOT the condition.
    # Condition: wires[k] is 1 implies wires[k+1] is 1.
    # i.e., wires[k] & ~wires[k+1] must be 0.
    
    # We can check this for all adjacent pairs.
    # If any pair violates, it's not sorted.
    # We want the count of failing inputs.
    # An input fails if ANY adjacent pair is out of order.
    # fail_mask = 0
    # for k in range(N-1):
    #    fail_mask |= (wires[k] & ~wires[k+1])
    
    fail_mask = np.zeros(NUM_INPUTS, dtype=np.uint8)
    for k in range(N - 1):
        fail_mask |= (wires[k] & ~wires[k+1])
        
    fail_count = np.sum(fail_mask)
    
    return (fail_count == 0, fail_count)

def construct():
    start_time = time.time()
    timeout = 85.0 # Safety margin for 90s limit
    
    # Seed network
    seed_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    seed_valid, seed_fail = validate_network(seed_network)
    best_network = seed_network
    best_score = 1.0 / len(seed_network) if seed_valid else 0.0
    best_fail = seed_fail
    
    # GA Parameters
    pop_size = 50
    max_generations = 1000
    mutation_rate = 0.3
    crossover_rate = 0.7
    
    # Initialize population
    population = [seed_network]
    # Add some random networks
    while len(population) < pop_size:
        size = random.randint(50, 150)
        net = []
        for _ in range(size):
            i = random.randint(0, N - 1)
            j = random.randint(i + 1, N - 1)
            net.append((i, j))
        population.append(net)
    
    # Evaluate initial population
    scores = []
    for net in population:
        valid, fail = validate_network(net)
        if valid:
            score = 1.0 / len(net)
        else:
            # Penalty: lower is worse. 
            # We want to minimize fail count and size.
            # Let's use a composite fitness.
            # If invalid, score is negative or very small.
            # To guide search, we want to reduce fail count first, then size.
            # score = -fail_count / 1000.0 - len(net) * 0.001
            score = -fail_count * 10.0 - len(net)
        scores.append(score)
        
    # Update best
    max_idx = np.argmax(scores)
    if scores[max_idx] > best_score:
        best_network = population[max_idx]
        best_score = scores[max_idx]
        valid, fail = validate_network(best_network)
        if valid:
            best_fail = fail # 0
        else:
            best_fail = fail
            
    gen = 0
    while gen < max_generations:
        if time.time() - start_time > timeout:
            break
            
        # Selection: Tournament
        new_population = []
        
        # Elitism: Keep top 2
        sorted_indices = np.argsort(scores)[::-1]
        for idx in sorted_indices[:2]:
            new_population.append(population[idx])
            
        while len(new_population) < pop_size:
            # Pick parents
            p1_idx = random.randint(0, pop_size - 1)
            p2_idx = random.randint(0, pop_size - 1)
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover
            if random.random() < crossover_rate and len(p1) > 10 and len(p2) > 10:
                # Simple crossover: take prefix of p1 and suffix of p2
                cut = random.randint(1, min(len(p1), len(p2)) - 1)
                child = p1[:cut] + p2[cut:]
            else:
                child = p1.copy()
                
            # Mutation
            if random.random() < mutation_rate:
                # Random mutation operations
                for _ in range(random.randint(1, 3)):
                    op = random.choice(['add', 'remove', 'swap', 'change'])
                    if op == 'add' and len(child) < 200:
                        i = random.randint(0, N - 1)
                        j = random.randint(i + 1, N - 1)
                        pos = random.randint(0, len(child))
                        child.insert(pos, (i, j))
                    elif op == 'remove' and len(child) > 10:
                        pos = random.randint(0, len(child) - 1)
                        child.pop(pos)
                    elif op == 'swap' and len(child) > 1:
                        i1 = random.randint(0, len(child) - 1)
                        i2 = random.randint(0, len(child) - 1)
                        child[i1], child[i2] = child[i2], child[i1]
                    elif op == 'change':
                        pos = random.randint(0, len(child) - 1)
                        i, j = child[pos]
                        # Change j or i slightly
                        if random.random() < 0.5:
                            new_j = random.randint(i + 1, N - 1)
                            child[pos] = (i, new_j)
                        else:
                            new_i = random.randint(0, j - 1)
                            child[pos] = (new_i, j)
                            
            new_population.append(child)
            
        population = new_population
        scores = []
        
        for net in population:
            valid, fail = validate_network(net)
            if valid:
                score = 1.0 / len(net)
            else:
                score = -fail * 10.0 - len(net)
            scores.append(score)
            
        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score:
            best_network = population[max_idx]
            best_score = scores[max_idx]
            valid, fail = validate_network(best_network)
            if valid:
                best_fail = 0
            else:
                best_fail = fail
                
        gen += 1
        
    return best_network

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# We need to ensure the interface matches.
# The task says: "Keep the function names and signatures of the seed program."
# Seed: def construct():
# So we just expose construct.
# EVOLVE-BLOCK-END
