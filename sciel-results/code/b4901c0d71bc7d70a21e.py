import numpy as np
import time
import random

N = 17
NUM_INPUTS = 1 << N
DEADLINE = 85  # seconds, safety margin

def _make_all_inputs():
    """Create array of all 2^N Boolean inputs as (NUM_INPUTS, N) bool array."""
    bits = np.arange(NUM_INPUTS, dtype=np.uint64)
    arr = np.empty((NUM_INPUTS, N), dtype=bool)
    for k in range(N):
        arr[:, k] = (bits >> k) & 1
    return arr

def _simulate(comparators, input_matrix):
    """Simulate the network on all inputs. Returns final state matrix."""
    wires = input_matrix.copy()
    for i, j in comparators:
        # min to i, max to j
        # Using bitwise operations for boolean min/max
        # min(a, b) = a & b
        # max(a, b) = a | b
        # We need to update both using the original values
        wi = wires[:, i]
        wj = wires[:, j]
        wires[:, i] = wi & wj
        wires[:, j] = wi | wj
    return wires

def _is_valid(comparators, input_matrix):
    """Check if network sorts all inputs using zero-one principle."""
    final_wires = _simulate(comparators, input_matrix)
    # Check if each row is sorted (non-decreasing)
    # A row is sorted if all adjacent pairs satisfy wires[:, k] <= wires[:, k+1]
    # For bools, this means no 1 is followed by a 0
    left = final_wires[:, :-1]
    right = final_wires[:, 1:]
    return not np.any(left & ~right)

def construct():
    start_time = time.time()
    random.seed(42) # Reproducibility
    
    input_matrix = _make_all_inputs()
    
    # Initial network: Odd-even transposition
    def _odd_even_network():
        comps = []
        for phase in range(N):
            if phase % 2 == 0:
                for i in range(0, N - 1, 2):
                    comps.append((i, i + 1))
            else:
                for i in range(1, N - 1, 2):
                    comps.append((i, i + 1))
        return comps

    best_network = _odd_even_network()
    best_score = len(best_network)
    
    # Verify best network
    if not _is_valid(best_network, input_matrix):
        return best_network

    # Genetic Algorithm parameters
    pop_size = 50
    mutation_rate = 0.2
    crossover_rate = 0.7
    
    # Initialize population
    population = [best_network.copy() for _ in range(pop_size)]
    # Add some random variations
    for idx in range(pop_size):
        net = best_network.copy()
        # Random mutations
        for _ in range(5):
            if random.random() < 0.5:
                # Delete
                if net:
                    pos = random.randint(0, len(net) - 1)
                    net.pop(pos)
            else:
                # Insert
                i = random.randint(0, N - 1)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(net))
                net.insert(pos, (i, j))
        population[idx] = net

    # Evaluate initial population
    scores = []
    for net in population:
        if _is_valid(net, input_matrix):
            scores.append(len(net))
        else:
            scores.append(float('inf'))
            
    # Sort population by score
    pop_with_scores = sorted(zip(scores, population), key=lambda x: x[0])
    population = [p for s, p in pop_with_scores]
    scores = [s for s, p in pop_with_scores]
    
    # Update best
    if scores[0] < best_score:
        best_score = scores[0]
        best_network = population[0]

    while time.time() - start_time < DEADLINE:
        new_population = []
        
        # Elitism: keep the best few
        elite_count = 5
        for i in range(elite_count):
            new_population.append(population[i].copy())
        
        # Generate new individuals
        while len(new_population) < pop_size:
            # Tournament selection
            def select():
                idx1 = random.randint(0, pop_size - 1)
                idx2 = random.randint(0, pop_size - 1)
                if scores[idx1] <= scores[idx2]:
                    return population[idx1]
                else:
                    return population[idx2]
            
            parent1 = select()
            parent2 = select()
            
            if random.random() < crossover_rate:
                # Crossover: cut at a random point
                if not parent1 or not parent2:
                    child = parent1 if random.random() < 0.5 else parent2
                else:
                    cut1 = random.randint(0, len(parent1))
                    cut2 = random.randint(0, len(parent2))
                    child = parent1[:cut1] + parent2[cut2:]
            else:
                child = parent1.copy() if random.random() < 0.5 else parent2.copy()
            
            # Mutation
            if random.random() < mutation_rate:
                # Multiple mutations
                num_mutations = random.randint(1, 3)
                for _ in range(num_mutations):
                    if not child:
                        # Insert a random comparator
                        i = random.randint(0, N - 1)
                        j = random.randint(i + 1, N - 1)
                        child.append((i, j))
                        continue
                    
                    mut_type = random.choice(['delete', 'insert', 'swap', 'change'])
                    if mut_type == 'delete':
                        if child:
                            pos = random.randint(0, len(child) - 1)
                            child.pop(pos)
                    elif mut_type == 'insert':
                        i = random.randint(0, N - 1)
                        j = random.randint(i + 1, N - 1)
                        pos = random.randint(0, len(child))
                        child.insert(pos, (i, j))
                    elif mut_type == 'swap':
                        if len(child) > 1:
                            i1 = random.randint(0, len(child) - 1)
                            i2 = random.randint(0, len(child) - 1)
                            child[i1], child[i2] = child[i2], child[i1]
                    elif mut_type == 'change':
                        pos = random.randint(0, len(child) - 1)
                        i = random.randint(0, N - 1)
                        j = random.randint(i + 1, N - 1)
                        child[pos] = (i, j)
            
            new_population.append(child)
        
        # Evaluate new population
        new_scores = []
        for net in new_population:
            if _is_valid(net, input_matrix):
                new_scores.append(len(net))
            else:
                new_scores.append(float('inf'))
        
        # Combine and select
        combined = list(zip(new_scores, new_population))
        combined.sort(key=lambda x: x[0])
        
        population = [p for s, p in combined[:pop_size]]
        scores = [s for s, p in combined[:pop_size]]
        
        # Update best
        if scores[0] < best_score:
            best_score = scores[0]
            best_network = population[0]

    return best_network
