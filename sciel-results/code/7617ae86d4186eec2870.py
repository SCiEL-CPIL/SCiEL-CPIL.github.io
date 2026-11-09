"""Genetic algorithm for size-efficient sorting networks on 18 wires."""
import time
import random
import numpy as np

N = 18
NUM_BOOLEANS = 1 << N  # 262144

# Precompute all boolean inputs as a (NUM_BOOLEANS, N) uint8 array
# Row r has bits of r distributed across N wires
def _build_inputs():
    idx = np.arange(NUM_BOOLEANS, dtype=np.uint32)
    inputs = np.empty((NUM_BOOLEANS, N), dtype=np.uint8)
    for w in range(N):
        inputs[:, w] = ((idx >> w) & 1).astype(np.uint8)
    return inputs

# Precompute target: sorted output for each input (0s then 1s)
def _build_targets(inputs):
    counts = inputs.sum(axis=1)  # number of 1s per input
    targets = np.zeros_like(inputs)
    for r in range(NUM_BOOLEANS):
        c = counts[r]
        targets[r, N - c:] = 1
    return targets

# Bit-parallel simulation: apply network to all inputs at once
def _simulate(network, inputs, targets):
    """Apply comparator network to all boolean inputs. Returns count of mismatches."""
    state = inputs.copy()
    for (i, j) in network:
        a = state[:, i]
        b = state[:, j]
        # min to i, max to j
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    # Count mismatches
    mismatches = (state != targets).any(axis=1).sum()
    return int(mismatches)

# Faster simulation using bit operations
def _simulate_fast(network, inputs, targets):
    state = inputs.copy()
    for (i, j) in network:
        a = state[:, i]
        b = state[:, j]
        # min(a,b) = a & b, max(a,b) = a | b for bits
        state[:, i] = a & b
        state[:, j] = a | b
    # Compare with targets
    diff = state ^ targets
    # An input is wrong if any bit differs
    wrong = diff.any(axis=1)
    return int(wrong.sum())

# Precomputed data
_INPUTS = _build_inputs()
_TARGETS = _build_targets(_INPUTS)

def _fitness(network):
    """Lower is better: number of unsorted boolean inputs."""
    return _simulate_fast(network, _INPUTS, _TARGETS)

def _random_network(length):
    """Generate a random comparator network."""
    net = []
    for _ in range(length):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        net.append((i, j))
    return net

def _odd_even_transposition():
    """Classic odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _crossover(parent1, parent2):
    """Uniform crossover."""
    length = max(len(parent1), len(parent2))
    child = []
    for k in range(length):
        if random.random() < 0.5:
            if k < len(parent1):
                child.append(parent1[k])
            else:
                child.append(None)
        else:
            if k < len(parent2):
                child.append(parent2[k])
            else:
                child.append(None)
    # Remove None entries
    child = [c for c in child if c is not None]
    return child

def _mutate(network, mutation_rate=0.1):
    """Apply mutations: add, remove, or change comparators."""
    net = list(network)
    n_ops = max(1, int(len(net) * mutation_rate))
    for _ in range(n_ops):
        op = random.random()
        if op < 0.3 and len(net) > 0:
            # Remove a random comparator
            idx = random.randint(0, len(net) - 1)
            net.pop(idx)
        elif op < 0.6:
            # Add a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(net))
            net.insert(pos, (i, j))
        elif op < 0.85 and len(net) > 0:
            # Change a random comparator
            idx = random.randint(0, len(net) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            net[idx] = (i, j)
        else:
            # Swap two adjacent comparators
            if len(net) >= 2:
                idx = random.randint(0, len(net) - 2)
                net[idx], net[idx + 1] = net[idx + 1], net[idx]
    return net

def _local_search(network, max_iters=50, deadline=None):
    """Greedy local search: try to remove or improve comparators."""
    net = list(network)
    current_fit = _fitness(net)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if current_fit == 0:
            break
        
        improved = False
        # Try removing each comparator
        for k in range(len(net)):
            if deadline is not None and time.time() > deadline:
                break
            trial = net[:k] + net[k+1:]
            trial_fit = _fitness(trial)
            if trial_fit < current_fit:
                net = trial
                current_fit = trial_fit
                improved = True
                break
        if improved:
            continue
        
        # Try changing each comparator
        for k in range(len(net)):
            if deadline is not None and time.time() > deadline:
                break
            i, j = net[k]
            # Try a few alternative comparators
            for _ in range(3):
                ni = random.randint(0, N - 2)
                nj = random.randint(ni + 1, N - 1)
                if (ni, nj) == (i, j):
                    continue
                trial = net[:k] + [(ni, nj)] + net[k+1:]
                trial_fit = _fitness(trial)
                if trial_fit < current_fit:
                    net[k] = (ni, nj)
                    current_fit = trial_fit
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        # Try adding a comparator
        if deadline is not None and time.time() > deadline:
            break
        for _ in range(5):
            ni = random.randint(0, N - 2)
            nj = random.randint(ni + 1, N - 1)
            pos = random.randint(0, len(net))
            trial = net[:pos] + [(ni, nj)] + net[pos:]
            trial_fit = _fitness(trial)
            if trial_fit < current_fit:
                net = trial
                current_fit = trial_fit
                improved = True
                break
        if not improved:
            break
    
    return net, current_fit

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    # Start with odd-even transposition
    best_net = _odd_even_transposition()
    best_fit = _fitness(best_net)
    
    # First, do local search to reduce size
    best_net, best_fit = _local_search(best_net, max_iters=200, deadline=deadline)
    
    if best_fit == 0:
        return best_net
    
    # Genetic algorithm
    pop_size = 40
    population = [best_net] + [_random_network(random.randint(60, 100)) for _ in range(pop_size - 1)]
    fitnesses = [_fitness(net) for net in population]
    
    # Track best
    best_idx = int(np.argmin(fitnesses))
    best_net = population[best_idx]
    best_fit = fitnesses[best_idx]
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Sort by fitness
        order = np.argsort(fitnesses)
        population = [population[i] for i in order]
        fitnesses = [fitnesses[i] for i in order]
        
        # Elitism: keep top 5
        elites = population[:5]
        
        new_population = list(elites)
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            p1_idx = int(np.random.choice(10, p=[1/(i+1) for i in range(10)]))
            p2_idx = int(np.random.choice(10, p=[1/(i+1) for i in range(10)]))
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            child = _crossover(p1, p2)
            child = _mutate(child, mutation_rate=0.15)
            new_population.append(child)
        
        # Evaluate new population
        population = new_population
        fitnesses = []
        for net in population:
            if time.time() > deadline:
                break
            f = _fitness(net)
            fitnesses.append(f)
            if f < best_fit:
                best_fit = f
                best_net = list(net)
                if best_fit == 0:
                    return best_net
        
        # Fill in any missing fitnesses
        while len(fitnesses) < len(population):
            fitnesses.append(best_fit)
        
        # Periodic local search on best
        if generation % 3 == 0 and best_fit > 0:
            improved_net, improved_fit = _local_search(best_net, max_iters=50, deadline=deadline)
            if improved_fit < best_fit:
                best_fit = improved_fit
                best_net = improved_net
                if best_fit == 0:
                    return best_net
                # Replace worst in population
                worst_idx = int(np.argmax(fitnesses))
                population[worst_idx] = best_net
                fitnesses[worst_idx] = best_fit
    
    return best_net
