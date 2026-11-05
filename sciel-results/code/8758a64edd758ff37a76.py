import numpy as np
import time
import random

N = 17
NUM_INPUTS = 1 << N
DEADLINE = 85  # seconds, safety margin before 90s kill

# Precompute all 2^N inputs as packed uint64 columns (one bit per input per wire)
def _make_packed_inputs():
    """Return array of shape (N,) of uint64, where bit k of column w is input k's wire w."""
    bits = np.arange(NUM_INPUTS, dtype=np.uint64)
    cols = np.empty(N, dtype=np.uint64)
    for k in range(N):
        cols[k] = (bits >> k) & ((1 << N) - 1)  # bit k set iff input has wire k = 1
    return cols

def _apply_comparator_packed(cols, i, j):
    """Apply compare-exchange on wires i,j. min->i, max->j. Returns new cols array."""
    a = cols[i]
    b = cols[j]
    cols[i] = a & b   # min (AND for booleans)
    cols[j] = a | b   # max (OR for booleans)
    return cols

def _is_sorted_packed(cols):
    """Check if all inputs are sorted: for each adjacent pair, no input has 1 on left and 0 on right."""
    for k in range(N - 1):
        # wire k should be <= wire k+1 for all inputs
        # violation: wire k = 1 AND wire k+1 = 0
        if (cols[k] & ~cols[k+1]) != 0:
            return False
    return True

def _validate_packed(comparators, cols):
    """Validate network by applying all comparators to packed columns."""
    state = cols.copy()
    for i, j in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
    return _is_sorted_packed(state)

def _depth(comparators):
    """Compute parallel depth of network."""
    if not comparators:
        return 0
    # Greedy coloring: assign each comparator to earliest layer where no conflict
    # A comparator (i,j) conflicts with (k,l) if they share a wire
    used = {}  # layer -> set of wires used
    layer = 0
    for comp in comparators:
        i, j = comp
        # Find earliest layer where wires i,j are free
        while layer in used and (i in used[layer] or j in used[layer]):
            layer += 1
        if layer not in used:
            used[layer] = set()
        used[layer].add(i)
        used[layer].add(j)
    return len(used)

def _odd_even_transposition():
    """Standard odd-even transposition network: 136 comparators."""
    comps = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps

def _random_network(n_comps):
    """Generate a random sorting network candidate."""
    comps = []
    for _ in range(n_comps):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comps.append((i, j))
    return comps

def _mutate(comparators, cols, mutation_rate=0.1):
    """Mutate a network: add/remove/swap comparators."""
    comps = list(comparators)
    n_ops = max(1, int(len(comps) * mutation_rate))
    for _ in range(n_ops):
        op = random.random()
        if op < 0.4 and comps:  # remove
            idx = random.randint(0, len(comps) - 1)
            comps.pop(idx)
        elif op < 0.7:  # add
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(comps))
            comps.insert(pos, (i, j))
        elif op < 0.9 and len(comps) >= 2:  # swap adjacent
            idx = random.randint(0, len(comps) - 2)
            comps[idx], comps[idx + 1] = comps[idx + 1], comps[idx]
        else:  # change a comparator's wires
            if comps:
                idx = random.randint(0, len(comps) - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                comps[idx] = (i, j)
    return comps

def _crossover(parent1, parent2, cols):
    """Uniform crossover: take each position from one parent or the other."""
    n = max(len(parent1), len(parent2))
    child = []
    for k in range(n):
        if random.random() < 0.5:
            if k < len(parent1):
                child.append(parent1[k])
        else:
            if k < len(parent2):
                child.append(parent2[k])
    return child

def _try_remove_greedy(comparators, cols):
    """Try to remove redundant comparators one by one."""
    comps = list(comparators)
    changed = True
    while changed:
        changed = False
        for idx in range(len(comps)):
            if len(comps) <= 1:
                break
            new_comps = comps[:idx] + comps[idx+1:]
            if _validate_packed(new_comps, cols):
                comps = new_comps
                changed = True
                break
    return comps

def construct():
    start_time = time.time()
    cols = _make_packed_inputs()
    
    # Start with odd-even transposition
    best = _odd_even_transposition()
    best_size = len(best)
    
    # First pass: greedy removal from seed
    best = _try_remove_greedy(best, cols)
    best_size = len(best)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return best
    
    # Genetic algorithm to find smaller networks
    population = [best]
    # Add some random initial individuals
    for _ in range(5):
        pop_size = random.randint(max(50, best_size - 20), best_size + 10)
        pop = _random_network(pop_size)
        if _validate_packed(pop, cols):
            pop = _try_remove_greedy(pop, cols)
            population.append(pop)
    
    generation = 0
    while time.time() - start_time < DEADLINE - 5:
        generation += 1
        new_population = []
        
        # Elitism: keep best
        new_population.append(best)
        
        # Create offspring
        n_offspring = min(10, max(3, int((DEADLINE - 5 - (time.time() - start_time)) / 2)))
        
        for _ in range(n_offspring):
            if len(population) < 2:
                # Just mutate best
                child = _mutate(best, cols, mutation_rate=0.15)
            else:
                # Select two parents (tournament)
                p1 = random.choice(population)
                p2 = random.choice(population)
                child = _crossover(p1, p2, cols)
                child = _mutate(child, cols, mutation_rate=0.1)
            
            if not child:
                continue
            
            # Validate
            if _validate_packed(child, cols):
                # Try to shrink
                child = _try_remove_greedy(child, cols)
                if len(child) < best_size:
                    best = child
                    best_size = len(child)
                new_population.append(child)
        
        # Update population
        population = new_population
        # Keep population diverse but bounded
        if len(population) > 8:
            # Keep best and some random others
            population = [best] + random.sample(population[1:], min(7, len(population) - 1))
        
        # Periodically try aggressive removal on best
        if generation % 5 == 0 and time.time() - start_time < DEADLINE - 10:
            improved = _try_remove_greedy(best, cols)
            if len(improved) < best_size:
                best = improved
                best_size = len(best)
    
    # Final: try removal one more time
    if time.time() - start_time < DEADLINE - 2:
        best = _try_remove_greedy(best, cols)
    
    return best
