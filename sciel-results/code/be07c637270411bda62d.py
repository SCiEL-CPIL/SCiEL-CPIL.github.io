"""Genetic algorithm with bitmask operations for 3-AP maximization with no 4-AP constraint."""
import numpy as np

N = 256

def precompute_bitmasks():
    """Precompute 3-AP and 4-AP as lists of integer bitmasks."""
    # 3-AP: (a, a+d, a+2d) -> bitmask with bits at positions a-1, a+d-1, a+2d-1
    three_aps = []
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            c = a + 2 * d
            if c <= N:
                mask = (1 << (a - 1)) | (1 << (a + d - 1)) | (1 << (c - 1))
                three_aps.append(mask)
    
    # 4-AP: (a, a+d, a+2d, a+3d) -> bitmask
    four_aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            c = a + 3 * d
            if c <= N:
                mask = (1 << (a - 1)) | (1 << (a + d - 1)) | (1 << (a + 2 * d - 1)) | (1 << (c - 1))
                four_aps.append(mask)
    
    return three_aps, four_aps

THREE_APS, FOUR_APS = precompute_bitmasks()
NUM_THREE_APS = len(THREE_APS)
NUM_FOUR_APS = len(FOUR_APS)

def count_three_aps_bitmask(bits):
    """Count 3-APs in an integer bitmask using bit operations."""
    count = 0
    for ap_mask in THREE_APS:
        if (bits & ap_mask) == ap_mask:
            count += 1
    return count

def has_four_ap_bitmask(bits):
    """Check if bitmask contains any 4-AP using bit operations."""
    for ap_mask in FOUR_APS:
        if (bits & ap_mask) == ap_mask:
            return True
    return False

def fitness_bitmask(bits):
    """Fitness = number of 3-APs if no 4-AP, else negative penalty."""
    if has_four_ap_bitmask(bits):
        return -100000
    return count_three_aps_bitmask(bits)

def repair(bits):
    """Remove elements to eliminate all 4-APs."""
    while has_four_ap_bitmask(bits):
        # Find a 4-AP
        for ap_mask in FOUR_APS:
            if (bits & ap_mask) == ap_mask:
                # Remove one random element from this 4-AP
                # Get the set bits in ap_mask
                temp = ap_mask
                positions = []
                while temp:
                    lowbit = temp & (-temp)
                    pos = lowbit.bit_length() - 1
                    positions.append(pos)
                    temp ^= lowbit
                # Pick a random position to remove
                pos_to_remove = positions[np.random.randint(len(positions))]
                bits &= ~(1 << pos_to_remove)
                break
    return bits

def greedy_construction(order=None):
    """Greedy: add elements in given order, skip if creates 4-AP."""
    if order is None:
        order = list(range(N))
    bits = 0
    for idx in order:
        new_bits = bits | (1 << idx)
        if not has_four_ap_bitmask(new_bits):
            bits = new_bits
    return bits

def local_search(bits, max_rounds=200):
    """Local search: try adding, removing, and swapping elements."""
    best_bits = bits
    best_fit = fitness_bitmask(best_bits)
    
    for _ in range(max_rounds):
        improved = False
        
        # Try adding each missing element
        for idx in range(N):
            if not (best_bits >> idx) & 1:
                new_bits = best_bits | (1 << idx)
                if not has_four_ap_bitmask(new_bits):
                    new_fit = count_three_aps_bitmask(new_bits)
                    if new_fit > best_fit:
                        best_bits = new_bits
                        best_fit = new_fit
                        improved = True
        
        # Try removing each present element
        for idx in range(N):
            if (best_bits >> idx) & 1:
                new_bits = best_bits & ~(1 << idx)
                new_fit = count_three_aps_bitmask(new_bits)
                if new_fit > best_fit:
                    best_bits = new_bits
                    best_fit = new_fit
                    improved = True
        
        if not improved:
            break
    
    return best_bits

def crossover(bits1, bits2):
    """Uniform crossover between two bitmasks."""
    child = 0
    for i in range(N):
        if np.random.random() < 0.5:
            if (bits1 >> i) & 1:
                child |= (1 << i)
        else:
            if (bits2 >> i) & 1:
                child |= (1 << i)
    return repair(child)

def mutate(bits, rate=0.05):
    """Random mutation: flip some bits, then repair."""
    child = bits
    for i in range(N):
        if np.random.random() < rate:
            child ^= (1 << i)
    return repair(child)

def genetic_algorithm(pop_size=40, generations=80, max_local_iter=100):
    """Run genetic algorithm to find a good 4-AP-free subset."""
    np.random.seed(42)
    
    # Initialize population with diverse greedy constructions
    population = []
    fitnesses = []
    
    # Greedy with natural order
    b = greedy_construction(list(range(N)))
    population.append(b)
    fitnesses.append(fitness_bitmask(b))
    
    # Greedy with reverse order
    b = greedy_construction(list(range(N-1, -1, -1)))
    population.append(b)
    fitnesses.append(fitness_bitmask(b))
    
    # Greedy with random orders
    for _ in range(pop_size - 2):
        order = np.random.permutation(N)
        b = greedy_construction(order.tolist())
        population.append(b)
        fitnesses.append(fitness_bitmask(b))
    
    fitnesses = np.array(fitnesses)
    best_idx = np.argmax(fitnesses)
    best_bits = population[best_idx]
    best_fit = fitnesses[best_idx]
    
    for gen in range(generations):
        new_population = []
        new_fitnesses = []
        
        for _ in range(pop_size):
            # Tournament selection
            t1 = np.random.randint(0, pop_size, 3)
            p1 = population[t1[np.argmax(fitnesses[t1])]]
            t2 = np.random.randint(0, pop_size, 3)
            p2 = population[t2[np.argmax(fitnesses[t2])]]
            
            # Crossover and mutation
            child = crossover(p1, p2)
            child = mutate(child, rate=0.03)
            
            # Local search
            child = local_search(child, max_rounds=max_local_iter)
            
            new_population.append(child)
            new_fitnesses.append(fitness_bitmask(child))
        
        new_fitnesses = np.array(new_fitnesses)
        
        # Elitism: keep the best individual
        old_best = population[np.argmax(fitnesses)]
        old_best_fit = np.max(fitnesses)
        
        if np.max(new_fitnesses) >= old_best_fit:
            population = new_population
            fitnesses = new_fitnesses
        else:
            # Replace worst with old best
            worst = np.argmin(new_fitnesses)
            new_population[worst] = old_best
            new_fitnesses[worst] = old_best_fit
            population = new_population
            fitnesses = new_fitnesses
        
        # Update global best
        cur_best = np.argmax(fitnesses)
        if fitnesses[cur_best] > best_fit:
            best_fit = fitnesses[cur_best]
            best_bits = population[cur_best]
    
    return best_bits

def construct():
    """Construct a 4-AP-free subset of [1, 256] maximizing 3-APs."""
    best_bits = genetic_algorithm(pop_size=30, generations=60, max_local_iter=80)
    
    # Convert bitmask to sorted list
    result = []
    for i in range(N):
        if (best_bits >> i) & 1:
            result.append(i + 1)
    
    if not result:
        result = [1]
    
    return result
