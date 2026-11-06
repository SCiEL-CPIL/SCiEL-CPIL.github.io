import time
import random
from collections import deque

N = 12

# Precompute subset transition tables for speed
# For each letter (0 or 1) and each subset (bitmask), compute the image subset
# We'll build these on-the-fly per automaton but cache within a single evaluation

def compute_threshold(trans_a, trans_b):
    """
    Compute the shortest reset word length using BFS on the subset automaton.
    trans_a, trans_b: lists of length N, destination states in 0..N-1.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    start = full_mask
    
    # Precompute transition tables for both letters for all subsets
    # This is expensive but allows very fast BFS
    # Actually, for N=12, 2^12 = 4096 subsets. Precomputing all transitions
    # for each automaton takes 4096 * 12 * 2 operations which is ~100k ops.
    # That's too slow per evaluation. Instead, we compute on the fly but use
    # a faster method.
    
    # Alternative: precompute for each state i, the bit 1<<trans_a[i] and 1<<trans_b[i]
    bit_a = [1 << trans_a[i] for i in range(N)]
    bit_b = [1 << trans_b[i] for i in range(N)]
    
    visited = {start: 0}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        if current.bit_count() == 1:
            return dist
            
        # Letter 0
        next_mask = 0
        for i in range(N):
            if (current >> i) & 1:
                next_mask |= bit_a[i]
        if next_mask not in visited:
            visited[next_mask] = dist + 1
            queue.append(next_mask)
            
        # Letter 1
        next_mask = 0
        for i in range(N):
            if (current >> i) & 1:
                next_mask |= bit_b[i]
        if next_mask not in visited:
            visited[next_mask] = dist + 1
            queue.append(next_mask)
            
    return -1


def _cerny_automaton():
    """Classical Černý automaton for N states."""
    a = [0] * N
    b = [0] * N
    for i in range(N):
        if i == 0:
            a[i] = 0
            b[i] = 1
        else:
            a[i] = i - 1
            b[i] = (i + 1) % N
    return a, b


def _pinwheel_automaton():
    """Pinwheel automaton variant."""
    a = [0] * N
    b = [0] * N
    for i in range(N):
        a[i] = (i + 1) % N
        if i == 0:
            b[i] = 0
        else:
            b[i] = i - 1
    return a, b


def _generate_structured_candidates():
    """Generate a pool of structured high-threshold candidates."""
    candidates = []
    
    # Černý automaton
    a, b = _cerny_automaton()
    candidates.append([a[:], b[:]])
    
    # Pinwheel
    a, b = _pinwheel_automaton()
    candidates.append([a[:], b[:]])
    
    # Reversed Černý (swap letters)
    a, b = _cerny_automaton()
    candidates.append([b[:], a[:]])
    
    # Various cycle + collapse structures
    for start in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [start] * N
        candidates.append([a[:], b[:]])
    
    # Partial cycle structures
    for split in range(2, N):
        a = []
        b = []
        for i in range(N):
            if i < split:
                a.append((i + 1) % split)
                b.append(i)
            else:
                a.append((i + 1) % N)
                b.append((i - split) % (N - split))
        candidates.append([a[:], b[:]])
    
    # Randomized structural variants
    for _ in range(30):
        # Base on Černý with random perturbations
        a, b = _cerny_automaton()
        for _ in range(random.randint(1, 4)):
            letter = random.randint(0, 1)
            state = random.randint(0, N - 1)
            if letter == 0:
                a[state] = random.randint(0, N - 1)
            else:
                b[state] = random.randint(0, N - 1)
        candidates.append([a[:], b[:]])
    
    return candidates


def _crossover(parent1, parent2):
    """
    Uniform crossover: for each (letter, state) position, pick from parent1 or parent2.
    """
    child_a = [0] * N
    child_b = [0] * N
    for i in range(N):
        if random.random() < 0.5:
            child_a[i] = parent1[0][i]
            child_b[i] = parent1[1][i]
        else:
            child_a[i] = parent2[0][i]
            child_b[i] = parent2[1][i]
    return child_a, child_b


def _crossover_block(parent1, parent2):
    """
    Block crossover: pick a contiguous range of states and swap entire blocks.
    """
    child_a = parent1[0][:]
    child_b = parent1[1][:]
    lo = random.randint(0, N - 2)
    hi = random.randint(lo + 1, N - 1)
    for i in range(lo, hi + 1):
        child_a[i] = parent2[0][i]
        child_b[i] = parent2[1][i]
    return child_a, child_b


def _mutate(a, b, mutation_rate=0.15):
    """
    Mutate each transition with probability mutation_rate.
    """
    new_a = a[:]
    new_b = b[:]
    for i in range(N):
        if random.random() < mutation_rate:
            new_a[i] = random.randint(0, N - 1)
        if random.random() < mutation_rate:
            new_b[i] = random.randint(0, N - 1)
    return new_a, new_b


def _mutate_single(a, b):
    """Mutate exactly one transition."""
    new_a = a[:]
    new_b = b[:]
    letter = random.randint(0, 1)
    state = random.randint(0, N - 1)
    new_dest = random.randint(0, N - 1)
    if letter == 0:
        new_a[state] = new_dest
    else:
        new_b[state] = new_dest
    return new_a, new_b


def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best_trans = None
    best_threshold = 0
    
    # Phase 1: Evaluate structured candidates
    candidates = _generate_structured_candidates()
    for cand in candidates:
        if time.time() > deadline - 15:
            break
        thresh = compute_threshold(cand[0], cand[1])
        if thresh > best_threshold:
            best_threshold = thresh
            best_trans = [cand[0][:], cand[1][:]]
    
    # Initialize population with best structured candidates
    population = []
    pop_size = 20
    
    # Add the best candidate
    if best_trans:
        population.append((best_trans[0][:], best_trans[1][:], best_threshold))
    
    # Add other high-scoring structured candidates
    scored_candidates = []
    for cand in candidates:
        if time.time() > deadline - 15:
            break
        thresh = compute_threshold(cand[0], cand[1])
        if thresh > 0:
            scored_candidates.append((cand[0][:], cand[1][:], thresh))
    
    scored_candidates.sort(key=lambda x: -x[2])
    for a, b, t in scored_candidates:
        if len(population) >= pop_size:
            break
        if not any(p[0] == a and p[1] == b for p in population):
            population.append((a, b, t))
    
    # Fill remaining population with random automata
    while len(population) < pop_size:
        a = [random.randint(0, N - 1) for _ in range(N)]
        b = [random.randint(0, N - 1) for _ in range(N)]
        thresh = compute_threshold(a, b)
        population.append((a, b, thresh))
    
    # Update best from population
    for a, b, t in population:
        if t > best_threshold:
            best_threshold = t
            best_trans = [a[:], b[:]]
    
    # Phase 2: Genetic Algorithm
    generation = 0
    stall_count = 0
    max_stall = 30
    
    while time.time() < deadline - 5:
        generation += 1
        
        # Selection: tournament selection
        def select_tournament(k=3):
            contenders = random.sample(population, min(k, len(population)))
            return max(contenders, key=lambda x: x[2])
        
        # Create next generation
        new_population = []
        
        # Elitism: keep top individuals
        population_sorted = sorted(population, key=lambda x: -x[2])
        elite_count = max(2, pop_size // 5)
        for i in range(elite_count):
            new_population.append(population_sorted[i])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
            
            p1 = select_tournament()
            p2 = select_tournament()
            
            # Crossover
            if random.random() < 0.5:
                child_a, child_b = _crossover(p1, p2)
            else:
                child_a, child_b = _crossover_block(p1, p2)
            
            # Mutation
            if random.random() < 0.7:
                child_a, child_b = _mutate(child_a, child_b, mutation_rate=0.1)
            else:
                child_a, child_b = _mutate_single(child_a, child_b)
            
            # Evaluate
            thresh = compute_threshold(child_a, child_b)
            new_population.append((child_a, child_b, thresh))
            
            # Update best
            if thresh > best_threshold:
                best_threshold = thresh
                best_trans = [child_a[:], child_b[:]]
                stall_count = 0
            else:
                stall_count += 1
        
        population = new_population
        
        # If stalled, inject diversity
        if stall_count > max_stall:
            for _ in range(pop_size // 3):
                a = [random.randint(0, N - 1) for _ in range(N)]
                b = [random.randint(0, N - 1) for _ in range(N)]
                thresh = compute_threshold(a, b)
                population.append((a, b, thresh))
                if thresh > best_threshold:
                    best_threshold = thresh
                    best_trans = [a[:], b[:]]
            population = population[:pop_size]
            stall_count = 0
        
        # Also try local refinement on the current best
        if time.time() < deadline - 10 and best_trans:
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                a, b = best_trans[0][:], best_trans[1][:]
                a, b = _mutate_single(a, b)
                thresh = compute_threshold(a, b)
                if thresh > best_threshold:
                    best_threshold = thresh
                    best_trans = [a[:], b[:]]
                    population.append((a, b, thresh))
                    population = population[:pop_size]
    
    # Fallback
    if best_trans is None:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]
        
    return best_trans
