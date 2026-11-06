import random
import time
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def calculate_threshold(trans_0, trans_1):
    """
    Calculates the exact shortest reset word length using BFS on the subset automaton
    with bitmask representation for speed.
    Returns -1 if not synchronizing.
    """
    start = FULL_MASK
    if start & (start - 1) == 0:  # single bit
        return 0
    
    visited = {start: 0}
    queue = deque([start])
    
    while queue:
        subset = queue.popleft()
        current_len = visited[subset]
        
        # Try adding '0'
        next_subset_0 = 0
        s = subset
        while s:
            lsb = s & (-s)
            idx = lsb.bit_length() - 1
            next_subset_0 |= (1 << trans_0[idx])
            s ^= lsb
        if next_subset_0 not in visited:
            if next_subset_0 & (next_subset_0 - 1) == 0:  # single bit
                return current_len + 1
            visited[next_subset_0] = current_len + 1
            queue.append(next_subset_0)
            
        # Try adding '1'
        next_subset_1 = 0
        s = subset
        while s:
            lsb = s & (-s)
            idx = lsb.bit_length() - 1
            next_subset_1 |= (1 << trans_1[idx])
            s ^= lsb
        if next_subset_1 not in visited:
            if next_subset_1 & (next_subset_1 - 1) == 0:  # single bit
                return current_len + 1
            visited[next_subset_1] = current_len + 1
            queue.append(next_subset_1)
            
    return -1

def _random_pinwheel():
    """Generate a random pinwheel-like automaton."""
    trans_0 = list(range(N))
    random.shuffle(trans_0)
    # Make trans_0 a permutation (cycle or near-cycle)
    # Actually, let's make it a random cycle
    cycle = list(range(N))
    random.shuffle(cycle)
    trans_0 = [0] * N
    for i in range(N):
        trans_0[cycle[i]] = cycle[(i + 1) % N]
    
    # trans_1: mostly maps to a single state, with some variation
    target = random.randint(0, N - 1)
    trans_1 = [target] * N
    # Add some randomness
    for _ in range(random.randint(1, 4)):
        idx = random.randint(0, N - 1)
        trans_1[idx] = random.randint(0, N - 1)
    
    return trans_0, trans_1

def _mutate(trans_0, trans_1, rate):
    """Mutate both transition functions."""
    new_t0 = trans_0[:]
    new_t1 = trans_1[:]
    for i in range(N):
        if random.random() < rate:
            new_t0[i] = random.randint(0, N - 1)
        if random.random() < rate:
            new_t1[i] = random.randint(0, N - 1)
    return new_t0, new_t1

def construct():
    start_time = time.time()
    max_time = 75.0  # Safety margin before 90s kill
    
    # Start with pinwheel
    best_t0, best_t1 = _random_pinwheel()
    best_score = calculate_threshold(best_t0, best_t1)
    
    # Also try the classic pinwheel
    classic_t0 = [(i + 1) % N for i in range(N)]
    classic_t1 = [0] + [(i - 1) % N for i in range(1, N)]
    classic_score = calculate_threshold(classic_t0, classic_t1)
    if classic_score > best_score:
        best_t0, best_t1, best_score = classic_t0, classic_t1, classic_score
    
    solution_pool = [(best_t0[:], best_t1[:], best_score)]
    
    # Simulated annealing
    temperature = 50.0
    min_temperature = 1.0
    cooling = 0.995
    mutation_rate = 0.15
    
    while time.time() - start_time < max_time:
        # Choose parent
        if random.random() < 0.3 and len(solution_pool) > 1:
            parent_t0, parent_t1, _ = random.choice(solution_pool)
        else:
            parent_t0, parent_t1 = best_t0, best_t1
        
        # Mutate
        new_t0, new_t1 = _mutate(parent_t0, parent_t1, mutation_rate)
        
        # Evaluate
        new_score = calculate_threshold(new_t0, new_t1)
        
        # Simulated annealing acceptance
        if new_score > best_score:
            best_t0, best_t1, best_score = new_t0, new_t1, new_score
            solution_pool.append((new_t0[:], new_t1[:], new_score))
            if len(solution_pool) > 100:
                solution_pool.sort(key=lambda x: x[2], reverse=True)
                solution_pool = solution_pool[:100]
        elif new_score > 0 and temperature > min_temperature:
            delta = new_score - best_score
            if random.random() < (2.718281828 ** (delta / temperature)):
                best_t0, best_t1, best_score = new_t0, new_t1, new_score
                solution_pool.append((new_t0[:], new_t1[:], new_score))
                if len(solution_pool) > 100:
                    solution_pool.sort(key=lambda x: x[2], reverse=True)
                    solution_pool = solution_pool[:100]
        
        # Cool down
        temperature = max(temperature * cooling, min_temperature)
        
        # If we hit theoretical max, stop
        if best_score >= (N - 1) ** 2:
            break
    
    return [best_t0, best_t1]
