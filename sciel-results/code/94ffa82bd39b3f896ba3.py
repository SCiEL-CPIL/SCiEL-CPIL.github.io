import time
import random
import sys
from collections import deque

N = 12
FULL_STATE = (1 << N) - 1
DEADLINE = 55.0  # Safety margin under 60s kill

def calculate_threshold(a_trans, b_trans):
    """
    Returns the shortest reset word length for the automaton defined by a_trans and b_trans,
    or -1 if not synchronizing.
    """
    # Precompute bit masks for single state transitions
    a_bits = [1 << a_trans[i] for i in range(N)]
    b_bits = [1 << b_trans[i] for i in range(N)]
    
    # Precompute transition tables for all possible subsets
    # There are 2^N = 4096 subsets.
    # For each subset, we need to compute the image under 'a' and 'b'.
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    # Optimization: Iterate through masks and use DP or bit manipulation.
    # Simple iteration is fast enough for N=12.
    for mask in range(1 << N):
        if mask == 0:
            continue
        res_a = 0
        res_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            res_a |= a_bits[idx]
            res_b |= b_bits[idx]
            m ^= lsb
        trans_a[mask] = res_a
        trans_b[mask] = res_b
        
    # BFS
    visited = [False] * (1 << N)
    visited[FULL_STATE] = True
    queue = deque()
    queue.append((FULL_STATE, 0))
    
    while queue:
        state, dist = queue.popleft()
        if state.bit_count() == 1:
            return dist
            
        next_a = trans_a[state]
        if not visited[next_a]:
            visited[next_a] = True
            queue.append((next_a, dist + 1))
            
        next_b = trans_b[state]
        if not visited[next_b]:
            visited[next_b] = True
            queue.append((next_b, dist + 1))
            
    return -1

def construct():
    start_time = time.time()
    
    def get_score(a, b):
        return calculate_threshold(a, b)

    def random_automaton():
        return [random.randint(0, N-1) for _ in range(N)]

    def pinsker_automaton():
        a = [(i + 1) % N for i in range(N)]
        # b: 0->0, i->i-1 for i>0
        b = [0] + [i-1 for i in range(1, N)]
        return a, b

    # Initialize population
    pop_size = 40
    population = []
    
    # Add some structured starts to ensure we have synchronizing automata early
    a_p, b_p = pinsker_automaton()
    population.append((a_p[:], b_p[:]))
    
    # Add random starts
    for _ in range(pop_size - 1):
        a = random_automaton()
        b = random_automaton()
        population.append((a, b))
    
    # Evaluate initial population
    scored_population = []
    for a, b in population:
        score = get_score(a, b)
        scored_population.append((score, a, b))
        
    # Sort by score descending
    scored_population.sort(key=lambda x: x[0], reverse=True)
    best_score = scored_population[0][0]
    best_a = scored_population[0][1][:]
    best_b = scored_population[0][2][:]
    
    if best_score == -1:
        # Fallback if no synchronizing automaton found (unlikely)
        return [best_a, best_b]

    # Genetic Algorithm
    # Parameters
    mutation_rate = 0.2
    crossover_rate = 0.8
    tournament_size = 3
    elitism_count = 5
    
    while time.time() - start_time < DEADLINE:
        # Selection
        new_population = []
        
        # Elitism
        for i in range(elitism_count):
            new_population.append((scored_population[i][1][:], scored_population[i][2][:]))
            
        # Fill the rest
        while len(new_population) < pop_size:
            # Tournament selection
            parents = []
            for _ in range(2):
                contenders = random.sample(scored_population, tournament_size)
                winner = max(contenders, key=lambda x: x[0])
                parents.append(winner)
                
            p1_a, p1_b = parents[0][1], parents[0][2]
            p2_a, p2_b = parents[1][1], parents[1][2]
            
            # Crossover
            if random.random() < crossover_rate:
                # 2-point crossover on the combined list? Or separate?
                # Let's do separate crossover for a and b
                c1 = random.randint(0, N-1)
                c2 = random.randint(0, N-1)
                if c1 > c2:
                    c1, c2 = c2, c1
                    
                child_a = p1_a[:]
                child_b = p1_b[:]
                
                for i in range(c1, c2 + 1):
                    child_a[i] = p2_a[i]
                    child_b[i] = p2_b[i]
            else:
                child_a = p1_a[:]
                child_b = p1_b[:]
                
            # Mutation
            if random.random() < mutation_rate:
                # Mutate 'a'
                idx = random.randint(0, N-1)
                child_a[idx] = random.randint(0, N-1)
            if random.random() < mutation_rate:
                # Mutate 'b'
                idx = random.randint(0, N-1)
                child_b[idx] = random.randint(0, N-1)
                
            new_population.append((child_a, child_b))
            
        # Evaluate new population
        scored_population = []
        for a, b in new_population:
            score = get_score(a, b)
            scored_population.append((score, a, b))
            
        # Update best
        current_best = max(scored_population, key=lambda x: x[0])
        if current_best[0] > best_score:
            best_score = current_best[0]
            best_a = current_best[1][:]
            best_b = current_best[2][:]
            
        # Sort for next generation
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
    return [best_a, best_b]
