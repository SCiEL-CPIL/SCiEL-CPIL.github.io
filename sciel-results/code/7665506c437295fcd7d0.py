import time
import random
import numpy as np
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _compute_threshold_fast(trans0, trans1):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Optimized for speed using precomputed transition tables for all subsets.
    Returns -1 if not synchronizing.
    """
    t0 = [1 << trans0[i] for i in range(N)]
    t1 = [1 << trans1[i] for i in range(N)]
    
    next_subsets = np.zeros((1 << N, 2), dtype=np.int32)
    
    for s in range(1 << N):
        res0 = 0
        temp_s = s
        while temp_s:
            lsb = temp_s & -temp_s
            idx = (lsb.bit_length() - 1)
            res0 |= t0[idx]
            temp_s ^= lsb
        next_subsets[s, 0] = res0
        
        res1 = 0
        temp_s = s
        while temp_s:
            lsb = temp_s & -temp_s
            idx = (lsb.bit_length() - 1)
            res1 |= t1[idx]
            temp_s ^= lsb
        next_subsets[s, 1] = res1

    visited = np.full(1 << N, -1, dtype=np.int32)
    queue = deque()
    
    start = FULL_MASK
    visited[start] = 0
    queue.append(start)
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        if current != 0 and (current & (current - 1)) == 0:
            return dist
            
        for letter in range(2):
            nxt = next_subsets[current, letter]
            if visited[nxt] == -1:
                visited[nxt] = dist + 1
                queue.append(nxt)
                
    return -1

def _generate_cerny_seed():
    """
    Generates a standard Cerny automaton which has threshold N-1.
    """
    trans0 = [(i + 1) % N for i in range(N)]
    trans1 = [0] * N
    for i in range(1, N):
        trans1[i] = i - 1
    return trans0, trans1

def _generate_ping_pong(rng):
    """
    Generates a ping-pong style automaton, often having high threshold.
    """
    # A cycle with one defect
    trans0 = [(i + 1) % N for i in range(N)]
    # Pick a random defect state
    defect = rng.randint(0, N - 1)
    # trans1 is mostly reverse, but defect state goes somewhere else
    trans1 = [0] * N
    for i in range(1, N):
        trans1[i] = i - 1
    trans1[0] = N - 1 # Close the cycle for 1 as well? No, let's make 1 a reset-ish or different cycle.
    
    # Let's try a specific high-threshold structure: 
    # 0: 0->1->2->...->N-1->0
    # 1: 0->0, 1->0, ..., N-1->N-2
    # This is Cerny.
    
    # Let's try a "Pinwheel" variant:
    # 0: i -> (i+1)%N
    # 1: i -> (i-1)%N except one state which is a sink or breaks the symmetry.
    
    trans1_alt = [(i - 1) % N for i in range(N)]
    # Break symmetry
    k = rng.randint(0, N-1)
    trans1_alt[k] = rng.randint(0, N-1)
    
    # Ensure synchronization? Not guaranteed.
    return trans0, trans1_alt

def _mutate_structure(parent0, parent1, rng, mutation_strength=0.1):
    """
    Mutates an automaton using structure-aware operations:
    1. Swap two states in one transition table.
    2. Change a random transition to a neighbor in the cycle.
    3. Randomly flip a few transitions.
    """
    mut0 = parent0[:]
    mut1 = parent1[:]
    
    num_mutations = rng.randint(1, 3)
    
    for _ in range(num_mutations):
        op = rng.choice(['swap', 'neighbor', 'random'])
        letter = rng.randint(0, 1)
        
        if op == 'swap':
            i, j = rng.sample(range(N), 2)
            if letter == 0:
                mut0[i], mut0[j] = mut0[j], mut0[i]
            else:
                mut1[i], mut1[j] = mut1[j], mut1[i]
        elif op == 'neighbor':
            i = rng.randint(0, N - 1)
            if letter == 0:
                # Change dest to a neighbor of current dest
                curr_dest = mut0[i]
                new_dest = (curr_dest + rng.choice([-1, 1])) % N
                mut0[i] = new_dest
            else:
                curr_dest = mut1[i]
                new_dest = (curr_dest + rng.choice([-1, 1])) % N
                mut1[i] = new_dest
        else: # random
            i = rng.randint(0, N - 1)
            if letter == 0:
                mut0[i] = rng.randint(0, N - 1)
            else:
                mut1[i] = rng.randint(0, N - 1)
                
    return mut0, mut1

def _crossover(parent_a, parent_b, rng):
    """
    Uniform crossover: for each state and letter, pick from parent A or B.
    """
    child0 = []
    child1 = []
    for i in range(N):
        if rng.random() < 0.5:
            child0.append(parent_a[0][i])
            child1.append(parent_a[1][i])
        else:
            child0.append(parent_b[0][i])
            child1.append(parent_b[1][i])
    return child0, child1

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    rng = random.Random(12345)
    
    # Initialize population
    pop_size = 20
    population = []
    
    # Seed with Cerny
    c0, c1 = _generate_cerny_seed()
    c_score = _compute_threshold_fast(c0, c1)
    population.append((c_score, c0, c1))
    
    # Seed with some random ping-pong variants
    for _ in range(pop_size - 1):
        t0, t1 = _generate_ping_pong(rng)
        score = _compute_threshold_fast(t0, t1)
        if score > 0:
            population.append((score, t0, t1))
        else:
            # If not synchronizing, try to fix or just add Cerny variant
            t0, t1 = _generate_cerny_seed()
            score = _compute_threshold_fast(t0, t1)
            population.append((score, t0, t1))
            
    # Sort by score descending
    population.sort(key=lambda x: x[0], reverse=True)
    
    best_score = population[0][0]
    best_0 = population[0][1][:]
    best_1 = population[0][2][:]
    
    # GA Parameters
    elite_size = 5
    tournament_size = 3
    mutation_rate = 0.3
    
    generation = 0
    
    while time.time() < deadline:
        new_population = []
        
        # 1. Elitism: keep top performers
        for i in range(min(elite_size, len(population))):
            new_population.append(population[i])
            
        # 2. Generate offspring to fill the population
        while len(new_population) < pop_size:
            # Selection: Tournament
            parents = rng.sample(population, min(tournament_size, len(population)))
            parents.sort(key=lambda x: x[0], reverse=True)
            parent_a = parents[0]
            
            # Occasionally pick second best for diversity
            if len(parents) > 1 and rng.random() < 0.3:
                parent_b = parents[1]
            else:
                parent_b = rng.choice(population)
                
            # Crossover
            child0, child1 = _crossover(parent_a, parent_b, rng)
            
            # Mutation
            if rng.random() < mutation_rate:
                child0, child1 = _mutate_structure(child0, child1, rng)
            else:
                # Light mutation: single random flip
                letter = rng.randint(0, 1)
                state = rng.randint(0, N - 1)
                if letter == 0:
                    child0[state] = rng.randint(0, N - 1)
                else:
                    child1[state] = rng.randint(0, N - 1)
                    
            score = _compute_threshold_fast(child0, child1)
            new_population.append((score, child0, child1))
            
        # Update population
        population = new_population
        population.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if population[0][0] > best_score:
            best_score = population[0][0]
            best_0 = population[0][1][:]
            best_1 = population[0][2][:]
            
        generation += 1
        
        # If stuck for many generations, inject random diversity
        if generation % 20 == 0:
            # Check if top score hasn't improved in last 20 gens? 
            # Simple heuristic: if best_score is low, inject randoms
            if best_score < 20:
                for _ in range(5):
                    t0 = list(range(N))
                    rng.shuffle(t0)
                    t1 = [rng.randint(0, N - 1) for _ in range(N)]
                    score = _compute_threshold_fast(t0, t1)
                    if score > 0:
                        population.append((score, t0, t1))
                population.sort(key=lambda x: x[0], reverse=True)
                population = population[:pop_size]

    return [best_0, best_1]

# EVOLVE-BLOCK-END
