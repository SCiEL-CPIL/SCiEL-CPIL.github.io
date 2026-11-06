"""
Strategy: Genetic Algorithm with Block Crossover and Greedy Repair
This approach maintains a population of candidate sets (represented as bitsets).
1. Initialization: Generate a diverse population using random greedy construction.
2. Selection: Tournament selection to pick parents.
3. Crossover: Block-based crossover. We divide the domain [0, N) into blocks.
   For each block, we inherit the bits from the parent with the larger set size
   (or randomly if tied). This preserves local structures that might be valid.
4. Mutation: 
   a. Greedy Fill: Try to add any missing element that doesn't create a 4-AP.
   b. Random Flip: With low probability, flip a random bit to introduce diversity.
5. Repair: If the offspring is infeasible (contains a 4-AP), remove elements 
   from the conflicting 4-APs until it is feasible.
6. Elitism: Keep the best individuals from the previous generation.

This differs from the current local search by exploring a wider landscape 
through population dynamics and recombination, rather than single-step moves.
"""
import time
import random
import numpy as np

N = 512
POP_SIZE = 20
BLOCK_SIZE = 64
NUM_BLOCKS = N // BLOCK_SIZE

# Precompute forbidden triples for each x
# forbidden[x] = list of (u, v, w) such that {x, u, v, w} is a 4-AP
def precompute_forbidden():
    forbidden = [[] for _ in range(N)]
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # 4-AP: a, b, c, e (1-indexed)
            # Indices: a-1, b-1, c-1, e-1
            forbidden[a - 1].append((b - 1, c - 1, e - 1))
            forbidden[b - 1].append((a - 1, c - 1, e - 1))
            forbidden[c - 1].append((a - 1, b - 1, e - 1))
            forbidden[e - 1].append((a - 1, b - 1, c - 1))
    return forbidden

FORBIDDEN = precompute_forbidden()

def can_add(bits, x):
    """Check if x (0-indexed) can be added to the set represented by bits."""
    for u, v, w in FORBIDDEN[x]:
        if (bits >> u) & 1 and (bits >> v) & 1 and (bits >> w) & 1:
            return False
    return True

def is_feasible(bits):
    """Check if the set represented by bits contains no 4-AP."""
    for x in range(N):
        if (bits >> x) & 1:
            if not can_add(bits & ~(1 << x), x):
                return False
    return True

def repair(bits):
    """
    Repair an infeasible set by removing elements from 4-APs.
    Returns the repaired bits.
    """
    changed = True
    while changed:
        changed = False
        for x in range(N):
            if (bits >> x) & 1:
                if not can_add(bits & ~(1 << x), x):
                    # x is part of a 4-AP. Remove it.
                    bits &= ~(1 << x)
                    changed = True
                    break # Restart scan after modification
    return bits

def greedy_fill(bits):
    """
    Try to add any element not in bits that doesn't conflict.
    Returns the new bits.
    """
    candidates = [i for i in range(N) if not (bits >> i) & 1]
    random.shuffle(candidates)
    for c in candidates:
        if can_add(bits, c):
            bits |= (1 << c)
    return bits

def greedy_init():
    """Generate a random greedy set."""
    bits = 0
    order = list(range(N))
    random.shuffle(order)
    for x in order:
        if can_add(bits, x):
            bits |= (1 << x)
    return bits

def block_crossover(parent1, parent2):
    """
    Block-based crossover.
    For each block, pick the parent with more bits in that block.
    """
    child = 0
    for b in range(NUM_BLOCKS):
        start = b * BLOCK_SIZE
        end = start + BLOCK_SIZE
        mask = ((1 << BLOCK_SIZE) - 1) << start
        
        bits1_in_block = parent1 & mask
        bits2_in_block = parent2 & mask
        
        count1 = bin(bits1_in_block).count('1')
        count2 = bin(bits2_in_block).count('1')
        
        if count1 > count2:
            child |= bits1_in_block
        elif count2 > count1:
            child |= bits2_in_block
        else:
            # Random choice
            if random.random() < 0.5:
                child |= bits1_in_block
            else:
                child |= bits2_in_block
    return child

def tournament_select(pop, k=3):
    """Tournament selection."""
    best = None
    for _ in range(k):
        candidate = random.choice(pop)
        if best is None or bin(candidate).count('1') > bin(best).count('1'):
            best = candidate
    return best

def construct():
    deadline = time.time() + 55.0  # Safety margin
    
    # 1. Initialize Population
    population = []
    for _ in range(POP_SIZE):
        population.append(greedy_init())
    
    best_bits = max(population, key=lambda b: bin(b).count('1'))
    best_size = bin(best_bits).count('1')
    
    # 2. Evolutionary Loop
    while time.time() < deadline:
        new_population = []
        
        # Elitism: Keep the best 10%
        elite_size = max(1, POP_SIZE // 10)
        sorted_pop = sorted(population, key=lambda b: bin(b).count('1'), reverse=True)
        for i in range(elite_size):
            new_population.append(sorted_pop[i])
            
        # Fill the rest of the population with offspring
        while len(new_population) < POP_SIZE:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            
            child = block_crossover(p1, p2)
            
            # Mutation: Greedy fill
            child = greedy_fill(child)
            
            # Mutation: Random flip (low probability)
            if random.random() < 0.1:
                flip_idx = random.randint(0, N - 1)
                child ^= (1 << flip_idx)
                
            # Repair if infeasible
            if not is_feasible(child):
                child = repair(child)
                # After repair, try to fill again
                child = greedy_fill(child)
                
            new_population.append(child)
            
        population = new_population
        
        # Update best
        current_best = max(population, key=lambda b: bin(b).count('1'))
        current_size = bin(current_best).count('1')
        if current_size > best_size:
            best_size = current_size
            best_bits = current_best

    # Convert best_bits to list
    result = [i + 1 for i in range(N) if (best_bits >> i) & 1]
    return result

# EVOLVE-BLOCK-END
