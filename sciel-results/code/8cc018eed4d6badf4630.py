"""
Strategy:
1. Algebraic Construction: Use a finite field-based coloring scheme to maximize colors.
   The constraint "no rainbow 4-cycle" is equivalent to saying that for any 4 distinct vertices,
   the 6 edges cannot all have distinct colors in a way that forms a rainbow cycle.
   Specifically, a 4-cycle (a,b,c,d) has edges (a,b), (b,c), (c,d), (d,a).
   We want to avoid c_ab != c_bc != c_cd != c_da and all distinct.
   
   A known extremal construction for avoiding rainbow 4-cycles (or similar) often uses additive groups.
   Let vertices be elements of Z_16 (or a product group).
   Color of edge (i,j) can be related to (j-i) mod 16 or similar.
   However, simple difference coloring c(i,j) = (j-i) mod k leads to rainbow cycles easily.
   
   Let's try a different structural approach:
   The problem is equivalent to finding a large subset of edges in K_16 such that no 4-cycle uses 4 distinct colors from that set? No, we assign colors from 0..119.
   
   Let's look at the "inspiration 1" which scored 3.0. It used a parametric approach.
   The current program scores 15.0 using hill climbing.
   
   A different strategy: **Greedy Construction with Backtracking / Constraint Propagation**.
   Instead of starting from a valid state and mutating, start from an empty state (or 1-color state) and add colors one by one, or assign edges in a specific order, ensuring constraints are met.
   
   Alternatively, use a **Differential Evolution** or **Genetic Algorithm** approach which explores the space differently than simple hill climbing.
   
   Or, a **Structured Heuristic**:
   Partition vertices into groups.
   Let's try a "Block" construction.
   Divide 16 vertices into 4 groups of 4: G0, G1, G2, G3.
   Edges within a group: assign distinct colors? No, if a group has 4 vertices, any 4-cycle within it uses 4 edges. If we use 4 distinct colors, it's a rainbow. So within a K4, we must reuse colors. Max distinct colors in a K4 without rainbow 4-cycle?
   In K4, there are 3 distinct 4-cycles.
   If we color K4 with 3 colors, can we avoid rainbow?
   Edges: 12.
   If we use 3 colors, say 0,1,2.
   It is possible.
   
   Let's try a **Randomized Restart with Intelligent Seeding** strategy.
   The current hill climbing might get stuck.
   We will use a **Simulated Annealing** but with a more sophisticated move set:
   - Swap colors between two edges.
   - Change color of one edge.
   - Change colors of two edges simultaneously.
   
   Also, we will use a **Diversity-Driven** objective.
   The current program maximizes distinct colors.
   
   Let's try a **Genetic Algorithm** approach.
   Population of 50 individuals.
   Crossover: Uniform crossover on the color array.
   Mutation: Random color change.
   Selection: Tournament selection based on (validity, distinct_count).
   Validity is a hard constraint. We can penalize invalid individuals heavily.
   
   This is structurally different from single-trajectory hill climbing.
"""

import time
import numpy as np
import random

N = 16
NEDGES = N * (N - 1) // 2
MAX_COLOR = 120

# Precompute edge indices and cycle structures
def _get_edge_indices():
    idx_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx
            idx += 1
    
    cycles = []
    from itertools import combinations
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        e1 = (idx_map[(a,b)], idx_map[(b,c)], idx_map[(c,d)], idx_map[(d,a)])
        cycles.append(e1)
        e2 = (idx_map[(a,b)], idx_map[(b,d)], idx_map[(c,d)], idx_map[(a,c)])
        cycles.append(e2)
        e3 = (idx_map[(a,c)], idx_map[(b,c)], idx_map[(b,d)], idx_map[(a,d)])
        cycles.append(e3)
    return np.array(cycles, dtype=np.int32)

CYCLES = _get_edge_indices()

def _is_valid_vectorized(colors_np):
    cycle_colors = colors_np[CYCLES]
    sorted_colors = np.sort(cycle_colors, axis=1)
    adjacent_equal = (sorted_colors[:, 0] == sorted_colors[:, 1]) | \
                     (sorted_colors[:, 1] == sorted_colors[:, 2]) | \
                     (sorted_colors[:, 2] == sorted_colors[:, 3])
    return np.all(adjacent_equal)

def _count_distinct(colors_np):
    return len(np.unique(colors_np))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initialize population
    # Start with a mix of valid and random solutions
    population = []
    
    # 1. Valid base: 2-coloring
    base = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
            if i >= 8 and j >= 8:
                base[idx] = 1
            else:
                base[idx] = 0
    population.append(base)
    
    # 2. Random valid-ish solutions (might be invalid, but GA handles it)
    # We can try to generate solutions that are close to valid.
    # Or just random.
    for _ in range(49):
        ind = np.random.randint(0, MAX_COLOR, size=NEDGES).astype(np.int32)
        population.append(ind)
    
    population = [ind for ind in population]
    
    best_colors = None
    best_distinct = 0
    
    # Evaluate initial population
    for ind in population:
        if _is_valid_vectorized(ind):
            d = _count_distinct(ind)
            if d > best_distinct:
                best_distinct = d
                best_colors = ind.copy()
    
    # GA Loop
    pop_size = 50
    mutation_rate = 0.1
    crossover_rate = 0.7
    tournament_size = 5
    
    while time.time() < deadline:
        # Selection
        # We need to evaluate fitness.
        # Fitness = (1 if valid else 0) * 1000 + distinct_count
        # But we want to prioritize validity first.
        
        # To save time, we can cache validity for the population?
        # Or just re-evaluate. 50 * 5460 checks is fast.
        
        # Let's do tournament selection.
        # First, compute scores for all individuals.
        scores = []
        for ind in population:
            if _is_valid_vectorized(ind):
                d = _count_distinct(ind)
                score = 1000 + d
            else:
                # Penalty for invalidity?
                # How many rainbow cycles?
                # cycle_colors = ind[CYCLES]
                # sorted_colors = np.sort(cycle_colors, axis=1)
                # adjacent_equal = ...
                # num_rainbows = np.sum(~adjacent_equal)
                # score = -num_rainbows
                score = 0 # Simplified: invalids score 0, valids score > 1000
            scores.append(score)
        
        # Select parents
        new_population = []
        
        # Elitism: keep the best individual
        best_idx = np.argmax(scores)
        new_population.append(population[best_idx].copy())
        
        # Check if best is valid and update global best
        if scores[best_idx] > 1000:
            d = scores[best_idx] - 1000
            if d > best_distinct:
                best_distinct = d
                best_colors = population[best_idx].copy()
        
        while len(new_population) < pop_size:
            # Tournament selection
            parents = []
            for _ in range(2):
                contenders = random.sample(range(pop_size), tournament_size)
                winner = max(contenders, key=lambda i: scores[i])
                parents.append(population[winner])
            
            p1, p2 = parents
            
            # Crossover
            if random.random() < crossover_rate:
                mask = np.random.random(NEDGES) < 0.5
                child = np.where(mask, p1, p2).astype(np.int32)
            else:
                child = p1.copy()
            
            # Mutation
            # Change some edges to random colors
            if random.random() < mutation_rate:
                num_mutations = random.randint(1, 5)
                for _ in range(num_mutations):
                    e_idx = random.randint(0, NEDGES - 1)
                    child[e_idx] = random.randint(0, MAX_COLOR - 1)
            
            new_population.append(child)
        
        population = new_population
        
        # If we have a valid best, we might want to inject some diversity
        # by mutating the best solution and adding it to the next population?
        # Elitism already keeps the best.
        
        # Additional: If best_distinct is stuck, try a "jump"
        # We can monitor stagnation.
        
    # Return the best valid coloring found
    if best_colors is not None:
        return best_colors.tolist()
    else:
        # Fallback: return a valid 1-coloring
        return [0] * NEDGES

# EVOLVE-BLOCK-END
