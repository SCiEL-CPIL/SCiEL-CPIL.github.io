import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    N = 256
    
    # Precompute all 4-term arithmetic progressions (0-indexed)
    # A 4-AP is (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < N
    # d >= 1
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            ap_list.append((a, a+d, a+2*d, a+3*d))
    
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Precompute the indices for vectorized score calculation
    a_idx = aps[:, 0]
    b_idx = aps[:, 1]
    c_idx = aps[:, 2]
    d_idx = aps[:, 3]
    
    def calc_score_vectorized(coloring):
        """
        Vectorized calculation of the number of monochromatic 4-APs.
        coloring: numpy array of 0s and 1s
        """
        ca = coloring[a_idx]
        cb = coloring[b_idx]
        cc = coloring[c_idx]
        cd = coloring[d_idx]
        
        # Monochromatic if all four are equal
        # (ca == cb) & (cb == cc) & (cc == cd)
        mono = (ca == cb) & (cb == cc) & (cc == cd)
        return int(np.sum(mono))
    
    # --- Initialization ---
    # We want a diverse initial population.
    # Mix Thue-Morse (structured, low AP density) and Random.
    
    def thue_morse(n):
        seq = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            c = 0
            x = i
            while x:
                c += x & 1
                x >>= 1
            seq[i] = c % 2
        return seq

    pop_size = 50
    population = []
    
    # Add Thue-Morse
    tm = thue_morse(N)
    population.append(tm)
    
    # Add a few random ones
    for _ in range(pop_size - 1):
        population.append(np.random.randint(0, 2, size=N).astype(np.uint8))
    
    # Evaluate initial population
    scores = np.array([calc_score_vectorized(ind) for ind in population])
    
    best_score = np.min(scores)
    best_coloring = population[np.argmin(scores)].copy()
    
    # GA Parameters
    mutation_rate = 0.02  # Start with low mutation
    tournament_size = 5
    
    # Track "diff" effectiveness for the "operator diff" question
    # We will measure how often a child improves upon the *average* of its parents
    # or simply track the raw improvement rate of children vs parents.
    # The prompt asks: "Does operator 'diff' change the chance that a child improves its island"
    # Interpretation: 'diff' likely refers to the difference in score between child and parent.
    # We will monitor the ratio of children that are better than their parents.
    
    num_generations = 0
    last_improvement_gen = 0
    stagnation_counter = 0
    
    while time.time() < deadline:
        if time.time() > deadline - 5:
            break
            
        new_population = []
        num_children = 0
        num_improvements = 0
        
        # Generate next generation
        for _ in range(pop_size):
            if time.time() > deadline - 5:
                break
                
            # Tournament Selection
            # Pick 'tournament_size' random individuals
            candidates = np.random.choice(pop_size, tournament_size, replace=False)
            candidate_scores = scores[candidates]
            winner_idx = candidates[np.argmin(candidate_scores)]
            parent1 = population[winner_idx]
            
            # Pick a second parent (different from first)
            parent2_idx = np.random.choice(pop_size - 1)
            if parent2_idx >= winner_idx:
                parent2_idx += 1
            parent2 = population[parent2_idx]
            
            # Crossover: Uniform crossover
            # For each bit, pick from parent1 or parent2 with 50% probability
            mask = np.random.randint(0, 2, size=N).astype(np.uint8)
            child = np.where(mask == 1, parent1, parent2).astype(np.uint8)
            
            # Mutation
            # Flip bits with probability mutation_rate
            mut_mask = np.random.random(N) < mutation_rate
            child[mut_mask] = 1 - child[mut_mask]
            
            # Evaluate Child
            child_score = calc_score_vectorized(child)
            
            # Track "diff" metric
            # Compare child to the *better* parent (the one that would be the "island" local best)
            parent_best_score = min(calc_score_vectorized(parent1), calc_score_vectorized(parent2))
            # Actually, we already have scores for parents in the population array
            # But parent1 and parent2 might not be the global best, just tournament winners.
            # Let's compare to the average of the two parents or the better one.
            # Let's use the better parent as the baseline for "improvement".
            p1_score = calc_score_vectorized(parent1)
            p2_score = calc_score_vectorized(parent2)
            baseline = min(p1_score, p2_score)
            
            if child_score < baseline:
                num_improvements += 1
            
            num_children += 1
            new_population.append(child)
            
            # Track global best
            if child_score < best_score:
                best_score = child_score
                best_coloring = child.copy()
                last_improvement_gen = num_generations
                stagnation_counter = 0
            else:
                stagnation_counter += 1
        
        if not new_population:
            break
            
        # Update population
        # Elitism: Keep the best individual from the old population
        old_best_idx = np.argmin(scores)
        old_best_individual = population[old_best_idx]
        old_best_score = scores[old_best_idx]
        
        # Replace the worst individual in the new population with the old best if it's better
        new_scores = np.array([calc_score_vectorized(ind) for ind in new_population])
        new_worst_idx = np.argmax(new_scores)
        
        if old_best_score < new_scores[new_worst_idx]:
            new_population[new_worst_idx] = old_best_individual
            new_scores[new_worst_idx] = old_best_score
        
        population = new_population
        scores = new_scores
        num_generations += 1
        
        # Adapt Mutation Rate based on "diff" (improvement rate)
        if num_children > 0:
            improvement_rate = num_improvements / num_children
            
            # If improvement rate is high, we are exploring well, maybe reduce mutation to exploit?
            # If improvement rate is low, we are stuck, increase mutation to explore?
            # Standard GA heuristic:
            # If many children are better than parents, we might be in a "valley" and should continue.
            # If few are better, we are likely stuck in local minima, increase mutation.
            
            # However, the "diff" question implies we should check if the *difference* itself is informative.
            # Let's use a simple adaptive scheme:
            # Target improvement rate ~ 0.2-0.4
            if improvement_rate < 0.1:
                mutation_rate = min(0.1, mutation_rate * 1.2) # Increase mutation
            elif improvement_rate > 0.5:
                mutation_rate = max(0.005, mutation_rate * 0.8) # Decrease mutation
            
            # Also, if stagnation is high, force a larger perturbation (restart-like)
            if stagnation_counter > 10:
                # Inject a random individual to break symmetry
                random_ind = np.random.randint(0, 2, size=N).astype(np.uint8)
                worst_idx = np.argmax(scores)
                population[worst_idx] = random_ind
                scores[worst_idx] = calc_score_vectorized(random_ind)
                stagnation_counter = 0

    return best_coloring.tolist()
# EVOLVE-BLOCK-END
