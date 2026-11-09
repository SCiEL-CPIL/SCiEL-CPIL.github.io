"""Hypergraph Ramsey R(4,5;3) construction via genetic algorithm."""
import numpy as np
from itertools import combinations
import time

def _precompute(n):
    """Precompute all triples and their indices, plus 4-subsets and 5-subsets."""
    triples = list(combinations(range(n), 3))
    ntriples = len(triples)
    # Map triple -> index
    tri_idx = {}
    for i, (a, b, c) in enumerate(triples):
        tri_idx[(a, b, c)] = i
    
    # For each 4-subset, store the 4 triple indices
    red_k4s = []
    for quad in combinations(range(n), 4):
        idxs = [tri_idx[(quad[i], quad[j], quad[k])] 
                for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)]
        red_k4s.append(idxs)
    
    # For each 5-subset, store the 10 triple indices
    blue_k5s = []
    for quint in combinations(range(n), 5):
        idxs = [tri_idx[(quint[i], quint[j], quint[k])] 
                for i in range(5) for j in range(i+1, 5) for k in range(j+1, 5)]
        blue_k5s.append(idxs)
    
    red_k4s = np.array(red_k4s, dtype=np.int32)
    blue_k5s = np.array(blue_k5s, dtype=np.int32)
    return triples, tri_idx, red_k4s, blue_k5s

def _count_violations(colors, red_k4s, blue_k5s):
    """Count red K4 violations and blue K5 violations."""
    colors = np.asarray(colors, dtype=np.int8)
    # Red K4: all 4 triples are 0
    red_viols = 0
    if len(red_k4s) > 0:
        # For each 4-subset, check if all 4 triples are red (0)
        # Vectorized: sum of (1 - colors[idx]) over the 4 triples should be 4
        red_sums = (1 - colors[red_k4s]).sum(axis=1)
        red_viols = np.count_nonzero(red_sums == 4)
    
    # Blue K5: all 10 triples are 1
    blue_viols = 0
    if len(blue_k5s) > 0:
        blue_sums = colors[blue_k5s].sum(axis=1)
        blue_viols = np.count_nonzero(blue_sums == 10)
    
    return red_viols, blue_viols

def _fitness(colors, red_k4s, blue_k5s):
    """Fitness = negative total violations (lower is better)."""
    r, b = _count_violations(colors, red_k4s, blue_k5s)
    return r + b

def _crossover_preserve_valid4(p1, p2, red_k4s, n):
    """Crossover that preserves valid 4-subsets from parents.
    
    For each 4-subset that is valid (not a red K4) in at least one parent,
    prefer the parent's coloring on those triples.
    """
    p1 = np.asarray(p1, dtype=np.int8)
    p2 = np.asarray(p2, dtype=np.int8)
    child = np.empty_like(p1)
    
    # Determine which 4-subsets are valid in each parent
    # A 4-subset is "invalid red" if all 4 triples are red
    if len(red_k4s) > 0:
        p1_red_sums = (1 - p1[red_k4s]).sum(axis=1)
        p2_red_sums = (1 - p2[red_k4s]).sum(axis=1)
        p1_valid = (p1_red_sums < 4)  # not a red K4
        p2_valid = (p2_red_sums < 4)
    else:
        p1_valid = np.array([])
        p2_valid = np.array([])
    
    # Start with random choice from parents
    mask = np.random.random(len(p1)) < 0.5
    child = np.where(mask, p1, p2)
    
    # For triples that are part of a valid 4-subset in p1 but invalid in p2, prefer p1
    # For triples that are part of a valid 4-subset in p2 but invalid in p1, prefer p2
    if len(red_k4s) > 0:
        # Build a preference score for each triple
        pref = np.zeros(len(p1), dtype=np.int8)
        # For each 4-subset, if valid in p1 and invalid in p2, add +1 to its triples
        # If valid in p2 and invalid in p1, add -1
        for i in range(len(red_k4s)):
            if p1_valid[i] and not p2_valid[i]:
                pref[red_k4s[i]] += 1
            elif p2_valid[i] and not p1_valid[i]:
                pref[red_k4s[i]] -= 1
        
        # Where preference is positive, take from p1; negative from p2
        take_p1 = pref > 0
        take_p2 = pref < 0
        child[take_p1] = p1[take_p1]
        child[take_p2] = p2[take_p2]
    
    return child

def _mutate_break_blue5(colors, blue_k5s, n, rate=0.05):
    """Mutation biased toward breaking blue K5 violations."""
    colors = np.asarray(colors, dtype=np.int8).copy()
    ntriples = len(colors)
    
    # Find blue K5 violations
    if len(blue_k5s) > 0:
        blue_sums = colors[blue_k5s].sum(axis=1)
        violating_k5s = np.where(blue_sums == 10)[0]
        
        if len(violating_k5s) > 0:
            # Pick a random violating K5 and flip one of its triples
            idx = violating_k5s[np.random.randint(len(violating_k5s))]
            triple_idxs = blue_k5s[idx]
            flip_idx = triple_idxs[np.random.randint(len(triple_idxs))]
            colors[flip_idx] = 1 - colors[flip_idx]
    
    # Random mutation
    if rate > 0:
        n_flips = max(1, int(ntriples * rate))
        flip_positions = np.random.choice(ntriples, size=n_flips, replace=False)
        colors[flip_positions] = 1 - colors[flip_positions]
    
    return colors

def construct():
    deadline = time.time() + 120  # 120 second safety margin
    
    # Start with n=20 and try to grow
    best_result = None
    best_n = 0
    
    # Try to find valid colorings for increasing n
    for n in range(15, 35):
        if time.time() > deadline - 30:
            break
        
        triples, tri_idx, red_k4s, blue_k5s = _precompute(n)
        ntriples = len(triples)
        
        # Initialize population
        pop_size = 30
        population = []
        
        # Seed with random colorings biased toward fewer violations
        for _ in range(pop_size):
            # Start with random coloring
            colors = np.random.randint(0, 2, size=ntriples, dtype=np.int8)
            population.append(colors)
        
        # Also add a structured initial coloring (sum mod 3)
        structured = np.array([0 if (a + b + c) % 3 == 0 else 1 
                              for a, b, c in triples], dtype=np.int8)
        population.append(structured)
        
        # Local search on each initial
        for i in range(len(population)):
            colors = population[i].copy()
            for _ in range(50):
                if time.time() > deadline - 30:
                    break
                # Find violations
                r, b = _count_violations(colors, red_k4s, blue_k5s)
                if r == 0 and b == 0:
                    break
                # Pick a violating structure and flip a triple
                if r > 0:
                    red_sums = (1 - colors[red_k4s]).sum(axis=1)
                    viol_idx = np.where(red_sums == 4)[0]
                    if len(viol_idx) > 0:
                        k4_idx = viol_idx[np.random.randint(len(viol_idx))]
                        triple_idxs = red_k4s[k4_idx]
                        flip_idx = triple_idxs[np.random.randint(len(triple_idxs))]
                        colors[flip_idx] = 1
                if b > 0:
                    blue_sums = colors[blue_k5s].sum(axis=1)
                    viol_idx = np.where(blue_sums == 10)[0]
                    if len(viol_idx) > 0:
                        k5_idx = viol_idx[np.random.randint(len(viol_idx))]
                        triple_idxs = blue_k5s[k5_idx]
                        flip_idx = triple_idxs[np.random.randint(len(triple_idxs))]
                        colors[flip_idx] = 0
                population[i] = colors
        
        # Check if any are valid
        valid_found = False
        for colors in population:
            r, b = _count_violations(colors, red_k4s, blue_k5s)
            if r == 0 and b == 0:
                valid_found = True
                if n > best_n:
                    best_n = n
                    best_result = {'n': n, 'colors': [int(c) for c in colors]}
                break
        
        if valid_found:
            continue  # Try larger n
        
        # Genetic algorithm for this n
        for gen in range(200):
            if time.time() > deadline - 30:
                break
            
            # Evaluate fitness
            fitnesses = []
            for colors in population:
                f = _fitness(colors, red_k4s, blue_k5s)
                fitnesses.append(f)
            
            # Check for valid solution
            for i, f in enumerate(fitnesses):
                if f == 0:
                    if n > best_n:
                        best_n = n
                        best_result = {'n': n, 'colors': [int(c) for c in population[i]]}
                    valid_found = True
                    break
            
            if valid_found:
                break
            
            # Selection: keep top half
            order = np.argsort(fitnesses)
            elite = [population[i].copy() for i in order[:pop_size//2]]
            
            # Create next generation
            new_pop = list(elite)
            while len(new_pop) < pop_size:
                # Crossover
                p1 = new_pop[np.random.randint(len(new_pop))]
                p2 = new_pop[np.random.randint(len(new_pop))]
                child = _crossover_preserve_valid4(p1, p2, red_k4s, n)
                
                # Mutation
                child = _mutate_break_blue5(child, blue_k5s, n, rate=0.03)
                
                # Local search
                for _ in range(20):
                    if time.time() > deadline - 30:
                        break
                    r, b = _count_violations(child, red_k4s, blue_k5s)
                    if r == 0 and b == 0:
                        break
                    if r > 0:
                        red_sums = (1 - child[red_k4s]).sum(axis=1)
                        viol_idx = np.where(red_sums == 4)[0]
                        if len(viol_idx) > 0:
                            k4_idx = viol_idx[np.random.randint(len(viol_idx))]
                            triple_idxs = red_k4s[k4_idx]
                            flip_idx = triple_idxs[np.random.randint(len(triple_idxs))]
                            child[flip_idx] = 1
                    if b > 0:
                        blue_sums = child[blue_k5s].sum(axis=1)
                        viol_idx = np.where(blue_sums == 10)[0]
                        if len(viol_idx) > 0:
                            k5_idx = viol_idx[np.random.randint(len(viol_idx))]
                            triple_idxs = blue_k5s[k5_idx]
                            flip_idx = triple_idxs[np.random.randint(len(triple_idxs))]
                            child[flip_idx] = 0
                
                new_pop.append(child)
            
            population = new_pop
        
        # After GA, check once more
        for colors in population:
            r, b = _count_violations(colors, red_k4s, blue_k5s)
            if r == 0 and b == 0:
                if n > best_n:
                    best_n = n
                    best_result = {'n': n, 'colors': [int(c) for c in colors]}
                break
    
    # If we found a valid coloring, return it
    if best_result is not None:
        return best_result
    
    # Fallback: return the best n we could get close on
    # Try one more time with the largest n we attempted
    n = min(34, max(15, best_n + 1))
    if n < 15:
        n = 15
    
    triples, tri_idx, red_k4s, blue_k5s = _precompute(n)
    ntriples = len(triples)
    
    # Quick local search from random start
    colors = np.random.randint(0, 2, size=ntriples, dtype=np.int8)
    for _ in range(1000):
        if time.time() > deadline - 5:
            break
        r, b = _count_violations(colors, red_k4s, blue_k5s)
        if r == 0 and b == 0:
            break
        if r > 0:
            red_sums = (1 - colors[red_k4s]).sum(axis=1)
            viol_idx = np.where(red_sums == 4)[0]
            if len(viol_idx) > 0:
                k4_idx = viol_idx[np.random.randint(len(viol_idx))]
                triple_idxs = red_k4s[k4_idx]
                flip_idx = triple_idxs[np.random.randint(len(triple_idxs))]
                colors[flip_idx] = 1
        if b > 0:
            blue_sums = colors[blue_k5s].sum(axis=1)
            viol_idx = np.where(blue_sums == 10)[0]
            if len(viol_idx) > 0:
                k5_idx = viol_idx[np.random.randint(len(viol_idx))]
                triple_idxs = blue_k5s[k5_idx]
                flip_idx = triple_idxs[np.random.randint(len(triple_idxs))]
                colors[flip_idx] = 0
    
    r, b = _count_violations(colors, red_k4s, blue_k5s)
    if r == 0 and b == 0:
        return {'n': n, 'colors': [int(c) for c in colors]}
    
    # Return a valid small coloring as fallback
    n = 9
    triples = list(combinations(range(n), 3))
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]}
