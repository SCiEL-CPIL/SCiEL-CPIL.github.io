import numpy as np
import itertools
import time
import sys

def construct():
    """
    Finds a 2-coloring of triples of n vertices with no red K4^(3) and no blue K5^(3).
    Maximizes n.
    """
    start_time = time.time()
    time_limit = 120  # seconds, safety margin for 150s limit
    
    # Try n from 34 downwards
    for n in range(34, 10, -1):
        if time.time() - start_time > time_limit - 10:
            break
            
        result = solve_for_n(n, time_limit - (time.time() - start_time))
        if result is not None:
            return result
            
    # Fallback: return a small valid coloring
    # The seed program returns n=9 with a specific pattern.
    # Let's ensure we return something valid even if GA fails.
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]
    return {'n': n, 'colors': colors}

def solve_for_n(n, time_budget):
    """
    Attempts to find a valid coloring for a specific n using GA + Local Search.
    Returns dict if successful, None otherwise.
    """
    if n < 4:
        return {'n': n, 'colors': [0] * (n * (n-1) * (n-2) // 6)}
        
    # 1. Precompute indices
    triples = list(itertools.combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute 4-subsets and their triple indices
    k4s = list(itertools.combinations(range(n), 4))
    num_k4 = len(k4s)
    k4_to_triples = []
    for k4 in k4s:
        idxs = [triple_to_idx[t] for t in itertools.combinations(k4, 3)]
        k4_to_triples.append(idxs)
    k4_to_triples = np.array(k4_to_triples, dtype=np.int32)
    
    # Precompute 5-subsets and their triple indices
    k5s = list(itertools.combinations(range(n), 5))
    num_k5 = len(k5s)
    k5_to_triples = []
    for k5 in k5s:
        idxs = [triple_to_idx[t] for t in itertools.combinations(k5, 3)]
        k5_to_triples.append(idxs)
    k5_to_triples = np.array(k5_to_triples, dtype=np.int32)
    
    # Precompute reverse mapping: triple index -> list of k4 indices and k5 indices
    triple_to_k4s = [[] for _ in range(num_triples)]
    for i, idxs in enumerate(k4_to_triples):
        for idx in idxs:
            triple_to_k4s[idx].append(i)
            
    triple_to_k5s = [[] for _ in range(num_triples)]
    for i, idxs in enumerate(k5_to_triples):
        for idx in idxs:
            triple_to_k5s[idx].append(i)
            
    # Convert to numpy arrays for faster access? 
    # Lists of lists are fine for now, but we need to iterate them.
    
    def calculate_fitness(colors):
        """
        colors: np.array of 0s and 1s
        Returns (red_violations, blue_violations, red_counts, blue_counts)
        """
        # Red K4: all 4 triples are 0
        # colors[k4_to_triples] has shape (num_k4, 4)
        red_counts = (colors[k4_to_triples] == 0).sum(axis=1)
        red_violations = np.sum(red_counts == 4)
        
        # Blue K5: all 10 triples are 1
        blue_counts = (colors[k5_to_triples] == 1).sum(axis=1)
        blue_violations = np.sum(blue_counts == 10)
        
        return red_violations, blue_violations, red_counts, blue_counts

    def local_search(colors, time_limit_inner):
        """
        Hill climbing local search.
        """
        red_viol, blue_viol, red_counts, blue_counts = calculate_fitness(colors)
        total_viol = red_viol + blue_viol
        
        # If already valid, return
        if total_viol == 0:
            return colors, 0
            
        # Local search loop
        # We try to flip bits to reduce violations
        # To speed up, we pick a random violating set and try to fix it
        # Or pick a random bit and check delta
        
        # Efficient approach: 
        # Pick a random bit that is part of a violation?
        # Or just iterate bits and calculate delta.
        
        # Let's use a greedy approach:
        # Find a violating K4 or K5. Pick a bit from it. Flip it if it improves.
        
        improved = True
        while improved and (time.time() < time_limit_inner):
            improved = False
            total_viol = red_viol + blue_viol
            
            # Pick a random violating set
            # If red_viol > 0, pick a random red violating K4
            # If blue_viol > 0, pick a random blue violating K5
            # Weight by count
            
            use_red = False
            if red_viol > 0 and (blue_viol == 0 or np.random.rand() < red_viol / (red_viol + blue_viol)):
                use_red = True
            
            if use_red:
                # Find indices of violating K4s
                viol_idx = np.where(red_counts == 4)[0]
                if len(viol_idx) == 0: continue
                k4_idx = viol_idx[np.random.randint(len(viol_idx))]
                triple_idxs = k4_to_triples[k4_idx]
            else:
                viol_idx = np.where(blue_counts == 10)[0]
                if len(viol_idx) == 0: continue
                k5_idx = viol_idx[np.random.randint(len(viol_idx))]
                triple_idxs = k5_to_triples[k5_idx]
                
            # Try flipping each triple in this set
            best_delta = 0
            best_flip_idx = -1
            
            for t_idx in triple_idxs:
                # Calculate delta if we flip t_idx
                # Current color: c
                # New color: 1-c
                
                # Effect on Red K4s:
                # t_idx is in some K4s.
                # If current color is 0 (Red), flipping to 1 (Blue) decreases red_count for those K4s.
                # If current color is 1 (Blue), flipping to 0 (Red) increases red_count for those K4s.
                
                c = colors[t_idx]
                
                delta_red_viol = 0
                delta_blue_viol = 0
                
                # Update Red Counts
                for k4_i in triple_to_k4s[t_idx]:
                    if c == 0:
                        # Was red, now blue. red_count decreases by 1.
                        # If red_count was 4, it becomes 3. Violation removed (-1).
                        # If red_count was 3, it becomes 2. No change in violation count.
                        # If red_count was < 3, no change.
                        if red_counts[k4_i] == 4:
                            delta_red_viol -= 1
                        # Note: It cannot create a new red violation by decreasing count.
                    else:
                        # Was blue, now red. red_count increases by 1.
                        # If red_count was 3, it becomes 4. Violation created (+1).
                        if red_counts[k4_i] == 3:
                            delta_red_viol += 1
                        
                # Update Blue Counts
                for k5_i in triple_to_k5s[t_idx]:
                    if c == 1:
                        # Was blue, now red. blue_count decreases by 1.
                        # If blue_count was 10, it becomes 9. Violation removed (-1).
                        if blue_counts[k5_i] == 10:
                            delta_blue_viol -= 1
                    else:
                        # Was red, now blue. blue_count increases by 1.
                        # If blue_count was 9, it becomes 10. Violation created (+1).
                        if blue_counts[k5_i] == 9:
                            delta_blue_viol += 1
                            
                total_delta = delta_red_viol + delta_blue_viol
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_flip_idx = t_idx
                    
            if best_flip_idx != -1:
                # Perform flip
                colors[best_flip_idx] = 1 - colors[best_flip_idx]
                
                # Update counts
                c_old = 1 - colors[best_flip_idx] # The color before flip
                c_new = colors[best_flip_idx]
                
                for k4_i in triple_to_k4s[best_flip_idx]:
                    if c_old == 0:
                        red_counts[k4_i] -= 1
                    else:
                        red_counts[k4_i] += 1
                        
                for k5_i in triple_to_k5s[best_flip_idx]:
                    if c_old == 1:
                        blue_counts[k5_i] -= 1
                    else:
                        blue_counts[k5_i] += 1
                        
                red_viol = np.sum(red_counts == 4)
                blue_viol = np.sum(blue_counts == 10)
                improved = True
                
        return colors, red_viol + blue_viol

    # GA Initialization
    pop_size = 50
    population = []
    fitnesses = []
    
    # Diverse initialization
    for i in range(pop_size):
        if i < 10:
            # Structured init: modular patterns
            # Try mod 3, mod 4, etc.
            mod = i + 3
            colors = np.array([0 if (a+b+c) % mod == 0 else 1 for a,b,c in triples], dtype=np.int8)
        else:
            colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            
        # Local search to improve
        colors, fit = local_search(colors, time.time() + 1.0) # 1s per individual initially
        population.append(colors)
        fitnesses.append(fit)
        
        if fit == 0:
            return {'n': n, 'colors': colors.tolist()}
            
    best_fit = min(fitnesses)
    if best_fit == 0:
        idx = fitnesses.index(0)
        return {'n': n, 'colors': population[idx].tolist()}
        
    # GA Loop
    gen = 0
    while time.time() < start_time + time_budget - 5:
        # Selection
        # Tournament selection
        parent1 = population[np.random.randint(pop_size)]
        parent2 = population[np.random.randint(pop_size)]
        # Ensure they are good? Just random for now, local search will fix.
        
        # Crossover
        # Uniform crossover
        mask = np.random.randint(0, 2, size=num_triples)
        child = np.where(mask == 1, parent1, parent2)
        
        # Mutation
        # Random bit flips
        # Bias mutation towards bits involved in violations?
        # For simplicity, random mutation with rate 0.01
        mut_mask = np.random.rand(num_triples) < 0.01
        child[mut_mask] = 1 - child[mut_mask]
        
        # Local Search
        child, fit = local_search(child, time.time() + 2.0) # 2s per offspring
        
        # Replacement
        # Replace the worst individual
        worst_idx = np.argmax(fitnesses)
        if fit < fitnesses[worst_idx]:
            population[worst_idx] = child
            fitnesses[worst_idx] = fit
            
            if fit == 0:
                return {'n': n, 'colors': child.tolist()}
                
        gen += 1
        
    # Return best found if not 0
    best_idx = np.argmin(fitnesses)
    if fitnesses[best_idx] == 0:
         return {'n': n, 'colors': population[best_idx].tolist()}
         
    return None

# Helper to run construct if called directly
if __name__ == "__main__":
    res = construct()
    print(res['n'])
