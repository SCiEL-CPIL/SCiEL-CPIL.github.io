import time
import random
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 32
    MAX_VAL = 512
    MAX_D = 255  # max possible d: 1 + 2*255 = 511 <= 512
    
    def time_ok():
        return time.time() < deadline
    
    # Bitmask approach: bit i represents element (i+1) in the set
    # So element value v corresponds to bit (v-1)
    
    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count
    
    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m
    
    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]
    
    def delta_replace(mask, old_val, new_val):
        """Return (delta, new_mask) when replacing old_val with new_val."""
        new_mask = mask & ~(1 << (old_val - 1)) | (1 << (new_val - 1))
        
        # Find d's that could be affected
        affected_ds = set()
        # d's involving old_val
        for d in range(1, MAX_D + 1):
            if old_val + 2 * d <= MAX_VAL:
                affected_ds.add(d)
            if old_val - d >= 1 and old_val + d <= MAX_VAL:
                affected_ds.add(d)
            if old_val - 2 * d >= 1:
                affected_ds.add(d)
        
        # d's involving new_val
        for d in range(1, MAX_D + 1):
            if new_val + 2 * d <= MAX_VAL:
                affected_ds.add(d)
            if new_val - d >= 1 and new_val + d <= MAX_VAL:
                affected_ds.add(d)
            if new_val - 2 * d >= 1:
                affected_ds.add(d)
        
        delta = 0
        for d in affected_ds:
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        
        return delta, new_mask
    
    # Simulated annealing local search
    def simulated_annealing(mask, s, s_set, max_rounds=100, t_start=2.0, t_end=0.1):
        lst = mask_to_list(mask)
        current_mask = mask
        current_s = s
        current_set = set(s_set)
        
        for round_idx in range(max_rounds):
            if not time_ok():
                break
            
            temperature = t_start * (t_end / t_start) ** (round_idx / max(1, max_rounds - 1))
            
            for _ in range(len(lst)):
                if not time_ok():
                    break
                
                i = random.randint(0, N - 1)
                old_val = lst[i]
                
                # Try several random replacements
                new_val = random.randint(1, MAX_VAL)
                if new_val == old_val or new_val in current_set:
                    continue
                
                delta, new_mask = delta_replace(current_mask, old_val, new_val)
                
                # Accept if better or with probability exp(delta/T)
                if delta >= 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(delta / temperature)
                
                if accept:
                    current_set.discard(old_val)
                    current_set.add(new_val)
                    lst[i] = new_val
                    current_mask = new_mask
                    current_s += delta
        
        return current_mask, current_s, current_set, lst
    
    # Genetic algorithm
    def genetic_algorithm(pop_size=20, generations=30, mutation_rate=0.3):
        if not time_ok():
            return None, None
        
        # Initialize population
        population = []
        for _ in range(pop_size):
            lst = random.sample(range(1, MAX_VAL + 1), N)
            mask = list_to_mask(lst)
            s = score_mask(mask)
            population.append((mask, s, set(lst), lst))
        
        # Sort by fitness
        population.sort(key=lambda x: x[1], reverse=True)
        
        for gen in range(generations):
            if not time_ok():
                break
            
            new_population = []
            
            # Elitism: keep top individuals
            for i in range(min(5, len(population))):
                new_population.append(population[i])
            
            # Create offspring
            while len(new_population) < pop_size:
                if not time_ok():
                    break
                
                # Tournament selection
                def select():
                    idx = random.randint(0, len(population) - 1)
                    return population[idx]
                
                parent1 = select()
                parent2 = select()
                
                # Crossover: mix elements from both parents
                child_lst = []
                used = set()
                for _ in range(N):
                    if random.random() < 0.5:
                        # Pick from parent1
                        candidates = [x for x in parent1[3] if x not in used]
                    else:
                        # Pick from parent2
                        candidates = [x for x in parent2[3] if x not in used]
                    
                    if candidates:
                        val = random.choice(candidates)
                    else:
                        # Fill with random
                        val = random.randint(1, MAX_VAL)
                        while val in used:
                            val = random.randint(1, MAX_VAL)
                    
                    child_lst.append(val)
                    used.add(val)
                
                # Mutation
                for i in range(N):
                    if random.random() < mutation_rate:
                        new_val = random.randint(1, MAX_VAL)
                        while new_val in used:
                            new_val = random.randint(1, MAX_VAL)
                        child_lst[i] = new_val
                        used.add(new_val)
                
                child_mask = list_to_mask(child_lst)
                child_s = score_mask(child_mask)
                child_set = set(child_lst)
                new_population.append((child_mask, child_s, child_set, child_lst))
            
            population = new_population
            population.sort(key=lambda x: x[1], reverse=True)
        
        return population[0][0], population[0][1]
    
    # Algebraic structure: union of two APs with optimized parameters
    def algebraic_construct():
        best_mask = 0
        best_s = 0
        best_list = []
        
        # Try various base structures
        for base in [2, 3, 4, 5, 6, 7, 8]:
            if not time_ok():
                break
            
            for offset1 in range(0, base * 2):
                if not time_ok():
                    break
                
                for offset2 in range(0, base * 2):
                    if not time_ok():
                        break
                    
                    # Create set as union of two APs
                    S = set()
                    # First AP: offset1 + i*base
                    for i in range(16):
                        v = offset1 + 1 + i * base
                        if 1 <= v <= MAX_VAL:
                            S.add(v)
                    # Second AP: offset2 + 1 + j*base
                    for j in range(16):
                        v = offset2 + 1 + j * base
                        if 1 <= v <= MAX_VAL:
                            S.add(v)
                    
                    if len(S) < N:
                        continue
                    
                    lst = sorted(S)[:N]
                    mask = list_to_mask(lst)
                    s = score_mask(mask)
                    
                    if s > best_s:
                        best_s = s
                        best_mask = mask
                        best_list = lst[:]
        
        return best_mask, best_s, best_list
    
    # Main: try multiple strategies
    random.seed(42)
    
    best_mask = 0
    best_s = 0
    best_list = []
    
    # Strategy 1: Interval start + simulated annealing
    if time_ok():
        mask = 0
        for i in range(1, 33):
            mask |= (1 << (i - 1))
        s = score_mask(mask)
        s_set = set(range(1, 33))
        print(f"Interval: {s}", file=sys.stderr)
        
        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = mask_to_list(mask)
        
        mask, s, s_set, lst = simulated_annealing(mask, s, s_set, max_rounds=80, t_start=3.0, t_end=0.05)
        print(f"Interval+SA: {s}", file=sys.stderr)
        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = lst[:]
    
    # Strategy 2: Algebraic structures
    if time_ok():
        mask, s, lst = algebraic_construct()
        print(f"Algebraic: {s}", file=sys.stderr)
        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = lst[:]
        
        # Polish with simulated annealing
        if s > 0 and time_ok():
            s_set = set(lst)
            mask, s, s_set, lst = simulated_annealing(mask, s, s_set, max_rounds=50, t_start=2.0, t_end=0.1)
            print(f"Algebraic+SA: {s}", file=sys.stderr)
            if s > best_s:
                best_mask = mask
                best_s = s
                best_list = lst[:]
    
    # Strategy 3: Genetic algorithm
    if time_ok():
        ga_mask, ga_s = genetic_algorithm(pop_size=25, generations=40, mutation_rate=0.25)
        if ga_s is not None:
            print(f"GA: {ga_s}", file=sys.stderr)
            if ga_s > best_s:
                best_mask = ga_mask
                best_s = ga_s
                best_list = mask_to_list(ga_mask)
            
            # Polish with simulated annealing
            if time_ok():
                s_set = set(best_list)
                mask, s, s_set, lst = simulated_annealing(best_mask, best_s, s_set, max_rounds=50, t_start=2.0, t_end=0.1)
                print(f"GA+SA: {s}", file=sys.stderr)
                if s > best_s:
                    best_mask = mask
                    best_s = s
                    best_list = lst[:]
    
    # Strategy 4: Random starts + simulated annealing
    for trial in range(15):
        if not time_ok():
            break
        lst = random.sample(range(1, MAX_VAL + 1), N)
        mask = list_to_mask(lst)
        s = score_mask(mask)
        s_set = set(lst)
        
        mask, s, s_set, lst = simulated_annealing(mask, s, s_set, max_rounds=60, t_start=2.5, t_end=0.1)
        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = lst[:]
            print(f"Random trial {trial}: {s}", file=sys.stderr)
    
    # Final: more simulated annealing on the best
    if time_ok():
        s_set = set(best_list)
        mask, s, s_set, lst = simulated_annealing(best_mask, best_s, s_set, max_rounds=80, t_start=1.5, t_end=0.05)
        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = lst[:]
    
    print(f"Final score: {best_s}", file=sys.stderr)
    return best_list
