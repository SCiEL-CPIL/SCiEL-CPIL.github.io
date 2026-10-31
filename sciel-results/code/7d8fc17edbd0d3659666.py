import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    def compute_score(U):
        if not U or 0 not in U or len(U) < 2:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        n_sum = len(sumset)
        n_diff = len(diffset)
        if n_sum == 0:
            return -1
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        return 1 + math.log(ratio) / math.log(base)
    
    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand or len(cand) < 2:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Known good Golomb rulers and structured sets
    known_sets = [
        [0, 1, 4, 10, 18, 23, 29, 34, 42, 47],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29],
        [0, 1, 4, 10, 18, 23],
        [0, 1, 4, 10, 18],
        [0, 1, 4, 10],
        [0, 1, 4],
        [0, 1, 3],
        [0, 1, 4, 9, 15, 22, 31],
        [0, 1, 4, 9, 15, 22],
        [0, 1, 4, 9, 15],
        [0, 1, 4, 9],
        [0, 2, 7, 15, 22, 31, 39],
        [0, 2, 7, 15, 22, 31],
        [0, 2, 7, 15, 22],
        [0, 2, 7, 15],
        [0, 1, 5, 12, 22, 30, 41],
        [0, 1, 5, 12, 22, 30],
        [0, 1, 5, 12, 22],
        [0, 1, 5, 12],
    ]
    
    for s in known_sets:
        try_set(s)
    
    # Base-B digit representation genetic algorithm
    def to_digits(val, base, num_digits):
        """Represent val in base, padded to num_digits digits"""
        digits = []
        v = val
        for _ in range(num_digits):
            digits.append(v % base)
            v //= base
        return digits
    
    def from_digits(digits, base):
        val = 0
        for d in reversed(digits):
            val = val * base + d
        return val
    
    def mutate_digits(digits, base, mutation_rate=0.3):
        """Mutate individual digits"""
        new_digits = list(digits)
        for i in range(len(new_digits)):
            if random.random() < mutation_rate:
                new_digits[i] = random.randint(0, base - 1)
        return new_digits
    
    def crossover(d1, d2):
        """Uniform crossover of digit vectors"""
        n = len(d1)
        child = []
        for i in range(n):
            child.append(d1[i] if random.random() < 0.5 else d2[i])
        return child
    
    def local_search(cand, max_iters=200):
        """Local search: try adding, removing, replacing elements"""
        curr = sorted(set(cand))
        if not curr or 0 not in curr or len(curr) < 2:
            return curr
        curr_score = compute_score(curr)
        no_improve = 0
        max_no_improve = 15
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            n_curr = len(curr)
            max_cur = max(curr) if curr else 0
            
            # Try adding
            if n_curr < 12:
                for new_el in range(1, max_cur + 15):
                    if new_el in curr:
                        continue
                    cand2 = sorted(curr + [new_el])
                    s = compute_score(cand2)
                    if s > curr_score:
                        curr = cand2
                        curr_score = s
                        improved = True
                        no_improve = 0
                        try_set(curr)
                        break
            
            # Try replacing
            if not improved and n_curr >= 2:
                for i in range(n_curr):
                    if curr[i] == 0:
                        continue
                    for new_el in range(1, max_cur + 15):
                        if new_el in curr:
                            continue
                        cand2 = list(curr)
                        cand2[i] = new_el
                        cand2 = sorted(set(cand2))
                        if 0 not in cand2 or len(cand2) < 2:
                            continue
                        s = compute_score(cand2)
                        if s > curr_score:
                            curr = cand2
                            curr_score = s
                            improved = True
                            no_improve = 0
                            try_set(curr)
                            break
                    if improved:
                        break
            
            # Try removing
            if not improved and n_curr > 2:
                for i in range(n_curr):
                    if curr[i] == 0:
                        continue
                    cand2 = [x for j, x in enumerate(curr) if j != i]
                    s = compute_score(cand2)
                    if s > curr_score:
                        curr = cand2
                        curr_score = s
                        improved = True
                        no_improve = 0
                        try_set(curr)
                        break
            
            if improved:
                continue
            no_improve += 1
            if no_improve >= max_no_improve:
                break
        
        return curr
    
    # Run local search from known sets
    for s in known_sets:
        if time.time() > deadline - 10:
            break
        result = local_search(s)
        try_set(result)
    
    # Genetic algorithm with base-B digit representation
    base = 16  # hex digits
    num_digits = 3  # 3 hex digits = up to 4095
    
    # Initialize population from known good sets
    population = []
    for s in known_sets[:8]:
        if time.time() > deadline - 10:
            break
        # Convert to digit representation
        max_val = max(s)
        if max_val < base ** num_digits:
            # Pad set to fixed length with zeros, then convert
            pass
        # Store as list of (value, digits) pairs
        pop_entry = list(s)
        population.append(pop_entry)
    
    # Add some random starts
    for _ in range(4):
        if time.time() > deadline - 10:
            break
        n = random.randint(3, 8)
        max_m = random.randint(10, 50)
        inner = random.sample(range(1, max_m + 1), min(n - 1, max_m))
        population.append(sorted([0] + inner))
    
    # Genetic algorithm loop
    gen = 0
    max_gens = 50
    while gen < max_gens and time.time() < deadline - 15:
        # Evaluate population
        scored = []
        for ind in population:
            ind = sorted(set(ind))
            if not ind or 0 not in ind or len(ind) < 2:
                scored.append((0, ind))
            else:
                s = compute_score(ind)
                scored.append((s, ind))
        
        scored.sort(key=lambda x: -x[0])
        
        # Elitism
        new_pop = [scored[0][1]]
        if len(scored) > 1:
            new_pop.append(scored[1][1])
        
        # Crossover and mutation
        while len(new_pop) < len(population):
            if time.time() > deadline - 15:
                break
            # Select parents
            p1 = scored[random.randint(0, min(3, len(scored)-1))][1]
            p2 = scored[random.randint(0, min(3, len(scored)-1))][1]
            
            # Mutate a copy
            child = list(p1)
            if random.random() < 0.5:
                # Add element
                if len(child) < 12:
                    max_c = max(child) if child else 0
                    new_el = random.randint(1, max_c + 10)
                    if new_el not in child:
                        child.append(new_el)
                else:
                    # Replace
                    idx = random.randint(1, len(child)-1)
                    max_c = max(child) if child else 0
                    child[idx] = random.randint(1, max_c + 10)
            elif random.random() < 0.5:
                # Remove element
                if len(child) > 2:
                    idx = random.randint(1, len(child)-1)
                    child.pop(idx)
            else:
                # Replace element
                if len(child) >= 2:
                    idx = random.randint(1, len(child)-1)
                    max_c = max(child) if child else 0
                    child[idx] = random.randint(1, max_c + 10)
            
            child = sorted(set(child))
            if child and 0 in child and len(child) >= 2:
                new_pop.append(child)
            else:
                new_pop.append(p1[:])
        
        population = new_pop
        gen += 1
        
        # Occasionally run local search on best
        if gen % 5 == 0 and time.time() < deadline - 15:
            best_ind = scored[0][1]
            improved = local_search(best_ind, max_iters=50)
            try_set(improved)
    
    # Final local search on best
    if time.time() < deadline - 5:
        improved = local_search(best_set, max_iters=300)
        try_set(improved)
    
    return list(best_set)
