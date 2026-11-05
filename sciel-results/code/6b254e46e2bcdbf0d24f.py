def construct():
    import random
    import time

    def count_nonzero_square(coeffs):
        # coeffs is list in ascending degree
        # compute square
        n = len(coeffs)
        square = [0] * (2*n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(n):
                if coeffs[j] == 0:
                    continue
                square[i+j] += coeffs[i] * coeffs[j]
        return sum(1 for c in square if c != 0)

    def normalize(coeffs):
        # Trim trailing zeros and ensure length <= 41
        # The problem states "length<=41", so if our internal representation is longer, we trim.
        # But we must ensure the *polynomial* has exactly 12 nonzero terms.
        # The list returned should be the coefficient list.
        # If we have trailing zeros, we should remove them to satisfy "nonzero final coefficient" if the evaluator checks the list length/end.
        # However, the contract says "Return coefficient list... length<=41... nonzero final coefficient".
        # This implies the list itself must end with a non-zero.
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        if not coeffs:
            return [0] # Fallback, though we ensure 12 non-zeros
        return coeffs

    def valid_candidate(coeffs):
        if len(coeffs) > 41:
            return False
        if not coeffs or coeffs[-1] == 0:
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        if any(abs(c) > 64 for c in coeffs):
            return False
        return True

    # Initialize population
    population = []
    
    # Seed with some structured candidates
    # 1. All ones
    c1 = [1]*12
    population.append(c1)
    
    # 2. Alternating
    c2 = [1 if i%2==0 else -1 for i in range(12)]
    population.append(c2)
    
    # 3. Random structured
    for _ in range(20):
        indices = random.sample(range(41), 12)
        indices.sort()
        c = [0]*41
        for idx in indices:
            c[idx] = random.choice([-1, 1, -2, 2, -3, 3])
        c = normalize(c)
        if valid_candidate(c):
            population.append(c)
            
    # Evaluate initial population
    scored_pop = []
    for c in population:
        score = count_nonzero_square(c)
        scored_pop.append((score, c))
        
    scored_pop.sort(key=lambda x: x[0])
    best_score = scored_pop[0][0]
    best_candidate = scored_pop[0][1][:]
    
    start_time = time.time()
    timeout = 55.0
    
    pop_size = 50
    # Fill population with randoms if needed
    while len(scored_pop) < pop_size:
        indices = random.sample(range(41), 12)
        indices.sort()
        c = [0]*41
        for idx in indices:
            c[idx] = random.choice([-1, 1, -2, 2, -3, 3, -4, 4])
        c = normalize(c)
        if valid_candidate(c):
            score = count_nonzero_square(c)
            scored_pop.append((score, c))
            
    scored_pop.sort(key=lambda x: x[0])
    
    while time.time() - start_time < timeout:
        # Selection: Keep top half
        scored_pop = scored_pop[:pop_size//2]
        
        # Generate new population via Crossover and Mutation
        new_pop = []
        
        # Add best candidate to new pop (elitism)
        new_pop.append((scored_pop[0][0], scored_pop[0][1][:]))
        
        for _ in range(pop_size - 1):
            # Pick two parents
            p1 = random.choice(scored_pop)
            p2 = random.choice(scored_pop)
            c1 = p1[1][:]
            c2 = p2[1][:]
            
            # Crossover: Uniform crossover on indices
            # Ensure length is 41 for manipulation
            c1_pad = c1 + [0]*(41-len(c1))
            c2_pad = c2 + [0]*(41-len(c2))
            
            child = [0]*41
            for i in range(41):
                if random.random() < 0.5:
                    child[i] = c1_pad[i]
                else:
                    child[i] = c2_pad[i]
            
            # Check if child has exactly 12 non-zeros
            # If not, we need to fix it or discard.
            # Let's try to fix by adding/removing terms.
            nz_indices = [i for i in range(41) if child[i] != 0]
            if len(nz_indices) < 12:
                # Add random terms
                while len(nz_indices) < 12:
                    idx = random.randint(0, 40)
                    if child[idx] == 0:
                        child[idx] = random.choice([-1, 1, -2, 2])
                        nz_indices.append(idx)
            elif len(nz_indices) > 12:
                # Remove random terms
                while len(nz_indices) > 12:
                    idx = random.choice(nz_indices)
                    child[idx] = 0
                    nz_indices.remove(idx)
            
            # Mutation
            # 1. Sign flip
            if random.random() < 0.5:
                idx = random.choice(nz_indices)
                child[idx] *= -1
            
            # 2. Magnitude change
            if random.random() < 0.3:
                idx = random.choice(nz_indices)
                child[idx] = random.randint(1, 10)
                if random.random() < 0.5:
                    child[idx] *= -1
            
            # 3. Move term
            if random.random() < 0.2:
                idx = random.choice(nz_indices)
                val = child[idx]
                new_idx = random.randint(0, 40)
                if child[new_idx] == 0:
                    child[idx] = 0
                    child[new_idx] = val
                    nz_indices.remove(idx)
                    nz_indices.append(new_idx)
            
            # 4. Swap two terms
            if random.random() < 0.2 and len(nz_indices) >= 2:
                i1, i2 = random.sample(nz_indices, 2)
                child[i1], child[i2] = child[i2], child[i1]
            
            # Normalize and Validate
            child = normalize(child)
            if valid_candidate(child):
                score = count_nonzero_square(child)
                new_pop.append((score, child))
            
        # Combine and Select
        scored_pop = new_pop
        if not scored_pop:
            break
        scored_pop.sort(key=lambda x: x[0])
        
        current_best = scored_pop[0]
        if current_best[0] < best_score:
            best_score = current_best[0]
            best_candidate = current_best[1][:]

    return best_candidate
