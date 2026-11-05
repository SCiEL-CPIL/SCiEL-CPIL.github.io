def construct():
    import random
    import time
    import math

    def count_nonzero_square(coeffs):
        n = len(coeffs)
        if n == 0:
            return 0
        square = [0] * (2*n - 1)
        nz_indices = [i for i in range(n) if coeffs[i] != 0]
        for i in nz_indices:
            ci = coeffs[i]
            for j in nz_indices:
                square[i+j] += ci * coeffs[j]
        return sum(1 for c in square if c != 0)

    def normalize(coeffs):
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        if not coeffs:
            return [0]
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

    def get_nz_indices(coeffs):
        return [i for i in range(len(coeffs)) if coeffs[i] != 0]

    def generate_initial_candidate():
        candidates = []
        
        # 1. Arithmetic progression
        c1 = [0]*41
        for i in range(12):
            c1[i] = 1
        candidates.append(normalize(c1))
        
        # 2. Random
        c2 = [0]*41
        indices = random.sample(range(41), 12)
        for idx in indices:
            c2[idx] = random.choice([-1, 1])
        candidates.append(normalize(c2))
        
        # 3. Clustered
        c3 = [0]*41
        base = random.randint(0, 20)
        indices = [base + random.randint(0, 5) for _ in range(12)]
        indices = list(set(indices))
        while len(indices) < 12:
            indices.append(random.randint(0, 40))
        indices = list(set(indices))[:12]
        for idx in indices:
            c3[idx] = random.choice([-1, 1])
        candidates.append(normalize(c3))
        
        best_c = None
        best_s = float('inf')
        for c in candidates:
            if valid_candidate(c):
                s = count_nonzero_square(c)
                if s < best_s:
                    best_s = s
                    best_c = c
        return best_c, best_s

    best_candidate, best_score = generate_initial_candidate()
    
    if not best_candidate:
        best_candidate = [1]*12
        best_score = count_nonzero_square(best_candidate)

    start_time = time.time()
    timeout = 55.0
    
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    current_candidate = best_candidate[:]
    current_score = best_score
    
    while time.time() - start_time < timeout and T > min_T:
        neighbor = current_candidate[:]
        nz_indices = get_nz_indices(neighbor)
        
        if not nz_indices:
            break
            
        move_type = random.randint(0, 3)
        
        if move_type == 0:
            idx = random.choice(nz_indices)
            neighbor[idx] *= -1
        elif move_type == 1:
            idx = random.choice(nz_indices)
            new_val = random.randint(1, 20)
            if random.random() < 0.5:
                new_val *= -1
            if neighbor[idx] != new_val:
                neighbor[idx] = new_val
        elif move_type == 2:
            idx = random.choice(nz_indices)
            val = neighbor[idx]
            new_idx = random.randint(0, 40)
            if new_idx != idx:
                if new_idx >= len(neighbor):
                    neighbor.extend([0] * (new_idx - len(neighbor) + 1))
                if neighbor[new_idx] == 0:
                    neighbor[idx] = 0
                    neighbor[new_idx] = val
                else:
                    neighbor[idx], neighbor[new_idx] = neighbor[new_idx], val
        else:
            if len(nz_indices) >= 2:
                i1, i2 = random.sample(nz_indices, 2)
                neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
            else:
                continue
        
        neighbor = normalize(neighbor)
        if not valid_candidate(neighbor):
            continue
            
        new_score = count_nonzero_square(neighbor)
        
        if new_score < current_score:
            current_candidate = neighbor
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_candidate = neighbor[:]
        else:
            delta = new_score - current_score
            if delta == 0:
                if random.random() < 0.3:
                    current_candidate = neighbor
            else:
                if T > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_candidate = neighbor
                        current_score = new_score
        
        T *= cooling_rate
        
    return best_candidate
