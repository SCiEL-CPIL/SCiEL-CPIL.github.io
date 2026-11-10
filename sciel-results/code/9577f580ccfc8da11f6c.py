import random

def count_nonzero(lst):
    return sum(1 for x in lst if x != 0)

def poly_square(coeffs):
    """Compute exact square of polynomial given as coeff list (ascending degree)."""
    n = len(coeffs)
    if n == 0:
        return []
    result = [0] * (2*n - 1)
    nz = [(i, c) for i, c in enumerate(coeffs) if c != 0]
    for k, (i, ci) in enumerate(nz):
        for l in range(k, len(nz)):
            j, cj = nz[l]
            val = ci * cj
            if i == j:
                result[i+j] += val
            else:
                result[i+j] += 2 * val
    return result

def valid_candidate(coeffs):
    if len(coeffs) == 0 or len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        return False
    if count_nonzero(coeffs) != 12:
        return False
    for c in coeffs:
        if c < -64 or c > 64:
            return False
    return True

def construct():
    best_score = float('inf')
    best_cand = None
    
    # Starting candidates
    starts = []
    
    # 1. Structured candidate with known cancellations
    starts.append([1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1])
    
    # 2. All 1s
    starts.append([1]*12)
    
    # 3. Alternating
    starts.append([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    
    # 4. Random starts
    random.seed(42)
    for _ in range(5):
        exps = sorted(random.sample(range(41), 12))
        p = [0]*41
        for e in exps:
            p[e] = random.choice([1, -1, 2, -2])
        while p and p[-1] == 0:
            p.pop()
        starts.append(p)

    for start in starts:
        current = start[:]
        # Ensure valid
        while current and current[-1] == 0:
            current.pop()
        if not valid_candidate(current):
            continue
            
        current_score = count_nonzero(poly_square(current))
        
        # Local Search
        for step in range(500):
            improved = False
            for _ in range(5):
                neighbor = current[:]
                # Ensure neighbor is long enough for moves
                if len(neighbor) < 41:
                    neighbor.extend([0] * (41 - len(neighbor)))
                
                move_type = random.choice(['val', 'exp'])
                
                if move_type == 'val':
                    nz_idx = [i for i, c in enumerate(neighbor) if c != 0]
                    if not nz_idx: continue
                    idx = random.choice(nz_idx)
                    neighbor[idx] = random.randint(-64, 64)
                    if neighbor[idx] == 0:
                        neighbor[idx] = 1
                else:
                    nz_idx = [i for i, c in enumerate(neighbor) if c != 0]
                    if not nz_idx: continue
                    idx = random.choice(nz_idx)
                    val = neighbor[idx]
                    neighbor[idx] = 0
                    
                    # Pick new index
                    new_idx = random.randint(0, 40)
                    if neighbor[new_idx] != 0:
                        # Collision, skip
                        continue
                    
                    neighbor[new_idx] = val
                    
                # Trim
                while neighbor and neighbor[-1] == 0:
                    neighbor.pop()
                
                if not valid_candidate(neighbor):
                    continue
                    
                nb_score = count_nonzero(poly_square(neighbor))
                
                if nb_score < current_score:
                    current_score = nb_score
                    current = neighbor
                    improved = True
                    break
                elif nb_score == current_score and random.random() < 0.2:
                    current = neighbor
                    improved = True
                    break
                    
            if not improved:
                break
                
        if current_score < best_score:
            best_score = current_score
            best_cand = current[:]
            
    return best_cand
