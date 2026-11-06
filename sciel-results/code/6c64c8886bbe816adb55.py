import time
import random
import numpy as np

def _square_count(coeffs_list):
    """
    Calculate the number of non-zero terms in the square of a polynomial.
    """
    if not coeffs_list:
        return 0
    # Use numpy convolve for speed
    sq = np.convolve(coeffs_list, coeffs_list)
    return np.count_nonzero(sq)

def _valid_poly(poly):
    """
    Check if the polynomial meets all constraints.
    """
    if len(poly) > 41:
        return False
    if poly[-1] == 0:
        return False
    nz_count = sum(1 for x in poly if x != 0)
    if nz_count != 12:
        return False
    if any(abs(x) > 64 for x in poly):
        return False
    return True

def _local_search(poly, max_iters=100000, deadline=None):
    """
    Perform local search with simulated annealing to minimize the number of non-zero terms in the square.
    """
    current_poly = poly[:]
    current_score = _square_count(current_poly)
    best_poly = current_poly[:]
    best_score = current_score
    
    # Pre-compute indices of non-zero terms
    nz_indices = [i for i, v in enumerate(current_poly) if v != 0]
    
    temperature = 1.0
    cooling_rate = 0.9999
    
    for i in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Change coefficient of a non-zero term
            if not nz_indices:
                break
            idx = random.choice(nz_indices)
            old_val = current_poly[idx]
            # Try other small values
            new_val = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            if new_val == old_val:
                continue
            if abs(new_val) > 64:
                continue
            current_poly[idx] = new_val
            new_score = _square_count(current_poly)
            
            delta = new_score - current_score
            if delta < 0 or (temperature > 0 and random.random() < np.exp(-delta / temperature)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_poly = current_poly[:]
            else:
                current_poly[idx] = old_val
                
        elif move_type < 0.6:
            # Shift exponent
            if not nz_indices:
                break
            idx = random.choice(nz_indices)
            val = current_poly[idx]
            shift = random.choice([-2, -1, 1, 2])
            new_idx = idx + shift
            if new_idx < 0 or new_idx >= len(current_poly):
                continue
            if current_poly[new_idx] != 0:
                continue
            
            current_poly[idx] = 0
            current_poly[new_idx] = val
            new_score = _square_count(current_poly)
            
            delta = new_score - current_score
            if delta < 0 or (temperature > 0 and random.random() < np.exp(-delta / temperature)):
                current_score = new_score
                nz_indices.remove(idx)
                nz_indices.append(new_idx)
                if new_score < best_score:
                    best_score = new_score
                    best_poly = current_poly[:]
            else:
                current_poly[new_idx] = 0
                current_poly[idx] = val
                
        elif move_type < 0.8:
            # Swap exponents of two non-zero terms
            if len(nz_indices) < 2:
                break
            i1, i2 = random.sample(nz_indices, 2)
            v1, v2 = current_poly[i1], current_poly[i2]
            # Swap positions
            current_poly[i1] = v2
            current_poly[i2] = v1
            new_score = _square_count(current_poly)
            
            delta = new_score - current_score
            if delta < 0 or (temperature > 0 and random.random() < np.exp(-delta / temperature)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_poly = current_poly[:]
            else:
                current_poly[i1] = v1
                current_poly[i2] = v2
                
        else:
            # Swap coefficients of two non-zero terms
            if len(nz_indices) < 2:
                break
            i1, i2 = random.sample(nz_indices, 2)
            v1, v2 = current_poly[i1], current_poly[i2]
            current_poly[i1], current_poly[i2] = v2, v1
            new_score = _square_count(current_poly)
            
            delta = new_score - current_score
            if delta < 0 or (temperature > 0 and random.random() < np.exp(-delta / temperature)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_poly = current_poly[:]
            else:
                current_poly[i1], current_poly[i2] = v1, v2
                
        temperature *= cooling_rate

    return best_poly, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Generate initial candidates
    candidates = []
    
    # 1. AP with alternating signs
    for step in [1, 2, 3]:
        for start in range(0, 10):
            signs = [1 if i % 2 == 0 else -1 for i in range(12)]
            max_exp = start + step * 11
            if max_exp < 41:
                poly = [0] * (max_exp + 1)
                for i in range(12):
                    poly[start + step * i] = signs[i]
                if _valid_poly(poly):
                    candidates.append(poly)
                    
    # 2. Random sparse polynomials
    for _ in range(100):
        exps = sorted(random.sample(range(0, 41), 12))
        signs = [1 if random.random() > 0.5 else -1 for _ in range(12)]
        coeffs = [random.choice([1, 2, 3, 4]) for _ in range(12)]
        poly = [0] * 41
        for i in range(12):
            poly[exps[i]] = signs[i] * coeffs[i]
        if _valid_poly(poly):
            candidates.append(poly)
            
    # 3. Symmetric structures
    for _ in range(50):
        # Try to create symmetric exponents
        base_exps = random.sample(range(0, 20), 6)
        exps = sorted(base_exps + [39 - e for e in base_exps])
        if len(set(exps)) < 12:
            continue
        signs = [1 if random.random() > 0.5 else -1 for _ in range(12)]
        coeffs = [1 for _ in range(12)]
        poly = [0] * 41
        for i in range(12):
            poly[exps[i]] = signs[i] * coeffs[i]
        if _valid_poly(poly):
            candidates.append(poly)

    if not candidates:
        return [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]

    best_poly = None
    best_score = 1000
    
    # Evaluate and refine each candidate
    for cand in candidates:
        if time.time() > deadline - 5.0:
            break
            
        score = _square_count(cand)
        if score < best_score:
            best_score = score
            best_poly = cand[:]
            
        # Local search
        if time.time() < deadline - 10.0:
            cand_deadline = time.time() + 2.0
            refined_poly, refined_score = _local_search(cand, max_iters=50000, deadline=cand_deadline)
            
            if refined_score < best_score:
                best_score = refined_score
                best_poly = refined_poly[:]
                
        # If we found a very good score, we can stop early
        if best_score <= 13:
            break

    # Final polish on the best solution
    if best_poly is not None and time.time() < deadline - 1.0:
        final_deadline = deadline - 0.5
        final_poly, final_score = _local_search(best_poly, max_iters=200000, deadline=final_deadline)
        if final_score < best_score:
            best_poly = final_poly
            best_score = final_score

    if best_poly is None:
        return [1] * 12
        
    return best_poly
