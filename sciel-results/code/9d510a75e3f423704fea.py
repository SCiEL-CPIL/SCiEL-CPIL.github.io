"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin: 55s out of 90s limit

    def fast_count_nonzero_from_terms(terms):
        """
        Fast evaluation: compute square of sparse polynomial and count nonzeros.
        terms: list of (pos, val) tuples, exactly 12 nonzero.
        Returns count of nonzero coefficients in the square.
        """
        # Build dense array
        max_deg = max(pos for pos, _ in terms)
        dense = np.zeros(max_deg + 1, dtype=np.int64)
        for pos, val in terms:
            dense[pos] = val
        # Square via convolution
        result = np.convolve(dense, dense, mode='full')
        return np.count_nonzero(result)

    def random_initial():
        """Generate a random valid 12-term polynomial"""
        positions = random.sample(range(41), 12)
        values = [random.choice([1, -1]) * random.randint(1, 64) for _ in range(12)]
        return list(zip(positions, values))

    def neighbor_position(terms):
        """Move one term to a new random position (most impactful for cancellation)"""
        terms = list(terms)
        idx = random.randint(0, 11)
        old_pos = terms[idx][0]
        # Try a few positions, pick one that doesn't collide
        for _ in range(10):
            new_pos = random.randint(0, 40)
            if new_pos == old_pos:
                continue
            collision = False
            for i, (pos, val) in enumerate(terms):
                if i != idx and pos == new_pos:
                    collision = True
                    break
            if not collision:
                terms[idx] = (new_pos, terms[idx][1])
                return terms
        return terms

    def neighbor_value(terms):
        """Change value of one term"""
        terms = list(terms)
        idx = random.randint(0, 11)
        new_val = random.choice([1, -1]) * random.randint(1, 64)
        terms[idx] = (terms[idx][0], new_val)
        return terms

    def neighbor_sign(terms):
        """Flip sign of one term"""
        terms = list(terms)
        idx = random.randint(0, 11)
        terms[idx] = (terms[idx][0], -terms[idx][1])
        return terms

    def neighbor(terms):
        """Generate a neighboring solution. Bias heavily toward position changes."""
        r = random.random()
        if r < 0.70:
            return neighbor_position(terms)
        elif r < 0.85:
            return neighbor_value(terms)
        else:
            return neighbor_sign(terms)

    # Simulated Annealing
    best_terms = random_initial()
    best_score = fast_count_nonzero_from_terms(best_terms)
    
    current_terms = best_terms[:]
    current_score = best_score
    
    # SA parameters
    T0 = 50.0
    Tend = 1.0
    total_steps = 200000  # Will be limited by time
    
    step = 0
    # Use time-based cooling
    t_start = time.time()
    
    while True:
        elapsed = time.time() - t_start
        if elapsed > deadline:
            break
        
        # Adaptive cooling: temperature based on remaining time
        remaining_frac = max(0.0, (deadline - elapsed) / deadline)
        T = Tend + (T0 - Tend) * remaining_frac
        
        # Generate neighbor
        new_terms = neighbor(current_terms)
        new_score = fast_count_nonzero_from_terms(new_terms)
        
        delta = new_score - current_score
        
        if delta <= 0:
            # Accept improvement
            current_terms = new_terms
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_terms = new_terms[:]
                if best_score <= 12:
                    break
        else:
            # Accept worse with probability exp(-delta/T)
            if T > 0:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_terms = new_terms
                    current_score = new_score
        
        step += 1
        if step % 5000 == 0:
            # Periodic restart if stuck
            if current_score > best_score + 5 and random.random() < 0.3:
                current_terms = best_terms[:]
                current_score = best_score
    
    # Final local search polish from best
    if time.time() < deadline - 2.0:
        polish_terms = best_terms[:]
        polish_score = best_score
        for _ in range(5000):
            if time.time() > deadline - 1.0:
                break
            new_terms = neighbor(polish_terms)
            new_score = fast_count_nonzero_from_terms(new_terms)
            if new_score < polish_score:
                polish_terms = new_terms
                polish_score = new_score
                if polish_score < best_score:
                    best_score = polish_score
                    best_terms = polish_terms[:]
                    if best_score <= 12:
                        break

    # Convert best terms back to coefficient list
    if not best_terms:
        return [1]
        
    max_pos = max(pos for pos, _ in best_terms)
    coeff_list = [0] * (max_pos + 1)
    for pos, val in best_terms:
        coeff_list[pos] = val
    
    # Ensure length <= 41 and nonzero final coefficient
    if len(coeff_list) > 41:
        coeff_list = coeff_list[:41]
    if coeff_list[-1] == 0:
        while coeff_list and coeff_list[-1] == 0:
            coeff_list.pop()
        if not coeff_list:
            coeff_list = [1]
            
    return coeff_list
# EVOLVE-BLOCK-END
