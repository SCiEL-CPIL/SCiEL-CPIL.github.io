import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    def compute_score(pairs):
        """
        pairs: list of 12 (exp, coeff) tuples.
        Returns number of non-zero terms in the square.
        """
        if len(pairs) != 12:
            return float('inf')
        
        max_deg = max(e for e, c in pairs)
        if max_deg > 40:
            return float('inf')
            
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in pairs:
            if c == 0:
                return float('inf')
            if abs(c) > 64:
                return float('inf')
            p[e] = c
        
        square = np.convolve(p, p)
        return int(np.count_nonzero(square))

    def get_dense(pairs):
        max_deg = max(e for e, c in pairs)
        dense = [0] * (max_deg + 1)
        for e, c in pairs:
            dense[e] = c
        return dense

    # Candidate 1: The "cancelled" polynomial derived in analysis
    # c0=1, c1=1, c2=-1, c3=1, c4=-3, c5=4, c6=-15, c7=22, c8=-64, c9=1, c10=1, c11=1
    initial_pairs = [
        (0, 1), (1, 1), (2, -1), (3, 1), (4, -3), (5, 4), 
        (6, -15), (7, 22), (8, -64), (9, 1), (10, 1), (11, 1)
    ]
    
    # Candidate 2: Trivial AP
    trivial_pairs = [(i, 1) for i in range(12)]
    
    best_score = float('inf')
    best_candidate = None
    
    def update_best(pairs, score):
        nonlocal best_score, best_candidate
        if score < best_score:
            best_score = score
            best_candidate = get_dense(pairs)

    def optimize(pairs, max_iters=5000):
        current_pairs = pairs[:]
        current_score = compute_score(current_pairs)
        update_best(current_pairs, current_score)
        
        temp = 1.0
        cooling = 0.999
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            new_pairs = current_pairs[:]
            move_type = random.random()
            
            if move_type < 0.7:
                # Change coefficient
                idx = random.randint(0, 11)
                old_exp, old_coeff = new_pairs[idx]
                # Try to adjust coefficient to cancel terms
                # Random step
                step = random.choice([-2, -1, 1, 2, 5, 10])
                new_coeff = old_coeff + step
                if abs(new_coeff) > 64 or new_coeff == 0:
                    continue
                new_pairs[idx] = (old_exp, new_coeff)
            else:
                # Change exponent (less likely to help with cancellation in AP structure, but allows escape)
                idx = random.randint(0, 11)
                old_exp, old_coeff = new_pairs[idx]
                delta = random.choice([-2, -1, 1, 2])
                new_exp = old_exp + delta
                if new_exp < 0 or new_exp > 40:
                    continue
                other_exps = [e for j, (e, c) in enumerate(new_pairs) if j != idx]
                if new_exp in other_exps:
                    continue
                new_pairs[idx] = (new_exp, old_coeff)
                
            new_score = compute_score(new_pairs)
            
            if new_score < current_score:
                current_pairs = new_pairs
                current_score = new_score
                update_best(current_pairs, current_score)
            else:
                delta_e = new_score - current_score
                if random.random() < np.exp(-delta_e / temp):
                    current_pairs = new_pairs
                    current_score = new_score
            
            temp *= cooling

    # Optimize the promising candidate
    optimize(initial_pairs)
    
    # Optimize the trivial candidate as fallback
    optimize(trivial_pairs)
    
    if best_candidate is None:
        best_candidate = get_dense(trivial_pairs)
        
    return best_candidate
