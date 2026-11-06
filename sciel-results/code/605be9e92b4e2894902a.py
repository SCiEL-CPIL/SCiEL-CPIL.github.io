import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    best_poly = None
    best_score = float('inf')
    
    # Initial candidate: the seed
    seed_poly = [0] * 41
    for i in range(12):
        seed_poly[i] = 1
    sq = _square_nonzero(seed_poly)
    best_score = sq
    best_poly = seed_poly[:]
    
    def _square_nonzero(poly):
        """Count nonzero terms in the square of poly (list of coeffs, ascending degree)"""
        n = len(poly)
        # Square has degree up to 2*(n-1)
        sq = [0] * (2*n - 1)
        for i in range(n):
            if poly[i] == 0:
                continue
            for j in range(n):
                if poly[j] == 0:
                    continue
                sq[i+j] += poly[i] * poly[j]
        return sum(1 for c in sq if c != 0)
    
    def _random_poly():
        """Generate random 12-term polynomial with degree < 41"""
        positions = random.sample(range(41), 12)
        coeffs = [random.randint(1, 64) for _ in range(12)]
        # Random signs
        for i in range(len(coeffs)):
            if random.random() < 0.5:
                coeffs[i] = -coeffs[i]
        poly = [0] * 41
        for pos, c in zip(positions, coeffs):
            poly[pos] = c
        return poly
    
    def _local_search(poly, max_iter=200):
        """Hill-climbing: try changing one coefficient or moving one term"""
        current = poly[:]
        current_score = _square_nonzero(current)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try changing a coefficient value
            nonzero_idx = [i for i in range(41) if current[i] != 0]
            idx = random.choice(nonzero_idx)
            old_val = current[idx]
            new_val = random.randint(1, 64)
            if random.random() < 0.5:
                new_val = -new_val
            if new_val == old_val:
                continue
            current[idx] = new_val
            new_score = _square_nonzero(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[idx] = old_val
            
            # Try moving a term to a new position
            if not improved and random.random() < 0.3:
                idx2 = random.choice(nonzero_idx)
                old_pos = idx2
                old_coeff = current[old_pos]
                new_pos = random.randint(0, 40)
                if current[new_pos] != 0:
                    continue
                current[old_pos] = 0
                current[new_pos] = old_coeff
                new_score = _square_nonzero(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[new_pos] = 0
                    current[old_pos] = old_coeff
        
        return current, current_score
    
    # Random restarts with local search
    num_restarts = 0
    while time.time() < deadline:
        poly = _random_poly()
        poly, score = _local_search(poly, max_iter=100)
        if score < best_score:
            best_score = score
            best_poly = poly[:]
        num_restarts += 1
        if time.time() > deadline:
            break
    
    return best_poly
