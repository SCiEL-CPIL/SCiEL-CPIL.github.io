import numpy as np
import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 140.0

    base = 51975
    divs = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    def time_left():
        return DEADLINE - time.time()

    def greedy_build(order):
        cover_count = np.zeros(base, dtype=np.int32)
        classes = []
        for m in order:
            # Find residue r that covers most currently uncovered
            zeros = np.where(cover_count == 0)[0]
            if len(zeros) == 0:
                classes.append([m, 0])
                continue
            
            # Count uncovered per residue class mod m
            A = np.bincount(zeros % m, minlength=m)
            best_r = int(np.argmax(A))
            classes.append([m, best_r])
            
            # Update cover_count
            cover_count[best_r::m] += 1
            
        # Calculate score
        score = base - int(np.sum(cover_count > 0))
        return classes, score, cover_count

    def local_search(classes, cover_count, cur_score):
        moduli = [c[0] for c in classes]
        residues = [c[1] for c in classes]
        n = len(moduli)
        
        max_sweeps = 20
        for sweep in range(max_sweeps):
            if time_left() < 3:
                break
            
            improved = False
            # Compute zeros once per sweep
            zeros = np.where(cover_count == 0)[0]
            
            for i in range(n):
                if time_left() < 2:
                    break
                m = moduli[i]
                r = residues[i]
                
                # Compute A: count of uncovered per residue mod m
                A = np.bincount(zeros % m, minlength=m)
                
                # Compute L_fixed: positions in P_r covered only by this class
                old_pos = np.arange(r, base, m)
                L_fixed = int(np.sum(cover_count[old_pos] == 1))
                
                # Find best new residue r' != r
                A_temp = A.copy()
                A_temp[r] = -1
                best_r_prime = int(np.argmax(A_temp))
                best_gain = A_temp[best_r_prime] - L_fixed
                
                if best_gain > 0:
                    new_pos = np.arange(best_r_prime, base, m)
                    cover_count[old_pos] -= 1
                    cover_count[new_pos] += 1
                    residues[i] = best_r_prime
                    cur_score -= best_gain
                    improved = True
                    # Recompute zeros since cover_count changed
                    zeros = np.where(cover_count == 0)[0]
            
            if not improved:
                break
        
        return [[moduli[i], residues[i]] for i in range(n)], cur_score

    best_score = base
    best_classes = None
    best_cover_count = None

    # Generate diverse orderings
    orderings = []
    
    # Sorted ascending
    orderings.append(sorted(divs))
    # Sorted descending
    orderings.append(sorted(divs, reverse=True))
    
    # Random orderings with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        rng = random.Random(seed)
        order = divs.copy()
        rng.shuffle(order)
        orderings.append(order)
    
    # Heuristic: prioritize moduli that are prime powers or have few divisors
    def divisor_count(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    orderings.append(sorted(divs, key=lambda x: divisor_count(x)))
    orderings.append(sorted(divs, key=lambda x: -divisor_count(x)))
    
    for order in orderings:
        if time_left() < 15:
            break
        
        classes, score, cover_count = greedy_build(order)
        if score < best_score:
            best_score = score
            best_classes = classes
            best_cover_count = cover_count.copy()
        
        # Local search
        if time_left() > 15:
            classes, score = local_search(classes, cover_count, score)
            if score < best_score:
                best_score = score
                best_classes = classes
                best_cover_count = cover_count.copy()

    return best_classes if best_classes is not None else [[3, 0]]
