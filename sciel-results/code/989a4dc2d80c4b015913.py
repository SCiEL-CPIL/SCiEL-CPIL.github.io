import numpy as np
import time
import random

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # 1. Generate divisors > 1
    divisors = []
    for i in range(3, base + 1, 2):
        if base % i == 0:
            divisors.append(i)
            
    # We use all divisors
    # classes: list of [m, r]
    
    def get_uncovered_count(classes):
        counts = np.zeros(base, dtype=np.int16)
        for m, r in classes:
            counts[r::m] += 1
        return np.count_nonzero(counts == 0)

    def optimize(classes, deadline):
        # classes is a list of [m, r]
        # Returns optimized classes and score
        counts = np.zeros(base, dtype=np.int16)
        for m, r in classes:
            counts[r::m] += 1
        
        improved = True
        no_improve_count = 0
        max_no_improve = 5 # kick after 5 full passes with no change
        
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(classes)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                m, r_old = classes[i]
                
                # Remove current class contribution
                counts[r_old::m] -= 1
                
                # Find best r for this m given the rest
                # is_uncovered is 1 where counts == 0
                is_uncovered = (counts == 0)
                
                # Reshape to (base//m, m)
                # mat[k, r] corresponds to index k*m + r
                mat = is_uncovered.reshape(base // m, m)
                gains = mat.sum(axis=0)
                best_r = np.argmax(gains)
                
                # Apply best_r
                counts[best_r::m] += 1
                classes[i] = [m, best_r]
                
                if best_r != r_old:
                    improved = True
                    no_improve_count = 0
            # End for i
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Kick: Randomly perturb a few residues
                    if len(classes) > 0:
                        # Perturb 1-3 random classes
                        num_perturb = random.randint(1, 3)
                        for _ in range(num_perturb):
                            i = random.randint(0, len(classes)-1)
                            m, r = classes[i]
                            new_r = random.randint(0, m-1)
                            if new_r != r:
                                counts[r::m] -= 1
                                counts[new_r::m] += 1
                                classes[i] = [m, new_r]
                        improved = True
                        no_improve_count = 0
                    else:
                        break
                        
        return classes, get_uncovered_count(classes)

    best_classes = None
    best_score = float('inf')
    
    # Strategy 1: Greedy Init (Largest moduli first, random residues)
    # This is a heuristic starting point
    divs_sorted_desc = sorted(divisors, reverse=True)
    classes_init = []
    for m in divs_sorted_desc:
        classes_init.append([m, random.randint(0, m-1)])
        
    c, s = optimize(classes_init, deadline)
    if s < best_score:
        best_score = s
        best_classes = [list(x) for x in c] # Deep copy
        
    # Strategy 2: Smallest moduli first
    divs_sorted_asc = sorted(divisors)
    classes_init2 = []
    for m in divs_sorted_asc:
        classes_init2.append([m, random.randint(0, m-1)])
        
    if time.time() < deadline - 5:
        c, s = optimize(classes_init2, deadline)
        if s < best_score:
            best_score = s
            best_classes = [list(x) for x in c]

    # Strategy 3: Random Restarts
    # Continue until deadline
    while time.time() < deadline - 5:
        classes_init3 = []
        for m in divisors:
            classes_init3.append([m, random.randint(0, m-1)])
        c, s = optimize(classes_init3, deadline)
        if s < best_score:
            best_score = s
            best_classes = [list(x) for x in c]

    return best_classes
