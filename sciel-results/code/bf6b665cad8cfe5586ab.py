import numpy as np
import time
import random

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Generate divisors
    divisors = []
    for i in range(3, base + 1, 2):
        if base % i == 0:
            divisors.append(i)
    divisors.sort()
    
    # 2. Greedy Initialization
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    
    for m in divisors:
        if time.time() > deadline:
            break
        # Vectorized gain calculation
        A = covered.reshape(base // m, m)
        gains = np.sum(A == 0, axis=0)
        best_r = np.argmax(gains)
        classes.append([m, best_r])
        covered[best_r::m] = 1

    # 3. Local Search (Vectorized Hill Climbing)
    counts = np.zeros(base, dtype=np.int16)
    for m, r in classes:
        counts[r::m] += 1
        
    best_score = np.count_nonzero(counts == 0)
    best_classes = [c[:] for c in classes]
    
    improved = True
    stagnation_counter = 0
    
    while time.time() < deadline:
        if not improved:
            # Perturb if stuck
            if stagnation_counter > 5:
                for _ in range(5):
                    idx = random.randint(0, len(classes) - 1)
                    m = classes[idx][0]
                    r_new = random.randint(0, m - 1)
                    r_old = classes[idx][1]
                    if r_new != r_old:
                        counts[r_old::m] -= 1
                        counts[r_new::m] += 1
                        classes[idx] = [m, r_new]
                stagnation_counter = 0
                improved = True
                continue
            else:
                stagnation_counter += 1
                if stagnation_counter > 20:
                    break
                else:
                    break # Break inner loop, but while condition will re-evaluate. 
                         # Actually, if not improved, we should break the while loop unless we plan to perturb.
                         # The logic above handles perturbation. If we don't perturb, we break.

        improved = False
        for i in range(len(classes)):
            if time.time() > deadline:
                return best_classes
            m, r_old = classes[i]
            A = counts.reshape(base // m, m)
            gains = np.sum(A == 0, axis=0)
            losses = np.sum(A == 1, axis=0)
            
            r_best = np.argmax(gains)
            gain_best = gains[r_best]
            loss_old = losses[r_old]
            
            if gain_best > loss_old and r_best != r_old:
                counts[r_old::m] -= 1
                counts[r_best::m] += 1
                classes[i] = [m, r_best]
                improved = True
                
        current_score = np.count_nonzero(counts == 0)
        if current_score < best_score:
            best_score = current_score
            best_classes = [c[:] for c in classes]

    return best_classes
