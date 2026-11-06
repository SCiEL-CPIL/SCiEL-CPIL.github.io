import numpy as np
import time
import random

def construct():
    N = 51975
    # Get divisors > 1
    divisors = []
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    # Add N itself if not added
    if N not in divisors:
        divisors.append(N)
    
    # Sort divisors for consistent processing
    divisors.sort()
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    def _count_uncovered(covered, m, r):
        """Count uncovered residues in class (m, r)."""
        return np.sum(1 - covered[r::m])
    
    def _find_best_residue(covered, m):
        """Find the residue r for modulus m that covers the most uncovered residues."""
        max_gain = 0
        best_r = 0
        for r in range(m):
            gain = np.sum(1 - covered[r::m])
            if gain > max_gain:
                max_gain = gain
                best_r = r
        return best_r, max_gain
    
    def _build_covered(classes):
        """Build the covered array from a list of classes."""
        covered = np.zeros(N, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return covered
    
    def _score(classes):
        """Compute the number of uncovered residues."""
        covered = _build_covered(classes)
        return np.sum(1 - covered)
    
    def _greedy_add_best(covered, used_moduli, classes, max_iters=1000):
        """Greedy: repeatedly add the class with the highest gain."""
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            best_gain = 0
            best_m = None
            best_r = None
            
            for m in divisors:
                if m in used_moduli:
                    continue
                r, gain = _find_best_residue(covered, m)
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
            
            if best_gain == 0:
                break
            
            classes.append([best_m, best_r])
            used_moduli.add(best_m)
            covered[best_r::best_m] = 1
        
        return classes
    
    def _local_search(classes, max_iters=50):
        """Local search: try swapping classes to improve coverage."""
        best_classes = classes[:]
        best_score = _score(best_classes)
        used_moduli = set(m for m, r in best_classes)
        unused_moduli = [m for m in divisors if m not in used_moduli]
        
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try removing each class and adding the best unused class
            for i in range(len(best_classes)):
                if time.time() > deadline - 3:
                    break
                
                m_old, r_old = best_classes[i]
                # Temporarily remove
                trial_classes = best_classes[:i] + best_classes[i+1:]
                trial_used = set(m for m, r in trial_classes)
                trial_unused = [m for m in divisors if m not in trial_used]
                
                # Find best new class
                covered_trial = _build_covered(trial_classes)
                best_new_gain = 0
                best_new_m = None
                best_new_r = None
                
                for m in trial_unused:
                    r, gain = _find_best_residue(covered_trial, m)
                    if gain > best_new_gain:
                        best_new_gain = gain
                        best_new_m = m
                        best_new_r = r
                
                if best_new_m is not None and best_new_gain > 0:
                    # Compute score of trial with new class
                    trial_with_new = trial_classes + [[best_new_m, best_new_r]]
                    new_score = _score(trial_with_new)
                    if new_score < best_score:
                        best_classes = trial_with_new
                        best_score = new_score
                        improved = True
            
            if not improved:
                break
        
        return best_classes, best_score
    
    # Main: greedy then local search
    covered = np.zeros(N, dtype=np.uint8)
    used_moduli = set()
    classes = []
    
    # Greedy phase
    classes = _greedy_add_best(covered, used_moduli, classes)
    
    # Local search phase
    if time.time() < deadline - 3:
        classes, final_score = _local_search(classes)
    else:
        final_score = _score(classes)
    
    return classes
