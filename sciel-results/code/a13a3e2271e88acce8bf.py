import numpy as np
import time
import random

def _get_divisors(N):
    """Get all divisors of N greater than 1, sorted."""
    divisors = set()
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.add(i)
            divisors.add(N // i)
    divisors.add(N)
    return sorted(divisors)

def _build_covered(classes, N):
    """Build the covered array from a list of classes."""
    covered = np.zeros(N, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return covered

def _score(classes, N):
    """Calculate the number of uncovered residues."""
    if not classes:
        return N
    covered = _build_covered(classes, N)
    return int(np.sum(1 - covered))

def _best_residue_for_modulus(m, covered, N):
    """Find the residue r that covers the most uncovered residues for modulus m."""
    # Vectorized approach: for each residue r in 0..m-1, count uncovered in covered[r::m]
    # We can do this by iterating, but let's try to be smart about it
    
    # For each r, we want sum of (1 - covered[r::m])
    # This is equivalent to (N//m or (N+1)//m) - sum(covered[r::m])
    # Let's just loop, it's fast enough with numpy slicing
    
    best_gain = -1
    best_r = 0
    for r in range(m):
        # Count uncovered residues in this class
        uncovered = np.count_nonzero(covered[r::m] == 0)
        if uncovered > best_gain:
            best_gain = uncovered
            best_r = r
    return best_r, best_gain

def _greedy_build(N, divisors, time_limit=100.0, start_time=None):
    """Run greedy algorithm to build a covering system."""
    if start_time is None:
        start_time = time.time()
    
    covered = np.zeros(N, dtype=np.uint8)
    used_moduli = set()
    classes = []
    
    while time.time() - start_time < time_limit:
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in divisors:
            if m in used_moduli:
                continue
            
            r, gain = _best_residue_for_modulus(m, covered, N)
            if gain > best_gain:
                best_gain = gain
                best_m = m
                best_r = r
        
        if best_gain == 0:
            break
        
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        covered[best_r::best_m] = 1
    
    return classes, covered

def _local_search(classes, N, divisors, time_limit=10.0, start_time=None):
    """Perform local search by trying to swap classes."""
    if start_time is None:
        start_time = time.time()
    
    current_classes = list(classes)
    current_score = _score(current_classes, N)
    used_moduli = set(m for m, r in current_classes)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        for i in range(len(current_classes)):
            if time.time() - start_time > time_limit:
                break
            
            m_old, r_old = current_classes[i]
            
            # Try replacing class i with each unused modulus
            for m_new in divisors:
                if m_new in used_moduli:
                    continue
                
                # Build trial classes: remove i, add new
                trial_classes = current_classes[:i] + current_classes[i+1:] + [[m_new, 0]]
                
                # Find best residue for m_new given the other classes
                # First, build covered state without class i
                covered_without = np.zeros(N, dtype=np.uint8)
                for j, (m, r) in enumerate(current_classes):
                    if j != i:
                        covered_without[r::m] = 1
                
                # Find best residue for m_new
                best_r_new = 0
                best_gain = -1
                for r_new in range(m_new):
                    gain = np.count_nonzero(covered_without[r_new::m_new] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r_new = r_new
                
                trial_classes[-1] = [m_new, best_r_new]
                trial_score = _score(trial_classes, N)
                
                if trial_score < current_score:
                    current_classes = trial_classes
                    current_score = trial_score
                    used_moduli = set(m for m, r in current_classes)
                    improved = True
                    break  # Restart the outer loop with the new state
    
    return current_classes, current_score

def construct():
    N = 51975
    divisors = _get_divisors(N)
    
    start_time = time.time()
    total_time_limit = 115.0  # Safety margin before 120s
    
    # Strategy 1: Standard greedy
    classes1, covered1 = _greedy_build(N, divisors, time_limit=60.0, start_time=start_time)
    score1 = _score(classes1, N)
    
    # Strategy 2: Greedy with different ordering (larger moduli first)
    elapsed = time.time() - start_time
    if elapsed < 80.0:
        divisors_desc = sorted(divisors, reverse=True)
        classes2, covered2 = _greedy_build(N, divisors_desc, time_limit=20.0, start_time=start_time)
        score2 = _score(classes2, N)
        if score2 < score1:
            classes1, score1 = classes2, score2
    
    # Local search on the best solution
    elapsed = time.time() - start_time
    if elapsed < total_time_limit - 2.0:
        local_time = min(15.0, total_time_limit - elapsed - 1.0)
        classes_final, score_final = _local_search(classes1, N, divisors, time_limit=local_time, start_time=start_time)
        if score_final < score1:
            classes1, score1 = classes_final, score_final
    
    return classes1
