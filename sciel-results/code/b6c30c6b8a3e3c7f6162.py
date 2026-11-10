import time
import numpy as np
import random

BASE = 51975

def get_divisors(n):
    divs = []
    for i in range(2, n + 1):
        if n % i == 0:
            divs.append(i)
    return divs

DIVISORS = get_divisors(BASE)
DIVISORS.sort()

def _compute_zero_counts(covered_count, m):
    """Compute for each residue r the count of x in r::m with covered_count[x]==0."""
    is_zero = (covered_count == 0)
    reshaped = is_zero.reshape(-1, m)
    return np.sum(reshaped, axis=0)

def _compute_all_zero_counts(covered_count, divisors):
    """Precompute zero counts for all divisors at once."""
    is_zero = (covered_count == 0)
    results = {}
    for m in divisors:
        reshaped = is_zero.reshape(-1, m)
        results[m] = np.sum(reshaped, axis=0)
    return results

def _local_search(covered_count, current_classes, current_score, deadline):
    """Perform local search: re-optimize residues, add/remove classes."""
    improved = True
    no_improve_count = 0
    max_no_improve = 10
    
    while improved and time.time() < deadline - 5:
        improved = False
        
        # A. Re-optimize residues - try all residues for each modulus
        for idx in range(len(current_classes)):
            if time.time() > deadline - 5:
                break
            
            m, r = current_classes[idx]
            
            # Temporarily remove this class
            covered_count[r::m] -= 1
            
            # Find best residue for this modulus
            zero_counts = _compute_zero_counts(covered_count, m)
            best_r_new = int(np.argmax(zero_counts))
            
            if best_r_new != r:
                # Try the swap
                covered_count[best_r_new::m] += 1
                new_score = int(np.sum(covered_count == 0))
                
                if new_score < current_score:
                    current_score = new_score
                    current_classes[idx] = [m, best_r_new]
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    covered_count[best_r_new::m] -= 1
            
            # Revert temporary removal
            covered_count[r::m] += 1
        
        # B. Add a new class if beneficial - use precomputed zero counts
        if not improved:
            used_set = set(m for m, r in current_classes)
            unused = [m for m in DIVISORS if m not in used_set]
            
            if unused:
                # Precompute zero counts for all unused moduli
                zero_counts_cache = {}
                best_m = None
                best_r = None
                best_gain = 0
                
                for m in unused:
                    if time.time() > deadline - 5:
                        break
                    zero_counts = _compute_zero_counts(covered_count, m)
                    zero_counts_cache[m] = zero_counts
                    max_gain = int(np.max(zero_counts))
                    if max_gain > best_gain:
                        best_gain = max_gain
                        best_m = m
                        best_r = int(np.argmax(zero_counts))
                
                if best_m is not None and best_gain > 0:
                    covered_count[best_r::best_m] += 1
                    current_classes.append([best_m, best_r])
                    current_score = int(np.sum(covered_count == 0))
                    improved = True
                    no_improve_count = 0
        
        # C. Remove redundant classes
        if not improved:
            for idx in range(len(current_classes)):
                if time.time() > deadline - 5:
                    break
                
                m, r = current_classes[idx]
                is_one = (covered_count == 1)
                unique_count = int(np.sum(is_one[r::m]))
                
                if unique_count == 0:
                    covered_count[r::m] -= 1
                    current_classes.pop(idx)
                    current_score = int(np.sum(covered_count == 0))
                    improved = True
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
        
        if current_score == 0:
            break
    
    return covered_count, current_classes, current_score

def _polish_pass(covered_count, current_classes, current_score, deadline):
    """Final high-precision polish: exhaustively try all residues for each class."""
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        for idx in range(len(current_classes)):
            if time.time() > deadline - 2:
                break
            
            m, r = current_classes[idx]
            
            # Try all possible residues for this modulus
            covered_count[r::m] -= 1
            zero_counts = _compute_zero_counts(covered_count, m)
            best_r_new = int(np.argmax(zero_counts))
            
            if best_r_new != r:
                covered_count[best_r_new::m] += 1
                new_score = int(np.sum(covered_count == 0))
                
                if new_score < current_score:
                    current_score = new_score
                    current_classes[idx] = [m, best_r_new]
                    improved = True
                    break
                else:
                    covered_count[best_r_new::m] -= 1
            
            covered_count[r::m] += 1
    
    return covered_count, current_classes, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for 120s limit
    
    best_classes = []
    best_score = BASE
    
    # Generate multiple diverse initializations
    num_restarts = 8
    used_orders = []
    
    for run_id in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Create diverse orders
        if run_id == 0:
            divs = DIVISORS[:]  # Ascending
        elif run_id == 1:
            divs = DIVISORS[::-1]  # Descending
        elif run_id == 2:
            # Primorial-like order: prioritize smaller prime factors
            divs = sorted(DIVISORS, key=lambda x: (x // 3, x // 5, x // 7, x // 11, x))
        elif run_id == 3:
            # Reverse primorial-like
            divs = sorted(DIVISORS, key=lambda x: (-(x // 3), -(x // 5), -(x // 7), -(x // 11), -x))
        else:
            divs = DIVISORS[:]
            random.shuffle(divs)
        
        covered_count = np.zeros(BASE, dtype=np.int8)
        current_classes = []
        
        # Greedy Phase: select (m, r) that maximizes newly_covered residues
        # Prioritize moduli that cover the most currently-uncovered positions
        available_moduli = set(DIVISORS)
        
        while available_moduli:
            if time.time() > deadline - 25:
                break
            
            best_m = None
            best_r = None
            best_gain = 0
            
            for m in available_moduli:
                if time.time() > deadline - 25:
                    break
                zero_counts = _compute_zero_counts(covered_count, m)
                max_gain = int(np.max(zero_counts))
                
                if max_gain > best_gain:
                    best_gain = max_gain
                    best_m = m
                    best_r = int(np.argmax(zero_counts))
            
            if best_m is None:
                break
            
            covered_count[best_r::best_m] += 1
            current_classes.append([best_m, best_r])
            available_moduli.remove(best_m)
        
        current_score = int(np.sum(covered_count == 0))
        
        if current_score < best_score:
            best_score = current_score
            best_classes = [c[:] for c in current_classes]
        
        # Local Search
        covered_count, current_classes, current_score = _local_search(
            covered_count, current_classes, current_score, deadline
        )
        
        if current_score < best_score:
            best_score = current_score
            best_classes = [c[:] for c in current_classes]
        
        if best_score == 0:
            break
    
    # Final polish on the best solution found
    if best_classes and best_score > 0 and time.time() < deadline - 3:
        # Reconstruct covered_count for best_classes
        covered_count = np.zeros(BASE, dtype=np.int8)
        for m, r in best_classes:
            covered_count[r::m] += 1
        
        current_score = int(np.sum(covered_count == 0))
        covered_count, best_classes, current_score = _polish_pass(
            covered_count, best_classes, current_score, deadline
        )
        
        if current_score < best_score:
            best_score = current_score
    
    return best_classes
