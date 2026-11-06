import time
import random
import numpy as np

def _get_divisors(n):
    """Get all divisors of n greater than 1."""
    divs = []
    i = 2
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.append(n)
    return sorted(divs)

def _greedy_init(base, divisors):
    """Greedy initialization: assign residues to maximize coverage."""
    covered = np.zeros(base, dtype=np.int32)
    residues = {}
    
    # Sort by size descending
    sorted_divs = sorted(divisors, reverse=True)
    
    for m in sorted_divs:
        best_r = 0
        best_count = -1
        for r in range(m):
            idx = np.arange(r, base, m)
            count = np.sum(covered[idx] == 0)
            if count > best_count:
                best_count = count
                best_r = r
        residues[m] = best_r
        covered[np.arange(best_r, base, m)] += 1
    
    score = int(np.sum(covered == 0))
    return covered, residues, score

def _local_search(covered, residues, score, divisors, base, deadline, seed=42):
    """Local search with simulated annealing."""
    random.seed(seed)
    current_temp = 1000.0
    cooling_rate = 0.999
    
    best_score = score
    best_residues = residues.copy()
    
    while time.time() < deadline:
        m = random.choice(divisors)
        current_r = residues[m]
        
        # For small moduli, try all residues; for large, sample
        if m <= 100:
            candidate_rs = list(range(m))
        else:
            candidate_rs = [current_r] + [random.randint(0, m-1) for _ in range(30)]
        
        best_r = current_r
        best_delta = 0
        
        for r in candidate_rs:
            if r == current_r:
                continue
            
            # Compute delta efficiently
            old_idx = np.arange(current_r, base, m)
            new_idx = np.arange(r, base, m)
            
            # Positions that lose coverage
            removed_mask = np.zeros(base, dtype=bool)
            removed_mask[old_idx] = True
            removed_mask[new_idx] = False
            removed_positions = np.where(removed_mask)[0]
            
            # Positions that gain coverage
            added_mask = np.zeros(base, dtype=bool)
            added_mask[new_idx] = True
            added_mask[old_idx] = False
            added_positions = np.where(added_mask)[0]
            
            delta = 0
            if len(removed_positions) > 0:
                delta += int(np.sum(covered[removed_positions] == 1))
            if len(added_positions) > 0:
                delta -= int(np.sum(covered[added_positions] == 0))
            
            if delta < best_delta:
                best_delta = delta
                best_r = r
        
        # Accept move with simulated annealing
        if best_r != current_r:
            accept = False
            if best_delta < 0:
                accept = True
            else:
                # Accept worse moves with probability exp(-delta/temp)
                if random.random() < np.exp(-best_delta / max(current_temp, 1e-10)):
                    accept = True
            
            if accept:
                old_idx = np.arange(current_r, base, m)
                new_idx = np.arange(best_r, base, m)
                covered[old_idx] -= 1
                covered[new_idx] += 1
                score += best_delta
                residues[m] = best_r
                
                if score < best_score:
                    best_score = score
                    best_residues = residues.copy()
        
        current_temp *= cooling_rate
        if current_temp < 1.0:
            current_temp = 1.0
    
    return covered, best_residues, best_score

def _polish(covered, residues, score, divisors, base, deadline):
    """Final polish: try all residues for each modulus."""
    best_score = score
    best_residues = residues.copy()
    
    while time.time() < deadline - 2:
        improved = False
        for m in divisors:
            current_r = residues[m]
            best_r = current_r
            best_delta = 0
            
            for r in range(m):
                if r == current_r:
                    continue
                
                old_idx = np.arange(current_r, base, m)
                new_idx = np.arange(r, base, m)
                
                removed_mask = np.zeros(base, dtype=bool)
                removed_mask[old_idx] = True
                removed_mask[new_idx] = False
                removed_positions = np.where(removed_mask)[0]
                
                added_mask = np.zeros(base, dtype=bool)
                added_mask[new_idx] = True
                added_mask[old_idx] = False
                added_positions = np.where(added_mask)[0]
                
                delta = 0
                if len(removed_positions) > 0:
                    delta += int(np.sum(covered[removed_positions] == 1))
                if len(added_positions) > 0:
                    delta -= int(np.sum(covered[added_positions] == 0))
                
                if delta < best_delta:
                    best_delta = delta
                    best_r = r
            
            if best_r != current_r:
                old_idx = np.arange(current_r, base, m)
                new_idx = np.arange(best_r, base, m)
                covered[old_idx] -= 1
                covered[new_idx] += 1
                score += best_delta
                residues[m] = best_r
                improved = True
                
                if score < best_score:
                    best_score = score
                    best_residues = residues.copy()
        
        if not improved:
            break
    
    return covered, best_residues, best_score

def construct():
    base = 51975
    divisors = _get_divisors(base)
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Greedy initialization
    covered, residues, score = _greedy_init(base, divisors)
    
    # Local search with simulated annealing
    covered, residues, score = _local_search(covered, residues, score, divisors, base, deadline - 10, seed=42)
    
    # Final polish
    covered, residues, score = _polish(covered, residues, score, divisors, base, deadline)
    
    return [[m, residues[m]] for m in divisors]
