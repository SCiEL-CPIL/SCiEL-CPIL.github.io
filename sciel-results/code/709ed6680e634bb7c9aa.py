import time
import random
from math import log, floor
import numpy as np

def fast_score(U):
    """
    Calculate the score for a set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2 * max(U) + 1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j)
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    num_pos_diffs = len(np.unique(diffs))
    num_diffs = 2 * num_pos_diffs + 1
    
    if num_sums == 0:
        return -1
        
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def difference_aware_candidates(U, max_val):
    """
    Generate candidates that are likely to improve the difference/sum ratio.
    Prioritize values that create new distinct differences.
    """
    U_set = set(U)
    existing_diffs = set()
    existing_sums = set()
    n = len(U)
    
    for i in range(n):
        for j in range(i + 1, n):
            existing_diffs.add(U[j] - U[i])
        for j in range(i, n):
            existing_sums.add(U[i] + U[j])
    
    candidates = []
    seen = set()
    
    # Try values that are differences of existing elements (shifted)
    for d in existing_diffs:
        for u in U:
            val = u + d
            if 0 < val <= max_val and val not in U_set and val not in seen:
                seen.add(val)
                candidates.append(val)
            val2 = abs(u - d)
            if 0 < val2 <= max_val and val2 not in U_set and val2 not in seen:
                seen.add(val2)
                candidates.append(val2)
    
    # Try values that are sums (for potential new structure)
    for s in existing_sums:
        if 0 < s <= max_val and s not in U_set and s not in seen:
            seen.add(s)
            candidates.append(s)
    
    # Add some random values in gaps
    if U:
        max_u = max(U)
        for _ in range(min(20, max_u // 10)):
            val = random.randint(1, max_val)
            if val not in U_set and val not in seen:
                seen.add(val)
                candidates.append(val)
    
    return candidates

def local_search_structured(U, deadline, max_iter=5000):
    """
    Structured local search using difference-aware candidate generation.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score = current_score
    start_time = time.time()
    
    # Temperature for simulated annealing
    temp = 0.5
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        n = len(current)
        move_type = random.random()
        
        if move_type < 0.4:
            # Replace an element with a difference-aware candidate
            if n < 2:
                continue
            i = random.randint(1, n - 1)
            old_val = current[i]
            
            max_val = max(100, max(current) * 2)
            candidates = difference_aware_candidates(current, max_val)
            
            if not candidates:
                new_val = random.randint(1, max_val)
            else:
                new_val = random.choice(candidates[:50])
            
            if new_val in current or new_val == 0:
                continue
                
            new_set = current[:]
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            
            if len(new_set) != n:
                continue
                
            new_score = fast_score(new_set)
            
        elif move_type < 0.6:
            # Add an element
            if n >= 25:
                continue
            max_val = max(100, max(current) * 2)
            candidates = difference_aware_candidates(current, max_val)
            
            if not candidates:
                new_val = random.randint(1, max_val)
            else:
                new_val = random.choice(candidates[:50])
            
            if new_val in current or new_val == 0:
                continue
                
            new_set = current + [new_val]
            new_set = sorted(set(new_set))
            new_score = fast_score(new_set)
            
        elif move_type < 0.8:
            # Remove an element (not 0)
            if n <= 3:
                continue
            removable = [x for x in current if x != 0]
            if not removable:
                continue
            val_to_remove = random.choice(removable)
            new_set = [x for x in current if x != val_to_remove]
            new_score = fast_score(new_set)
            
        else:
            # Small perturbation
            i = random.randint(1, n - 1)
            delta = random.randint(-10, 10)
            new_val = current[i] + delta
            if new_val <= 0 or new_val in current:
                continue
            new_set = current[:]
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
            new_score = fast_score(new_set)
        
        if new_score > best_score:
            best_score = new_score
            best = new_set[:]
        
        # Simulated annealing acceptance
        if new_score >= current_score:
            current = new_set
            current_score = new_score
        else:
            delta_score = current_score - new_score
            if random.random() < np.exp(-delta_score / temp):
                current = new_set
                current_score = new_score
        
        # Cool down
        temp *= 0.999
        
    return best, best_score

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Structured seeds: Golomb rulers (minimize max for given n)
    golomb_rulers = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 41],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 64],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 64, 72],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 64, 72, 78],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 64, 72, 78, 84],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 64, 72, 78, 84, 92],
    ]
    
    # Geometric seeds
    geometric_seeds = []
    for b in [1.5, 1.7, 1.9, 2.0, 2.2, 2.5, 3.0]:
        for m in range(3, 15):
            U = geometric_set(b, m)
            if len(U) >= 3:
                geometric_seeds.append(U)
    
    # Sidon-like seeds (all pair sums distinct)
    sidon_seeds = [
        [0, 1, 4, 10, 18, 23, 25, 29, 30, 32],
        [0, 1, 4, 10, 18, 28, 36, 41, 43, 46],
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
    ]
    
    best_set = None
    best_score = -1
    
    # Evaluate all seeds
    all_seeds = golomb_rulers + geometric_seeds + sidon_seeds
    
    for U in all_seeds:
        if time.time() > deadline - 30:
            break
        s = fast_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    if best_set is None:
        best_set = [0, 1, 4, 9, 11]
        best_score = fast_score(best_set)
    
    # Refine top seeds with structured local search
    scored_seeds = []
    for U in all_seeds:
        s = fast_score(U)
        scored_seeds.append((s, U))
    scored_seeds.sort(key=lambda x: x[0], reverse=True)
    
    top_seeds = [U for s, U in scored_seeds[:10]]
    
    for seed in top_seeds:
        if time.time() > deadline - 15:
            break
        refined, refined_score = local_search_structured(seed, deadline, max_iter=3000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # Final refinement on the best set
    if time.time() < deadline - 10:
        final, final_score = local_search_structured(best_set, deadline, max_iter=5000)
        if final_score > best_score:
            best_score = final_score
            best_set = final
    
    return best_set

if __name__ == "__main__":
    result = construct()
    print(f"Best set: {result}")
    print(f"Score: {fast_score(result)}")
