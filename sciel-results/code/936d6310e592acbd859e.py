import time
import random
from math import log
import numpy as np

def fast_score(U):
    """
    Compute the score: 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    U must contain 0.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    # For small sets, Python sets are faster due to lower overhead
    if n <= 12:
        sums_set = set()
        diffs_set = set()
        for i in range(n):
            ui = U[i]
            for j in range(i, n):
                uj = U[j]
                sums_set.add(ui + uj)
            for j in range(i + 1, n):
                uj = U[j]
                diffs_set.add(uj - ui)
        num_sums = len(sums_set)
        num_pos_diffs = len(diffs_set)
        num_diffs = 2 * num_pos_diffs + 1
    else:
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
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def local_search(U, deadline, max_iter=10000):
    """
    Perform local search to improve the set U.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score = current_score
    
    T = 0.1
    cooling = 0.999
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        r = random.random()
        
        # Move selection
        if r < 0.7:
            # Perturb an element
            if n <= 1:
                continue
            i = random.randint(1, n-1)
            old_val = current[i]
            # Step size relative to the gap or magnitude
            if n > 1:
                gap = current[i] - current[i-1] if i > 0 else current[i]
                # Ensure step is at least 1
                step = max(1, int(gap * 0.5))
                # Sometimes take a larger jump
                if random.random() < 0.2:
                    step = max(1, int(current[-1] * 0.1))
            else:
                step = 1
            
            new_val = old_val + random.randint(-step, step)
            if new_val <= 0:
                new_val = old_val + random.randint(1, step)
            
            new_set = current[:]
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
        elif r < 0.85:
            # Add an element
            if n >= 20:
                continue
            max_val = current[-1]
            # Try adding near existing elements or extending the range
            if random.random() < 0.5:
                idx = random.randint(1, n-1)
                new_val = current[idx] + random.randint(1, max(1, int(current[idx]*0.1)))
            else:
                new_val = max_val + random.randint(1, max(2, int(max_val*0.1)))
            
            new_set = current + [new_val]
            new_set = sorted(set(new_set))
            if len(new_set) == n:
                continue
        else:
            # Remove an element
            if n <= 3:
                continue
            i = random.randint(1, n-1)
            new_set = current[:i] + current[i+1:]
        
        new_score = fast_score(new_set)
        
        if new_score < 0:
            # Invalid move, unlikely to accept
            if random.random() < T * 0.1:
                current = new_set
                current_score = new_score
        else:
            if new_score > current_score:
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current[:]
            else:
                delta = current_score - new_score
                if delta < 0.1:
                    # Accept small regressions
                    if random.random() < np.exp(-delta / max(T, 0.001)):
                        current = new_set
                        current_score = new_score
        
        T *= cooling
        if T < 0.001:
            T = 0.001
            
    return best, best_score

def polish(U, deadline):
    """
    Final polish with small steps.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    best = current[:]
    best_score = current_score
    
    n = len(current)
    # Try small perturbations
    for _ in range(100):
        if time.time() > deadline:
            break
        for i in range(1, n):
            if time.time() > deadline:
                break
            for offset in [-1, 1]:
                new_val = current[i] + offset
                if new_val <= 0:
                    continue
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    continue
                s = fast_score(new_set)
                if s > best_score:
                    best_score = s
                    best = new_set[:]
                    current = new_set[:]
    
    return best, best_score

def construct():
    start_time = time.time()
    # Leave 10s for safety
    deadline = start_time + 140
    
    best_set = [0, 1]
    best_score = -1
    
    # 1. Generate diverse initial candidates
    candidates = []
    
    # Golomb rulers (maximize differences)
    golomb_rulers = {
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 41],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67],
    }
    for n, ruler in golomb_rulers.items():
        candidates.append(ruler)
        
    # Geometric progressions
    for base in [1.5, 2.0, 2.5, 3.0]:
        U = [0]
        val = 1
        for _ in range(15):
            U.append(val)
            val = int(val * base)
            if val > 10000:
                break
        candidates.append(U)
        
    # Random sparse sets
    random.seed(42)
    for _ in range(20):
        n_elem = random.randint(4, 15)
        max_val = random.randint(50, 500)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
        
    # Evaluate candidates
    scored_candidates = []
    for U in candidates:
        U_clean = sorted(set(U))
        if 0 not in U_clean or len(U_clean) < 2:
            continue
        s = fast_score(U_clean)
        if s > best_score:
            best_score = s
            best_set = U_clean
        scored_candidates.append((s, U_clean))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Run local search on top candidates
    # Allocate time based on remaining budget
    sa_budget = deadline - 10
    top_k = 10
    
    for idx, (score, U) in enumerate(scored_candidates[:top_k]):
        if time.time() > sa_budget:
            break
        
        # Allocate more time to better candidates
        remaining = sa_budget - time.time()
        if remaining < 5:
            break
            
        # Simple heuristic: give proportional time
        alloc = max(5, min(remaining / (top_k - idx), 20))
        
        refined, refined_score = local_search(U, time.time() + alloc)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined[:]
            
    # 3. Final polish
    if time.time() < deadline - 2:
        refined, refined_score = polish(best_set, deadline - 1)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined[:]
            
    return best_set

# Test
if __name__ == "__main__":
    print(f"Best set: {construct()}")
