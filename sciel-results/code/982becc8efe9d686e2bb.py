import time
import random
from math import log, log2
import numpy as np
from itertools import combinations

def fast_score(U):
    """
    Score a set U using numpy for efficiency.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j)
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    num_pos_diffs = len(np.unique(diffs))
    # Total diffs = 2 * num_pos_diffs + 1 (for 0)
    num_diffs = 2 * num_pos_diffs + 1
    
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def exact_local_search(U, deadline, max_rounds=50):
    """
    Exact local search: for each element, try all possible small perturbations.
    This is a hill-climbing approach with exact evaluation of neighbors.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score = current_score
    max_val = max(current) if current else 1
    n = len(current)
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        max_val = max(current)
        
        for i in range(1, n):
            if time.time() > deadline:
                break
            old_val = current[i]
            
            # Try a range of new values around old_val
            # Range: from max(1, old_val - max_val//5) to old_val + max_val//5
            lower = max(1, old_val - max(1, max_val // 10))
            upper = old_val + max(1, max_val // 10)
            
            for new_val in range(lower, upper + 1):
                if new_val == old_val:
                    continue
                test_set = current[:]
                test_set[i] = new_val
                test_set = sorted(set(test_set))
                if len(test_set) != n:
                    continue
                s = fast_score(test_set)
                if s > best_score:
                    best_score = s
                    best = test_set[:]
                    current = test_set[:]
                    old_val = new_val
                    improved = True
        
        if not improved:
            # Try adding a new element
            if n < 15:
                for new_val in range(1, max_val + max(1, max_val // 5)):
                    if time.time() > deadline:
                        break
                    test_set = current + [new_val]
                    test_set = sorted(set(test_set))
                    if len(test_set) == n:
                        continue
                    s = fast_score(test_set)
                    if s > best_score:
                        best_score = s
                        best = test_set[:]
                        current = test_set[:]
                        n = len(current)
                        max_val = max(current)
                        improved = True
                if improved:
                    continue
            
            # Try removing an element
            if n > 3:
                for i in range(1, n):
                    if time.time() > deadline:
                        break
                    test_set = current[:i] + current[i+1:]
                    s = fast_score(test_set)
                    if s > best_score:
                        best_score = s
                        best = test_set[:]
                        current = test_set[:]
                        n = len(current)
                        max_val = max(current)
                        improved = True
                if improved:
                    continue
            
            break
    
    return best, best_score

def simulated_annealing(U, deadline, max_iter=5000):
    """
    Simulated annealing with more aggressive moves.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score = current_score
    
    T = 0.15
    cooling = 0.998
    
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        max_val = max(current)
        
        r = random.random()
        if r < 0.35:
            move_type = 'perturb'
        elif r < 0.55:
            move_type = 'add'
        elif r < 0.7:
            move_type = 'remove'
        elif r < 0.85:
            move_type = 'swap'
        else:
            move_type = 'scale'
        
        new_set = current[:]
        
        if move_type == 'perturb' and n > 1:
            i = random.randint(1, n-1)
            old_val = current[i]
            # Multiple perturbation scales
            scale_choice = random.random()
            if scale_choice < 0.4:
                scale = max(1, int(max_val * 0.03))
            elif scale_choice < 0.7:
                scale = max(1, int(max_val * 0.1))
            else:
                scale = max(1, int(max_val * 0.25))
            new_val = old_val + random.randint(-scale, scale)
            if new_val <= 0:
                new_val = old_val + random.randint(1, scale)
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
        
        elif move_type == 'add' and n < 20:
            if random.random() < 0.3:
                new_val = max_val + random.randint(1, max(2, int(max_val * 0.3)))
            elif random.random() < 0.6:
                new_val = random.randint(1, max_val)
            else:
                new_val = random.randint(max_val + 1, max_val * 2)
            new_set.append(new_val)
            new_set = sorted(set(new_set))
            if len(new_set) == n:
                continue
        
        elif move_type == 'remove' and n > 3:
            i = random.randint(1, n-1)
            new_set.pop(i)
        
        elif move_type == 'swap' and n > 2:
            i, j = random.sample(range(1, n), 2)
            new_set[i], new_set[j] = new_set[j], new_set[i]
            new_set = sorted(new_set)
        
        elif move_type == 'scale' and n > 1:
            factor = random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.5])
            new_set = [0] + [max(1, int(v * factor)) for v in current[1:]]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
        
        else:
            continue
            
        new_score = fast_score(new_set)
        
        if new_score < 0:
            if random.random() < T:
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
                if delta < 0.3 and random.random() < np.exp(-delta / max(T, 0.001)):
                    current = new_set
                    current_score = new_score
        
        T *= cooling
        if T < 0.005:
            T = 0.005
            
    return best, best_score

def geometric_set(base, m):
    """
    Construct a set U = {0} U { floor(base^k) for k=1..m }
    """
    U = [0]
    val = 1
    for _ in range(m):
        val = int(val * base)
        if val <= 0:
            break
        if val not in U:
            U.append(val)
    return sorted(set(U))

def generate_candidates(deadline):
    """
    Generate diverse candidate sets.
    """
    candidates = []
    
    # Golomb rulers
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
    
    # Power-of-prime constructions
    for p in [2, 3, 5, 7]:
        for m in range(3, 15):
            U = [0]
            val = 1
            for _ in range(m):
                val *= p
                U.append(val)
            candidates.append(sorted(set(U)))
    
    # Geometric sets with various bases
    for base in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0]:
        for m in range(3, 18):
            U = geometric_set(base, m)
            if len(U) >= 3:
                candidates.append(U)
    
    # Random sparse sets
    random.seed(42)
    for _ in range(30):
        n_elem = random.randint(4, 12)
        max_val = random.randint(20, 200)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
    
    # Mixed constructions
    for b1 in [2, 3]:
        for b2 in [3, 5, 7]:
            U = [0]
            for k in range(1, 12):
                if k % 2 == 1:
                    U.append(b1 ** k)
                else:
                    U.append(b2 ** (k // 2))
            candidates.append(sorted(set(U)))
    
    # Small dense sets
    for n in range(4, 10):
        for max_val in [10, 20, 30, 50]:
            U = [0] + sorted(random.sample(range(1, max_val), n - 1))
            candidates.append(U)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Generate candidates
    candidates = generate_candidates(deadline)
    
    # Score all candidates
    scored_candidates = []
    for U in candidates:
        U_clean = sorted(set(U))
        if 0 not in U_clean or len(U_clean) < 2:
            continue
        s = fast_score(U_clean)
        if s > 0:
            scored_candidates.append((s, U_clean))
    
    scored_candidates.sort(reverse=True)
    
    if not scored_candidates:
        return [0, 1, 4, 9, 11]
    
    best_score, best_set = scored_candidates[0]
    
    # Run SA on top candidates
    sa_budget = deadline - 40  # Leave 40s for exact local search
    for idx, (score, U) in enumerate(scored_candidates[:8]):
        if time.time() > sa_budget:
            break
        remaining = sa_budget - time.time()
        alloc = max(5, min(remaining, 20))
        refined, refined_score = simulated_annealing(U, time.time() + alloc)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined[:]
    
    # Run exact local search on the best set
    if time.time() < deadline - 5:
        refined, refined_score = exact_local_search(best_set, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined[:]
    
    # Final polish: try a few more SA restarts if time permits
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            # Perturb the best set and run SA
            perturbed = best_set[:]
            if len(perturbed) > 2:
                i = random.randint(1, len(perturbed)-1)
                perturbed[i] = max(1, perturbed[i] + random.randint(-5, 5))
                perturbed = sorted(set(perturbed))
            refined, refined_score = simulated_annealing(perturbed, deadline - 5)
            if refined_score > best_score:
                best_score = refined_score
                best_set = refined[:]
    
    return best_set

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Best set: {result}")
    print(f"Score: {fast_score(result)}")
