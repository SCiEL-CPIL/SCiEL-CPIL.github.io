import time
import random
from math import log
import numpy as np

def fast_score(U):
    """
    Faster scoring using numpy for larger sets, falling back to python for small.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    # Compute all pairwise sums
    # i <= j
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences
    # i < j
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    # Unique positive diffs
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

def structured_set(n):
    """
    Generate structured sets from additive combinatorics constructions.
    Uses modular arithmetic patterns that tend to have good sumset/difference-set properties.
    """
    candidates = []
    
    # Golomb-ruler-like: all pairwise differences distinct
    # Known optimal Golomb rulers for small n
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
    if n in golomb_rulers:
        candidates.append(golomb_rulers[n])
    
    # Power-of-prime-like constructions
    for p in [2, 3, 5, 7]:
        U = [0]
        val = 1
        for k in range(1, n):
            val *= p
            U.append(val)
        candidates.append(sorted(set(U)))
    
    # Mixed base constructions
    bases = [(2, 3), (3, 5), (2, 5), (3, 7)]
    for b1, b2 in bases:
        U = [0]
        for k in range(1, n):
            if k % 2 == 1:
                U.append(b1 ** k)
            else:
                U.append(b2 ** (k // 2))
        candidates.append(sorted(set(U)))
    
    return candidates

def ternary_search_geometric(deadline):
    """
    Search over base b and length m for geometric-like sets.
    """
    best_score = -1
    best_set = [0, 1]
    
    # Range of bases: 1.1 to 5.0
    # Range of m: 3 to 30
    
    # Coarse search
    b_low, b_high = 1.1, 5.0
    m_low, m_high = 3, 30
    
    # We can't do full 2D ternary easily without many evaluations.
    # Let's iterate over m and ternary search b for each m.
    
    for m in range(m_low, m_high + 1):
        if time.time() > deadline - 10:
            break
        
        b_low, b_high = 1.1, 5.0
        # Ternary search for optimal base for this m
        for _ in range(10):  # 10 iterations of ternary search is ~30 evaluations
            if time.time() > deadline - 5:
                break
            b1 = b_low + (b_high - b_low) / 3
            b2 = b_high - (b_high - b_low) / 3
            
            U1 = geometric_set(b1, m)
            U2 = geometric_set(b2, m)
            
            s1 = fast_score(U1)
            s2 = fast_score(U2)
            
            if s1 > s2:
                b_high = b2
            else:
                b_low = b1
                
            if s1 > best_score:
                best_score = s1
                best_set = U1
            if s2 > best_score:
                best_score = s2
                best_set = U2
                
    return best_set, best_score

def simulated_annealing(U, deadline, max_iter=2000):
    """
    Simulated annealing to improve the set U.
    Moves:
    1. Perturb an element (local move).
    2. Add an element.
    3. Remove an element.
    4. Swap two elements.
    5. Scale entire set (change max element).
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score = current_score
    
    T = 0.1  # Initial temperature
    cooling = 0.995
    
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        # Weighted move selection: perturb most common, then add, remove, swap, scale
        r = random.random()
        if r < 0.4:
            move_type = 'perturb'
        elif r < 0.6:
            move_type = 'add'
        elif r < 0.75:
            move_type = 'remove'
        elif r < 0.9:
            move_type = 'swap'
        else:
            move_type = 'scale'
        
        new_set = current[:]
        
        if move_type == 'perturb' and n > 1:
            i = random.randint(1, n-1)
            old_val = current[i]
            max_val = max(current)
            # Use multiple perturbation scales
            scale_choice = random.random()
            if scale_choice < 0.5:
                scale = max(1, int(max_val * 0.05))
            elif scale_choice < 0.8:
                scale = max(1, int(max_val * 0.15))
            else:
                scale = max(1, int(max_val * 0.3))
            new_val = old_val + random.randint(-scale, scale)
            if new_val <= 0:
                new_val = old_val + random.randint(1, scale)
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
        
        elif move_type == 'add' and n < 25:
            max_val = max(current)
            if random.random() < 0.4:
                new_val = max_val + random.randint(1, max(2, int(max_val * 0.2)))
            elif random.random() < 0.7:
                new_val = random.randint(1, max_val)
            else:
                new_val = random.randint(max_val + 1, max_val * 3)
            new_set.append(new_val)
            new_set = sorted(set(new_set))
            if len(new_set) == n:
                continue
        
        elif move_type == 'remove' and n > 4:
            i = random.randint(1, n-1)
            new_set.pop(i)
        
        elif move_type == 'swap' and n > 2:
            i, j = random.sample(range(1, n), 2)
            new_set[i], new_set[j] = new_set[j], new_set[i]
            new_set = sorted(new_set)
        
        elif move_type == 'scale' and n > 1:
            # Scale all elements except 0 by a small factor
            factor = random.choice([0.8, 0.9, 1.1, 1.2, 1.5])
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
                if delta < 0.5 and random.random() < np.exp(-delta / max(T, 0.001)):
                    current = new_set
                    current_score = new_score
        
        T *= cooling
        if T < 0.001:
            T = 0.001
            
    return best, best_score

def high_precision_polish(U, deadline):
    """
    Final high-precision polish: try all single-element perturbations with small steps.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score = current_score
    
    n = len(current)
    max_val = max(current)
    
    # Try small perturbations of each element
    for i in range(1, n):
        if time.time() > deadline:
            break
        old_val = current[i]
        # Try offsets from -5 to +5
        for offset in range(-5, 6):
            if offset == 0:
                continue
            new_val = old_val + offset
            if new_val <= 0:
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
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin: evaluator kills at 150s, return well before
    
    best_set = [0, 1, 4, 9, 11]
    best_score = fast_score(best_set)
    
    # 1. Geometric Search
    geo_set, geo_score = ternary_search_geometric(deadline)
    if geo_score > best_score:
        best_score = geo_score
        best_set = geo_set[:]
        
    # 2. Add known good candidates (Golomb Rulers, structured sets, etc.)
    candidates = [
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 41],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
        geo_set[:]
    ]
    
    # Add structured sets from additive combinatorics
    for n in range(4, 13):
        candidates.extend(structured_set(n))
    
    # Generate some random sparse sets with diverse parameters
    random.seed(42)
    for _ in range(15):
        n_elem = random.randint(4, 14)
        max_val = random.randint(30, 300)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
        
    # Geometric sets with specific ratios
    for ratio in [1.5, 2.0, 2.5, 3.0, 4.0]:
        U = [0]
        val = 1
        for _ in range(12):
            U.append(val)
            val = int(val * ratio)
            if val > 5000:
                break
        candidates.append(U)
    
    # Evaluate candidates
    scored_candidates = []
    for U in candidates:
        U_clean = sorted(set(U))
        if 0 not in U_clean or len(U_clean) < 2:
            continue
        s = fast_score(U_clean)
        if s > 0:
            scored_candidates.append((s, U_clean))
            
    scored_candidates.sort(reverse=True)
    
    # 3. Run SA on top candidates
    # Prioritize the geometric best and the global best
    sa_budget = deadline - 30  # Leave 30s for polish
    for idx, (score, U) in enumerate(scored_candidates[:12]):
        if time.time() > sa_budget:
            break
        # Allocate more time to better candidates
        remaining = sa_budget - time.time()
        alloc = max(5, min(remaining, 15))
        refined, refined_score = simulated_annealing(U, time.time() + alloc)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined[:]
            
    # 4. Final high-precision polish on the best set
    if time.time() < deadline - 5:
        refined, refined_score = high_precision_polish(best_set, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined[:]
    
    return best_set

# Test
if __name__ == "__main__":
    print(f"Best set: {construct()}")
