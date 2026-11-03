import time
import random
import numpy as np
from math import log

def score_set(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    # Compute all pairwise sums (i <= j)
    sums = set()
    diffs = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(int(arr[i]) + int(arr[j]))
        for j in range(i + 1, n):
            d = int(arr[j]) - int(arr[i])
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = int(arr[-1])
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def fast_score(U):
    """Faster score using numpy where possible."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    
    # Use numpy for sums
    i_idx, j_idx = np.triu_indices(n)
    sums = arr[i_idx] + arr[j_idx]
    n_sums = len(np.unique(sums))
    
    # Use numpy for diffs
    i_idx2, j_idx2 = np.triu_indices(n, k=1)
    diffs = arr[j_idx2] - arr[i_idx2]
    n_diffs = len(np.unique(diffs)) * 2 + 1  # symmetric + zero
    
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return -1
    max_u = int(arr[-1])
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def geometric_set(start, ratio, length):
    """Generate a geometric-like set."""
    U = [0]
    val = start
    for _ in range(length):
        U.append(val)
        val = int(val * ratio)
        if val > 200000:
            break
    return U

def ternary_search_geometric(start, ratio, length, deadline):
    """Ternary search over ratio for a given start and length."""
    lo, hi = 1.01, 5.0
    best_score = -1
    best_ratio = ratio
    
    for _ in range(30):
        if time.time() > deadline:
            break
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        s1 = fast_score(geometric_set(start, m1, length))
        s2 = fast_score(geometric_set(start, m2, length))
        if s1 > s2:
            hi = m2
        else:
            lo = m1
        if s1 > best_score:
            best_score = s1
            best_ratio = m1
        if s2 > best_score:
            best_score = s2
            best_ratio = m2
    
    return geometric_set(start, best_ratio, length), best_score

def simulated_annealing(U, deadline, start_temp=5.0, cool_rate=0.995):
    """Simulated annealing to optimize the set."""
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = list(current)
    best_score = current_score
    temp = start_temp
    n = len(current)
    
    while time.time() < deadline and temp > 0.01:
        n = len(current)
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.5 and n > 1:
            # Perturb an element
            i = random.randint(1, n - 1)
            old_val = current[i]
            # Random perturbation
            if random.random() < 0.3:
                # Large jump
                new_val = random.randint(1, int(old_val * 2) + 10)
            else:
                # Small perturbation
                delta = random.randint(-max(1, old_val // 5), max(1, old_val // 5))
                new_val = old_val + delta
            
            if new_val <= 0 or new_val in current:
                temp *= cool_rate
                continue
            
            new_set = [current[j] if j != i else new_val for j in range(n)]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                temp *= cool_rate
                continue
            
        elif move_type < 0.75 and n < 25:
            # Add an element
            max_val = max(current)
            if random.random() < 0.5:
                new_val = random.randint(max_val + 1, max_val * 2 + 20)
            else:
                # Insert in a gap
                i = random.randint(0, n - 1)
                if i == 0:
                    new_val = random.randint(1, current[0])
                else:
                    new_val = random.randint(current[i-1] + 1, current[i] - 1)
                    if new_val <= current[i-1] or new_val >= current[i]:
                        temp *= cool_rate
                        continue
            
            if new_val <= 0 or new_val in current:
                temp *= cool_rate
                continue
            new_set = sorted(current + [new_val])
            
        elif move_type < 0.9 and n > 4:
            # Remove an element
            i = random.randint(1, n - 1)
            new_set = [current[j] for j in range(n) if j != i]
            
        else:
            # Swap two elements
            if n < 2:
                temp *= cool_rate
                continue
            i, j = random.sample(range(1, n), 2)
            new_set = list(current)
            new_set[i], new_set[j] = new_set[j], new_set[i]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                temp *= cool_rate
                continue
        
        new_score = fast_score(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        else:
            # Accept with probability exp(-delta/temp)
            delta = current_score - new_score
            if delta < 0:
                current = new_set
                current_score = new_score
            else:
                if random.random() < np.exp(-delta / max(temp, 0.001)):
                    current = new_set
                    current_score = new_score
        
        temp *= cool_rate
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (150s limit, keep 40s buffer)
    
    best_set = [0, 1, 4, 9, 11]
    best_score = fast_score(best_set)
    
    # Generate diverse base candidates
    candidates = []
    
    # 1. Geometric sequences with various parameters
    for start in [1, 2, 3, 5]:
        for ratio in [1.3, 1.5, 1.7, 1.9, 2.0, 2.1, 2.3, 2.5, 2.8, 3.0, 3.5, 4.0]:
            for length in [8, 10, 12, 14, 16]:
                U = geometric_set(start, ratio, length)
                if len(U) >= 4:
                    candidates.append(U)
    
    # 2. Powers of 2
    for k in range(5, 18):
        U = [0] + [2**i for i in range(1, k)]
        candidates.append(U)
    
    # 3. Exponential growth (2^k - 1)
    for k in range(4, 16):
        U = [0]
        val = 1
        for _ in range(k):
            U.append(val)
            val = 2 * val + 1
        candidates.append(U)
    
    # 4. Fibonacci
    a, b = 0, 1
    U = [0, 1]
    while b < 50000:
        U.append(b)
        a, b = b, a + b
    candidates.append(U)
    
    # 5. Squares
    U = [i*i for i in range(15)]
    candidates.append(U)
    
    # 6. Triangular numbers
    U = [i*(i+1)//2 for i in range(15)]
    candidates.append(U)
    
    # 7. Randomized exponential with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        random.seed(seed)
        for _ in range(5):
            U = [0]
            val = random.randint(1, 10)
            for _ in range(random.randint(6, 16)):
                U.append(val)
                val = random.randint(int(val * 1.2), int(val * 3.0))
                if val > 100000:
                    break
            if len(U) >= 4:
                candidates.append(U)
    
    # 8. Sparse random sets
    random.seed(999)
    for _ in range(30):
        n_elem = random.randint(6, 20)
        max_val = random.randint(100, 5000)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
    
    # 9. Mixed growth patterns
    for seed in [11, 22, 33, 44, 55, 66, 77, 88]:
        random.seed(seed)
        U = [0]
        val = random.randint(1, 8)
        for _ in range(random.randint(8, 18)):
            if random.random() < 0.5:
                val = int(val * random.uniform(1.3, 2.5))
            else:
                val += random.randint(1, max(2, int(val * 0.3)))
            U.append(val)
            if val > 100000:
                break
        if len(U) >= 4:
            candidates.append(U)
    
    # 10. Sets with controlled gaps (Fibonacci-like gaps)
    for gap_pattern in [[1, 2, 4, 8, 16], [2, 3, 5, 8, 13], [1, 3, 7, 15, 31], [2, 4, 8, 16, 32], [3, 5, 8, 13, 21]]:
        U = [0]
        val = 0
        for _ in range(4):
            for gap in gap_pattern:
                val += gap
                U.append(val)
                if val > 50000:
                    break
            if val > 50000:
                break
        if len(U) >= 4:
            candidates.append(U)
    
    # Evaluate all candidates
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline - 30:
            break
        s = fast_score(U)
        if s > 0:
            scored_candidates.append((s, U))
    
    scored_candidates.sort(reverse=True)
    
    # Process top candidates with simulated annealing
    for score, U in scored_candidates[:40]:
        if time.time() > deadline - 15:
            break
        refined, refined_score = simulated_annealing(U, deadline - 5)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # If we have time, do additional refinement on the best set
    if time.time() < deadline - 10:
        refined, refined_score = simulated_annealing(best_set, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # Final check
    if time.time() < deadline - 5:
        final_score = fast_score(best_set)
        if final_score > best_score:
            best_score = final_score
    
    return best_set
