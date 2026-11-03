import time
import random
from math import log
import itertools
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    # Use NumPy for faster sumset and difference set calculation
    # Convert to numpy array for vectorized operations
    U_arr = np.array(U)
    
    # Calculate all pairwise sums
    # sums = {u_i + u_j for i, j in range(n)}
    # This is equivalent to U_arr[:, None] + U_arr[None, :]
    sums = np.unique(U_arr[:, None] + U_arr[None, :])
    
    # Calculate all pairwise differences
    # diffs = {u_j - u_i for i, j in range(n)}
    # This is equivalent to U_arr[:, None] - U_arr[None, :]
    diffs = np.unique(U_arr[:, None] - U_arr[None, :])
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def local_search(U, deadline, max_iter=2000):
    current = sorted(set(U))
    current_score = score_set(current)
    if current_score < 0:
        return current, current_score
    
    # Simulated Annealing parameters
    temperature = 0.05
    cooling_rate = 0.99
    
    improved = True
    iteration = 0
    while time.time() < deadline and iteration < max_iter:
        iteration += 1
        n = len(current)
        
        # Randomly select a mutation strategy
        # Strategy 1: Perturb individual element
        # Strategy 2: Add element
        # Strategy 3: Remove element
        # Strategy 4: Swap two elements (if applicable, or just replace)
        
        move_type = random.randint(1, 4)
        new_set = current
        
        if move_type == 1: # Perturb
            i = random.randint(1, n - 1) # Don't touch 0
            old_val = current[i]
            # Choose a delta
            # Mix of small and large jumps
            if random.random() < 0.7:
                delta = random.randint(-10, 10)
            else:
                delta = random.randint(-int(old_val*0.5), int(old_val*0.5))
            
            new_val = old_val + delta
            if new_val <= 0:
                continue
            new_set = [current[j] if j != i else new_val for j in range(n)]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
                
        elif move_type == 2: # Add
            if n >= 25: # Limit size to keep computation fast
                continue
            max_val = max(current)
            # Try adding near max or in gaps
            if random.random() < 0.5:
                new_val = random.randint(max_val + 1, max_val + 50)
            else:
                # Pick a random gap
                i = random.randint(1, n - 1)
                if current[i] > current[i-1] + 2:
                    new_val = random.randint(current[i-1] + 1, current[i] - 1)
                else:
                    new_val = random.randint(1, max_val + 50)
            
            if new_val in current:
                continue
            new_set = sorted(current + [new_val])
            
        elif move_type == 3: # Remove
            if n <= 5:
                continue
            i = random.randint(1, n - 1) # Don't remove 0
            new_set = [current[j] for j in range(n) if j != i]
            
        else: # Swap/Replace
            i = random.randint(1, n - 1)
            old_val = current[i]
            # Replace with a random value from a wider range
            new_val = random.randint(1, max(10, int(max(current) * 1.5)))
            if new_val == old_val:
                continue
            new_set = [current[j] if j != i else new_val for j in range(n)]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue

        new_score = score_set(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
        else:
            # Accept worse solution with probability exp(-delta/T)
            delta = current_score - new_score
            if random.random() < np.exp(-delta / max(temperature, 1e-9)):
                current = new_set
                current_score = new_score
        
        temperature *= cooling_rate

    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)
    
    # Generate diverse base candidates
    candidates = []
    
    # 1. Exponential growth (2^k - 1)
    for k in range(1, 18):
        U = [0]
        val = 1
        for _ in range(k):
            U.append(val)
            val = 2 * val + 1
        candidates.append(U)
        
    # 2. Powers of 2
    for k in range(1, 18):
        U = [0]
        val = 1
        for _ in range(k):
            U.append(val)
            val = 2 * val
        candidates.append(U)
        
    # 3. Fibonacci
    a, b = 0, 1
    U = [0, 1]
    while b < 50000:
        U.append(b)
        a, b = b, a + b
    candidates.append(U)
    
    # 4. Squares
    U = [i*i for i in range(20)]
    candidates.append(U)
    
    # 5. Triangular numbers
    U = [i*(i+1)//2 for i in range(20)]
    candidates.append(U)
    
    # 6. Prime numbers
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    U = [0] + [i for i in range(1, 500) if is_prime(i)]
    candidates.append(U)
    
    # 7. Randomized exponential with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        random.seed(seed)
        for _ in range(10):
            U = [0]
            val = random.randint(1, 10)
            for _ in range(random.randint(5, 20)):
                U.append(val)
                val = random.randint(int(val * 1.1), int(val * 4.0))
            candidates.append(U)
    
    # 8. Geometric sequences with various ratios
    for ratio in [1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.8, 3.0, 3.5, 4.0, 5.0]:
        U = [0]
        val = 1
        for _ in range(15):
            U.append(val)
            val = int(val * ratio)
            if val > 200000:
                break
        candidates.append(U)
    
    # 9. Sparse random sets
    random.seed(999)
    for _ in range(30):
        n_elem = random.randint(8, 25)
        max_val = random.randint(200, 5000)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
    
    # 10. Mixed growth patterns
    for seed in [11, 22, 33, 44, 55, 66, 77, 88]:
        random.seed(seed)
        U = [0]
        val = random.randint(1, 5)
        for _ in range(random.randint(10, 20)):
            # Alternate between different growth rates
            if random.random() < 0.5:
                val = int(val * random.uniform(1.5, 3.0))
            else:
                val += random.randint(1, max(2, int(val * 0.5)))
            U.append(val)
        candidates.append(U)
    
    # 11. Sets with controlled gaps
    for gap_pattern in [[1, 2, 4, 8, 16], [2, 3, 5, 8, 13], [1, 3, 7, 15, 31], [2, 4, 8, 16, 32], [3, 6, 12, 24, 48]]:
        U = [0]
        val = 0
        for gap in gap_pattern * 4:  # Repeat pattern
            val += gap
            U.append(val)
            if val > 100000:
                break
        candidates.append(U)
        
    # 12. Random walks with varying step sizes
    for seed in [101, 202, 303]:
        random.seed(seed)
        U = [0]
        val = 0
        for _ in range(20):
            step = random.randint(1, 20) * random.choice([1, -1])
            val += step
            if val > 0:
                U.append(val)
        U = sorted(set(U))
        candidates.append(U)

    # Evaluate and refine
    # Sort candidates by initial score to prioritize better starting points
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline - 30:
            break
        s = score_set(U)
        if s > 0:
            scored_candidates.append((s, U))
    scored_candidates.sort(reverse=True)
    
    # Process top candidates first
    # Allocate time budget per candidate based on remaining time
    num_candidates_to_process = min(len(scored_candidates), 30)
    time_per_candidate = (deadline - time.time() - 10) / num_candidates_to_process if num_candidates_to_process > 0 else 0
    
    for i, (score, U) in enumerate(scored_candidates[:num_candidates_to_process]):
        if time.time() > deadline - 10:
            break
        
        # Set a deadline for this specific local search run
        candidate_deadline = min(deadline - 10, time.time() + time_per_candidate * 1.5)
        
        refined, refined_score = local_search(U, candidate_deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # If we have time, do additional refinement on the best set
    if time.time() < deadline - 5:
        refined, refined_score = local_search(best_set, deadline - 1)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    return best_set
