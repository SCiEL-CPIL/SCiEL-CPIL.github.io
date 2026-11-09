"""Program for integer sum-product extremal sets using enhanced local search."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0  # Safety margin
    n = 20
    max_val = 10000
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy for speed."""
        A = np.array(sorted(A), dtype=np.int64)
        # Generate all pairs (i, j) with i <= j
        i_idx, j_idx = np.triu_indices(n)
        sums = A[i_idx] + A[j_idx]
        prods = A[i_idx] * A[j_idx]
        
        # Count unique values
        unique_sums = np.unique(sums)
        unique_prods = np.unique(prods)
        
        return max(len(unique_sums), len(unique_prods))

    best_set = None
    best_score = float('inf')
    
    # Generate high-quality initial candidates
    candidates = []
    
    # 1. Geometric Progressions
    for base in [1, 2, 3]:
        for ratio in [2, 3]:
            cand = []
            val = base
            for i in range(n):
                if val > max_val:
                    break
                cand.append(val)
                val *= ratio
            if len(cand) == n:
                candidates.append(cand)
                
    # 2. Arithmetic Progressions
    for start in [1, 2, 3, 5]:
        for step in [1, 2, 3, 4, 5, 10]:
            cand = [start + i * step for i in range(n)]
            if max(cand) <= max_val:
                candidates.append(cand)
                
    # 3. Perturbed APs (Hybrid)
    for start in [1, 2]:
        for step in [2, 3]:
            base_cand = [start + i * step for i in range(n)]
            for _ in range(5):
                cand = list(base_cand)
                for i in range(n):
                    if random.random() < 0.3:
                        offset = random.randint(-2, 2)
                        new_val = cand[i] + offset
                        if 1 <= new_val <= max_val and new_val not in cand:
                            cand[i] = new_val
                if len(set(cand)) == n:
                    candidates.append(sorted(cand))

    # 4. Random small integers
    for _ in range(50):
        cand = sorted(random.sample(range(1, 150), n))
        candidates.append(cand)
        
    # 5. Structured candidates: mix of small numbers and geometric
    # Try sets that have structure in both sums and products
    # e.g., powers of 2 with some perturbations
    for base in [1, 2, 3, 4, 5]:
        for _ in range(10):
            cand = []
            val = base
            for i in range(n):
                if val > max_val:
                    break
                cand.append(val)
                val = int(val * random.uniform(1.5, 2.5))
            if len(cand) == n:
                cand = sorted(set(cand))
                if len(cand) == n:
                    candidates.append(cand)
    
    # 6. Try sets with many collisions in sums
    # e.g., symmetric sets around a center
    for center in [10, 20, 50, 100]:
        for spread in [1, 2, 5, 10]:
            cand = []
            for i in range(n):
                offset = i - n // 2
                val = center + offset * spread
                if 1 <= val <= max_val:
                    cand.append(val)
            if len(set(cand)) == n:
                candidates.append(sorted(cand))
        
    # Evaluate candidates
    for cand in candidates:
        if len(set(cand)) != n or max(cand) > max_val:
            continue
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = list(cand)
            
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = evaluate(best_set)
        
    # Multi-start simulated annealing
    num_starts = 5
    current_set = list(best_set)
    current_score = best_score
    
    for start_iter in range(num_starts):
        # Reset current to best found so far (or random restart for later starts)
        if start_iter == 0:
            current_set = list(best_set)
            current_score = best_score
        else:
            # Random restart
            current_set = sorted(random.sample(range(1, 200), n))
            current_score = evaluate(current_set)
            if current_score < best_score:
                best_score = current_score
                best_set = list(current_set)
        
        T = 15.0  # Initial temperature
        cooling_rate = 0.998
        min_T = 0.5
        
        while time.time() < deadline:
            if T < min_T:
                T = min_T
                
            # Generate a neighbor
            move_type = random.random()
            new_set = list(current_set)
            
            if move_type < 0.4:
                # Nudge: change one element slightly
                idx = random.randint(0, n - 1)
                old_val = new_set[idx]
                for _ in range(5):
                    delta = random.randint(-15, 15)
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if 1 <= new_val <= max_val and new_val not in new_set:
                        new_set[idx] = new_val
                        break
                if new_set == current_set:
                    continue
                    
            elif move_type < 0.7:
                # Swap: exchange two elements
                i, j = random.sample(range(n), 2)
                new_set[i], new_set[j] = new_set[j], new_set[i]
                
            else:
                # Random Replace: change one element to a random value
                idx = random.randint(0, n - 1)
                old_val = new_set[idx]
                # Bias towards smaller values since they tend to work better
                if random.random() < 0.7:
                    new_val = random.randint(1, 100)
                else:
                    new_val = random.randint(1, 500)
                if new_val == old_val or new_val in new_set:
                    for _ in range(10):
                        if random.random() < 0.7:
                            new_val = random.randint(1, 100)
                        else:
                            new_val = random.randint(1, 500)
                        if new_val != old_val and new_val not in new_set:
                            break
                if new_val == old_val or new_val in new_set:
                    continue
                new_set[idx] = new_val

            # Evaluate neighbor
            new_score = evaluate(new_set)
            
            # Acceptance criterion
            delta_score = new_score - current_score
            if delta_score < 0:
                # Improvement
                current_set = new_set
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = list(current_set)
            else:
                # Worse, accept with probability exp(-delta/T)
                if random.random() < math.exp(-delta_score / T):
                    current_set = new_set
                    current_score = new_score
                    
            T *= cooling_rate
            
        # Check if we should continue with more starts
        if time.time() >= deadline - 5.0:
            break
            
    return best_set
# EVOLVE-BLOCK-END
