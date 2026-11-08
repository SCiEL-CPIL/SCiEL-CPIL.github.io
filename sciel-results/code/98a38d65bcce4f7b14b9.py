import time
from math import log
import random
import numpy as np

def score_set(U):
    """Compute the score for a set U using efficient bitset operations."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    max_u = max(U)
    
    # Compute sumset using bitset convolution
    sum_mask = 1 << 0
    for x in U:
        sum_mask |= (sum_mask << x)
    
    # Count bits in sum_mask
    sum_count = bin(sum_mask).count('1')
    
    # Compute difference set
    diff_mask = 1 << max_u  # represent 0 as bit at position max_u
    for i in range(n):
        for j in range(i + 1, n):
            d = U[j] - U[i]
            diff_mask |= (1 << (max_u + d))
            diff_mask |= (1 << (max_u - d))
    
    diff_count = bin(diff_mask).count('1')
    
    if sum_count == 0 or diff_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def random_seed(n_min=3, n_max=8, max_val=100):
    """Generate a random seed set containing 0."""
    n = random.randint(n_min, n_max)
    while True:
        elements = [0] + sorted(random.sample(range(1, max_val + 1), n - 1))
        if len(set(elements)) == n:
            return elements

def structured_seeds():
    """Generate diverse structured seed sets."""
    seeds = []
    
    # Powers of 2
    for m in range(3, 10):
        seeds.append([0] + [2**k for k in range(m)])
    
    # Squares
    for m in range(3, 10):
        seeds.append([0] + [k**2 for k in range(1, m + 1)])
    
    # Fibonacci-like
    fib = [0, 1]
    for _ in range(12):
        fib.append(fib[-1] + fib[-2])
    for m in range(3, 10):
        seeds.append([0] + fib[1:m + 1])
    
    # Random structured
    for _ in range(5):
        base = random.choice([2, 3, 4])
        m = random.randint(4, 12)
        vals = [0]
        for k in range(1, m + 1):
            vals.append(int(base ** k * random.uniform(0.5, 1.5)))
        vals = sorted(set([v for v in vals if v >= 0]))
        if len(vals) >= 2:
            seeds.append(vals)
    
    return seeds

def simulated_annealing(U, deadline, start_temp=1.0, cool_rate=0.95):
    """Simulated annealing local search."""
    U = sorted(set(U))
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    temperature = start_temp
    max_val = max(U) * 2 + 50
    
    while time.time() < deadline - 3 and temperature > 0.001:
        n = len(U)
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4 and n < 15:
            # Add an element
            new_val = random.randint(1, max_val)
            if new_val in U:
                continue
            new_U = sorted(U + [new_val])
        elif move_type < 0.7 and n > 2:
            # Remove an element
            idx = random.randint(1, n - 1)
            new_U = [U[i] for i in range(n) if i != idx]
        else:
            # Replace an element
            if n < 2:
                continue
            idx = random.randint(1, n - 1)
            old_val = U[idx]
            new_val = max(1, old_val + random.randint(-10, 10))
            new_set = [U[i] if i != idx else new_val for i in range(n)]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
            new_U = new_set
        
        new_score = score_set(new_U)
        delta = new_score - current_score
        
        # Accept if improvement or with probability exp(-delta/T)
        if delta > 0 or random.random() < np.exp(-delta / temperature):
            U = new_U
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
        
        temperature *= cool_rate
    
    return best_score, best_set

def multi_start_search(deadline, n_starts=None):
    """Run multiple independent searches from different seeds."""
    start_time = time.time()
    
    if n_starts is None:
        # Estimate number of starts based on time
        n_starts = max(5, int((deadline - start_time - 10) / 5))
    
    all_seeds = structured_seeds()
    for _ in range(10):
        all_seeds.append(random_seed())
    
    best_score = -1
    best_set = None
    
    for i in range(n_starts):
        if time.time() > deadline - 5:
            break
        
        seed = all_seeds[i % len(all_seeds)]
        local_score, local_set = simulated_annealing(
            seed, 
            deadline - 5,
            start_temp=random.uniform(0.5, 2.0),
            cool_rate=random.uniform(0.9, 0.98)
        )
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
    
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_score, best_set = multi_start_search(deadline)
    
    # Final polish: try small perturbations on the best set
    if best_set and time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        polish_deadline = deadline - 1
        
        while improved and time.time() < polish_deadline:
            improved = False
            n = len(current)
            
            # Try swapping two elements
            for i in range(1, n):
                if time.time() > polish_deadline:
                    break
                for j in range(i + 1, n):
                    old_i, old_j = current[i], current[j]
                    new_i, new_j = old_j, old_i
                    new_set = sorted(set([current[k] if k != i and k != j else (new_i if k == i else new_j) for k in range(n)]))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try adding a new element near existing ones
                if n < 12:
                    for x in current:
                        for delta in range(-5, 6):
                            new_val = x + delta
                            if new_val <= 0 or new_val in current:
                                continue
                            new_set = sorted(current + [new_val])
                            new_score = score_set(new_set)
                            if new_score > current_score:
                                current = new_set
                                current_score = new_score
                                improved = True
                                break
                        if improved:
                            break
            
            if not improved and n > 3:
                # Try removing an element
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
        
        if current_score > best_score:
            best_score = current_score
            best_set = current
    
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
