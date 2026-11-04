import time
import math
import random
import numpy as np
from itertools import combinations

def compute_score(U):
    U = tuple(sorted(set(U)))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def fast_score(U):
    """Fast scoring using numpy for vectorized set size computation."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    if 0 not in U or len(U) < 2:
        return -1.0
    max_u = int(U.max())
    if max_u == 0:
        return -1.0
    
    n = len(U)
    
    # For small n, NumPy broadcasting is very fast
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    # Count unique elements using sort (faster than np.unique for small arrays)
    n_sums = len(np.unique(sums.ravel()))
    n_diffs = len(np.unique(diffs.ravel()))
    
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def generate_hamming_like_set(n_bits, num_codewords):
    """
    Generate a set inspired by Hamming code structure.
    Uses binary patterns with controlled Hamming weight to create
    large difference sets while minimizing sum collisions.
    
    Strategy:
    - Start with binary patterns of controlled weight
    - Map to integers
    - Apply local perturbations to break symmetry
    """
    if n_bits < 2:
        return [0, 1]
    
    # Generate candidate binary patterns
    candidates = set()
    candidates.add(0)  # Always include 0
    
    # Add patterns with controlled Hamming weight
    for weight in range(1, min(n_bits + 1, 4)):
        for bits in combinations(range(n_bits), weight):
            val = 0
            for b in bits:
                val |= (1 << b)
            candidates.add(val)
    
    # Also add some higher-weight patterns
    if num_codewords > len(candidates) and n_bits >= 4:
        for _ in range(20):
            val = random.getrandbits(n_bits)
            candidates.add(val)
    
    candidates = sorted(candidates)
    
    # Select the best subset by score
    if len(candidates) <= num_codewords:
        U = candidates
    else:
        # Greedy selection: add elements that maximize score increment
        U = [0]
        remaining = [c for c in candidates if c != 0]
        
        while len(U) < num_codewords and remaining:
            best_score = -1
            best_elem = None
            for elem in remaining:
                test_U = U + [elem]
                s = fast_score(test_U)
                if s > best_score:
                    best_score = s
                    best_elem = elem
            if best_elem is not None:
                U.append(best_elem)
                remaining.remove(best_elem)
            else:
                break
    
    # Apply small perturbations to break symmetry
    U = sorted(set(U))
    for i in range(1, len(U)):
        if random.random() < 0.3:
            delta = random.randint(-3, 3)
            new_val = U[i] + delta
            if new_val > 0 and new_val not in U:
                U[i] = new_val
    
    return sorted(set(U))

def generate_ldpc_structure(n_bits, density=0.3):
    """
    Generate a set inspired by LDPC code structure.
    Uses sparse binary patterns to create controlled difference sets.
    """
    if n_bits < 3:
        return [0, 1, 2]
    
    # Create sparse binary patterns
    U = {0}
    num_patterns = max(4, int(n_bits * 2))
    
    for _ in range(num_patterns):
        # Create a sparse pattern
        val = 0
        num_ones = random.randint(1, max(1, int(n_bits * density)))
        positions = random.sample(range(n_bits), num_ones)
        for pos in positions:
            val |= (1 << pos)
        U.add(val)
    
    # Ensure 0 is present
    U.add(0)
    
    return sorted(U)

def local_search(U, deadline, max_steps=500, step_range=5):
    """
    Local search to improve the score of U.
    Tries to move each element by a small amount.
    """
    U = sorted(set(U))
    if 0 not in U:
        U.insert(0, 0)
    U = sorted(U)
    
    current_score = fast_score(U)
    if current_score <= 0:
        return U, current_score

    n = len(U)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        improved = False
        indices = list(range(1, n))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            old_val = U[i]
            low = max(0, old_val - step_range)
            high = old_val + step_range
            
            best_new_val = old_val
            best_new_score = current_score
            
            candidates = set()
            candidates.add(old_val - 1)
            candidates.add(old_val + 1)
            if step_range > 1:
                for _ in range(10):
                    candidates.add(random.randint(low, high))
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                if new_val < 0:
                    continue
                
                new_U = U[:i] + [new_val] + U[i+1:]
                if len(set(new_U)) != len(new_U):
                    continue
                
                s = fast_score(new_U)
                if s > best_new_score:
                    best_new_score = s
                    best_new_val = new_val
            
            if best_new_val != old_val:
                U[i] = best_new_val
                U = sorted(U)
                current_score = best_new_score
                improved = True
        
        if not improved:
            step_range += 2
            if step_range > 20:
                break
                
    return U, current_score

def generate_random_set(n, max_val):
    """Generate a random set of size n containing 0."""
    if n == 1:
        return [0]
    if max_val < n:
        return list(range(n))
        
    S = {0}
    while len(S) < n:
        x = random.randint(1, max_val)
        S.add(x)
    return sorted(S)

def generate_geometric_set(m, b):
    """Generate {0, 1, b, b^2, ...}"""
    U = [0, 1]
    cur = 1
    for i in range(m-1):
        cur = int(cur * b)
        if cur in U:
            cur += 1
            while cur in U:
                cur += 1
        U.append(cur)
    return sorted(set(U))

def construct():
    start = time.time()
    # Evaluator kills at 150s. Keep margin.
    deadline = start + 120
    
    best_set = [0, 1, 2, 4, 8]
    best_score = fast_score(best_set)
    
    random.seed(42)
    
    # Phase 1: Explore small sets with local search
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        strategy = random.random()
        
        if strategy < 0.3:
            n = random.randint(3, 8)
            max_val = random.randint(10, 100)
            U = generate_random_set(n, max_val)
        elif strategy < 0.5:
            n = random.randint(3, 7)
            b = random.uniform(1.5, 3.0)
            U = generate_geometric_set(n, b)
            for i in range(1, len(U)):
                if random.random() < 0.3:
                    U[i] += random.randint(-2, 2)
                    if U[i] < 0: U[i] = 0
            U = sorted(set(U))
        elif strategy < 0.7:
            # Hamming code inspired
            n_bits = random.randint(4, 10)
            num_codewords = random.randint(4, 12)
            U = generate_hamming_like_set(n_bits, num_codewords)
        elif strategy < 0.85:
            # LDPC structure
            n_bits = random.randint(5, 12)
            density = random.uniform(0.2, 0.4)
            U = generate_ldpc_structure(n_bits, density)
        else:
            # Golomb-like
            n = random.randint(3, 8)
            U = [0, 1]
            for i in range(1, n-1):
                if len(U) >= 2:
                    idx = max(0, len(U) - 1 - U[-1])
                    new_val = U[-1] + U[idx] + 1
                else:
                    new_val = U[-1] + 1
                if new_val not in U:
                    U.append(new_val)
        
        U, score = local_search(U, deadline, max_steps=100, step_range=5)
        
        if score > best_score:
            best_score = score
            best_set = U[:]
    
    # Phase 2: Simulated Annealing on the best set found
    U = np.array(best_set, dtype=np.int64)
    current_score = best_score
    
    T = 0.03
    T_min = 0.0005
    cooling = 0.998
    
    max_size = 15
    
    for iteration in range(50000):
        if time.time() > deadline - 2:
            break
            
        if T < T_min:
            T = T_min
            
        n = len(U)
        move_type = random.random()
        
        if move_type < 0.40 and n > 2:
            idx = random.randint(1, n-1)
            old_val = U[idx]
            sigma = max(1, old_val * 0.15)
            new_val = max(0, int(old_val + random.gauss(0, sigma)))
            if new_val == old_val:
                continue
            new_U = U.copy()
            new_U[idx] = new_val
            if len(set(new_U)) != len(new_U):
                continue
            s = fast_score(new_U)
            delta = s - current_score
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                U = new_U
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = sorted(U.tolist())
                    
        elif move_type < 0.70 and n < max_size:
            max_u = int(U.max())
            if random.random() < 0.4:
                # Binary-structured addition
                bit_pos = random.randint(1, min(15, max_u.bit_length() + 3))
                new_val = 1 << bit_pos
                if random.random() < 0.5:
                    new_val |= (1 << max(0, bit_pos - 1))
                if random.random() < 0.3:
                    new_val |= (1 << max(0, bit_pos - 2))
            elif random.random() < 0.3:
                # Sparse binary pattern
                num_ones = random.randint(1, 3)
                bit_len = max(4, max_u.bit_length() + 1)
                positions = random.sample(range(bit_len), num_ones)
                new_val = 0
                for pos in positions:
                    new_val |= (1 << pos)
            else:
                new_val = random.randint(1, max_u + 25)
            if new_val in U:
                continue
            new_U = np.append(U, new_val)
            new_U = np.sort(new_U)
            s = fast_score(new_U)
            delta = s - current_score
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                U = new_U
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = sorted(U.tolist())
                    
        else:
            if n > 3:
                idx = random.randint(1, n-1)
                new_U = np.delete(U, idx)
                s = fast_score(new_U)
                delta = s - current_score
                if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                    U = new_U
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_set = sorted(U.tolist())
                        
        T *= cooling
    
    # Phase 3: Binary-structure refinement
    # Try to optimize the binary representation of elements
    U = list(best_set)
    for _ in range(50):
        if time.time() > deadline - 3:
            break
        
        improved = False
        n = len(U)
        if n < 2:
            break
        
        # For each element, try flipping individual bits
        for i in range(1, n):
            old_val = U[i]
            bit_len = max(4, old_val.bit_length() + 2)
            
            for bit in range(bit_len):
                if time.time() > deadline - 3:
                    break
                
                # Try setting the bit
                new_val = old_val | (1 << bit)
                if new_val == old_val or new_val in U:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U.sort()
                s = fast_score(new_U)
                if s > best_score:
                    best_score = s
                    U = new_U
                    improved = True
                    break
                
                # Try clearing the bit
                new_val = old_val & ~(1 << bit)
                if new_val == old_val or new_val < 0 or new_val in U:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U.sort()
                s = fast_score(new_U)
                if s > best_score:
                    best_score = s
                    U = new_U
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Final Polish
    U = list(best_set)
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        n = len(U)
        for i in range(1, n):
            old_val = U[i]
            for delta in range(-8, 9):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 0 or new_val in U:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U.sort()
                s = fast_score(new_U)
                if s > best_score:
                    best_score = s
                    U = new_U
                    improved = True
                    break
            if improved:
                break
        if improved:
            best_set = U[:]

    return best_set

if __name__ == "__main__":
    res = construct()
    print(compute_score(res))
