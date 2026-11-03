import time
import math
import random
import numpy as np

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

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin before 150s limit
    
    # Helper to check time
    def time_left():
        return time.time() < deadline
    
    random.seed(12345)
    
    # Phase 1: Generate diverse seeds based on geometric progressions and Golomb-like structures
    seeds = []
    
    # Geometric progressions with various bases
    for b in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]:
        for m in range(3, 15):
            U = [0]
            val = 1
            for k in range(1, m+1):
                val = int(round(b**k))
                if val not in U:
                    U.append(val)
            if len(U) >= 2:
                seeds.append(U)
    
    # Powers of 2
    for m in range(3, 16):
        U = [1 << i for i in range(m)]
        seeds.append(U)
    
    # "Almost" geometric with slight perturbations
    for base_b in [2.0, 2.5, 3.0, 4.0]:
        for m in range(4, 12):
            U = [0]
            val = 1
            for k in range(1, m+1):
                val = int(round(base_b**k * (1 + random.uniform(-0.1, 0.1))))
                val = max(1, val)
                if val not in U:
                    U.append(val)
            if len(U) >= 2:
                seeds.append(U)
    
    # Golomb-like sequences (distinct differences)
    for m in range(3, 14):
        U = [0]
        for k in range(1, m):
            # Golomb: a(n) = a(n-1) + a(n - a(n-1) - 1) + 1 (approximate)
            if len(U) >= 2:
                idx = max(0, len(U) - 1 - U[-1])
                new_val = U[-1] + U[idx] + 1
            else:
                new_val = U[-1] + 1
            if new_val not in U:
                U.append(new_val)
        if len(U) >= 2:
            seeds.append(U)
    
    # Binary linear independence seeds: select numbers with distinct bit patterns
    # to minimize sumset collisions (binary carries)
    for m in range(4, 16):
        U = [0]
        used_bits = set()
        for k in range(1, m+1):
            # Try to find a number with a new highest bit
            base = 1 << (k - 1)
            candidates = [base]
            # Add some variations
            for delta in range(-3, 4):
                candidates.append(base + delta)
            for c in candidates:
                if c > 0 and c not in U:
                    U.append(c)
                    break
        if len(U) >= 2:
            seeds.append(U)
    
    # Sparse binary patterns: numbers with few bits set
    for m in range(4, 14):
        U = [0]
        for k in range(1, m+1):
            # Use numbers with exactly 2 bits set
            for i in range(k):
                val = (1 << i) | (1 << (i + 2))
                if val > 0 and val not in U:
                    U.append(val)
                    break
        if len(U) >= 2:
            seeds.append(U)
    
    # Evaluate and sort seeds
    scored_seeds = []
    for U in seeds:
        U = sorted(set(U))
        if 0 not in U:
            U.insert(0, 0)
        s = fast_score(U)
        scored_seeds.append((s, U))
    
    scored_seeds.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top seeds
    top_seeds = [U for s, U in scored_seeds[:10]]
    
    best_set = top_seeds[0] if top_seeds else [0, 1, 3]
    best_score = fast_score(best_set)
    
    # Phase 2: Evolutionary Crossover - combine structural features from different parents
    # This can find non-constructive islands that greedy methods miss
    if time_left():
        # Take top diverse seeds for crossover
        parent_pool = [U for s, U in scored_seeds[:10]]
        
        for gen in range(5):
            if not time_left():
                break
            
            offspring_pool = []
            
            # Crossover between pairs of parents
            for i in range(len(parent_pool)):
                for j in range(i+1, len(parent_pool)):
                    if not time_left():
                        break
                    A = sorted(parent_pool[i])
                    B = sorted(parent_pool[j])
                    
                    # Uniform crossover: pick elements from A or B
                    max_len = max(len(A), len(B))
                    child = [0]
                    for k in range(1, max_len):
                        if random.random() < 0.5:
                            if k < len(A):
                                val = A[k]
                            else:
                                val = B[min(k, len(B)-1)]
                        else:
                            if k < len(B):
                                val = B[k]
                            else:
                                val = A[min(k, len(A)-1)]
                        if val > 0 and val not in child:
                            child.append(val)
                    
                    if len(child) >= 2:
                        child.sort()
                        s = fast_score(child)
                        if s > -1:
                            offspring_pool.append((s, child))
            
            # Add some random mutations of top parents
            for U in parent_pool[:5]:
                if not time_left():
                    break
                for _ in range(3):
                    child = list(U)
                    # Random mutation: modify 1-2 elements
                    for _ in range(random.randint(1, 2)):
                        if len(child) > 1:
                            idx = random.randint(1, len(child)-1)
                            old = child[idx]
                            new_val = max(1, old + random.randint(-5, 5))
                            if new_val not in child:
                                child[idx] = new_val
                    child.sort()
                    s = fast_score(child)
                    if s > -1:
                        offspring_pool.append((s, child))
            
            # Add offspring to pool and update best
            if offspring_pool:
                offspring_pool.sort(key=lambda x: x[0], reverse=True)
                for s, U in offspring_pool[:5]:
                    if s > best_score:
                        best_score = s
                        best_set = U[:]
                # Merge into parent pool for next generation
                parent_pool = [U for s, U in offspring_pool[:5]] + parent_pool[:5]
                parent_pool = list(set(tuple(U) for U in parent_pool))
                parent_pool = [list(U) for U in parent_pool]
    
    # Phase 3: Beam Search with expansion
    beam = [(best_score, best_set)]
    for s, U in scored_seeds[1:10]:
        beam.append((s, U))
    
    beam.sort(key=lambda x: x[0], reverse=True)
    beam = beam[:12]
    
    max_size = 14  # Limit set size to keep computation fast
    
    for step in range(25):
        if not time_left():
            break
            
        new_beam = []
        for score, U in beam:
            if len(U) >= max_size:
                new_beam.append((score, U))
                continue
                
            max_u = max(U) if U else 0
            
            # Generate candidates
            candidates = set()
            # Small numbers
            for i in range(1, min(12, max_u + 1)):
                candidates.add(i)
            # Numbers near max
            for i in range(max(1, max_u - 8), max_u + 12):
                candidates.add(i)
            # Geometric-like extensions
            if max_u > 0:
                for factor in [1.5, 2.0, 2.5, 3.0]:
                    candidates.add(int(round(max_u * factor)))
            # Binary-like extensions: numbers with sparse bit patterns
            if max_u > 0:
                for i in range(1, 8):
                    candidates.add(1 << i)
                    candidates.add((1 << i) | (1 << (i - 1)))
            # Random candidates
            for _ in range(8):
                candidates.add(random.randint(1, max_u + 30))
            
            for c in candidates:
                if c in U:
                    continue
                new_U = U + [c]
                new_U.sort()
                s = fast_score(new_U)
                if s > -1:
                    new_beam.append((s, new_U))
        
        if not new_beam:
            break
            
        # Sort and prune
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:12]
        
        # Update global best
        if beam[0][0] > best_score:
            best_score = beam[0][0]
            best_set = beam[0][1]

    # Phase 4: Local Search (First Improvement)
    U = list(best_set)
    current_score = best_score
    
    improved = True
    while improved and time_left():
        improved = False
        n = len(U)
        max_u = max(U)
        
        # 1. Try modifying existing elements
        for i in range(1, n):  # Don't modify 0
            old_val = U[i]
            # Try nearby values with wider range
            for delta in range(-12, 13):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 0:
                    continue
                if new_val in U:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U.sort()
                s = fast_score(new_U)
                if s > current_score:
                    current_score = s
                    U = new_U
                    improved = True
                    break
            if improved:
                break
        
        # 2. Try swapping two elements
        if not improved and n >= 3:
            for i in range(1, n):
                for j in range(i+1, n):
                    if not time_left():
                        break
                    new_U = U[:]
                    new_U[i], new_U[j] = new_U[j], new_U[i]
                    new_U.sort()
                    if len(set(new_U)) != len(new_U):
                        continue
                    s = fast_score(new_U)
                    if s > current_score:
                        current_score = s
                        U = new_U
                        improved = True
                        break
                if improved:
                    break
        
        # 3. Try adding a new element
        if not improved and len(U) < max_size:
            best_add_score = -1
            best_add_val = -1
            candidates = set()
            for i in range(1, max_u + 20):
                candidates.add(i)
            # Geometric extensions
            for factor in [1.5, 2.0, 2.5, 3.0, 4.0]:
                candidates.add(int(round(max_u * factor)))
            # Binary extensions
            for i in range(1, 10):
                candidates.add(1 << i)
                candidates.add((1 << i) | (1 << (i - 1)))
            for _ in range(15):
                candidates.add(random.randint(1, max_u + 60))
                
            for c in candidates:
                if c in U:
                    continue
                new_U = U + [c]
                new_U.sort()
                s = fast_score(new_U)
                if s > best_add_score:
                    best_add_score = s
                    best_add_val = c
            
            if best_add_score > current_score:
                current_score = best_add_score
                U.append(best_add_val)
                U.sort()
                improved = True
        
        # 4. Try removing an element
        if not improved and len(U) > 3:
            best_rem_score = -1
            best_rem_idx = -1
            for i in range(1, n):  # Don't remove 0
                new_U = U[:i] + U[i+1:]
                s = fast_score(new_U)
                if s > best_rem_score:
                    best_rem_score = s
                    best_rem_idx = i
            
            if best_rem_score > current_score:
                current_score = best_rem_score
                U = U[:best_rem_idx] + U[best_rem_idx+1:]
                improved = True

        if current_score > best_score:
            best_score = current_score
            best_set = U[:]

    # Phase 5: Simulated Annealing
    U = np.array(best_set, dtype=np.int64)
    current_score = best_score
    
    T = 0.03
    T_min = 0.0005
    cooling = 0.998
    
    for iteration in range(80000):
        if not time_left():
            break
            
        if T < T_min:
            T = T_min
            
        n = len(U)
        move_type = random.random()
        
        if move_type < 0.45 and n > 2:
            # Modify
            idx = random.randint(1, n-1)
            old_val = U[idx]
            # Propose new value with Gaussian noise
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
                    
        elif move_type < 0.75 and n < max_size:
            # Add
            max_u = int(U.max())
            # Bias toward binary-sparse values
            if random.random() < 0.3:
                # Pick a power of 2 or near
                bit_pos = random.randint(1, min(15, max_u.bit_length() + 2))
                new_val = 1 << bit_pos
                if random.random() < 0.5:
                    new_val |= (1 << max(0, bit_pos - 1))
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
            # Remove
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

    # Final Polish: Tighter local search
    U = list(best_set)
    improved = True
    while improved and time_left():
        improved = False
        n = len(U)
        for i in range(1, n):
            old_val = U[i]
            for delta in range(-5, 6):
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
