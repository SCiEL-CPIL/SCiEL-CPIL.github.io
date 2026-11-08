import time
import math
import random
import numpy as np

def compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # Use numpy for speed
    arr = np.array(U, dtype=np.int64)
    sums = arr[:, None] + arr[None, :]
    diffs = arr[:, None] - arr[None, :]
    sum_count = len(np.unique(sums))
    diff_count = len(np.unique(diffs))
    
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / math.log(2 * maxU + 1)
    return score

def simulated_annealing(initial_set, deadline, max_iter=20000, T0=1.0, cooling=0.9995, min_T=1e-6):
    """Simulated annealing for set optimization"""
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U
    
    best_score = compute_score(U)
    best_set = U[:]
    current_score = best_score
    current_set = U[:]
    max_val = max(U)
    
    T = T0
    no_improve = 0
    max_no_improve = 2000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate a neighbor
        n = len(current_set)
        move_type = random.random()
        
        if move_type < 0.5:
            # Replace an element
            if n > 2:
                i = random.randint(1, n - 1)  # Don't replace 0
                # Try different replacement strategies
                r = random.random()
                if r < 0.3:
                    new_val = random.randint(0, max_val * 2)
                elif r < 0.6:
                    new_val = random.randint(0, max_val)
                elif r < 0.8:
                    new_val = max(0, current_set[i] + random.randint(-5, 5))
                else:
                    new_val = current_set[i] * random.choice([2, 3]) if current_set[i] > 0 else random.randint(1, max_val)
                
                if new_val in current_set and new_val != current_set[i]:
                    continue
                
                new_set = sorted(set(current_set[:i] + [new_val] + current_set[i+1:]))
                if 0 not in new_set:
                    new_set = sorted([0] + [x for x in new_set if x > 0])
                if len(new_set) < 2:
                    continue
                new_set = new_set  # Keep as is
                
            else:
                continue
                
        elif move_type < 0.7:
            # Add an element
            if len(current_set) < 40:
                r = random.random()
                if r < 0.4:
                    new_val = max_val + random.randint(1, max(1, max_val // 3))
                elif r < 0.7:
                    new_val = random.randint(1, max_val * 2)
                else:
                    new_val = random.randint(1, max_val)
                if new_val in current_set:
                    continue
                new_set = sorted(current_set + [new_val])
            else:
                continue
                
        elif move_type < 0.85:
            # Remove an element
            if len(current_set) > 3:
                idx = random.choice([i for i in range(len(current_set)) if current_set[i] != 0])
                new_set = current_set[:idx] + current_set[idx+1:]
            else:
                continue
                
        else:
            # Swap two elements
            if len(current_set) > 3:
                i, j = random.sample(range(1, len(current_set)), 2)
                temp = current_set[i]
                new_set = current_set[:]
                new_set[i] = random.randint(0, max_val * 2)
                new_set[j] = random.randint(0, max_val * 2)
                new_set = sorted(set(new_set))
                if 0 not in new_set:
                    new_set = sorted([0] + [x for x in new_set if x > 0])
                if len(new_set) < 2:
                    continue
            else:
                continue
        
        # Evaluate new solution
        new_score = compute_score(new_set)
        delta = new_score - current_score
        
        # Accept or reject
        if delta > 0 or random.random() < math.exp(delta / max(T, min_T)):
            current_set = new_set
            current_score = new_score
            max_val = max(current_set)
            
            if current_score > best_score + 1e-12:
                best_score = current_score
                best_set = current_set[:]
                no_improve = 0
            else:
                no_improve += 1
        
        else:
            no_improve += 1
        
        # Cool down
        T = max(T * cooling, min_T)
        
        if no_improve > max_no_improve:
            break
    
    return best_score, best_set

def multi_start_sa(deadline, num_starts=10, sa_iter=5000):
    """Run simulated annealing from multiple starting points"""
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Generate diverse starting points
    starts = []
    
    # Structured starts
    structured = [
        [0, 1, 3, 9],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 7, 15],
        [0, 1, 4, 9, 16, 25],
        [0, 1, 2, 3, 5, 8, 13],
        [0, 1, 3, 6, 10, 15, 21],
        [0, 2, 4, 6, 10, 15],
        [0, 1, 5, 10, 15, 20],
    ]
    starts.extend(structured)
    
    # Random starts
    for seed in range(8):
        random.seed(seed)
        n = random.randint(3, 12)
        U = sorted(random.sample(range(1, 100), n - 1) + [0])
        starts.append(U)
    
    # Geometric starts
    for base in [2, 3, 4, 5]:
        for n in range(3, 8):
            U = [0] + [base**k for k in range(1, n)]
            starts.append(U)
    
    # Run SA from each start
    for i, start in enumerate(starts):
        if time.time() > deadline - 5:
            break
        
        s, U = simulated_annealing(start, deadline, max_iter=sa_iter, T0=1.0, cooling=0.9998, min_T=1e-6)
        if s > best_score:
            best_score = s
            best_set = U
    
    return best_score, best_set

def hill_climb_polish(initial_set, deadline, max_iter=3000):
    """Local search polish with greedy moves"""
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U
    
    best_score = compute_score(U)
    best_set = U[:]
    max_val = max(U)
    stagnant = 0
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        n = len(U)
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            candidates = set()
            for _ in range(8):
                candidates.add(random.randint(0, max_val * 2))
            candidates.add(max(0, U[i] - 1))
            candidates.add(U[i] + 1)
            if U[i] > 0:
                candidates.add(max(0, U[i] // 2))
                candidates.add(U[i] * 2)
            if i > 0 and i < n - 1:
                candidates.add((U[i-1] + U[i+1]) // 2)
            
            for new_val in candidates:
                if time.time() > deadline:
                    break
                if new_val in U and new_val != U[i]:
                    continue
                new_U = sorted(set(U[:i] + [new_val] + U[i+1:]))
                if 0 not in new_U:
                    new_U = sorted([0] + [x for x in new_U if x > 0])
                if len(new_U) < 2:
                    continue
                s = compute_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    max_val = max(U)
                    improved = True
                    stagnant = 0
                    break
        
        if not improved:
            stagnant += 1
            
            # Try adding
            if len(U) < 50 and stagnant < 15:
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    r = random.random()
                    if r < 0.3:
                        new_val = max_val + random.randint(1, max(1, max_val // 4))
                    elif r < 0.6:
                        new_val = random.randint(1, max_val * 3)
                    else:
                        new_val = random.randint(1, max_val)
                    if new_val in U:
                        continue
                    new_U = sorted(U + [new_val])
                    s = compute_score(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_set = new_U
                        U = new_U
                        max_val = max(U)
                        improved = True
                        stagnant = 0
                        break
            
            # Try removing
            if not improved and len(U) > 3 and stagnant < 15:
                for _ in range(6):
                    if time.time() > deadline:
                        break
                    idx = random.choice([i for i in range(len(U)) if U[i] != 0])
                    new_U = U[:idx] + U[idx+1:]
                    s = compute_score(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_set = new_U
                        U = new_U
                        max_val = max(U)
                        improved = True
                        stagnant = 0
                        break
            
            if not improved and stagnant > 50:
                break
    
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Phase 1: Multi-start simulated annealing
    if time.time() < deadline - 30:
        s, U = multi_start_sa(deadline - 30, num_starts=15, sa_iter=4000)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Phase 2: Hill climb polish from best found
    if time.time() < deadline - 10:
        s, U = hill_climb_polish(best_set, deadline - 10, max_iter=5000)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Phase 3: More SA from diverse starts with remaining time
    remaining = deadline - time.time()
    if remaining > 15:
        random.seed(int(time.time() * 1000) % 10000)
        for _ in range(min(5, int(remaining / 5))):
            if time.time() > deadline - 3:
                break
            n = random.randint(3, 10)
            U = sorted(random.sample(range(1, 50), n - 1) + [0])
            s, U2 = simulated_annealing(U, deadline - 3, max_iter=3000, T0=0.5, cooling=0.9997, min_T=1e-6)
            if s > best_score:
                best_score = s
                best_set = U2
    
    # Final polish
    if time.time() < deadline - 2:
        s, U = hill_climb_polish(best_set, deadline - 2, max_iter=2000)
        if s > best_score:
            best_score = s
            best_set = U
    
    return best_set

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
