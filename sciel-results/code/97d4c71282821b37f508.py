import math
import time
import random

def _score_of(U):
    """Fast score using set operations for sum/difference sets."""
    if not U:
        return 0.0
    u_list = sorted(set(U))
    if 0 not in u_list:
        return 0.0
    max_u = u_list[-1]
    if max_u == 0:
        return 0.0
    n = len(u_list)
    if n < 2:
        return 0.0
    # Use set comprehension for speed
    sums = {u_list[i] + u_list[j] for i in range(n) for j in range(n)}
    diffs = {u_list[i] - u_list[j] for i in range(n) for j in range(n)}
    num_sums = len(sums)
    num_diffs = len(diffs)
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return 1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / log_denom

def _simulated_annealing(U, deadline, max_iter=2000):
    """Simulated annealing to escape local optima."""
    current = sorted(set(U))
    current_score = _score_of(current)
    best = list(current)
    best_score = current_score
    n = len(current)
    
    temperature = 0.05
    cooling = 0.995
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if temperature < 0.001:
            break
            
        # Random move: perturb, add, remove, or swap
        move_type = random.randint(0, 3)
        candidate = list(current)
        
        if move_type == 0 and n > 1:
            # Perturb a random element
            i = random.randint(1, n - 1)
            old_val = candidate[i]
            # Try various perturbations
            perturbations = []
            for mult in [0.5, 0.7, 1.3, 1.5, 2.0, 0.3, 3.0]:
                new_val = int(old_val * mult)
                if new_val > 0:
                    perturbations.append(new_val)
            for delta in range(-5, 6):
                if delta != 0:
                    new_val = old_val + delta
                    if new_val > 0:
                        perturbations.append(new_val)
            if perturbations:
                new_val = random.choice(perturbations)
                candidate[i] = new_val
                candidate = sorted(set(candidate))
                if len(candidate) != n:
                    continue
        elif move_type == 1 and n < 25:
            # Add a new element
            max_val = max(current)
            candidates_to_add = []
            for v in range(max_val + 1, max_val + 20):
                candidates_to_add.append(v)
            for _ in range(5):
                candidates_to_add.append(random.randint(max_val + 1, max_val * 2 + 10))
            # Also try filling gaps
            for i in range(1, n):
                gap = current[i] - current[i-1]
                if gap > 2:
                    candidates_to_add.append(random.randint(current[i-1]+1, current[i]-1))
            if candidates_to_add:
                new_val = random.choice(candidates_to_add)
                if new_val not in current:
                    candidate = sorted(current + [new_val])
                    n = len(candidate)
        elif move_type == 2 and n > 3:
            # Remove an element
            i = random.randint(1, n - 1)
            candidate = [current[j] for j in range(n) if j != i]
            n = len(candidate)
        elif move_type == 3 and n > 1:
            # Swap an element
            i = random.randint(1, n - 1)
            old_val = candidate[i]
            max_val = max(current)
            new_val = random.randint(1, max_val * 2 + 5)
            if new_val not in current:
                candidate[i] = new_val
                candidate = sorted(set(candidate))
                if len(candidate) != n:
                    continue
        
        if not candidate or 0 not in candidate:
            continue
            
        new_score = _score_of(candidate)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(delta/temperature) if worse
        if delta > 0 or random.random() < math.exp(delta / temperature):
            current = candidate
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = list(current)
        
        temperature *= cooling
    
    return best, best_score

def _local_search(U, deadline, max_steps=500):
    """Local search with add, remove, and swap moves."""
    current = sorted(set(U))
    best_score = _score_of(current)
    n = len(current)
    
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        improved = False
        
        # Try adding a new element
        if n < 25:
            max_val = max(current)
            candidates = []
            for v in range(max_val + 1, max_val + 30):
                candidates.append(v)
            for i in range(1, n):
                gap = current[i] - current[i-1]
                if gap > 3:
                    candidates.append(current[i-1] + gap // 2)
            for _ in range(5):
                candidates.append(random.randint(max_val + 1, max_val * 2 + 10))
            random.shuffle(candidates)
            for new_val in candidates:
                if new_val <= 0 or new_val in current:
                    continue
                candidate = sorted(current + [new_val])
                s = _score_of(candidate)
                if s > best_score:
                    best_score = s
                    current = candidate
                    n = len(current)
                    improved = True
                    break
        if improved:
            continue
        
        # Try removing an element
        if n > 3:
            for i in range(1, n):
                candidate = [current[j] for j in range(n) if j != i]
                s = _score_of(candidate)
                if s > best_score:
                    best_score = s
                    current = candidate
                    n = len(current)
                    improved = True
                    break
        if improved:
            continue
        
        # Try perturbing individual elements
        for i in range(1, n):
            if time.time() > deadline:
                break
            old_val = current[i]
            perturbations = []
            for mult in [0.5, 0.7, 1.3, 1.5, 2.0, 0.3, 3.0]:
                new_val = int(old_val * mult)
                if new_val > 0:
                    perturbations.append(new_val)
            for delta in range(-8, 9):
                if delta != 0:
                    new_val = old_val + delta
                    if new_val > 0:
                        perturbations.append(new_val)
            random.shuffle(perturbations)
            for new_val in perturbations:
                if new_val in current:
                    continue
                candidate = [current[j] if j != i else new_val for j in range(n)]
                candidate = sorted(set(candidate))
                if len(candidate) != n:
                    continue
                s = _score_of(candidate)
                if s > best_score:
                    best_score = s
                    current = candidate
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        # Try swapping
        for i in range(1, n):
            if time.time() > deadline:
                break
            old_val = current[i]
            max_val = max(current)
            for new_val in range(1, max_val * 2 + 10):
                if new_val in current or new_val == old_val:
                    continue
                candidate = [current[j] if j != i else new_val for j in range(n)]
                candidate = sorted(set(candidate))
                if len(candidate) != n:
                    continue
                s = _score_of(candidate)
                if s > best_score:
                    best_score = s
                    current = candidate
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
            
    return current, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    random.seed(42)
    
    # Generate diverse base candidates
    candidates = []
    
    # 1. Geometric sequences with various ratios
    for ratio in [1.3, 1.5, 1.7, 1.9, 2.0, 2.1, 2.3, 2.5, 2.8, 3.0, 3.5, 4.0]:
        U = [0]
        val = 1
        for _ in range(15):
            U.append(val)
            val = int(val * ratio)
            if val > 100000:
                break
        candidates.append(U)
    
    # 2. Powers of 2
    U = [0]
    val = 1
    for _ in range(15):
        U.append(val)
        val *= 2
    candidates.append(U)
    
    # 3. Fibonacci
    a, b = 0, 1
    U = [0, 1]
    while b < 50000:
        U.append(b)
        a, b = b, a + b
    candidates.append(U)
    
    # 4. Squares
    U = [i*i for i in range(15)]
    candidates.append(U)
    
    # 5. Cubes
    U = [i**3 for i in range(10)]
    candidates.append(U)
    
    # 6. Triangular numbers
    U = [i*(i+1)//2 for i in range(15)]
    candidates.append(U)
    
    # 7. Randomized exponential growth
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
        random.seed(seed)
        for _ in range(6):
            U = [0]
            val = random.randint(1, 8)
            for _ in range(random.randint(5, 14)):
                U.append(val)
                val = random.randint(int(val * 1.2), int(val * 3.5))
            candidates.append(U)
    
    # 8. Sparse random sets
    random.seed(999)
    for _ in range(15):
        n_elem = random.randint(6, 18)
        max_val = random.randint(100, 5000)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
    
    # 9. Mixed growth patterns
    for seed in [11, 22, 33, 44, 55, 66, 77]:
        random.seed(seed)
        U = [0]
        val = random.randint(1, 5)
        for _ in range(random.randint(8, 15)):
            if random.random() < 0.5:
                val = int(val * random.uniform(1.5, 2.5))
            else:
                val += random.randint(1, max(2, int(val * 0.3)))
            U.append(val)
        candidates.append(U)
    
    # 10. Sets with controlled gaps
    for gap_pattern in [[1, 2, 4, 8, 16], [2, 3, 5, 8, 13], [1, 3, 7, 15, 31], [2, 4, 8, 16, 32], [3, 5, 9, 17, 33]]:
        U = [0]
        val = 0
        for gap in gap_pattern * 3:
            val += gap
            U.append(val)
            if val > 50000:
                break
        candidates.append(U)
    
    # 11. Prime-based
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    U = [0] + [i for i in range(1, 200) if is_prime(i)]
    candidates.append(U)
    
    # Evaluate and sort candidates by initial score
    scored_candidates = []
    for U in candidates:
        s = _score_of(U)
        if s > 0:
            scored_candidates.append((s, U))
    scored_candidates.sort(reverse=True)
    
    best_set = scored_candidates[0][1] if scored_candidates else [0, 1, 3]
    best_score = scored_candidates[0][0] if scored_candidates else _score_of([0, 1, 3])
    
    # Refine top candidates
    for score, U in scored_candidates[:30]:
        if time.time() > deadline - 30:
            break
        
        # Local search
        refined, refined_score = _local_search(U, deadline - 10)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
        
        # Simulated annealing on the best so far
        if time.time() > deadline - 15:
            break
        sa_result, sa_score = _simulated_annealing(best_set, deadline - 5, max_iter=1500)
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_result
        
        # Final local search after SA
        if time.time() > deadline - 5:
            break
        final_refined, final_score = _local_search(best_set, deadline - 2, max_steps=300)
        if final_score > best_score:
            best_score = final_score
            best_set = final_refined
    
    # Final polish
    if time.time() < deadline - 2:
        final, final_s = _local_search(best_set, deadline - 1, max_steps=200)
        if final_s > best_score:
            best_set = final
            best_score = final_s
    
    return best_set
