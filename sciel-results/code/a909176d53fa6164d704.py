import time
import math

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Start with a good base set
    # Try geometric-like sets and structured sets
    best_score = 0
    best_U = [0, 1, 3]
    
    # Evaluate the seed
    s = compute_score(best_U)
    if s > best_score:
        best_score = s
    
    # Try various structured sets
    # Powers of 2
    U_powers2 = [0] + [2**i for i in range(1, 12)]  # 0,1,2,4,8,...,2048
    s = compute_score(U_powers2)
    if s > best_score and (time.time() - start_time) < deadline:
        best_score = s
        best_U = U_powers2[:]
    
    # Powers of 3
    U_powers3 = [0] + [3**i for i in range(1, 8)]  # 0,1,3,9,27,81,243,729
    s = compute_score(U_powers3)
    if s > best_score and (time.time() - start_time) < deadline:
        best_score = s
        best_U = U_powers3[:]
    
    # Mixed: combine small dense set with sparse large elements
    # Try: {0,1,2,...,k} union {large elements}
    for k in range(2, 15):
        if time.time() - start_time > deadline:
            break
        base = list(range(k + 1))
        # Add some large sparse elements
        for num_large in range(1, 8):
            if time.time() - start_time > deadline:
                break
            for step in [3, 5, 7, 11]:
                if time.time() - start_time > deadline:
                    break
                U = base[:]
                max_base = k
                for j in range(1, num_large + 1):
                    val = max_base + j * step
                    U.append(val)
                s = compute_score(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
    
    # Try arithmetic progression with gaps
    for n in range(5, 30):
        if time.time() - start_time > deadline:
            break
        for gap in [2, 3, 5]:
            if time.time() - start_time > deadline:
                break
            U = [i * gap for i in range(n)]
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Try to optimize: start from best_U and try adding/removing elements
    # Greedy improvement
    current_U = best_U[:]
    current_score = best_score
    improved = True
    max_iterations = 50
    
    for iteration in range(max_iterations):
        if time.time() - start_time > deadline:
            break
        if not improved:
            break
        improved = False
        
        # Try adding new elements
        max_val = max(current_U) if current_U else 0
        # Try values from 1 to max_val * 2
        candidates = set()
        for v in range(1, max_val * 2 + 1):
            candidates.add(v)
        
        best_candidate = None
        best_candidate_score = current_score
        
        # Sample candidates to keep it fast
        import random
        random.seed(42)
        candidate_list = list(candidates)
        if len(candidate_list) > 200:
            candidate_list = random.sample(candidate_list, 200)
        
        for v in candidate_list:
            if v in current_U:
                continue
            new_U = current_U + [v]
            s = compute_score(new_U)
            if s > best_candidate_score + 1e-9:
                best_candidate_score = s
                best_candidate = v
        
        if best_candidate is not None:
            current_U.append(best_candidate)
            current_score = best_candidate_score
            best_U = current_U[:]
            best_score = current_score
            improved = True
    
    return best_U
# EVOLVE-BLOCK-END
