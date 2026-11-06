import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 90s limit, 5s margin
    
    n = 128
    target_start = 128
    target_end = 256
    
    def evaluate(A_set):
        """
        Returns (max_mult, num_uncovered, total_pairs)
        """
        if not A_set:
            return 1000, 100, 0
        A_list = sorted(A_set)
        
        v = np.zeros(129, dtype=np.int32)
        for x in A_list:
            if 0 <= x <= 128:
                v[x] = 1
            else:
                return 1000, 100, 0
        
        conv = np.convolve(v, v, mode='full')
        
        max_mult = 0
        num_uncovered = 0
        
        for t in range(target_start, target_end + 1):
            ordered = conv[t]
            diag = 1 if (t % 2 == 0 and (t // 2) in A_set) else 0
            unordered = (ordered + diag) // 2
            
            if unordered == 0:
                num_uncovered += 1
            elif unordered > max_mult:
                max_mult = unordered
                
        total_pairs = len(A_list) * (len(A_list) + 1) // 2
        return max_mult, num_uncovered, total_pairs

    def admm_solve(rho=1.0, max_iters=500, tol=1e-6):
        """
        ADMM for BQP: minimize max multiplicity subject to coverage.
        Variables: x in {0,1}^129 (indicator for A)
        Constraints: for each t in [128,256], sum_{a+b=t} x_a x_b >= 1
                     (using unordered pairs)
        Objective: minimize max_t (sum_{a+b=t} x_a x_b)
        
        We relax to continuous [0,1] and use ADMM with penalty.
        """
        # Precompute pair indices for each target
        # For target t, unordered pairs (a,b) with a<=b, a+b=t, 0<=a,b<=128
        target_pairs = {}
        for t in range(target_start, target_end + 1):
            pairs = []
            for a in range(0, 129):
                b = t - a
                if b < a:
                    continue
                if 0 <= b <= 128:
                    pairs.append((a, b))
            target_pairs[t] = pairs
        
        # Initialize x in [0,1]
        x = np.zeros(129, dtype=np.float64)
        x[128] = 1.0  # Must have 128 for 256=128+128
        
        # ADMM variables
        # We'll use a simple approach: minimize ||x - z||^2 + rho * penalty
        # where z is the binary projection
        
        best_x = x.copy()
        best_score = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Compute multiplicities for current x
            mults = np.zeros(257, dtype=np.float64)
            for a in range(129):
                for b in range(a, 129):
                    t = a + b
                    if target_start <= t <= target_end:
                        # Unordered: if a==b, count once; else count once for (a,b)
                        # But we want the actual multiplicity
                        if a == b:
                            mults[t] += x[a] * x[b]
                        else:
                            # (a,b) and (b,a) are the same unordered pair
                            # In the sum over unordered pairs, we count it once
                            mults[t] += x[a] * x[b]
            
            # Coverage check
            uncovered = 0
            for t in range(target_start, target_end + 1):
                if mults[t] < 0.5:  # Not covered
                    uncovered += 1
            
            # Max multiplicity
            max_mult = np.max(mults[target_start:target_end+1])
            
            # Score for tracking
            score = uncovered * 10000 + max_mult
            if score < best_score:
                best_score = score
                best_x = x.copy()
            
            # ADMM update: x_{k+1} = argmin_x (1/2)||x - z_k||^2 + rho * (uncovered_penalty + max_penalty)
            # This is complex; use a simpler subgradient approach
            # For each element a, compute gradient-like update
            
            # Simpler: use projected gradient descent on the relaxed problem
            # Objective: sum_t max(0, 1 - mult_t)^2 + lambda * max(mults)
            # Gradient with respect to x_a
            
            # Let's use a different approach: iterative refinement
            # Try to add/remove elements to improve coverage and reduce max
            
            # Find elements to potentially add
            candidates = list(range(129))
            random.shuffle(candidates)
            
            improved = False
            for a in candidates:
                if time.time() > deadline - 5:
                    break
                if x[a] > 0.9:
                    continue  # Already in set
                
                # Try adding a
                x_test = x.copy()
                x_test[a] = 1.0
                
                # Evaluate
                mults_test = np.zeros(257, dtype=np.float64)
                for i in range(129):
                    if x_test[i] < 0.5:
                        continue
                    for j in range(i, 129):
                        if x_test[j] < 0.5:
                            continue
                        t = i + j
                        if target_start <= t <= target_end:
                            mults_test[t] += 1.0
                
                unc_test = sum(1 for t in range(target_start, target_end+1) if mults_test[t] < 0.5)
                max_test = np.max(mults_test[target_start:target_end+1])
                score_test = unc_test * 10000 + max_test
                
                if score_test < score:
                    x = x_test
                    score = score_test
                    improved = True
                    break
            
            if not improved:
                # Try removing elements
                for a in candidates:
                    if time.time() > deadline - 5:
                        break
                    if x[a] < 0.1:
                        continue  # Not in set
                    
                    # Try removing a
                    x_test = x.copy()
                    x_test[a] = 0.0
                    
                    mults_test = np.zeros(257, dtype=np.float64)
                    for i in range(129):
                        if x_test[i] < 0.5:
                            continue
                        for j in range(i, 129):
                            if x_test[j] < 0.5:
                                continue
                            t = i + j
                            if target_start <= t <= target_end:
                                mults_test[t] += 1.0
                    
                    unc_test = sum(1 for t in range(target_start, target_end+1) if mults_test[t] < 0.5)
                    max_test = np.max(mults_test[target_start:target_end+1])
                    score_test = unc_test * 10000 + max_test
                    
                    if score_test <= score:
                        x = x_test
                        score = score_test
                        improved = True
                        break
            
            if not improved:
                # Stagnation: try random perturbation
                a = random.randint(0, 128)
                if x[a] > 0.5:
                    x[a] = 0.0
                else:
                    x[a] = 1.0
        
        # Convert to set
        A_set = set()
        for i in range(129):
            if best_x[i] > 0.5:
                A_set.add(i)
        
        if 128 not in A_set:
            A_set.add(128)
        
        return A_set

    def local_search_polish(A_set, max_iters=500):
        """Local search to polish the solution."""
        current_A = set(A_set)
        cur_max, cur_unc, _ = evaluate(current_A)
        cur_cost = cur_unc * 10000 + cur_max
        best_A = set(current_A)
        best_cost = cur_cost
        
        T = 5.0
        T_min = 0.1
        cooling = 0.98
        
        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            move_type = random.random()
            new_A = set(current_A)
            
            if move_type < 0.4 and len(current_A) > 2:
                x = random.choice(list(current_A))
                new_A.remove(x)
            elif move_type < 0.8:
                x = random.randint(0, 128)
                if x in current_A:
                    continue
                new_A.add(x)
            else:
                if len(current_A) < 1:
                    continue
                x = random.choice(list(current_A))
                y = random.randint(0, 128)
                if y == x:
                    continue
                new_A.remove(x)
                new_A.add(y)
            
            if len(new_A) == 0:
                continue
            
            new_max, new_unc, _ = evaluate(new_A)
            new_cost = new_unc * 10000 + new_max
            delta = new_cost - cur_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_A = new_A
                cur_cost = new_cost
                cur_max, cur_unc, _ = evaluate(current_A)
                
                if cur_cost < best_cost:
                    best_A = set(current_A)
                    best_cost = cur_cost
            
            T = max(T * cooling, T_min)
        
        return best_A

    # Run ADMM with different initializations
    best_overall_A = set()
    best_overall_cost = float('inf')
    
    # Multiple restarts with different random seeds
    for restart in range(10):
        if time.time() > deadline - 10:
            break
        
        # Initialize with a random subset
        init_A = set()
        init_A.add(128)  # Must have 128
        for _ in range(random.randint(15, 30)):
            init_A.add(random.randint(0, 128))
        
        # Run ADMM
        admm_A = admm_solve(max_iters=200)
        
        # Polish with local search
        polished_A = local_search_polish(admm_A, max_iters=100)
        
        c, u, _ = evaluate(polished_A)
        cost = u * 10000 + c
        
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_A = polished_A
        
        # Also try polishing the initial set
        polished_init = local_search_polish(init_A, max_iters=100)
        c2, u2, _ = evaluate(polished_init)
        cost2 = u2 * 10000 + c2
        if cost2 < best_overall_cost:
            best_overall_cost = cost2
            best_overall_A = polished_init
    
    # Fallback
    if not best_overall_A:
        best_overall_A = set(range(64, 129))
    
    # Final polish: try removing redundant elements
    m, u, _ = evaluate(best_overall_A)
    if u == 0:
        for x in sorted(best_overall_A, reverse=True):
            if time.time() > deadline - 2:
                break
            test_A = set(best_overall_A) - {x}
            if not test_A:
                continue
            m_t, u_t, _ = evaluate(test_A)
            if u_t == 0 and m_t <= m:
                best_overall_A = test_A
                m, u, _ = evaluate(best_overall_A)
    
    return sorted(best_overall_A)
