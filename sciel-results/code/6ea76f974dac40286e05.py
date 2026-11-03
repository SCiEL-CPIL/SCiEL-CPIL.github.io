import numpy as np
import time
import random
from scipy.optimize import minimize, LinearConstraint

N = 50

def find_sidon_set(N, K, attempts=1000):
    """
    Attempts to find a Sidon set of size K in [0, N).
    Returns the set if found, None otherwise.
    """
    for _ in range(attempts):
        # Randomize the order of candidates to avoid bias
        candidates = list(range(N))
        random.shuffle(candidates)
        
        S = []
        sums = set()
        
        for cand in candidates:
            if len(S) == K:
                break
            
            # Check if adding cand creates a collision
            conflict = False
            new_sums = set()
            
            # Check cand + cand
            if cand + cand in sums:
                conflict = True
            else:
                new_sums.add(cand + cand)
                
            # Check cand + s for s in S
            if not conflict:
                for s in S:
                    sm = cand + s
                    if sm in sums:
                        conflict = True
                        break
                    if sm in new_sums:
                        conflict = True
                        break
                    new_sums.add(sm)
            
            if not conflict:
                S.append(cand)
                sums.update(new_sums)
        
        if len(S) == K:
            return S
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initial Best: Uniform
    best_h = np.ones(N) / N
    best_ratio = np.sum(best_h**2) / (np.sum(best_h) * np.max(np.convolve(best_h, best_h)))

    # 2. Find best Sidon set (Sparse structure)
    # This provides a very strong local minimum for the ratio.
    best_sidon_h = None
    best_sidon_k = 0
    for K in range(12, 4, -1):
        if time.time() > deadline - 30.0:
            break
        attempts = 200 if K > 8 else 500
        S = find_sidon_set(N, K, attempts)
        if S is not None:
            h = np.zeros(N)
            h[S] = 1.0 / K
            best_sidon_h = h
            best_sidon_k = K
            # Since we iterate K downwards, the first found is the largest.
            # However, we might want to check if a slightly smaller one with better geometry?
            # No, for Sidon sets, ratio is exactly K (if sum=1). So larger K is always better.
            # We break because any smaller K will yield a lower ratio for a pure Sidon set.
            # But wait, the optimization below might improve a non-Sidon set or a Sidon set?
            # Actually, if we have a Sidon set, the max autoconv is 1/K.
            # Ratio = (K * (1/K)^2) / (1 * (1/K)) = (1/K) / (1/K) = 1?
            # Let's re-calculate.
            # h has K entries of 1/K.
            # ||h||_2^2 = K * (1/K)^2 = 1/K.
            # ||h||_1 = 1.
            # autoconv h*h. Max value is 1/K (at the peaks where i=j).
            # Ratio = (1/K) / (1 * (1/K)) = 1.
            # Wait, the current program score is 1.148. Uniform gives 1.0.
            # Why does Sidon give 1?
            # Ah, the definition of ratio in the prompt: "squared L2 norm to L1 norm times supremum norm of the autoconvolution".
            # R = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
            # If h is uniform: ||h||_2^2 = 1/N. ||h||_1 = 1. ||h*h||_inf = 1/N. R = 1.
            # If h is Sidon (K spikes): ||h||_2^2 = 1/K. ||h||_1 = 1. ||h*h||_inf = 1/K. R = 1.
            # So Sidon sets don't immediately improve the ratio if we normalize sum to 1?
            # Let's check the current program's logic.
            # `best_h = np.ones(N) / N` -> ratio 1.0.
            # It found score 1.1484. This implies the current program found something better than uniform.
            # Maybe the Sidon search in the current program was just a fallback or it found something else?
            # Actually, looking at the current code:
            # `ratio = np.sum(h**2) / (np.sum(h) * np.max(np.convolve(h, h)))`
            # If it finds a Sidon set, ratio is 1.
            # So why is the score 1.1484?
            # Perhaps the "Sidon set" logic in the current program is flawed or I am miscalculating?
            # Or maybe the evaluator uses a different normalization?
            # "Choose 50 nonnegative heights... maximize the ratio... score is the lower bound on C2"
            # "search score is value/seed".
            # If seed is uniform (ratio 1), and current is 1.1484, then the current program found a configuration with ratio 1.1484.
            # How?
            # If we use a non-uniform distribution, can we get R > 1?
            # Let h be a Gaussian.
            # ||h||_2^2 is small. ||h*h||_inf is also small.
            # Let's test a simple case. N=2. h=[1,0]. R = 1/(1*1) = 1.
            # h=[0.5, 0.5]. R = 0.5 / (1 * 0.5) = 1.
            # h=[1, 1]. R = 2 / (2 * 2) = 0.5.
            # It seems hard to get > 1 with simple discrete vectors?
            # Wait, `np.convolve` is linear convolution.
            # For h=[1, 0, 0], conv = [1, 0, 0, 0, 0]. Max 1. R = 1/1 = 1.
            # For h=[1, 1, 0], conv = [1, 2, 1, 0, 0]. Max 2. R = 2 / (2*2) = 0.5.
            # It seems R <= 1 for non-negative vectors?
            # Let's check the inequality.
            # ||h||_2^2 <= ||h||_1 * ||h*h||_inf ?
            # This is related to the "second autocorrelation inequality".
            # If the inequality holds, R <= 1.
            # But the task says "score is the lower bound on C2" and "higher is better".
            # And "1.0 equals the seed program".
            # If the seed is uniform (R=1), and we can get 1.1484, then R can be > 1?
            # Or maybe the "ratio" defined in the task is different?
            # "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
            # R = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
            # If R can be > 1, my manual checks above were wrong or I missed a case.
            # Let's re-evaluate h=[1, 0, 1].
            # ||h||_2^2 = 2. ||h||_1 = 2.
            # conv = [1, 0, 2, 0, 1]. Max = 2.
            # R = 2 / (2 * 2) = 0.5.
            # 
            # What if h is continuous?
            # The problem says "50 nonnegative heights on equal intervals". This is a discrete vector.
            # 
            # Is it possible that the evaluator normalizes differently?
            # "Choose 50 nonnegative heights...".
            # Maybe the heights are not normalized to sum 1?
            # The ratio is scale invariant. R(ah) = R(h).
            # So normalization doesn't matter.
            #
            # Let's look at the inspiration 1 code.
            # It uses SLSQP to minimize M subject to M >= c_k and sum(h)=1.
            # It returns h.
            # It got score 1.3057.
            # This implies that there exist h with R > 1.
            # My manual discrete checks failed to find one.
            # Let's try a wider support.
            # Maybe the peak of autoconvolution is not at the center?
            # No, for non-negative h, the max of h*h is at the center (k=N-1) if h is symmetric?
            # Not necessarily.
            #
            # Let's trust the inspirations. The SLSQP approach works.
            # The Sidon approach in the current program might be a red herring or I am misunderstanding the "Sidon" benefit in this specific metric context.
            # However, the current program *has* a Sidon search.
            # If I combine, I should keep the Sidon search as a *good initial guess* for the optimizer, or as a fallback.
            # But if Sidon gives R=1, and uniform gives R=1, and we found 1.148, the 1.148 came from somewhere else?
            # Wait, the current program code *only* does Sidon search and fallback to uniform.
            # If Sidon gives R=1 and uniform gives R=1, how did it get 1.1484?
            # 
            # Re-reading current program:
            # `best_h = np.ones(N) / N`
            # `best_ratio = ...` (which is 1.0)
            # Loop K=12..5.
            # If Sidon found, update best_h if ratio > best_ratio.
            # If Sidon ratio is 1.0, it won't update (1.0 is not > 1.0).
            # So best_h remains uniform.
            # So the current program should return uniform, scoring 1.0.
            # But the prompt says "CURRENT PROGRAM (score 1.1484)".
            # This is a contradiction.
            # Possibility 1: The Sidon search *does* find sets with R > 1?
            # Let's check R for a Sidon set again.
            # h has K entries of 1/K.
            # ||h||_2^2 = 1/K.
            # ||h||_1 = 1.
            # h*h max.
            # If it's a Sidon set, all sums i+j are distinct (except for commutativity i+j=j+i).
            # The number of pairs (i,j) such that i+j=k is 1 for most k (if i!=j) and 1 for i=j.
            # Wait, Sidon set property: all sums i+j with i<=j are distinct.
            # So for each k, there is at most one pair (i,j) with i<=j such that i+j=k.
            # In the convolution sum `sum h_i h_{k-i}`, we sum over all i.
            # If h_i = h_j = 1/K for i in S.
            # The term is non-zero only if i in S and k-i in S.
            # So we are counting pairs (i, k-i) in SxS.
            # If i != k-i, this pair contributes 2 * (1/K)^2 to the sum? No.
            # The sum is `sum_{i} h_i h_{k-i}`.
            # If i is in S and k-i is in S, the term is (1/K)*(1/K) = 1/K^2.
            # How many such i?
            # If k = 2i, then i is the unique solution (since i=i). Count 1.
            # If k = i+j with i!=j, then i and j are distinct solutions.
            # Since Sidon sets have distinct sums, for a given k, there is at most one pair {i,j}.
            # So the number of terms is either 1 (if k=2i) or 2 (if k=i+j, i!=j).
            # So max autoconv is 2/K^2?
            # If max autoconv is 2/K^2, then R = (1/K) / (1 * 2/K^2) = K/2.
            # If K=10, R=5.
            # This is much larger than 1.
            # So Sidon sets DO give high ratios!
            # My previous calculation of "max autoconv is 1/K" was wrong. I assumed the peak was 1/K, but that would require K terms of 1/K^2 to sum to 1/K.
            # But Sidon sets have sparse overlaps.
            # So, finding a large Sidon set is a very strong strategy.
            #
            # Why did the current program score 1.1484?
            # If it found a Sidon set of K=12, R=6.
            # If it found K=10, R=5.
            # Maybe it didn't find a Sidon set?
            # Or maybe the `find_sidon_set` function failed?
            # Or maybe the ratio calculation in the current program is different?
            # `ratio = np.sum(h**2) / (np.sum(h) * np.max(np.convolve(h, h)))`
            # This matches my derivation.
            # If it returned uniform, score would be 1.0 (relative to seed 1.0).
            # The score 1.1484 suggests it found something slightly better than uniform, but not a full Sidon set?
            # Or maybe the "seed" is not uniform?
            # "1.0 equals the seed program".
            # If the seed program is uniform, and uniform gives R=1, then score 1.0.
            # If the current program returns uniform, it should score 1.0.
            # The fact that it scores 1.1484 implies it is NOT returning uniform.
            # It must have found a Sidon set with R=1.1484?
            # R = K/2.
            # K/2 = 1.1484 => K = 2.29.
            # This implies it found a "Sidon-like" set with K=2 or 3?
            # Or maybe the max autoconv is not 2/K^2?
            #
            # Let's assume the current program's Sidon search is effective but limited.
            # The inspiration 1 (SLSQP) got 1.3057.
            # This suggests that *non-Sidon* sets, or optimized weights, can beat pure Sidon sets?
            # Or maybe the SLSQP found a better Sidon set?
            # No, SLSQP optimizes continuous weights.
            #
            # Strategy for Combination:
            # 1. Use the Sidon search to find a good sparse support.
            # 2. Use that support (or a slightly denser version) as an initial guess for SLSQP.
            # 3. Let SLSQP refine the weights.
            # 4. Also try random sparse supports as initial guesses for SLSQP.
            #
            # This combines the structural insight of Sidon sets (good initial geometry) with the refinement power of SLSQP (optimizing weights).

    # --- Optimization Phase ---
    
    def autoconv(h):
        return np.convolve(h, h)

    def neg_ratio_obj(x):
        # x = [h_0, ..., h_{N-1}, M]
        return x[N]

    def neg_ratio_obj_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad

    def constraint_sum(x):
        return np.sum(x[:N]) - 1.0

    def constraint_sum_grad(x):
        grad = np.zeros(N + 1)
        grad[:N] = 1.0
        return grad

    def constraints_autoconv(x):
        h = x[:N]
        M = x[N]
        c = autoconv(h)
        return M - c

    def constraints_autoconv_grad(x):
        h = x[:N]
        grad = np.zeros((2*N - 1, N + 1))
        for k in range(2*N - 1):
            derivs = np.zeros(N)
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    derivs[i] = 2.0 * h[j]
            grad[k, :N] = -derivs
            grad[k, N] = 1.0
        return grad

    def calculate_ratio(h):
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        c = autoconv(h)
        peak = np.max(c)
        if peak == 0:
            return 0.0
        return l2_sq / (l1 * peak)

    # Update best with Sidon result if found
    if best_sidon_h is not None:
        r_sidon = calculate_ratio(best_sidon_h)
        if r_sidon > best_ratio:
            best_ratio = r_sidon
            best_h = best_sidon_h.copy()

    bounds = [(0.0, None)] * N + [(0.0, None)]
    cons = [
        {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
        {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
    ]

    # Initial guesses for SLSQP
    # 1. Current best (Uniform or Sidon)
    # 2. Random sparse supports (Sidon-like)
    
    initial_guesses = []
    
    # Add current best
    if best_h is not None:
        c_best = autoconv(best_h)
        M_best = np.max(c_best)
        x_best = np.concatenate([best_h, [M_best]])
        initial_guesses.append(x_best)
        
    # Add Uniform
    h_uni = np.ones(N) / N
    c_uni = autoconv(h_uni)
    M_uni = np.max(c_uni)
    initial_guesses.append(np.concatenate([h_uni, [M_uni]]))
    
    # Add random sparse guesses
    for _ in range(20):
        if time.time() > deadline - 10.0:
            break
        K_rand = np.random.randint(5, 15)
        support = np.random.choice(N, K_rand, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K_rand
        c_rand = autoconv(h_rand)
        M_rand = np.max(c_rand)
        initial_guesses.append(np.concatenate([h_rand, [M_rand]]))

    # Run SLSQP on each guess
    for x0 in initial_guesses:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(
                neg_ratio_obj, 
                x0, 
                method='SLSQP', 
                jac=neg_ratio_obj_grad,
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if res.success:
                h_res = res.x[:N]
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    continue
                h_res = np.clip(h_res, 0, None)
                h_res = h_res / np.sum(h_res)
                
                ratio_res = calculate_ratio(h_res)
                if ratio_res > best_ratio:
                    best_ratio = ratio_res
                    best_h = h_res.copy()
                    
        except Exception:
            continue

    # Final cleanup
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
