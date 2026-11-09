import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    min_sum = 128
    max_sum = 256
    
    # Precompute FFT sizes for speed
    # We need convolution of length 129 with length 129 -> length 257
    # Next power of 2 is 512
    fft_size = 512
    
    def get_reps_from_probs(p):
        """
        Compute expected representation counts using FFT.
        p is a numpy array of shape (n+1,).
        Returns R array of shape (max_sum+1,).
        """
        # Zero pad p
        p_pad = np.zeros(fft_size, dtype=np.float64)
        p_pad[:n+1] = p
        
        # FFT
        P_fft = np.fft.fft(p_pad)
        
        # Convolution is inverse FFT of P^2
        # Note: This gives ordered pairs (i,j). 
        # For unordered pairs with i<=j, the count is:
        # R_s = 0.5 * (OrderedCount_s + SelfCount_s) ? 
        # Actually, OrderedCount_s = sum_{i+j=s} p_i p_j.
        # SelfCount_s = sum_{2i=s} p_i.
        # Unordered R_s = 0.5 * (OrderedCount_s + SelfCount_s) is not quite right for expectation.
        # E[R_s] for unordered pairs {i,j} with i<=j:
        # If i != j, prob is p_i p_j.
        # If i == j, prob is p_i (since it's a set, element i is either in or out).
        # Wait, if A is a set, and we pick elements with prob p_i, 
        # The number of pairs (i,j) in A with i<=j and i+j=s is:
        # Sum_{i<j, i+j=s} I_i I_j + Sum_{i, 2i=s} I_i
        # E[ I_i I_j ] = p_i p_j for i!=j.
        # E[ I_i ] = p_i.
        # So E[R_s] = Sum_{i<j, i+j=s} p_i p_j + Sum_{i, 2i=s} p_i.
        
        # Let's compute OrderedConv = p * p.
        # OrderedConv[s] = Sum_{i+j=s} p_i p_j = Sum_{i!=j} p_i p_j + Sum_{2i=s} p_i^2.
        # We want Sum_{i!=j} p_i p_j + Sum_{2i=s} p_i.
        # Sum_{i!=j} p_i p_j = OrderedConv[s] - Sum_{2i=s} p_i^2.
        # So E[R_s] = OrderedConv[s] - Sum_{2i=s} p_i^2 + Sum_{2i=s} p_i.
        
        R_ordered = np.real(np.fft.ifft(P_fft * P_fft))
        
        # Compute self terms
        # self_term[s] is p_i if s=2i, else 0
        self_terms = np.zeros(fft_size, dtype=np.float64)
        # i ranges from 0 to n
        # s = 2*i
        valid_i = np.arange(0, n+1)
        valid_s = 2 * valid_i
        mask_s = (valid_s >= 0) & (valid_s < fft_size)
        self_terms[valid_s[mask_s]] = p[valid_i[mask_s]]
        
        # p_i^2 terms
        p_sq = p * p
        self_sq_terms = np.zeros(fft_size, dtype=np.float64)
        self_sq_terms[valid_s[mask_s]] = p_sq[valid_i[mask_s]]
        
        # E[R_s] = R_ordered[s] - self_sq_terms[s] + self_terms[s]
        R_expected = R_ordered - self_sq_terms + self_terms
        
        return R_expected[min_sum:max_sum+1]

    def get_gradient(p, R_expected):
        """
        Compute gradient of Sum(R_expected^2) + Penalty(Coverage) w.r.t p.
        Simplified gradient for descent.
        """
        # Gradient of Sum(R_s^2) w.r.t p_i
        # R_s depends on p_i linearly in the 'self' part and quadratically in 'cross' part?
        # Actually, R_expected is linear in p for the self part, and quadratic for cross part?
        # No, R_expected[s] = Sum_{i<j} p_i p_j + Sum_{2i=s} p_i.
        # This is not linear.
        # Let's use numerical gradient or just use the "spectral" proxy:
        # Minimize ||p * p||^2 is equivalent to minimizing energy.
        # Gradient of ||conv(p,p)||^2 w.r.t p is 4 * conv(p, conv(p,p)) (roughly, for ordered).
        # Let's just use the gradient of the continuous quadratic form Q = p^T M p where M encodes the sums.
        # But R_expected is not exactly p^T M p because of the linear self-term.
        
        # A simpler, robust gradient step:
        # We want to reduce high R_s and increase low R_s.
        # Delta R_s / Delta p_i = 1 if 2i=s (self) + sum_{j!=i, i+j=s} p_j (cross).
        # This is exactly the "force" or gradient of the linear part.
        # For the squared sum objective, gradient is 2 * R_s * (dR_s/dp_i).
        
        grad = np.zeros(n+1, dtype=np.float64)
        
        # Calculate dR_s/dp_i for each i
        # dR_s/dp_i = (1 if 2i=s) + sum_{j in A, j!=i, i+j=s} p_j
        # This sum is effectively the convolution of p with p, minus the self term.
        # Let C = p * p (ordered convolution).
        # C[s] = sum_{j} p_j p_{s-j}.
        # Term where j=i is p_i^2.
        # So sum_{j!=i} p_j p_{s-j} = C[s] - p_i^2.
        # dR_s/dp_i = (1 if 2i=s else 0) + (C[s] - p_i^2) / p_i ? No.
        # R_s = Sum_{j<k, j+k=s} p_j p_k + Sum_{k, 2k=s} p_k.
        # dR_s/dp_i:
        # If 2i=s: derivative of p_i is 1.
        # For pairs (j,k) with j+k=s, j!=k:
        # If i is one of them, say j=i, term is p_i p_k. Deriv is p_k.
        # So dR_s/dp_i = (1 if 2i=s) + Sum_{k!=i, i+k=s} p_k.
        
        # We can compute Sum_{k!=i, i+k=s} p_k efficiently.
        # It is essentially the cross-correlation.
        
        # Let's just compute the gradient numerically for a few steps or use a simpler heuristic:
        # Move p_i in direction that reduces the variance of R.
        # Heuristic: grad_i = Sum_{s} R_s * (dR_s/dp_i).
        
        # Approximation:
        # dR_s/dp_i is 1 if 2i=s.
        # dR_s/dp_i is p_k if i+k=s, k!=i.
        
        # Let's use FFT to compute the "influence" vector.
        # influence_i = Sum_s R_s * (dR_s/dp_i)
        
        # dR_s/dp_i = delta_{2i,s} + (C[s] - p_i^2) / p_i ? No, C[s] includes p_i^2.
        # Sum_{k!=i} p_k where i+k=s is C[s] - p_i^2 (if 2i=s) or C[s] (if 2i!=s, but C[s] doesn't have p_i^2 term unless 2i=s).
        # Actually C[s] = sum_{j=0}^n p_j p_{s-j}.
        # If 2i=s, C[s] includes p_i^2.
        # If 2i!=s, C[s] does not include p_i^2 (since j=i implies s-j=i, so s=2i).
        
        # So:
        # if 2i == s: dR_s/dp_i = 1 + (C[s] - p_i^2)
        # if 2i != s: dR_s/dp_i = C[s] (where s=i+k, so k=s-i. The term in C[s] corresponding to j=i is p_i p_{s-i}. Wait.
        # C[s] = sum_j p_j p_{s-j}.
        # We want sum_{k!=i} p_k such that i+k=s.
        # This is just p_{s-i} if s-i != i.
        # If s-i == i, it's 0 (since k!=i).
        
        # So dR_s/dp_i = (1 if 2i=s) + (p_{s-i} if 0<=s-i<=n and s-i!=i else 0).
        
        # This is O(N^2) to compute for all i, s.
        # N=129, S=129. 129*129 = 16k operations. Very fast.
        
        for i in range(n+1):
            # Sum over s in [128, 256]
            # s = i + k, where k is in [0, n] and k != i.
            # Also self term if 2i=s.
            
            # Range of s for this i:
            # s_min = max(128, i+0)
            # s_max = min(256, i+n)
            
            # Iterate s
            # Contribution to grad[i] = Sum_s R_expected[s] * dR_s/dp_i
            
            # dR_s/dp_i:
            # 1. Self: if 2i == s, add 1.
            # 2. Cross: k = s - i. If 0 <= k <= n and k != i, add p[k].
            
            s_start = max(128, i)
            s_end = min(256, i + n)
            
            if s_start > s_end:
                continue
                
            # Vectorize s loop
            s_vals = np.arange(s_start, s_end + 1)
            k_vals = s_vals - i
            
            # Mask for valid k
            valid_k = (k_vals >= 0) & (k_vals <= n)
            
            # Get R_expected for these s
            # R_expected is indexed by s - 128
            R_slice = R_expected[s_vals - min_sum]
            
            # Calculate dR
            dR = np.zeros(len(s_vals), dtype=np.float64)
            
            # Self term
            self_mask = (2 * i == s_vals)
            dR[self_mask] += 1.0
            
            # Cross term
            # We need p[k_vals] where valid_k and k_vals != i
            cross_mask = valid_k & (k_vals != i)
            dR[cross_mask] += p[k_vals[cross_mask]]
            
            grad[i] += np.dot(R_slice, dR)
            
        # Add coverage penalty gradient
        # Penalty = Sum_{s} max(0, 1 - R_s)
        # Grad = Sum_s (-1 if R_s < 1 else 0) * dR_s/dp_i
        # This is similar to above but with weights w_s = -1 if R_s < 1 else 0
        
        # We can combine:
        # Objective = Sum R_s^2 + Lambda * Sum max(0, 1-R_s)
        # We already computed grad for Sum R_s^2 (implicitly, if we assume R is the weight? No, grad was Sum R * dR).
        # Wait, d/dp Sum R_s^2 = Sum 2 R_s dR_s/dp_i.
        # My loop computed Sum R_s dR_s/dp_i. So multiply by 2.
        
        grad *= 2.0
        
        # Add penalty gradient
        # w_s = -1 if R_s < 1 else 0
        w = np.where(R_expected < 1.0, -1.0, 0.0)
        
        for i in range(n+1):
            s_start = max(128, i)
            s_end = min(256, i + n)
            if s_start > s_end: continue
            
            s_vals = np.arange(s_start, s_end + 1)
            k_vals = s_vals - i
            valid_k = (k_vals >= 0) & (k_vals <= n)
            
            dR = np.zeros(len(s_vals), dtype=np.float64)
            self_mask = (2 * i == s_vals)
            dR[self_mask] += 1.0
            cross_mask = valid_k & (k_vals != i)
            dR[cross_mask] += p[k_vals[cross_mask]]
            
            grad[i] += np.dot(w[s_vals - min_sum], dR)
            
        return grad

    # 1. Continuous Optimization
    # Initialize p
    # Bias towards middle values to cover sums 128-256
    # Sums 128-256 require elements roughly 64-128.
    # Let's start with p=0.5 for all, or slightly biased.
    p = np.full(n+1, 0.5, dtype=np.float64)
    
    # Gradient Descent
    lr = 0.05
    lambda_cov = 10.0 # Weight for coverage
    
    # We need R_expected for gradient
    R_expected = get_reps_from_probs(p)
    
    steps = 0
    max_steps = 2000
    while steps < max_steps:
        if time.time() > deadline - 20: # Reserve 20s for refinement
            break
            
        grad = get_gradient(p, R_expected)
        
        # Update p
        p -= lr * grad
        p = np.clip(p, 0.0, 1.0)
        
        # Update R_expected
        R_expected = get_reps_from_probs(p)
        
        steps += 1
        
        # Adaptive learning rate
        if steps % 100 == 0:
            lr *= 0.95

    # 2. Threshold to Discrete Set
    # Threshold at 0.5
    A = set(np.where(p > 0.5)[0])
    
    # 3. Discrete Refinement (Local Search)
    # Objective: Minimize Max(R_s), subject to All(R_s > 0).
    # If invalid, prioritize coverage.
    
    def calc_reps_discrete(A_set):
        R = np.zeros(max_sum + 1, dtype=np.int32)
        A_list = list(A_set)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if min_sum <= s <= max_sum:
                    R[s] += 1
        return R

    R_curr = calc_reps_discrete(A)
    
    def get_cost(R):
        uncovered = np.sum(R[min_sum:max_sum+1] == 0)
        max_mult = np.max(R[min_sum:max_sum+1])
        return 1000 * uncovered + max_mult

    current_cost = get_cost(R_curr)
    best_cost = current_cost
    best_A = set(A)
    
    # Local Search: Try adding/removing elements
    # Prioritize fixing uncovered sums by adding elements that cover them.
    # Then minimize max mult by removing elements that are in high-multiplicity sums.
    
    for step in range(5000):
        if time.time() > deadline - 5:
            break
            
        # 1. Check coverage
        uncovered_sums = np.where(R_curr[min_sum:max_sum+1] == 0)[0] + min_sum
        
        if len(uncovered_sums) > 0:
            # Try to add an element that covers an uncovered sum
            # Pick a random uncovered sum
            s_target = random.choice(uncovered_sums)
            
            # Find candidate x such that x + y = s_target for some y in A, or 2x = s_target
            # Or just try random x not in A
            candidates = []
            # Elements that could form s_target with existing A
            for y in A:
                x = s_target - y
                if 0 <= x <= n and x not in A:
                    candidates.append(x)
            # Self pair
            if s_target % 2 == 0:
                x = s_target // 2
                if 0 <= x <= n and x not in A:
                    candidates.append(x)
            
            if not candidates:
                # Try random
                x = random.randint(0, n)
                if x not in A:
                    candidates.append(x)
            
            if not candidates:
                continue
                
            x = random.choice(candidates)
            
            # Simulate add
            # Add x to A
            A.add(x)
            # Update R
            for y in A:
                if y == x: continue
                s = x + y
                if min_sum <= s <= max_sum:
                    R_curr[s] += 1
            if 2*x >= min_sum and 2*x <= max_sum:
                R_curr[2*x] += 1
                
            new_cost = get_cost(R_curr)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = set(A)
            else:
                # Revert
                A.remove(x)
                for y in A:
                    s = x + y
                    if min_sum <= s <= max_sum:
                        R_curr[s] -= 1
                if 2*x >= min_sum and 2*x <= max_sum:
                    R_curr[2*x] -= 1
        else:
            # Covered. Minimize Max Mult.
            # Find sum with max multiplicity
            max_val = np.max(R_curr[min_sum:max_sum+1])
            if max_val <= 1:
                break # Optimal?
                
            max_sums = np.where(R_curr[min_sum:max_sum+1] == max_val)[0] + min_sum
            s_target = random.choice(max_sums)
            
            # Try to remove an element involved in s_target
            # Elements x such that x+y=s_target for y in A, or 2x=s_target
            candidates = []
            for y in A:
                x = s_target - y
                if x in A and x != y:
                    candidates.append(x)
                elif x == y: # 2x = s_target
                    candidates.append(x)
            
            # Remove duplicates
            candidates = list(set(candidates))
            
            if not candidates:
                continue
                
            x = random.choice(candidates)
            
            # Simulate remove
            A.remove(x)
            for y in A:
                s = x + y
                if min_sum <= s <= max_sum:
                    R_curr[s] -= 1
            if 2*x >= min_sum and 2*x <= max_sum:
                R_curr[2*x] -= 1
                
            new_cost = get_cost(R_curr)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = set(A)
            else:
                # Revert
                A.add(x)
                for y in A:
                    if y == x: continue
                    s = x + y
                    if min_sum <= s <= max_sum:
                        R_curr[s] += 1
                if 2*x >= min_sum and 2*x <= max_sum:
                    R_curr[2*x] += 1

    # Final check
    R_final = calc_reps_discrete(best_A)
    if np.any(R_final[min_sum:max_sum+1] == 0):
        # Fallback
        return list(range(64, 129))
        
    return list(best_A)
