import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try different K values
    for K in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() - start_time > 550:
            break
            
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        ks = np.arange(2, K + 1)
        c = np.log(ks) / ks
        
        max_x = 10 * K
        
        # Exploit periodicity: floor(x/k) has period k, so {x/k} has period k.
        # The constraint M(x) <= 1.0001 depends on floor(x/k), which is periodic with period lcm(1..K).
        # But we only need x in [1, 10K]. The number of distinct patterns of {x/k} for k in [2,K]
        # is at most K (actually much less, but let's use a safe upper bound).
        # Actually, the function floor(x/k) changes at multiples of k. The distinct values of the vector
        # (floor(x/2), floor(x/3), ..., floor(x/K)) occur at most O(K) times in [1, 10K].
        # More precisely, the number of distinct "states" is bounded by the number of distinct sets
        # of intervals, which is O(K). Let's sample x values where any floor(x/k) changes.
        
        # Find all critical points x where floor(x/k) changes for some k in [2,K]
        # floor(x/k) changes when x is a multiple of k.
        # So critical points are all multiples of any k in [2,K] that are in [1, 10K].
        # This is too many. Instead, note that the pattern of {x/k} for all k repeats with period LCM(2..K).
        # But LCM is huge. However, in [1, 10K], the number of distinct patterns is at most 10K, but we can do better.
        
        # Alternative: The constraint is sum_k f(k) * floor(x/k) <= 1.0001.
        # floor(x/k) = (x - {x*k})/k. So sum_k f(k) * (x - {x*k})/k = x * sum_k f(k)/k - sum_k f(k) * {x*k}/k.
        # We know sum_k f(k)/k = 0 (after the f(1) adjustment), so M(x) = -sum_k f(k) * {x*k}/k.
        # Wait, that's not right. Let me re-derive.
        # M(x) = sum_k f(k) * floor(x/k).
        # floor(x/k) = x/k - {x/k}.
        # So M(x) = sum_k f(k) * x/k - sum_k f(k) * {x/k} = x * S - sum_k f(k) * {x/k}, where S = sum_k f(k)/k.
        # After adjustment, S = 0, so M(x) = -sum_k f(k) * {x/k}.
        # The constraint is M(x) <= 1.0001, i.e., -sum_k f(k) * {x/k} <= 1.0001, or sum_k f(k) * {x/k} >= -1.0001.
        # Also, we need to ensure the clipped values work. But the LP already enforces bounds.
        
        # So the constraint is: for all x in [1, 10K], sum_{k=2}^K g(k) * {x/k} >= -1.0001.
        # Note: {x/k} = (x mod k) / k.
        
        # The function {x/k} is periodic in x with period k. The combined function h(x) = sum_k g(k) * {x/k}
        # is periodic with period LCM(2..K), but in [1, 10K], we can find the minimum of h(x).
        # The minimum occurs at some x. Since {x/k} is piecewise linear (actually step function), the minimum
        # of h(x) occurs at some critical point. The critical points are where any {x/k} has a jump, i.e.,
        # when x is a multiple of k for some k.
        
        # But there are too many such points. Instead, let's use the fact that in [1, 10K], the number of
        # distinct values of the vector ({x/2}, {x/3}, ..., {x/K}) is at most O(K log K) or similar.
        # Actually, a simpler approach: the function h(x) is determined by the values of x mod k for k=2..K.
        # The number of distinct residue vectors (x mod 2, x mod 3, ..., x mod K) in [1, 10K] is at most 10K,
        # but we can find a smaller set.
        
        # Key insight: h(x+K) is not necessarily related to h(x). But h(x) depends on x mod k for each k.
        # The period of the entire system is LCM(2..K), which is huge. However, in [1, 10K], we can just
        # check all x? No, 10K can be 2,000,000 for K=200,000, and we have K-1 variables, so the LP would be
        # 2,000,000 x 199,999 which is way too big.
        
        # Better insight: The constraint sum_k g(k) * {x/k} >= -1.0001 for all x in [1, 10K].
        # Since {x/k} >= 0, and g(k) can be negative, the worst case is when {x/k} is small for large positive g(k)
        # and large for large negative g(k).
        
        # Actually, let's reconsider. The original constraint in the LP was:
        # M(x) = sum_k f(k) * floor(x/k) <= 1.0001.
        # With f(1) adjusted so that sum_k f(k)/k = 0, we have M(x) = -sum_{k=2}^K g(k) * {x/k}.
        # So the constraint is -sum_{k=2}^K g(k) * {x/k} <= 1.0001, i.e., sum_{k=2}^K g(k) * {x/k} >= -1.0001.
        
        # Now, {x/k} takes values in [0, 1). The minimum of sum_k g(k) * {x/k} over x in [1, 10K]
        # is what we need to constrain.
        
        # Since {x/k} = (x - k*floor(x/k))/k, and we're summing g(k)*{x/k}, this is a piecewise linear function
        # of x (actually, it's a sum of sawtooth functions). The minimum occurs at a point where the derivative
        # changes sign, or at a discontinuity.
        
        # A simpler approach: note that for x in [1, 10K], the value of {x/k} for k > x is just x/k (since floor(x/k)=0).
        # For k <= x, {x/k} = (x mod k)/k.
        
        # Let's try a different strategy: instead of using all x in [1, 10K], use only x in [1, K] plus x in [K+1, 10K]
        # sampled sparsely. But we need to ensure correctness.
        
        # Actually, the key observation is that the function phi(x) = sum_{k=2}^K g(k) * (x/k) is linear in x,
        # and psi(x) = sum_{k=2}^K g(k) * {x/k} is periodic in each variable. The constraint is phi(x) - psi(x) <= 1.0001.
        # Since phi(x) is linear, the maximum of phi(x) - psi(x) on [1, 10K] occurs at x=10K or at a critical point of psi.
        
        # Let's use the following: the number of distinct values of the vector (x mod 2, x mod 3, ..., x mod K)
        # for x in [1, N] is at most N, but for N=10K, it's 10K. However, we can reduce this by noting that
        # if x and x' are congruent modulo lcm(2,3,...,K), they give the same residues. But lcm is huge.
        
        # Practical approach: for large K, the constraint matrix is too large. Let's limit K to a manageable size
        # where 10K * K is feasible, say K <= 10000 (100000 * 10000 = 1e9 elements, ~8GB, too much).
        # For K=5000, 50000 * 5000 = 2.5e8, ~2GB. Still too much for dense LP.
        
        # Alternative: use the fact that the constraint sum_k g(k) * {x/k} >= -1.0001 can be reformulated.
        # Note that {x/k} >= 0, so if all g(k) >= 0, the constraint is automatically satisfied.
        # The issue is when g(k) < 0.
        
        # Let's try a completely different approach: instead of solving the LP with all constraints,
        # use a cutting-plane method or just check a subset of x values.
        
        # For now, let's keep the same approach but limit K to avoid memory issues.
        # For K > 5000, use a sparse constraint set.
        
        if K <= 3000:
            # Use all x in [1, 10K]
            x_vals = np.arange(1, max_x + 1, dtype=np.float64)
            A_ub = -(x_vals[:, None] % ks[None, :]) / ks[None, :]
            b_ub = np.ones(max_x)
        else:
            # Use a reduced set of x values: x in [1, K] and x = K, 2K, 3K, ..., 10K
            # plus some random samples
            x_vals_list = list(np.arange(1, K + 1))
            x_vals_list.extend(list(np.arange(K, max_x + 1, K)))
            # Add some random samples
            n_random = min(10000, max_x - len(x_vals_list))
            if n_random > 0:
                x_random = np.random.randint(1, max_x + 1, size=n_random)
                x_vals_list.extend(x_random.tolist())
            x_vals = np.array(x_vals_list, dtype=np.float64)
            A_ub = -(x_vals[:, None] % ks[None, :]) / ks[None, :]
            b_ub = np.ones(len(x_vals))
        
        try:
            # Add constraints to ensure f(1) = -sum(g/k) is in [-10, 10]
            row_pos = 1.0 / ks
            row_neg = -1.0 / ks
            
            A_ub_full = np.vstack([A_ub, row_pos, row_neg])
            b_ub_full = np.concatenate([b_ub, [10.0, 10.0]])
            
            bounds = [(-10, 10)] * n_vars
            
            res = linprog(c, A_ub=A_ub_full, b_ub=b_ub_full, bounds=bounds, method='highs')
            
            if res.success:
                g = res.x
                f = np.zeros(K + 1)
                f[2:K+1] = g
                S = np.sum(g / ks)
                f[1] = -S
                
                # Verify f(1) is within bounds
                if abs(f[1]) <= 10.0 + 1e-9:
                    # Verify feasibility for all x in [1, 10K]
                    x_check = np.arange(1, max_x + 1, dtype=np.float64)
                    M_vals = -np.sum(g * (x_check[:, None] % ks[None, :]) / ks[None, :], axis=1)
                    if np.max(M_vals) <= 1.0001 + 1e-9:
                        a = -np.sum(g * c)
                        if a > best_score:
                            best_score = a
                            domain = np.arange(1, K + 1)
                            values = np.clip(f[1:K+1], -10, 10)
                            best_domain = domain
                            best_values = values

        except Exception as e:
            pass

    if best_domain is not None:
        return best_domain, best_values
    else:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
