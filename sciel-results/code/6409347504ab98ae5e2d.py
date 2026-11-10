import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600.0
    
    best_a = -np.inf
    best_domain = None
    best_values = None
    
    # Candidate K values
    # Based on analysis, a = c * sum(log k / k) and c ~ 1/K, so a ~ (log K)^2 / K
    # This peaks at small K. But let's check a range.
    K_candidates = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    
    for K in K_candidates:
        if time.time() > deadline - 10:
            break
            
        # Compute H_K
        k_arr = np.arange(1, K + 1, dtype=np.float64)
        H_K = np.sum(1.0 / k_arr)
        
        # Compute D(x) for x in [1, 10K]
        max_x = 10 * K
        # d(x) = number of divisors of x
        d = np.zeros(max_x + 1, dtype=np.int64)
        # Sieve for divisor counts
        for i in range(1, max_x + 1):
            d[i::i] += 1
        
        # D(x) = cumsum(d)
        D = np.cumsum(d)
        D = D[1:] # D[0] is 0, we want D[1]..D[max_x]
        
        # Q(x) = x * H_K - D(x)
        x_arr = np.arange(1, max_x + 1, dtype=np.float64)
        Q = x_arr * H_K - D.astype(np.float64)
        
        Q_max = np.max(Q)
        
        if Q_max <= 0:
            c = 10.0
        else:
            c = 1.0001 / Q_max
            if c > 10.0:
                c = 10.0
        
        # Construct f
        # f(k) = -c for k > 1
        # f(1) = -c - S, where S = sum(f(k)/k)
        # S = f(1)/1 + sum_{k=2}^K (-c)/k
        # Let g(k) = -c for all k.
        # S_g = -c * H_K
        # f(1) = g(1) - S_g = -c - (-c H_K) = c(H_K - 1)
        # f(k) = -c for k > 1
        
        values = np.full(K, -c, dtype=np.float64)
        values[0] = c * (H_K - 1)
        
        # Clip
        values = np.clip(values, -10, 10)
        
        # Re-calculate S after clipping?
        # The problem says: "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S..."
        # So we must clip first, then compute S, then adjust f(1).
        
        values_clipped = np.clip(values, -10, 10)
        S = np.sum(values_clipped / k_arr)
        
        values_corrected = values_clipped.copy()
        values_corrected[0] = values_clipped[0] - S
        
        # Check constraint M(x) <= 1.0001
        # M(x) = sum f(k) floor(x/k)
        # We can compute this using the difference array method or direct sum.
        # Direct sum is O(K * max_x) which is too slow.
        # Use the property M(x) = c * Q(x) only if clipping didn't change anything.
        # Clipping might change f(1) if c(H_K-1) > 10.
        # If c is limited by 10, then f(k)=-10, f(1)=10(H_K-1).
        # If 10(H_K-1) > 10, i.e., H_K > 2, then f(1) is clipped to 10.
        # H_K > 2 for K >= 4.
        # So for K >= 4, f(1) is likely clipped.
        # If f(1) is clipped, the linear relationship M(x) = c Q(x) breaks.
        # We need to compute M(x) explicitly for the corrected values.
        
        # Efficient M(x) computation
        # M(x) = sum_{k=1}^K f(k) floor(x/k)
        # We can use the difference array technique on the values.
        # Let diff be an array of size max_x + 2.
        # For each k, f(k) adds to M(x) for x in [k, 2k-1] with value f(k)*1, [2k, 3k-1] with f(k)*2, etc.
        # This is what the original code did.
        # Let's use a faster vectorized approach.
        
        # M(x) = sum_{j=1}^{max_x} (sum_{k: j <= x/k < j+1} f(k)) ? No.
        # M(x) = sum_{k=1}^K f(k) floor(x/k).
        
        # Let's just compute M for the specific K if it's small enough, or use the approximation.
        # Given the time, let's use the vectorized diff array from the original code, it was O(m * 10K / K) roughly?
        # No, the original code looped over k and did np.add.at.
        
        # Alternative:
        # M(x) can be computed as:
        # M = np.zeros(max_x+1)
        # for k in range(1, K+1):
        #     f_k = values_corrected[k-1]
        #     if f_k == 0: continue
        #     # x ranges from k to max_x
        #     # floor(x/k) is constant in intervals [j*k, (j+1)*k - 1]
        #     # This is complex to vectorize per k.
        
        # Let's stick to the previous method but optimized.
        # Actually, if K is small (e.g., < 1000), we can just do it directly.
        # If K is large, we rely on the fact that the constant solution is robust.
        # Let's compute M(x) for the best candidate K at the end?
        # No, we need to verify constraint for each K to ensure validity.
        
        # Let's use the property that if no clipping occurred, M(x) = c Q(x).
        # Clipping occurs if c > 10 or f(1) > 10.
        # If c <= 10 and f(1) <= 10, then M(x) = c Q(x) holds?
        # f(1) = c(H_K - 1). If this is <= 10, and -c >= -10, then no clipping.
        # In that case, we already know max M(x) = c Q_max <= 1.0001.
        
        # If clipping occurred, we need to check.
        # Clipping of f(1) to 10 reduces M(x) compared to the unclipped case (since f(1) is positive and contributes to M(x) positively? No, f(1) is positive, M(x) includes f(1)*x.
        # Wait, f(1) = c(H_K-1). This is positive.
        # M(x) = f(1) x + sum_{k=2} f(k) floor(x/k).
        # If f(1) is clipped to 10, and unclipped was > 10, then M(x) decreases.
        # So if unclipped M(x) <= 1.0001, clipped is also <= 1.0001.
        # If unclipped M(x) > 1.0001, clipping might bring it down.
        
        # So:
        # 1. Check if no clipping. If so, valid.
        # 2. If clipping, we must compute M(x) explicitly.
        
        no_clipping = (c <= 10.0) and (values[0] <= 10.0) and (values[0] >= -10.0)
        
        if no_clipping:
            valid = True
        else:
            # Compute M(x) explicitly
            # This is expensive. Only do it if K is small or if we are close to the end.
            # For now, assume valid if K is small, else skip?
            # Let's implement a fast check for small K.
            if K <= 1000:
                M = np.zeros(max_x + 1)
                for k in range(1, K + 1):
                    f_k = values_corrected[k-1]
                    if f_k == 0: continue
                    # Add f_k * floor(x/k)
                    # x from 1 to max_x
                    # We can use:
                    # M += f_k * (np.arange(1, max_x+1) // k)
                    # This is O(max_x) per k. Total O(K * max_x) = O(10 K^2).
                    # For K=1000, 10^7 ops. Fast.
                    M += f_k * (np.arange(1, max_x + 1, dtype=np.int64) // k)
                
                if np.max(M[1:]) > 1.0001:
                    valid = False
                else:
                    valid = True
            else:
                # For large K, assume the constraint is tight and clipping helps.
                # Or just skip large K if we have a good small K.
                valid = False # Conservative
        
        if valid:
            # Compute objective a
            log_k = np.log(k_arr)
            a = -np.sum(values_corrected * log_k / k_arr)
            
            if a > best_a:
                best_a = a
                best_domain = np.arange(1, K + 1, dtype=np.int64)
                best_values = values_corrected

    if best_domain is None:
        # Fallback
        domain = np.array([1], dtype=np.int64)
        values = np.array([0.0], dtype=np.float64)
        return domain, values

    return best_domain, best_values
