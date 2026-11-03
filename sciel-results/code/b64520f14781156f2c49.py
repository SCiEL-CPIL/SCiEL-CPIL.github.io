N = 100

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def construct():
    start_time = time.time()
    
    # We will use L-BFGS-B with a smoothed max objective (LogSumExp)
    # and a penalty for the sum constraint.
    
    # Initial guess: uniform 0.5
    h0 = np.full(N, 0.5)
    
    # Add small noise to break symmetry if needed, but uniform is a good start for L-BFGS
    # Actually, uniform is a saddle point or local max for some metrics? 
    # For max overlap, uniform gives N/4. 
    # Let's try to start with a sinusoid to help escape.
    # But L-BFGS is good. Let's just use 0.5.
    
    beta = 10.0 # Temperature for LogSumExp
    lambda_sum = 100.0 # Penalty for sum constraint
    
    def get_correlations(h):
        # h is N x 1 or N,
        # c_k for k=1..N-1
        # c_k = sum_i h_i (1 - h_{i+k})
        # We can compute all circular correlations or just linear?
        # The task says "shifted 1-h". Usually implies circular or linear?
        # "step heights h on [0,2]" ... "integral 1".
        # The seed used np.correlate(h, g, mode='full') and took max excluding center.
        # This is linear correlation.
        # Let's stick to linear correlation to match the seed's definition of "overlap".
        
        g = 1 - h
        c = np.correlate(h, g, mode='full')
        # Exclude shift 0 (index N-1)
        # The seed took max(c[:N-1], c[N:])
        # This includes all non-zero shifts.
        return c[:N-1], c[N:]
        
    def objective_and_grad(h):
        # h is a flat array
        c1, c2 = get_correlations(h)
        c_all = np.concatenate([c1, c2])
        
        # LogSumExp for numerical stability
        max_c = np.max(c_all)
        if max_c > 500: # Prevent overflow
             max_c = 500
        lse = max_c + np.log(np.sum(np.exp(c_all - max_c))) / beta
        
        # Sum penalty
        sum_h = np.sum(h)
        sum_pen = lambda_sum * (sum_h - N/2)**2
        
        val = lse + sum_pen
        
        # Gradient
        # Grad of lse: (1/beta) * sum_k exp(beta c_k - lse_beta) * grad c_k
        # Let p_k = exp(beta c_k) / sum_j exp(beta c_j)
        # grad lse = sum_k p_k * grad c_k
        
        exp_c = np.exp(beta * c_all - beta * max_c) # Unnormalized
        sum_exp = np.sum(exp_c)
        p = exp_c / sum_exp
        
        # grad c_k w.r.t h_j
        # c_k = sum_i h_i - sum_i h_i h_{i+k}
        # d c_k / d h_j = 1 - h_{j+k} - h_{j-k} (with appropriate bounds)
        # We need to sum p_k * (1 - h_{j+k} - h_{j-k}) over k
        
        grad_lse = np.zeros(N)
        
        # Efficient calculation:
        # sum_k p_k * 1 = sum(p) = 1
        # sum_k p_k * h_{j+k}
        # This is a correlation of p and h.
        
        # Let's compute the correlation of p (shifted) and h.
        # p is indexed by k (shift). 
        # c_k corresponds to shift k.
        # In np.correlate(h, 1-h, 'full'), the shift k is represented by index.
        # We need to be careful with indices.
        
        # Alternative: Numerical gradient? 
        # N=100. L-BFGS-B with numerical gradient is slow but N is small.
        # However, we can compute the gradient analytically.
        
        # Let's reconstruct the gradient carefully.
        # c_k = sum_i h_i (1 - h_{i+k})
        # Note: in linear correlation, the shift k is such that h_{i+k} is defined only if 0 <= i+k < N.
        # np.correlate(h, g, 'full') computes sum_i h_i g_{i+k} for various k.
        # The indices in 'full' are 0 to 2N-2.
        # Index N-1 is k=0.
        # Index N-1+k is shift k.
        # Index N-1-k is shift -k.
        
        # We have c_all constructed from c[:N-1] (shifts -1 to -(N-1)?) and c[N:] (shifts 1 to N-1?).
        # Actually, np.correlate(a, v, 'full') at index k is sum_i a_i v_{i+k-N+1}?
        # Let's just use the property:
        # grad c_k (for shift k > 0) w.r.t h_j:
        # c_k = sum_{i=0}^{N-k-1} h_i (1 - h_{i+k})
        # d c_k / d h_j = (1 - h_{j+k}) * [j < N-k] - h_{j-k} * [j-k >= 0 and j-k < N-k]
        #                 = (1 - h_{j+k}) if 0 <= j < N-k
        #                 - h_{j-k} if k <= j < N
        #                 Note: if j+k >= N, first term is 0. If j-k < 0, second term is 0.
        
        # This is getting complex to vectorize perfectly without errors.
        # Given N=100, we can just use numerical gradient provided by L-BFGS-B.
        # It's robust and N is small enough.
        
        return val, None # None tells scipy to compute numerical gradient

    # We will run L-BFGS-B
    # Bounds: 0 to 1
    bounds = [(0, 1) for _ in range(N)]
    
    best_h = h0
    best_score = float('inf')
    
    # Run for a few seconds
    # L-BFGS-B can be sensitive to beta. Start with lower beta, increase?
    # Or just use a fixed reasonable beta.
    
    for restart in range(5):
        if time.time() - start_time > 100:
            break
            
        if restart == 0:
            h_curr = np.full(N, 0.5)
        else:
            # Perturb
            h_curr = np.random.rand(N) * 0.5 + 0.25
            h_curr = h_curr / np.sum(h_curr) * (N/2)
            h_curr = np.clip(h_curr, 0, 1)
            
        # Increase beta for sharper max approximation
        beta = 5.0 + restart * 5.0
        
        # Re-define objective with current beta
        def obj_local(h):
            c1, c2 = get_correlations(h)
            c_all = np.concatenate([c1, c2])
            max_c = np.max(c_all)
            lse = max_c + np.log(np.sum(np.exp(c_all - max_c))) / beta
            sum_h = np.sum(h)
            sum_pen = lambda_sum * (sum_h - N/2)**2
            return lse + sum_pen

        res = scipy.optimize.minimize(obj_local, h_curr, method='L-BFGS-B', 
                                      bounds=bounds, 
                                      options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})
        
        h_opt = np.clip(res.x, 0, 1)
        
        # Correct sum
        current_sum = np.sum(h_opt)
        diff = N/2 - current_sum
        h_opt += diff / N
        h_opt = np.clip(h_opt, 0, 1)
        
        # Final fix
        for _ in range(5):
            current_sum = np.sum(h_opt)
            diff = N/2 - current_sum
            if abs(diff) < 1e-12: break
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
            
        # Calculate true score (max overlap)
        c1, c2 = get_correlations(h_opt)
        c_all = np.concatenate([c1, c2])
        max_c = np.max(c_all)
        
        if max_c < best_score:
            best_score = max_c
            best_h = h_opt

    return best_h
# EVOLVE-BLOCK-END
