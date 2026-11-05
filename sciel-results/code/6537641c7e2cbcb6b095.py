"""Rewritten program for the third autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 400


def construct():
    """
    Construct step heights h of length N=400 to minimize:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    
    Key insight: The value is scale-invariant, so we can fix sum(h) = 1.
    Then we minimize max(np.convolve(h, h)).
    
    The autoconvolution entries are quadratic in h. We use projected gradient
    descent with a smooth approximation of the max function (log-sum-exp).
    """
    
    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10
        ac = np.convolve(h, h)
        return abs(2 * N * np.max(ac) / (s ** 2))
    
    # Use log-sum-exp as a smooth approximation of max
    # soft_max(x, t) = (1/t) * log(sum(exp(t*x)))
    # For large t, this approximates max(x)
    
    t = 50.0  # temperature parameter for soft max
    
    def objective(h):
        ac = np.convolve(h, h)
        # Use log-sum-exp for smooth max
        t_ac = t * ac
        t_ac = t_ac - np.max(t_ac)  # for numerical stability
        return (1.0 / t) * np.log(np.sum(np.exp(t_ac)))
    
    def objective_grad(h):
        ac = np.convolve(h, h)
        t_ac = t * ac
        t_ac = t_ac - np.max(t_ac)
        exp_ac = np.exp(t_ac)
        sum_exp = np.sum(exp_ac)
        # Gradient of soft_max w.r.t. ac: exp(t*ac) / sum(exp(t*ac))
        grad_ac = exp_ac / sum_exp
        # Gradient of ac w.r.t. h: ac[k] = sum_i h[i]*h[k-i]
        # d(ac[k])/d(h[j]) = h[k-j] + h[j-k] (with appropriate bounds)
        # We can compute this via convolution with a reversed h
        # d(ac)/d(h) = convolve(h, grad_ac) + convolve(h, flip(grad_ac))
        # Actually, ac = convolve(h, h), so d(ac[k])/d(h[j]) = h[k-j] if k-j in range, plus h[j-(k)] ... 
        # Let's use: d(ac)/dh = convolve(grad_ac, h) + convolve(grad_ac, h[::-1])[::-1]
        # Simpler: ac[k] = sum_i h[i] h[k-i], so d(ac[k])/d(h[j]) = h[k-j] + (h[j-k] if j-k != k-j else 0)
        # For j = k-i: d(ac[k])/d(h[j]) includes h[k-j] when i=j, and h[j-k] when k-i=j i.e. i=k-j
        # So d(ac[k])/d(h[j]) = h[k-j] + h[j-k] if both indices valid, else just one term
        # This is equivalent to: grad_h = convolve(grad_ac, h) where we handle the full conv
        # Actually, let's just do it directly with the formula:
        # grad_h[j] = sum_k grad_ac[k] * (h[k-j] if 0<=k-j<N else 0 + h[j-k] if 0<=j-k<N else 0)
        # = sum_k grad_ac[k] * h[k-j] (k from j to j+N-1) + sum_k grad_ac[k] * h[j-k] (k from 0 to j)
        # First term: for each j, sum over k where k-j in [0,N-1], i.e., k in [j, j+N-1]
        # This is a convolution-like operation.
        
        # Use FFT for efficiency
        # ac = ifft(fft(h)^2), so d(ac)/dh can be computed via FFT
        # But for simplicity, let's use the direct approach with np.convolve
        # grad_h[j] = sum_k grad_ac[k] * h[k-j] (k-j in [0,N-1]) + sum_k grad_ac[k] * h[j-k] (j-k in [0,N-1])
        # First sum: for fixed j, k ranges from j to min(2N-2, j+N-1)
        # This is: (grad_ac * h)[j] where * is cross-correlation
        # Let's compute: term1[j] = sum_{k} grad_ac[k] * h[k-j] for valid k-j
        # = convolve(grad_ac, h)[j + N - 1] (full conv, index offset)
        # Actually, let me just use a simple approach:
        # Since N=400, direct computation is O(N^2) which is 160000, acceptable
        
        grad_h = np.zeros(N)
        for j in range(N):
            # d(ac[k])/d(h[j]) = h[k-j] if 0 <= k-j < N, plus h[j-k] if 0 <= j-k < N and j-k != k-j
            # But since ac is symmetric (ac[k] = ac[2N-2-k]), and h is not necessarily symmetric,
            # let's just compute directly:
            # ac[k] = sum_i h[i] h[k-i]
            # d(ac[k])/d(h[j]) = h[k-j] (when i=j, so k-j must be valid index) 
            #               + h[j-k] (when k-i=j, so i=k-j, and we need j=k-i means i=k-j, 
            #                 but this is the same as i=j when k=2j... let me be careful)
            # ac[k] = sum_{i: 0<=i<N, 0<=k-i<N} h[i] h[k-i]
            # d(ac[k])/d(h[j]) = h[k-j] if 0<=k-j<N (from term where i=j)
            #                  + h[j-k] if 0<=j-k<N and j != k-j (from term where k-i=j, i.e., i=k-j)
            #                  + if j == k-j, i.e., k=2j, then it's h[j]^2, derivative is 2*h[j]
            # So: d(ac[k])/d(h[j]) = h[k-j] * I(0<=k-j<N) + h[j-k] * I(0<=j-k<N)
            #   but when k-j == j-k, i.e., k=2j, we count it once, not twice.
            #   Actually when i=j and i=k-j, these are the same term only when j=k-j.
            #   So: d(ac[k])/d(h[j]) = h[k-j] * I(0<=k-j<N) + h[j-k] * I(0<=j-k<N) 
            #     minus h[j] * I(k=2j) if both conditions hold (to avoid double counting)
            # Wait, no. If k-j is valid and j-k is valid, and k-j != j-k, they're different terms.
            # If k-j == j-k (i.e., k=2j), it's the same term h[j]*h[j], so derivative is 2*h[j], not h[j]+h[j]... 
            # Actually h[j]*h[j] derivative w.r.t. h[j] is 2*h[j], and h[j]+h[j] = 2*h[j], so it's fine!
            # So: d(ac[k])/d(h[j]) = h[k-j] * I(0<=k-j<N) + h[j-k] * I(0<=j-k<N)
            
            val = 0.0
            k_minus_j = k - j  # this is wrong, k is not defined here
            # Let me redo: for each k, contribute grad_ac[k] * (h[k-j] if valid + h[j-k] if valid)
            for k in range(2 * N - 1):
                contrib = 0.0
                k_j = k - j
                if 0 <= k_j < N:
                    contrib += h[k_j]
                j_k = j - k
                if 0 <= j_k < N:
                    contrib += h[j_k]
                val += grad_ac[k] * contrib
            grad_h[j] = val
        
        return grad_h
    
    # This direct gradient is O(N^2) per evaluation, which for N=400 is 160k ops, fine.
    # But let's use a smarter approach with FFT or np.convolve
    
    def objective_grad_v2(h):
        ac = np.convolve(h, h)
        t_ac = t * ac
        t_ac = t_ac - np.max(t_ac)
        exp_ac = np.exp(t_ac)
        sum_exp = np.sum(exp_ac)
        grad_ac = exp_ac / sum_exp  # shape (2N-1,)
        
        # grad_h[j] = sum_k grad_ac[k] * (h[k-j] if 0<=k-j<N else 0 + h[j-k] if 0<=j-k<N else 0)
        # First term: sum_k grad_ac[k] * h[k-j] for k-j in [0,N-1]
        # = sum_{i=0}^{N-1} grad_ac[j+i] * h[i]  (where i = k-j, k = j+i, k in [j, j+N-1])
        # This is a cross-correlation.
        # Let's compute using np.convolve:
        # term1[j] = sum_i h[i] * grad_ac[j+i] = (h * grad_ac_rev)[something]
        # Actually, let's just use FFT-based approach or direct sum.
        
        # For N=400, let's do it with numpy operations:
        # Create a matrix where M[j,k] = 1 if 0<=k-j<N else 0, then term1 = M @ (h * something)
        # Simpler: term1[j] = sum_{i=0}^{N-1} h[i] * grad_ac[j+i]
        # This is like a sliding dot product.
        
        # Use np.convolve: 
        # term1 = np.convolve(grad_ac, h)[N-1 : N-1+N]  # need to check indexing
        # Actually, np.convolve(a, b)[k] = sum_i a[i] b[k-i]
        # So np.convolve(grad_ac, h)[k] = sum_i grad_ac[i] h[k-i]
        # We want term1[j] = sum_i h[i] grad_ac[j+i] = sum_i grad_ac[j+i] h[i]
        # Let i' = j+i, then i = i'-j, term1[j] = sum_{i'} grad_ac[i'] h[i'-j] where i' from j to j+N-1
        # = sum_{i'=0}^{2N-2} grad_ac[i'] h[i'-j] * I(j <= i' <= j+N-1)
        # This is not a standard convolution because of the indicator.
        
        # Let's just do it directly with broadcasting for clarity, N=400 is small enough
        # grad_h[j] = sum_k grad_ac[k] * (h[k-j] * I(0<=k-j<N) + h[j-k] * I(0<=j-k<N))
        
        # Create index arrays
        j_idx = np.arange(N)
        k_idx = np.arange(2 * N - 1)
        
        # For each j, compute sum over k
        # h[k-j] valid when 0 <= k-j < N, i.e., j <= k <= j+N-1
        # h[j-k] valid when 0 <= j-k < N, i.e., j-N+1 <= k <= j
        
        term1 = np.zeros(N)
        term2 = np.zeros(N)
        
        for j in range(N):
            # term1[j] = sum_{k=j}^{min(2N-2, j+N-1)} grad_ac[k] * h[k-j]
            k_start = j
            k_end = min(2 * N - 1, j + N)  # exclusive
            if k_start < k_end:
                term1[j] = np.dot(grad_ac[k_start:k_end], h[0:k_end - k_start])
            
            # term2[j] = sum_{k=max(0, j-N+1)}^{j} grad_ac[k] * h[j-k]
            k_start = max(0, j - N + 1)
            k_end = j + 1  # exclusive
            if k_start < k_end:
                term2[j] = np.dot(grad_ac[k_start:k_end], h[j-k_end+1:j+1][::-1])
        
        return term1 + term2
    
    # Use scipy.optimize with L-BFGS-B for better convergence
    # Objective: soft_max(convolve(h,h)) with constraint sum(h)=1
    
    # We'll use a parameterization h = h0 + v - mean(v) to enforce sum(h)=1
    # where h0 = ones(N)/N
    
    h0 = np.ones(N) / N
    
    def obj_and_grad(v):
        h = h0 + v - np.mean(v)
        ac = np.convolve(h, h)
        t_ac = t * ac
        t_ac = t_ac - np.max(t_ac)
        exp_ac = np.exp(t_ac)
        sum_exp = np.sum(exp_ac)
        f = (1.0 / t) * np.log(sum_exp)
        
        # Gradient w.r.t. h
        grad_ac = exp_ac / sum_exp
        
        # grad_h[j] = sum_k grad_ac[k] * (h[k-j] if 0<=k-j<N else 0 + h[j-k] if 0<=j-k<N else 0)
        # Use FFT-based computation
        # ac = convolve(h, h), so in frequency: AC = |H|^2
        # But we need grad_h, not grad_ac directly.
        # Let's use the direct formula but vectorized:
        
        # For efficiency, note that:
        # term1[j] = sum_{k=j}^{j+N-1} grad_ac[k] * h[k-j]
        # This is a cross-correlation of grad_ac with h, evaluated at specific lags.
        
        # Use np.correlate or manual FFT
        # Let's use FFT: 
        # Define A = grad_ac (length 2N-1), B = h (length N)
        # We want C[j] = sum_{i=0}^{N-1} A[j+i] * B[i] for j=0..N-1
        # This is like a sliding inner product.
        
        # Pad and use FFT:
        size = 2 * N
        A_pad = np.zeros(size)
        A_pad[:2*N-1] = grad_ac
        B_pad = np.zeros(size)
        B_pad[:N] = h
        
        # C[j] = sum_i A[j+i] B[i] = sum_i A[j+i] B[i]
        # This is the cross-correlation of A and B at lag j... 
        # Actually, cross-correlation R_AB[j] = sum_i A[i+j] B[i]
        # So C = R_AB[0:N]
        
        # Compute via FFT: R_AB = ifft(conj(fft(B)) * fft(A))
        FA = np.fft.fft(A_pad)
        FB = np.fft.fft(B_pad)
        R = np.fft.ifft(np.conj(FB) * FA).real
        term1 = R[:N]
        
        # term2[j] = sum_{k=max(0,j-N+1)}^{j} grad_ac[k] * h[j-k]
        # = sum_{i=0}^{j} grad_ac[j-i] * h[i] (where i = j-k, k = j-i)
        # But k >= max(0, j-N+1) means j-i >= j-N+1, i.e., i <= N-1, which is always true for i<=j<N
        # And k >= 0 means j-i >= 0, i.e., i <= j, which is satisfied.
        # So term2[j] = sum_{i=0}^{j} grad_ac[j-i] * h[i]
        # This is the standard convolution (grad_ac * h)[j]
        
        # conv[j] = sum_i grad_ac[i] * h[j-i]
        # We want sum_i grad_ac[j-i] * h[i] = conv_rev[j] where conv_rev is reverse convolution
        # Actually, sum_i grad_ac[j-i] h[i] = (h * grad_ac)[j] where * is standard conv
        # Let's verify: (h * g)[j] = sum_i h[i] g[j-i]. Yes!
        
        # So term2 = convolve(h, grad_ac) truncated to first N elements
        term2_full = np.convolve(h, grad_ac)
        term2 = term2_full[:N]
        
        grad_h = term1 + term2
        
        # Gradient w.r.t. v: h = h0 + v - mean(v), so dh/dv = I - (1/N)J
        # grad_v = (I - (1/N)J) @ grad_h = grad_h - mean(grad_h)
        grad_v = grad_h - np.mean(grad_h)
        
        return f, grad_v
    
    # Use L-BFGS-B
    try:
        result = minimize(obj_and_grad, np.zeros(N), jac=True, method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        v_opt = result.x
        h_opt = h0 + v_opt - np.mean(v_opt)
        val = compute_value(h_opt)
        if val < 2.0:
            return h_opt
    except Exception as e:
        pass
    
    return np.ones(N)
