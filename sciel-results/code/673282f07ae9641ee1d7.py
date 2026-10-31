import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 115.0

def _score_from_set(U):
    """
    Exact score calculation for a discrete set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2*max(U) + 1)
    """
    if not U:
        return 0.0
    max_val = max(U)
    if max_val < 1:
        return 0.0
    
    L = max_val + 1
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len *= 2
        
    vec = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        vec[u] = 1.0
        
    f = np.fft.fft(vec)
    
    # Sumset: Convolution of vec with itself
    conv_sum = np.fft.ifft(f * f).real
    # Count non-zero entries in the first L-1 indices? 
    # Actually |U+U| is the size of the set of sums. 
    # The convolution gives counts. We need the number of positions with count > 0.
    # Since we are dealing with sets, counts are integers.
    sums_count = np.sum(conv_sum[:L] > 0.5) # Sums range from 0 to 2*max(U), but max sum is 2*max(U).
    # Wait, U contains 0. Max sum is 2*max(U). 
    # The FFT length is >= 2*L-1, which covers up to index 2*L-2 = 2*max(U).
    # So we should check the whole relevant range.
    sums_count = np.sum(conv_sum > 0.5)
    
    # Difference set: Autocorrelation
    # U-U corresponds to autocorrelation of vec.
    # Autocorrelation is ifft(conj(f) * f).
    conv_diff = np.fft.ifft(np.conj(f) * f).real
    diffs_count = np.sum(np.abs(conv_diff) > 0.5)
    
    if sums_count == 0:
        return 0.0
        
    ratio = diffs_count / float(sums_count)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _surrogate_score(z, L, fft_len):
    """
    Compute a smooth surrogate score for continuous vector z in [0,1]^L.
    We want to maximize |U-U| / |U+U|.
    Approximate |S| for a set S represented by indicator vector s as sum(s).
    For continuous z, we use sum(z) as a proxy for the "mass" of the set.
    However, the ratio of L1 norms of the correlation/convolution is a better proxy.
    
    Let A = autocorrelation of z. A[k] = sum_i z[i] z[i+k].
    Let C = convolution of z. C[k] = sum_i z[i] z[k-i].
    
    We want to maximize sum(A) / sum(C).
    Note: sum(A) = sum(z)^2.
    sum(C) = sum(z)^2.
    This ratio is always 1. This is a constant!
    
    The issue is that |U-U| is the *support size* (number of non-zero elements), not the L1 norm.
    The L1 norm of the autocorrelation is the additive energy, not the size of the difference set.
    
    To approximate the support size of a binary vector s, we can use a smooth approximation.
    One common approach is to use a penalty that encourages sparsity or specific structures.
    
    Let's look at the structure of the problem again.
    Maximize log(|U-U|/|U+U|) / log(2M+1).
    
    If we use the L1 norm of the *binary* indicators of the correlations:
    |U-U| = sum_k 1[A[k] > 0]
    |U+U| = sum_k 1[C[k] > 0]
    
    A smooth surrogate for 1[x>0] is sigmoid(beta * x).
    Let S_diff = sum_k sigmoid(beta * A[k])
    Let S_sum = sum_k sigmoid(beta * C[k])
    
    Objective: J(z) = log(S_diff / S_sum) / log(2 * max(z) * L + 1)
    The max(z) term is tricky. We can approximate the "span" by the last non-zero element.
    Or simply fix L (the length of the vector) and optimize z within that length.
    If we fix L, the denominator is constant. We just maximize log(S_diff / S_sum).
    
    We can optimize z for a fixed L, then try different Ls.
    """
    vec = np.zeros(fft_len, dtype=np.float64)
    vec[:L] = z
    
    f = np.fft.fft(vec)
    
    A = np.fft.ifft(np.conj(f) * f).real
    C = np.fft.ifft(f * f).real
    
    beta = 5.0 # Sharpness of the sigmoid approximation
    
    # Clip to avoid overflow in exp
    A_clipped = np.clip(A, -100, 100)
    C_clipped = np.clip(C, -100, 100)
    
    S_diff = np.sum(1.0 / (1.0 + np.exp(-beta * A_clipped)))
    S_sum = np.sum(1.0 / (1.0 + np.exp(-beta * C_clipped)))
    
    if S_sum < 1e-6:
        return -1e6, S_diff, S_sum
        
    ratio = S_diff / S_sum
    if ratio <= 0:
        return -1e6, S_diff, S_sum
        
    return math.log(ratio), S_diff, S_sum

def _grad_surrogate(z, L, fft_len):
    """
    Compute gradient of the surrogate score with respect to z.
    This is expensive to do analytically, so we use finite differences for a few key dimensions
    or rely on the fact that the objective is smooth enough for a simple gradient ascent step
    if we can derive the gradient.
    
    Let's derive the gradient of J(z) = log(S_diff) - log(S_sum).
    dJ/dz_i = (1/S_diff) * dS_diff/dz_i - (1/S_sum) * dS_sum/dz_i
    
    S_diff = sum_k sigma(beta * A[k])
    dS_diff/dz_i = sum_k sigma'(beta * A[k]) * beta * dA[k]/dz_i
    
    A[k] = sum_j z[j] z[j+k] (circular or linear? Linear autocorrelation for difference set size)
    For linear autocorrelation, A[k] = sum_j z[j] z[j+k] for k>=0.
    The FFT method computes circular autocorrelation if the vector is zero-padded sufficiently.
    With zero-padding, circular autocorrelation equals linear autocorrelation.
    
    dA[k]/dz_i:
    A[k] = <z, shift_k(z)>
    If k > 0: A[k] = sum_{j} z[j] z[j+k]
    dA[k]/dz_i = z[i+k] (from first term if j=i) + z[i-k] (from second term if j+k=i => j=i-k)
    So dA[k]/dz_i = z[i+k] + z[i-k] (with boundary checks).
    
    This is O(L^2) to compute for all i and k.
    Given L is small (e.g., 100-500), this is feasible.
    """
    vec = np.zeros(fft_len, dtype=np.float64)
    vec[:L] = z
    
    f = np.fft.fft(vec)
    A = np.fft.ifft(np.conj(f) * f).real
    C = np.fft.ifft(f * f).real
    
    beta = 5.0
    
    S_diff = np.sum(1.0 / (1.0 + np.exp(-beta * A)))
    S_sum = np.sum(1.0 / (1.0 + np.exp(-beta * C)))
    
    if S_sum < 1e-6 or S_diff < 1e-6:
        return np.zeros_like(z)
    
    grad = np.zeros_like(z)
    
    # Compute derivatives of sigmoid
    # sigma'(x) = sigma(x) * (1 - sigma(x))
    sigma_A = 1.0 / (1.0 + np.exp(-beta * A))
    sigma_C = 1.0 / (1.0 + np.exp(-beta * C))
    
    # dS_diff/dz_i = sum_k beta * sigma_A[k] * (1-sigma_A[k]) * (z[i+k] + z[i-k])
    # This is a convolution of the gradient weights with z.
    
    w_diff = beta * sigma_A * (1.0 - sigma_A)
    w_sum = beta * sigma_C * (1.0 - sigma_C)
    
    # w_diff[k] is the weight for A[k].
    # dA[k]/dz_i involves z[i+k] and z[i-k].
    # Summing over k:
    # sum_k w_diff[k] * z[i+k] is a convolution of w_diff and z.
    # sum_k w_diff[k] * z[i-k] is a correlation, which is convolution of w_diff and reversed z.
    
    # We only care about z[0:L].
    # Let's compute the gradient for the first L elements.
    
    # Efficient computation using FFT for the convolution part?
    # L is small, so direct computation or numpy convolution is fine.
    
    # w_diff is defined on [0, L-1] effectively (A[k] is non-zero only for k in [-L+1, L-1])
    # But A from FFT is circular. We must be careful.
    # Since we zero-padded to fft_len >= 2L-1, A[k] for k in [0, L-1] is the linear autocorrelation.
    # A[-k] = A[k].
    
    # Let's just compute the gradient directly for each i in range(L).
    # O(L^2) is fine for L < 1000.
    
    for i in range(L):
        dS_diff_di = 0.0
        dS_sum_di = 0.0
        
        for k in range(-L + 1, L):
            # A[k] corresponds to index k in linear autocorrelation.
            # In the FFT output A, the index for lag k is k if k>=0, or fft_len+k if k<0.
            idx_A = k if k >= 0 else fft_len + k
            idx_C = k if k >= 0 else fft_len + k
            
            val_A = A[idx_A]
            val_C = C[idx_C]
            
            # Derivative of A[k] w.r.t z[i]
            # A[k] = sum_j z[j] z[j+k]
            # Term 1: j=i => z[i+k]
            # Term 2: j+k=i => j=i-k => z[i-k]
            
            val_z_ik = z[i+k] if 0 <= i+k < L else 0.0
            val_z_imk = z[i-k] if 0 <= i-k < L else 0.0
            
            dA_di = val_z_ik + val_z_imk
            dC_di = val_z_ik + val_z_imk # Convolution derivative is similar for real signals
            
            # Wait, C is convolution. C[k] = sum_j z[j] z[k-j].
            # dC[k]/dz[i]:
            # Term 1: j=i => z[k-i]
            # Term 2: k-j=i => j=k-i => z[i]
            # So dC[k]/dz[i] = z[k-i] + z[i] (if k-i in range and i in range)
            
            # Let's re-evaluate dC_di.
            # C[k] = sum_{j} z[j] z[k-j]
            # Partial derivative w.r.t z[i]:
            # 1. j=i: z[k-i]
            # 2. k-j=i => j=k-i: z[i]
            # So dC[k]/dz[i] = z[k-i] + z[i]
            
            val_z_ki = z[k-i] if 0 <= k-i < L else 0.0
            dC_di = val_z_ki + z[i]
            
            # Weights
            w_d = w_diff[idx_A]
            w_s = w_sum[idx_C]
            
            dS_diff_di += w_d * dA_di
            dS_sum_di += w_s * dC_di
            
        grad[i] = (1.0 / S_diff) * dS_diff_di - (1.0 / S_sum) * dS_sum_di
        
    return grad

def construct():
    global _DEADLINE
    best_score = 0.0
    best_set = [0, 1]
    
    # Try different lengths L
    # L is the max element + 1.
    # We want to maximize the score.
    # The denominator log(2M+1) grows with L.
    # The numerator log(|U-U|/|U+U|) is bounded.
    # For large L, the score tends to 1.
    # We want to find L and U that maximize the ratio.
    
    # Let's search over L in a range.
    # Small L: few elements, hard to get large difference set relative to sum set.
    # Large L: more room, but denominator grows.
    
    # Optimal L is likely in the range of 50 to 500.
    
    for L in [20, 50, 100, 200, 300, 500]:
        if time.time() > _DEADLINE - 20.0:
            break
            
        fft_len = 1
        while fft_len < 2 * L - 1:
            fft_len *= 2
            
        # Initialize z with random values in [0,1]
        # Or start with a Golomb ruler-like structure?
        # Random start is fine for gradient descent if we use multiple restarts.
        
        best_local_score = 0.0
        best_local_z = None
        
        # Multiple restarts
        for restart in range(5):
            if time.time() > _DEADLINE - 10.0:
                break
                
            # Initialize z
            # Start with sparse random initialization to encourage set-like structure
            z = np.zeros(L)
            # Pick ~sqrt(L) random positions to be 1
            num_elements = max(2, int(math.sqrt(L)))
            indices = np.random.choice(L, num_elements, replace=False)
            z[indices] = 1.0
            z[0] = 1.0 # Ensure 0 is in the set
            
            # Gradient Descent
            lr = 0.1
            steps = 1000
            start_time = time.time()
            
            for step in range(steps):
                if time.time() - start_time > 5.0: # 5s per restart
                    break
                    
                # Compute gradient
                grad = _grad_surrogate(z, L, fft_len)
                
                # Update
                z += lr * grad
                
                # Project to [0,1]
                z = np.clip(z, 0.0, 1.0)
                
                # Ensure 0 is 1
                z[0] = 1.0
                
                # Optional: Sparsify? 
                # The sigmoid surrogate handles this implicitly by pushing values to 0 or 1.
                
            # Evaluate final z
            # Threshold z to get binary vector
            threshold = 0.5
            binary_z = (z > threshold).astype(int)
            
            # Ensure 0 is included
            binary_z[0] = 1
            
            # Get indices
            indices = np.where(binary_z > 0)[0]
            U = [int(i) for i in indices]
            
            if len(U) < 2:
                continue
                
            score = _score_from_set(U)
            
            if score > best_local_score:
                best_local_score = score
                best_local_z = z.copy()
                best_local_U = U[:]
            
        if best_local_score > best_score:
            best_score = best_local_score
            best_set = best_local_U[:]
            
    # Final check
    if len(best_set) < 2:
        best_set = [0, 1]
        
    return sorted(set(best_set))

# To make it faster, we can reduce the number of restarts or steps if time is tight.
# The gradient computation is O(L^2). For L=500, that's 250,000 ops per step.
# 1000 steps * 5 restarts * 6 Ls = 30,000 gradient computations.
# 30,000 * 250,000 = 7.5e9 ops. This might be too slow in Python.
# We need to optimize the gradient computation.

def _grad_surrogate_fast(z, L, fft_len):
    """
    Optimized gradient computation using vectorized operations.
    """
    vec = np.zeros(fft_len, dtype=np.float64)
    vec[:L] = z
    
    f = np.fft.fft(vec)
    A = np.fft.ifft(np.conj(f) * f).real
    C = np.fft.ifft(f * f).real
    
    beta = 5.0
    
    S_diff = np.sum(1.0 / (1.0 + np.exp(-beta * A)))
    S_sum = np.sum(1.0 / (1.0 + np.exp(-beta * C)))
    
    if S_sum < 1e-6 or S_diff < 1e-6:
        return np.zeros_like(z)
    
    grad = np.zeros_like(z)
    
    sigma_A = 1.0 / (1.0 + np.exp(-beta * A))
    sigma_C = 1.0 / (1.0 + np.exp(-beta * C))
    
    w_diff = beta * sigma_A * (1.0 - sigma_A)
    w_sum = beta * sigma_C * (1.0 - sigma_C)
    
    # We need to compute:
    # dS_diff/dz_i = sum_k w_diff[k] * (z[i+k] + z[i-k])
    # This is a convolution of w_diff with z, plus w_diff with reversed z.
    
    # Let's use numpy convolve.
    # w_diff is defined on indices 0..fft_len-1.
    # We only care about the linear part, which is indices 0..L-1 and fft_len-L+1..fft_len-1.
    # But z is only defined on 0..L-1.
    
    # It's safer to stick to the O(L^2) loop if L is small, or optimize the loop with numpy.
    # Let's try to vectorize the inner loop.
    
    # For each i, we sum over k.
    # This is equivalent to:
    # grad_diff[i] = sum_k w_diff[k] * (z[i+k] + z[i-k])
    
    # Let's create a matrix W of size L x (2L-1) where W[i, k] = w_diff[k]
    # And a matrix Z of size L x (2L-1) where Z[i, k] = z[i+k] + z[i-k]
    # Then grad_diff = W @ Z
    
    # This is O(L^2) memory and time, but vectorized.
    
    k_range = np.arange(-L + 1, L)
    
    # Create indices for z[i+k] and z[i-k]
    # i is row index 0..L-1
    # k is column index -L+1..L-1
    
    i_idx = np.arange(L)[:, None]
    k_idx = k_range[None, :]
    
    # i + k
    idx_ik = i_idx + k_idx
    # i - k
    idx_imk = i_idx - k_idx
    
    # Mask valid indices
    mask_ik = (idx_ik >= 0) & (idx_ik < L)
    mask_imk = (idx_imk >= 0) & (idx_imk < L)
    
    # Safe indices
    idx_ik_safe = np.where(mask_ik, idx_ik, 0)
    idx_imk_safe = np.where(mask_imk, idx_imk, 0)
    
    z_ik = np.where(mask_ik, z[idx_ik_safe], 0.0)
    z_imk = np.where(mask_imk, z[idx_imk_safe], 0.0)
    
    # w_diff for each k
    # k_range is -L+1..L-1
    # Map k to FFT index
    k_fft_idx = np.where(k_range >= 0, k_range, fft_len + k_range)
    w_diff_vec = w_diff[k_fft_idx]
    w_sum_vec = w_sum[k_fft_idx]
    
    # w_diff_row is w_diff_vec repeated for each i
    # We can just multiply element-wise
    
    # dA_di = z_ik + z_imk
    dA_di = z_ik + z_imk
    
    # dC_di = z[k-i] + z[i]
    # k - i
    idx_ki = k_idx - i_idx
    mask_ki = (idx_ki >= 0) & (idx_ki < L)
    idx_ki_safe = np.where(mask_ki, idx_ki, 0)
    z_ki = np.where(mask_ki, z[idx_ki_safe], 0.0)
    
    # z[i] is just z repeated for each k
    z_i = np.repeat(z[:, None], L - 1, axis=1) # Wait, k_range has 2L-1 elements
    
    # Correct shape: i_idx is (L, 1), k_idx is (1, 2L-1)
    # So idx_ki is (L, 2L-1)
    # z[i] should be (L, 2L-1) where each row i is z[i]
    z_i_matrix = np.tile(z[:, None], (1, 2 * L - 1))
    
    dC_di = z_ki + z_i_matrix
    
    # Sum over k (axis 1)
    dS_diff_di = np.sum(w_diff_vec[None, :] * dA_di, axis=1)
    dS_sum_di = np.sum(w_sum_vec[None, :] * dC_di, axis=1)
    
    grad = (1.0 / S_diff) * dS_diff_di - (1.0 / S_sum) * dS_sum_di
    
    return grad

# Replace the slow gradient with the fast one
_grad_surrogate = _grad_surrogate_fast

if __name__ == "__main__":
    # Test
    U = construct()
    print(U)
    print(_score_from_set(U))
