import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 50

def _exact_ratio(h):
    """
    Compute the exact ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _optimize_weights_on_support(support, h_init=None):
    """
    Given a support (set of indices), optimize the weights on those indices
    to maximize the ratio. We fix L1 norm to 1 for scale invariance.
    h is non-negative.
    """
    n_support = len(support)
    if n_support == 0:
        return None, 0.0
    
    # If only one element, ratio is 1.0 (trivially)
    if n_support == 1:
        h = np.zeros(N)
        h[support[0]] = 1.0
        return h, 1.0

    # Parameterize h such that sum(h) = 1 and h >= 0
    # Use softmax or simplex projection.
    # Let's use a variable z in R^n_support and map to h via softmax?
    # Softmax is always positive and sums to 1.
    # h_i = exp(z_i) / sum(exp(z_j))
    
    def neg_ratio_smooth(z):
        # Compute h from z
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        sum_exp = np.sum(exp_z)
        h_support = exp_z / sum_exp
        
        # Place h into full vector
        h = np.zeros(N)
        h[support] = h_support
        
        l1 = np.sum(h) # Should be 1
        l2_sq = np.sum(h**2)
        
        # Autoconvolution
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        
        if max_conv < 1e-15:
            return 1e6
            
        ratio = l2_sq / (l1 * max_conv)
        return -ratio

    # Initial guess: uniform weights on support
    if h_init is not None:
        # Project h_init onto support
        h_supp_init = h_init[support]
        # Convert to z: log(h_i) approximately
        # Avoid log(0)
        h_supp_init = np.maximum(h_supp_init, 1e-10)
        z_init = np.log(h_supp_init / np.sum(h_supp_init))
    else:
        z_init = np.zeros(n_support)
        
    try:
        res = minimize(
            neg_ratio_smooth,
            z_init,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        z_opt = res.x
        z_max = np.max(z_opt)
        exp_z = np.exp(z_opt - z_max)
        h_support = exp_z / np.sum(exp_z)
        
        h = np.zeros(N)
        h[support] = h_support
        
        score = _exact_ratio(h)
        return h, score
    except Exception:
        return None, 0.0

def _greedy_sidon_search():
    """
    Search for supports that minimize the peak of the autoconvolution.
    A Sidon set (or B2 set) has the property that all pairwise sums are distinct.
    If the support is a Sidon set, the autoconvolution peaks at the diagonal (self-sums)
    and off-diagonals are distinct and potentially lower.
    Actually, for non-negative weights, the peak is often at the self-sums if weights are large,
    but if weights are optimized, we want to avoid constructive interference in the conv.
    
    Strategy: 
    1. Generate candidate supports of size k.
    2. For each, optimize weights.
    3. Keep the best.
    
    To make it efficient, we can use a greedy construction or random sampling.
    Given N=50, we can try small k (2 to 10).
    """
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try small supports
    for k in range(2, 12):
        # Number of random tries per k
        num_tries = 50 if k < 6 else 20
        
        for _ in range(num_tries):
            # Random support
            support = np.random.choice(N, size=k, replace=False)
            h, score = _optimize_weights_on_support(support)
            if h is not None and score > best_score:
                best_score = score
                best_h = h.copy()
                
            # Check time
            if time.time() > DEADLINE - 10:
                return best_h, best_score

    return best_h, best_score

def construct():
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + 110 # Safety margin
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # 1. Try single spike (score 1.0)
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        score = _exact_ratio(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
            
    # 2. Greedy/Sidon search for sparse supports
    np.random.seed(42)
    h_sparse, score_sparse = _greedy_sidon_search()
    if score_sparse > best_score:
        best_score = score_sparse
        best_h = h_sparse.copy()
        
    # 3. If time permits, try a few dense but structured vectors
    # e.g. Gaussian, or optimized uniform-like
    if time.time() < DEADLINE - 20:
        # Try a Gaussian bump
        x = np.linspace(-0.5, 0.5, N)
        for sigma in [0.1, 0.2, 0.3, 0.5]:
            h_gauss = np.exp(-x**2 / (2 * sigma**2))
            score_gauss = _exact_ratio(h_gauss)
            if score_gauss > best_score:
                best_score = score_gauss
                best_h = h_gauss.copy()
                
    # 4. Try a few random dense vectors and optimize locally?
    # Maybe just random dense vectors
    if time.time() < DEADLINE - 10:
        for _ in range(100):
            h_rand = np.random.rand(N)
            score_rand = _exact_ratio(h_rand)
            if score_rand > best_score:
                best_score = score_rand
                best_h = h_rand.copy()
                
    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    
    # If still zero, return uniform
    if np.sum(best_h) < 1e-10:
        best_h = np.ones(N) / N
        best_score = _exact_ratio(best_h)
        
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print(f"Score: {_exact_ratio(np.array(h))}")
    print(f"Time taken: {time.time() - time.time()}")
