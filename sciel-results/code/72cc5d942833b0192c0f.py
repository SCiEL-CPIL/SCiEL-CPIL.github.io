"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50
# Equal intervals on [-1/4, 1/4]
# Step size h = (1/4 - (-1/4)) / (N - 1) = (1/2) / 49
h = (0.5) / (N - 1)
x = -0.25 + h * np.arange(N)

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sequence of nonnegative step heights to maximize:
    (||f||_2^2) / (||f||_1 * ||f * f||_inf)
    
    where * denotes autoconvolution on the grid.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute the convolution indices
    # The autoconvolution (f*f)(k) = sum_j f(j) f(j+k)
    # We want to maximize ||f||_2^2 / (||f||_1 * max_k |(f*f)(k)|)
    
    # Strategy: Use a sparse, structured sequence.
    # A "Sidon-like" or low-convolution structure minimizes the peak of the autoconvolution
    # relative to the L2 norm.
    
    # Let's try a few different structural candidates and pick the best one quickly.
    # Candidate 1: Uniform (baseline)
    # Candidate 2: Sparse random (Poisson-like)
    # Candidate 3: Structured sparse (e.g., every k-th point)
    # Candidate 4: Exponentially decaying
    
    def compute_score(f):
        f = np.asarray(f, dtype=float)
        l1 = np.sum(f)
        l2_sq = np.sum(f**2)
        if l1 == 0:
            return 0.0
        # Compute autoconvolution
        # f*f at lag k is sum_{i} f[i] f[i+k]
        # We need the supremum norm of the autoconvolution.
        # The autoconvolution is defined on the same grid? 
        # The task says "supremum norm of the autoconvolution".
        # Usually, autoconvolution of a function on a domain is defined on the domain or a larger one.
        # Given the context of "heights on equal intervals", it likely refers to the discrete autocorrelation.
        # Let's assume the autoconvolution is computed as the circular or linear convolution?
        # "Autoconvolution" usually implies linear convolution f*f.
        # The support of f*f is [-1/2, 1/2]. The supremum is the max value.
        
        # Efficient computation of max autocorrelation:
        # Use FFT for speed, but for N=50, direct computation is fine and exact.
        max_conv = 0.0
        for k in range(-N + 1, N):
            # (f*f)(k) = sum_i f[i] f[i+k]
            # i ranges such that 0 <= i < N and 0 <= i+k < N
            i_min = max(0, -k)
            i_max = min(N, N - k)
            if i_min < i_max:
                val = np.dot(f[i_min:i_max], f[i_min + k:i_max + k])
                if val > max_conv:
                    max_conv = val
        
        if max_conv == 0:
            return 0.0
        return l2_sq / (l1 * max_conv)
    
    # Generate candidates
    
    best_score = 0.0
    best_f = np.ones(N)
    
    # 1. Uniform
    f_uniform = np.ones(N)
    s = compute_score(f_uniform)
    if s > best_score:
        best_score = s
        best_f = f_uniform
    
    # 2. Sparse: Only a few points active
    # Try different sparsity levels and patterns
    
    # Pattern: Every k-th point
    for k in range(2, 10):
        f_sparse = np.zeros(N)
        f_sparse[::k] = 1.0
        s = compute_score(f_sparse)
        if s > best_score:
            best_score = s
            best_f = f_sparse.copy()
    
    # 3. Random sparse
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        sparsity = np.random.uniform(0.1, 0.5)
        mask = np.random.rand(N) < sparsity
        if not np.any(mask):
            mask[0] = True
        f_rand = np.zeros(N)
        f_rand[mask] = 1.0
        s = compute_score(f_rand)
        if s > best_score:
            best_score = s
            best_f = f_rand.copy()
    
    # 4. Exponentially decaying
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
        f_exp = np.exp(-alpha * np.arange(N))
        s = compute_score(f_exp)
        if s > best_score:
            best_score = s
            best_f = f_exp.copy()
        
        # Also try centered exponential
        center = N // 2
        f_exp_c = np.exp(-alpha * np.abs(np.arange(N) - center))
        s = compute_score(f_exp_c)
        if s > best_score:
            best_score = s
            best_f = f_exp_c.copy()
    
    # 5. Greedy construction: Start with 0, add weights one by one to maximize the ratio
    # This is complex to optimize directly, but we can try a simple heuristic:
    # Place a unit weight at a position that minimizes the increase in max_conv relative to l2_sq.
    
    # Let's try a specific structured pattern: "Prime" indices or similar low-correlation sets
    # For N=50, primes are: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47
    f_prime = np.zeros(N)
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]
    for p in primes:
        if p < N:
            f_prime[p] = 1.0
    s = compute_score(f_prime)
    if s > best_score:
        best_score = s
        best_f = f_prime.copy()
    
    # 6. Try a "Golay" like sequence or Barker sequence?
    # Barker sequences have low autocorrelation sidelobes.
    # Barker sequence of length 11: 1 1 1 -1 1 -1 -1 ... (but we need nonnegative)
    # So we can't use negative weights.
    
    # 7. Try a "comb" with varying period
    # Maybe a combination of sparse and dense?
    
    # Let's do a local search on the best found so far
    # Perturb the best_f slightly
    if time.time() < deadline - 5:
        f_current = best_f.copy()
        s_current = best_score
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                # Try increasing f[i]
                f_trial = f_current.copy()
                f_trial[i] += 0.1
                s_trial = compute_score(f_trial)
                if s_trial > s_current + 1e-6:
                    f_current = f_trial
                    s_current = s_trial
                    improved = True
                # Try decreasing f[i]
                if f_current[i] > 0.1:
                    f_trial = f_current.copy()
                    f_trial[i] -= 0.1
                    s_trial = compute_score(f_trial)
                    if s_trial > s_current + 1e-6:
                        f_current = f_trial
                        s_current = s_trial
                        improved = True
        if s_current > best_score:
            best_score = s_current
            best_f = f_current
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    
    return list(best_f)
# EVOLVE-BLOCK-END
