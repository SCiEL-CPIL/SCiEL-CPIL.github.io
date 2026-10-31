import numpy as np
import time

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Strategy:
    # 1. Construct a "Sidon-like" set S greedily. A Sidon set minimizes the number of
    #    collisions in the autoconvolution (i.e., distinct sums i+j). For a set of size K,
    #    if all sums are distinct (except trivial symmetries), the peak of the autoconvolution
    #    is minimized relative to the energy.
    # 2. For the chosen support S, optimize the weights w_i (i in S) to maximize the ratio.
    #    We use a fast coordinate ascent / hill climbing method on the simplex (sum w = 1, w >= 0).
    #    This avoids the slow stochastic search for supports and focuses computational budget
    #    on optimizing the weights for a structurally good support.

    def greedy_sidon_search(max_K=15, rng_seed=42):
        """
        Greedily build a set of indices to minimize autoconvolution collisions.
        We prefer indices that are 'far apart' in a modular sense or simply
        minimize the number of pairs (i, j) such that i+j equals another pair's sum.
        """
        rng = np.random.default_rng(rng_seed)
        S = []
        
        # Start with a random element
        S.append(rng.integers(0, N))
        
        for _ in range(max_K - 1):
            best_candidate = None
            best_score = -np.inf
            
            # Try adding each remaining index
            candidates = [i for i in range(N) if i not in S]
            if not candidates:
                break
                
            for cand in candidates:
                new_S = S + [cand]
                # Evaluate "Sidon-ness": count collisions in sums
                sums = {}
                for i_idx in range(len(new_S)):
                    for j_idx in range(i_idx, len(new_S)):
                        s = new_S[i_idx] + new_S[j_idx]
                        if s in sums:
                            sums[s] += 1
                        else:
                            sums[s] = 1
                
                # We want to minimize the max collision count or total collisions
                # A better metric: minimize the peak of the autoconvolution of the indicator vector
                h_ind = np.zeros(N)
                for idx in new_S:
                    h_ind[idx] = 1.0
                ratio = compute_ratio(h_ind)
                # For indicator vectors, l1=K, l2_sq=K, so ratio = K / (K * peak) = 1/peak.
                # Maximizing ratio is equivalent to minimizing peak.
                if ratio > best_score:
                    best_score = ratio
                    best_candidate = cand
            
            if best_candidate is not None:
                S.append(best_candidate)
            else:
                break
                
        return S

    # Step 1: Find a good support set
    # We might try a few different greedy starts or sizes
    best_support = []
    best_support_score = -np.inf
    
    # Try different max_K values to see which yields the best ratio with uniform weights first
    for max_K in range(5, 16):
        S = greedy_sidon_search(max_K=max_K)
        if not S:
            continue
        h_unif = np.zeros(N)
        for idx in S:
            h_unif[idx] = 1.0 / len(S)
        
        score_unif = compute_ratio(h_unif)
        if score_unif > best_support_score:
            best_support_score = score_unif
            best_support = S.copy()
            
        # Check time
        if time.time() > deadline - 10.0:
            break

    if not best_support:
        # Fallback: random set
        rng = np.random.default_rng(42)
        best_support = list(rng.choice(N, size=10, replace=False))
        best_support_score = -np.inf

    # Step 2: Optimize weights for the best support
    # We work on the simplex: w_i >= 0, sum(w_i) = 1
    K = len(best_support)
    w = np.ones(K) / K
    
    def get_h(w_vec):
        h = np.zeros(N)
        for i, idx in enumerate(best_support):
            h[idx] = w_vec[i]
        return h

    current_score = compute_ratio(get_h(w))
    
    # Coordinate Ascent / Hill Climbing
    # For each coordinate i, we can adjust w_i by taking mass from another coordinate j.
    # To maximize efficiency, we can try to increase w_i and decrease w_j proportionally or to zero.
    # A simpler robust method: Random perturbation on the simplex with acceptance if improved.
    
    rng_opt = np.random.default_rng(123)
    T = 0.05
    decay = 0.999
    
    for step in range(20000):
        if time.time() > deadline - 5.0:
            break
            
        # Perturb: pick two indices i, j and transfer mass
        i = rng_opt.integers(0, K)
        j = rng_opt.integers(0, K)
        if i == j:
            continue
            
        # Transfer amount
        # We can transfer a fraction of w_j to w_i
        # Or add random noise and project
        w_new = w.copy()
        
        # Strategy: Add small noise to all, project to simplex
        noise = rng_opt.normal(0, 0.05 * T, size=K)
        w_new += noise
        w_new = np.maximum(w_new, 0)
        s = np.sum(w_new)
        if s < 1e-10:
            continue
        w_new /= s
        
        h_new = get_h(w_new)
        score_new = compute_ratio(h_new)
        
        if score_new > current_score:
            w = w_new
            current_score = score_new
            T *= 0.99 # Cool down slightly on improvement
        else:
            # Accept with probability exp((new - old)/T)
            delta = score_new - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng_opt.random() < prob:
                    w = w_new
                    current_score = score_new
            T *= decay # Cool down
            
    # Final polish: Deterministic coordinate ascent
    # For each i, try to increase w_i by taking from the largest other w_j
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(K):
            # Find the best j to transfer mass from
            best_j = -1
            best_score_i = current_score
            
            # Try transferring from each other j
            for j in range(K):
                if i == j:
                    continue
                if w[j] < 1e-6:
                    continue
                    
                # Try transferring a fraction of w[j] to w[i]
                # We can optimize the fraction alpha in [0, 1]
                # New w_i = w_i + alpha * w_j
                # New w_j = w_j * (1 - alpha)
                # This keeps sum constant.
                
                # Simple heuristic: try alpha = 0.1, 0.5, 0.9
                for alpha in [0.1, 0.5, 0.9]:
                    w_trial = w.copy()
                    w_trial[i] += alpha * w[j]
                    w_trial[j] *= (1 - alpha)
                    
                    h_trial = get_h(w_trial)
                    score_trial = compute_ratio(h_trial)
                    
                    if score_trial > best_score_i:
                        best_score_i = score_trial
                        best_j = j
                        best_alpha = alpha
                        
            if best_j != -1:
                w[i] += best_alpha * w[best_j]
                w[best_j] *= (1 - best_alpha)
                current_score = best_score_i
                improved = True

    # Construct final h
    h_final = np.zeros(N)
    for i, idx in enumerate(best_support):
        h_final[idx] = w[i]
        
    # Ensure non-negative and valid
    h_final = np.maximum(h_final, 0)
    
    # If for some reason score is 0, fallback to uniform on support
    if compute_ratio(h_final) < 1e-6:
        h_final = np.zeros(N)
        for idx in best_support:
            h_final[idx] = 1.0 / len(best_support)

    return list(h_final)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
