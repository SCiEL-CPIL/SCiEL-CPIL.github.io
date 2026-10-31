import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # The score is 1 + log(|U-U|/|U+U|)/log(2 max U + 1).
    # We want to maximize |U-U|/|U+U| while keeping max U small.
    # We model the set U as a level set of a trigonometric polynomial.
    
    # We will search over a few different "spectral profiles" and lengths L.
    # The set U is determined by the sign of f(x) = sum c_k cos(2*pi*k*x/L)
    # or by taking the top N values.
    
    best_score = 0
    best_set = [0, 1] # Default valid set
    
    def compute_score(U):
        if not U or 0 not in U or len(U) < 2:
            return 0
        U = list(set(U))
        max_u = max(U)
        if max_u == 0:
            return 0
        
        # Use sets for exact calculation
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return 0
            
        base = 2 * max_u + 1
        if base <= 1:
            return 0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
            
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def update_best(U):
        nonlocal best_score, best_set
        # Ensure 0 is in U and U is sorted
        U = sorted(list(set(U)))
        if 0 not in U:
            # Shift to include 0? No, the problem requires 0 in U.
            # Our spectral method generates sets in [0, L-1].
            # If 0 is not in the top N, we can't just shift because the structure matters.
            # However, for the score, shifting U by -min(U) changes max(U) and the sets.
            # U-U is invariant under translation. U+U is not (it shifts).
            # Actually, |U+U| is invariant under translation.
            # Wait, if U = {1, 2}, U+U = {2, 3, 4}. If V = {0, 1}, V+V = {0, 1, 2}.
            # Sizes are the same.
            # But max(U) changes. The score depends on max(U).
            # To maximize score, we want small max(U).
            # So if we find a set, we should shift it so min(U)=0.
            if not U:
                return
            min_u = min(U)
            U = [x - min_u for x in U]
            if 0 not in U:
                return
        
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U

    # Spectral Optimization Strategy
    # We define f(x) on x in [0, L-1].
    # f(x) = sum_{k=1}^K c_k cos(2*pi*k*x/L)
    # We want to choose coefficients c_k to maximize the "difference-ness" of the level set.
    # Proxy objective: Maximize ||Autocorr(f)||_1 / ||Conv(f)||_1
    # Autocorr(f) = f * f(-x). Conv(f) = f * f(x).
    # For real even f, these are related.
    # Actually, let's just optimize the coefficients to make the set "spread out" (large diff)
    # and "clustered" (small sum)? No, we want large diff and small sum.
    # Sets with large difference sets and small sum sets are "Sidon-like" or "Bhattacharya-like"?
    # Actually, random sets have |U+U| ~ N^2/2 and |U-U| ~ N^2/2. Ratio ~ 1.
    # To get ratio > 1, we need |U-U| > |U+U|.
    # This happens if the set has a structure that makes differences unique but sums collide?
    # Or differences spread out but sums cluster?
    # If U is an arithmetic progression {0, 1, ..., N-1}:
    # U+U = {0, ..., 2N-2}, size 2N-1.
    # U-U = {-(N-1), ..., N-1}, size 2N-1.
    # Ratio = 1. Score = 1.
    
    # To get ratio > 1, we need |U-U| > |U+U|.
    # Consider U = {0, 1, 3, 4, 6, 7, ...} (blocks of 2).
    # Or U = {0, 2, 4, ...} (even numbers).
    # U = {0, 2, ..., 2N-2}.
    # U+U = {0, 2, ..., 4N-4}. Size 2N-1.
    # U-U = {-2N+2, ..., 2N-2} even. Size 2N-1.
    # Ratio 1.
    
    # It turns out for any set, |U-U| >= |U+U| is NOT always true.
    # But for sets containing 0, |U-U| = |U+U|?
    # No. Let U = {0, 1, 3}.
    # U+U = {0, 1, 2, 3, 4, 6}. Size 6.
    # U-U = {-3, -2, -1, 0, 1, 2, 3}. Size 7.
    # Ratio 7/6 > 1.
    # Score = 1 + log(7/6)/log(7) = 1 + 0.154/1.946 = 1.079.
    
    # So we want sets where sums collide more than differences.
    # This suggests sets that are "convex" or have gaps?
    # {0, 1, 3} has a gap at 2.
    
    # Let's try to optimize the coefficients c_k to maximize this ratio proxy.
    # We will use a simple gradient ascent on the coefficients.
    
    # Define a proxy objective function using FFT for speed.
    def proxy_objective(coeffs, L):
        # coeffs are c_1...c_K
        # f(x) = sum c_k cos(2*pi*k*x/L)
        # We want to evaluate ||corr||_1 / ||conv||_1
        # Note: For real signals, autocorrelation is real.
        # We can compute f as a vector of length L.
        
        x = np.arange(L)
        f = np.zeros(L)
        for k, c in enumerate(coeffs, 1):
            f += c * np.cos(2 * np.pi * k * x / L)
        
        # Compute autocorrelation and convolution
        # Use FFT
        F = np.fft.fft(f, n=2*L)
        power = np.abs(F)**2
        corr = np.fft.ifft(power).real[:L]
        # corr[k] = sum f[i] f[i+k]
        
        # Convolution f * f
        # This is related to the power spectrum but not the same.
        # conv = ifft( F * F )? No, that's circular convolution.
        # Linear convolution:
        conv = np.fft.ifft(F * F).real[:L]
        
        # We want to maximize sum |corr| / sum |conv| ?
        # Actually, the score uses the SIZE of the sets.
        # The size of the difference set is the number of k where corr[k] != 0?
        # No, that's for the indicator function.
        # If f is the indicator function (0/1), then corr[k] is the number of pairs (i, j) with i-j=k.
        # So |U-U| is the number of non-zero entries in corr.
        # Similarly |U+U| is the number of non-zero entries in conv.
        
        # So we want to maximize count(corr != 0) / count(conv != 0).
        # This is a step function, hard to optimize.
        # Approximation: Maximize (sum |corr|)^2 / L / (sum |conv|)^2 / L ?
        # Or simply maximize the "spread" of corr and "compactness" of conv.
        
        # Let's try: Objective = sum(|corr|) / sum(|conv|)
        # This favors large dynamic range in corr (many large values) and small in conv?
        # No, we want many non-zeros in corr and few in conv.
        
        # Alternative: Use the L1 norm of the *indicator* of the support.
        # support_corr = (np.abs(corr) > 1e-9).sum()
        # support_conv = (np.abs(conv) > 1e-9).sum()
        # This is discontinuous.
        
        # Let's use a smooth proxy:
        # Obj = sum(exp(-|corr|/scale)) ... no.
        
        # Let's stick to the simple idea:
        # We want the set to be "difference rich" and "sum poor".
        # This is characteristic of sets with large gaps.
        # Let's try optimizing the coefficients to maximize the variance of the autocorrelation
        # while minimizing the variance of the convolution?
        
        # Actually, let's just do a random search over the coefficients and pick the best one
        # based on the exact score of the resulting top-N set.
        # This is "Black-box optimization over the spectral coefficients".
        
        return 0.0 # Placeholder

    # Since the proxy is hard, let's use the spectral model to GENERATE candidate sets
    # and then use the exact score to select the best one.
    # We will sample random coefficient vectors, generate the set, and score it.
    # This is a different search space than integer local search.
    
    # We will explore a few L values and K values.
    # L: 10 to 50
    # K: 1 to 5
    
    # To make it efficient, we will do a coarse random search in coefficient space,
    # then a local search in the integer domain on the best candidates.
    
    # But the directive says "different design". So let's stick to the spectral generation.
    
    # Strategy:
    # 1. Randomly pick L and K.
    # 2. Randomly pick coefficients c_1...c_K in [-1, 1].
    # 3. Compute f(x) for x in 0..L-1.
    # 4. Threshold f(x) to get a set U.
    #    - Option A: Top N values.
    #    - Option B: f(x) > 0.
    # 5. Shift U to start at 0.
    # 6. Score U.
    # 7. Keep best.
    
    # To improve this, we can do a simple gradient ascent on the coefficients
    # using a smooth approximation of the score.
    # Approx Score = log( (sum |corr|)^2 / L ) / log( (sum |conv|)^2 / L ) ?
    # No, that's not right.
    
    # Let's try a different proxy:
    # Maximize the ratio of the number of non-zero autocorrelation lags to non-zero convolution lags.
    # We can approximate the "number of non-zeros" by the L1 norm if we assume the values are 0 or 1.
    # But they are not.
    
    # Let's just do a smarter random search.
    # We will bias the coefficients to create "spiky" functions (high peaks, low valleys).
    # This corresponds to sets with clusters and gaps.
    
    random.seed(42)
    
    for L in [10, 15, 20, 25, 30, 40, 50]:
        for K in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 10:
                break
            
            # Try 100 random coefficient vectors
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Generate random coefficients
                # Use a distribution that favors high frequencies?
                coeffs = [random.uniform(-1, 1) for _ in range(K)]
                
                x = np.arange(L)
                f = np.zeros(L)
                for k, c in enumerate(coeffs, 1):
                    f += c * np.cos(2 * np.pi * k * x / L)
                
                # Get top N values
                # Try different N
                # N should be around L/2
                for N in [L//4, L//3, L//2, 2*L//3]:
                    if N < 2 or N > L:
                        continue
                    # Get indices of top N values
                    top_indices = np.argsort(f)[-N:]
                    U = sorted([int(i) for i in top_indices])
                    
                    # Shift to start at 0
                    if U:
                        min_u = min(U)
                        U = [x - min_u for x in U]
                        
                        # Ensure 0 is in U
                        if 0 in U:
                            update_best(U)

    # After spectral generation, do a quick local search on the best set found
    # to polish it.
    if time.time() < deadline - 5 and best_set:
        current = list(best_set)
        curr_score = best_score
        
        # Simple local search
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            improved = False
            max_cur = max(current)
            
            # Try adding
            if len(current) < 20:
                for new_el in range(1, max_cur + 5):
                    if new_el not in current:
                        cand = sorted(current + [new_el])
                        s = compute_score(cand)
                        if s > curr_score:
                            current = cand
                            curr_score = s
                            improved = True
                            break
            # Try replacing
            if not improved:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    for new_el in range(1, max_cur + 5):
                        if new_el not in current:
                            cand = list(current)
                            cand[i] = new_el
                            cand = sorted(set(cand))
                            if 0 in cand:
                                s = compute_score(cand)
                                if s > curr_score:
                                    current = cand
                                    curr_score = s
                                    improved = True
                                    break
                    if improved:
                        break
            # Try removing
            if not improved and len(current) > 2:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    if s > curr_score:
                        current = cand
                        curr_score = s
                        improved = True
                        break
            
            if not improved:
                break
        
        if curr_score > best_score:
            best_score = curr_score
            best_set = current

    return list(best_set)
