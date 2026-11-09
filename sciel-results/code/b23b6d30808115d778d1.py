"""Erdos minimum overlap: Bimodal Simulated Annealing Strategy."""

import numpy as np
import time

N = 100

def _evaluate_binary(binary_seq):
    """
    Evaluate the maximum overlap for a given binary sequence of 0s and 1s.
    The heights are a where bit=1, and b where bit=0.
    Sum constraint: sum(h) = N/2.
    If k ones, k*a + (N-k)*b = N/2.
    Also a+b=1 (complementarity for the '1-h' part? No, '1-h' is just 1-h_i).
    
    Wait, the problem says "shifted 1-h".
    h_i in [0,1].
    Overlap at shift k: sum_i h_i * (1 - h_{(i+k)%N}).
    
    If we restrict h to be bimodal {a, b}, we can solve for a and b given the count of ones.
    Let x be the count of indices where h_i = a.
    Then (N-x) indices have h_i = b.
    Sum = x*a + (N-x)*b = N/2.
    
    We also have the freedom to choose a and b as long as 0 <= a,b <= 1.
    To maximize the flexibility of the optimization, we can fix the structure (binary mask)
    and then optimize a and b analytically or via a small search?
    Actually, for a fixed binary mask, the overlap is a quadratic function of a and b.
    Let's analyze the overlap for a fixed mask.
    
    Let S be the set of indices where h_i = a. Let T be the set where h_i = b.
    |S| = x, |T| = N-x.
    Constraint: x*a + (N-x)*b = N/2.
    
    Overlap O_k = sum_{i in S} a * (1 - h_{i+k}) + sum_{i in T} b * (1 - h_{i+k}).
    
    Let's break down the terms based on where i+k falls.
    This depends on the autocorrelation of the mask.
    
    Alternatively, and more simply:
    Since we are doing Simulated Annealing on the binary mask, we can just pick a and b
    that satisfy the sum constraint and are within [0,1].
    To minimize the max overlap, we might want a and b to be close to 0.5?
    Or maybe far apart?
    
    If a=0.5, b=0.5, overlap is always 0.25 * N = 25. Score 1.0.
    If we vary a and b, we can potentially lower the peak.
    
    For a fixed mask, let's find the optimal a and b.
    Let u = a - 0.5, v = b - 0.5.
    Sum constraint: x(0.5+u) + (N-x)(0.5+v) = N/2
    => N/2 + x*u + (N-x)*v = N/2
    => x*u + (N-x)*v = 0
    => v = -x/(N-x) * u
    
    So b is determined by a (or u).
    We need 0 <= a <= 1 and 0 <= b <= 1.
    a = 0.5 + u
    b = 0.5 - x/(N-x) * u
    
    If x=0 or x=N, this is degenerate (constant sequence).
    Assume 0 < x < N.
    
    The overlap O_k = sum h_i (1 - h_{i+k}) = sum h_i - sum h_i h_{i+k}.
    Sum h_i = N/2 is constant.
    So minimizing max O_k is equivalent to maximizing min (sum h_i h_{i+k})?
    No, we want to MINIMIZE the MAX overlap.
    O_k = N/2 - C_k, where C_k = sum h_i h_{i+k} is the autocorrelation at lag k.
    Minimizing max O_k is equivalent to maximizing min C_k?
    No, max (N/2 - C_k) = N/2 - min C_k.
    So we want to MAXIMIZE the minimum autocorrelation C_k.
    
    Wait, is that right?
    If C_k is large, O_k is small.
    We want all O_k to be small.
    So we want all C_k to be large.
    Specifically, we want the minimum C_k to be as large as possible.
    
    So the objective is: Maximize min_{k} C_k(h).
    
    For a fixed binary mask and variable u:
    h_i = 0.5 + u * s_i, where s_i = 1 if i in S, s_i = -x/(N-x) if i in T?
    No, let's stick to a and b.
    h_i = a if s_i=1, b if s_i=0.
    
    C_k = sum h_i h_{i+k}.
    This is a quadratic in u.
    C_k(u) = A_k * u^2 + B_k * u + C_k(0).
    
    We want to choose u (and thus a,b) to maximize the minimum of these quadratics over k=0..N-1.
    This is a convex optimization problem (maximizing a concave function? No, min of convex functions is convex, we want to maximize it?
    Wait, C_k(u) is quadratic.
    h_i h_{i+k} is linear in a,b for fixed i? No, product of two variables.
    But a and b are linked by a linear constraint.
    So C_k is quadratic in u.
    The minimum of a set of convex functions is convex.
    We want to maximize a convex function? That's generally hard (non-convex optimization).
    However, the range of u is small. We can just scan u or use a ternary search if it's unimodal?
    It might not be unimodal.
    
    Alternative: Just include u in the SA search?
    Or, simpler:
    Fix the binary mask.
    Calculate the "structure" of the mask.
    Actually, for the purpose of SA, we can just evaluate the "best possible score" for a given mask by optimizing u.
    Since N=100, calculating C_k for all k is O(N^2) = 10,000 ops.
    Optimizing u: We can sample u or do a golden section search on the function f(u) = min_k C_k(u).
    
    Let's refine the plan:
    1. State: Binary mask s (length N, 0/1).
    2. Move: Flip a random bit.
    3. Evaluation:
       a. Determine x = sum(s). If x=0 or x=N, return score 1.0 (or bad).
       b. Define function C_min(u):
          - Calculate a = 0.5 + u, b = 0.5 - x/(N-x)*u.
          - Check bounds 0<=a,b<=1. If invalid, return -inf (or large penalty).
          - Construct h vector (or just use s and a,b to compute correlations efficiently).
          - Compute autocorrelation C_k for all k.
          - Return min(C_k).
       c. Optimize u to maximize C_min(u).
          - The domain of u is determined by bounds on a and b.
          - a in [0,1] => u in [-0.5, 0.5].
          - b in [0,1] => 0.5 - r*u in [0,1] => -0.5/r <= u <= 0.5/r where r = x/(N-x).
          - So u is in [max(-0.5, -0.5/r), min(0.5, 0.5/r)].
          - We can use `scipy.optimize.minimize_scalar` on -C_min(u) or just a simple grid search + refinement since it's 1D.
          - Given the 120s limit, and SA steps ~10k, we need eval to be fast.
          - Grid search of 100 points for u might be too slow if done in Python loops.
          - Vectorized numpy:
            h = np.where(s==1, a, b)
            C = np.correlate(h, h, mode='full')[N-1:] # Circular?
            np.correlate is linear. We need circular.
            FFT is faster: O(N log N).
            H = np.fft.fft(h)
            C = np.fft.ifft(H * np.conj(H)).real
            C_min = np.min(C)
            
    4. SA Loop:
       - Start with random mask.
       - Optimize u for initial mask.
       - Iterate:
         - Flip bit.
         - Optimize u for new mask.
         - Accept if better or with probability exp((new-old)/T).
    
    Is optimizing u at every step too slow?
    N=100. FFT is fast.
    Grid search for u: 50 points.
    50 * (FFT + min) per step.
    10,000 steps * 50 * (100 log 100) ~ 10^4 * 50 * 700 ~ 3.5 * 10^8 ops.
    Might be borderline in Python/Numpy if not careful.
    
    Optimization:
    We don't need to re-optimize u from scratch. We can keep the previous u as a starting point?
    Or, we can observe that the optimal u often stays near 0 if the mask is balanced?
    Actually, if we just fix u=0 (a=b=0.5), we are back to the trivial solution.
    The gain comes from u != 0.
    
    Let's try a simpler approach first:
    Just search the binary mask, and for each mask, find the optimal u via a fast 1D optimization.
    To speed up, we can limit the number of u samples.
    
    Also, note that the score is `seed / value`.
    Seed is likely 25 (for N=100, max overlap 25).
    Wait, the task says "score is the upper bound on C5".
    And "1.0 equals the seed program".
    Seed program returns 0.5 everywhere.
    Overlap = sum 0.5 * 0.5 = 0.25 * 100 = 25.
    So seed score = 25/25 = 1.0.
    We want to minimize the overlap value.
    
    Let's implement the SA with FFT-based evaluation and a coarse grid search for u.
    
    Deadline guard:
    Start time.
    While time < deadline:
       ...
    
    Return best h found.
    """
    start_time = time.time()
    deadline = 110.0  # 10s safety margin
    
    # Initial state
    # Try to start with a pattern that has good autocorrelation properties?
    # Random is fine for SA.
    s = np.random.randint(0, 2, N)
    
    def get_optimal_u_and_score(mask):
        """
        Returns (best_u, best_min_c, best_h) for a given binary mask.
        """
        x = np.sum(mask)
        if x == 0 or x == N:
            return 0.0, -np.inf, np.full(N, 0.5)
        
        r = x / (N - x)
        
        # Bounds for u
        # a = 0.5 + u in [0,1] => u in [-0.5, 0.5]
        # b = 0.5 - r*u in [0,1] => u in [-0.5/r, 0.5/r]
        
        u_min = max(-0.5, -0.5 / r)
        u_max = min(0.5, 0.5 / r)
        
        if u_min >= u_max:
            return 0.0, -np.inf, np.full(N, 0.5)
            
        # Coarse grid search
        # We can use 20 points for speed
        us = np.linspace(u_min, u_max, 20)
        
        best_min_c = -np.inf
        best_u = 0.0
        best_h = None
        
        for u in us:
            a = 0.5 + u
            b = 0.5 - r * u
            
            # Construct h
            # Instead of full vector construction every time, we can precompute?
            # No, mask changes.
            h = np.where(mask == 1, a, b)
            
            # Circular Autocorrelation
            # C_k = sum h_i h_{i+k}
            # Using FFT
            H = np.fft.fft(h)
            C = np.fft.ifft(H * np.conj(H)).real
            # Note: C from ifft is the circular autocorrelation.
            # The values might have tiny imaginary parts, we take real.
            min_c = np.min(C)
            
            if min_c > best_min_c:
                best_min_c = min_c
                best_u = u
                best_h = h.copy()
                
        # Refine around best_u?
        # Given time constraints, coarse grid might be enough to escape the local min of u=0.
        # Let's do a small refinement.
        if best_h is not None:
            # Local search around best_u
            step = (u_max - u_min) / 20.0
            for _ in range(3):
                candidates = [best_u - step, best_u, best_u + step]
                candidates = [c for c in candidates if u_min <= c <= u_max]
                local_best_min_c = best_min_c
                local_best_u = best_u
                local_best_h = best_h
                
                for u in candidates:
                    a = 0.5 + u
                    b = 0.5 - r * u
                    h = np.where(mask == 1, a, b)
                    H = np.fft.fft(h)
                    C = np.fft.ifft(H * np.conj(H)).real
                    min_c = np.min(C)
                    if min_c > local_best_min_c:
                        local_best_min_c = min_c
                        local_best_u = u
                        local_best_h = h.copy()
                        
                if local_best_u == best_u and local_best_min_c == best_min_c:
                    break
                best_u = local_best_u
                best_min_c = local_best_min_c
                best_h = local_best_h
                step /= 2.0

        return best_u, best_min_c, best_h

    # Initial evaluation
    current_u, current_min_c, current_h = get_optimal_u_and_score(s)
    best_u, best_min_c, best_h = current_u, current_min_c, current_h
    best_score_val = N/2 - best_min_c # Overlap value
    
    # SA Parameters
    T = 0.1
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        # Flip a random bit
        idx = np.random.randint(N)
        s[idx] = 1 - s[idx]
        
        new_u, new_min_c, new_h = get_optimal_u_and_score(s)
        
        # Acceptance
        if new_min_c > current_min_c:
            current_u, current_min_c, current_h = new_u, new_min_c, new_h
            if current_min_c > best_min_c:
                best_u, best_min_c, best_h = current_u, current_min_c, current_h
        else:
            # Delta is in min_c. We want to maximize min_c.
            # So delta = new - current (negative)
            delta = new_min_c - current_min_c
            if np.random.rand() < np.exp(delta / T):
                current_u, current_min_c, current_h = new_u, new_min_c, new_h
                
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart? Or just keep going.
            
    # Final check
    if best_h is None:
        return np.full(N, 0.5)
        
    return best_h

def construct():
    return _evaluate_binary_main()

# Wrapper to handle the function name requirement
def _evaluate_binary_main():
    # The logic above was inside a docstring. Let's move it to a proper function.
    start_time = time.time()
    deadline = 110.0
    
    s = np.random.randint(0, 2, N)
    
    def get_optimal_u_and_score(mask):
        x = np.sum(mask)
        if x == 0 or x == N:
            return 0.0, -np.inf, np.full(N, 0.5)
        
        r = x / (N - x)
        
        u_min = max(-0.5, -0.5 / r)
        u_max = min(0.5, 0.5 / r)
        
        if u_min >= u_max:
            return 0.0, -np.inf, np.full(N, 0.5)
            
        us = np.linspace(u_min, u_max, 20)
        
        best_min_c = -np.inf
        best_u = 0.0
        best_h = None
        
        for u in us:
            a = 0.5 + u
            b = 0.5 - r * u
            h = np.where(mask == 1, a, b)
            H = np.fft.fft(h)
            C = np.fft.ifft(H * np.conj(H)).real
            min_c = np.min(C)
            
            if min_c > best_min_c:
                best_min_c = min_c
                best_u = u
                best_h = h.copy()
                
        if best_h is not None:
            step = (u_max - u_min) / 20.0
            for _ in range(3):
                candidates = [best_u - step, best_u, best_u + step]
                candidates = [c for c in candidates if u_min <= c <= u_max]
                local_best_min_c = best_min_c
                local_best_u = best_u
                local_best_h = best_h
                
                for u in candidates:
                    a = 0.5 + u
                    b = 0.5 - r * u
                    h = np.where(mask == 1, a, b)
                    H = np.fft.fft(h)
                    C = np.fft.ifft(H * np.conj(H)).real
                    min_c = np.min(C)
                    if min_c > local_best_min_c:
                        local_best_min_c = min_c
                        local_best_u = u
                        local_best_h = h.copy()
                        
                if local_best_u == best_u and local_best_min_c == best_min_c:
                    break
                best_u = local_best_u
                best_min_c = local_best_min_c
                best_h = local_best_h
                step /= 2.0

        return best_u, best_min_c, best_h

    current_u, current_min_c, current_h = get_optimal_u_and_score(s)
    best_u, best_min_c, best_h = current_u, current_min_c, current_h
    
    T = 0.1
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        idx = np.random.randint(N)
        s[idx] = 1 - s[idx]
        
        new_u, new_min_c, new_h = get_optimal_u_and_score(s)
        
        if new_min_c > current_min_c:
            current_u, current_min_c, current_h = new_u, new_min_c, new_h
            if current_min_c > best_min_c:
                best_u, best_min_c, best_h = current_u, current_min_c, current_h
        else:
            delta = new_min_c - current_min_c
            if np.random.rand() < np.exp(delta / T):
                current_u, current_min_c, current_h = new_u, new_min_c, new_h
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    if best_h is None:
        return np.full(N, 0.5)
        
    return best_h
