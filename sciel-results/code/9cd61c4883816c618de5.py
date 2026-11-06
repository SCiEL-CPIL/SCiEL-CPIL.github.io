"""Optimized program: GR sequence with verification and local polish."""
import time
import math

def construct():
    start = time.time()
    deadline = 120.0  # safety margin
    
    # Golay-Rudin-Shapiro construction for 24 coefficients
    # P_0 = [1], Q_0 = [1]
    # P_{n+1} = P_n + Q_n, Q_{n+1} = P_n - Q_n
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    coeffs = p[:24]
    
    elapsed = time.time() - start
    if elapsed > deadline:
        return coeffs
    
    # Local search: try single-coefficient flips to see if any improve
    # Evaluate sup-norm by sampling on unit circle
    def evaluate(coeffs):
        n = len(coeffs)
        # Dense sampling: 2048 points
        angles = [2.0 * math.pi * k / 2048 for k in range(2048)]
        max_val = 0.0
        for theta in angles:
            c, s = math.cos(theta), math.sin(theta)
            # Compute P(e^{i*theta}) = sum c_k * e^{i*k*theta}
            # Use iterative complex multiplication
            re, im = 0.0, 0.0
            ck, sk = 1.0, 0.0  # e^{i*k*theta} for k=0
            for k in range(n):
                re += coeffs[k] * ck
                im += coeffs[k] * sk
                # Update ck, sk for next k
                new_ck = ck * c - sk * s
                new_sk = ck * s + sk * c
                ck, sk = new_ck, new_sk
            val = math.sqrt(re*re + im*im)
            if val > max_val:
                max_val = val
        return max_val / math.sqrt(n)
    
    current_score = evaluate(coeffs)
    best_coeffs = list(coeffs)
    best_score = current_score
    
    # Try single flips
    for i in range(24):
        if time.time() - start > deadline - 10:
            break
        trial = list(coeffs)
        trial[i] = -trial[i]
        trial_score = evaluate(trial)
        if trial_score < best_score - 1e-12:
            best_score = trial_score
            best_coeffs = trial
    
    # Try double flips (limited)
    if time.time() - start < deadline - 30:
        for i in range(24):
            for j in range(i+1, 24):
                if time.time() - start > deadline - 15:
                    break
                trial = list(best_coeffs)
                trial[i] = -trial[i]
                trial[j] = -trial[j]
                trial_score = evaluate(trial)
                if trial_score < best_score - 1e-12:
                    best_score = trial_score
                    best_coeffs = trial
            if time.time() - start > deadline - 15:
                break
    
    return best_coeffs
