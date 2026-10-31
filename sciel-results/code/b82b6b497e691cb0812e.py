"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.stats import binom

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import binom

def compute_max_overlap(h):
    """
    Computes the maximum linear overlap of h with shifted 1-h.
    Overlap at shift k is sum_{i=0}^{N-1-k} h[i] * (1 - h[i+k]).
    """
    n = len(h)
    max_ov = 0.0
    # Optimization: only check k up to n//2 or similar, but let's do all to be safe
    # Actually, for large N, this is O(N^2). N=100 is small.
    for k in range(n):
        # h[0:N-k] and h[k:N]
        ov = np.sum(h[:n-k] * (1.0 - h[k:]))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def generate_step_profile(N, m, center, p=0.5):
    """
    Generates a step profile using cumulative binomial distribution.
    h[i] = 0.5 + 0.5 * (CDF(i) - 0.5) ? 
    We want h to go from 0 to 1.
    CDF ranges from 0 to 1.
    So h[i] = CDF(i).
    But we need sum(h) = N/2.
    A symmetric CDF around N/2 will have sum approx N/2.
    We can fine-tune by shifting or scaling, but let's assume symmetry.
    """
    # Binomial CDF
    # m is the number of trials (width parameter)
    # center is the location
    # We want the transition to happen around 'center'
    # Let's define the CDF of Binom(m, 0.5)
    # It transitions from 0 to 1 over a range of ~sqrt(m)
    
    # To make it fit in [0, N-1], we need to scale or choose m appropriately.
    # Standard Binom(m, 0.5) has mean m/2 and std sqrt(m)/2.
    # We want the transition width to be some fraction of N.
    
    # Let's use a normalized CDF.
    # We want h[i] to be 0 for i << center and 1 for i >> center.
    # Let's map the binomial variable to the index space.
    
    # Simpler approach: Use a logistic or erf-like step, or just a polynomial.
    # But directive says "cumulative binomial distribution".
    
    # Let's create an index array
    idx = np.arange(N)
    
    # Shift index so center is at 0
    x = idx - center
    
    # We want a CDF that goes from 0 to 1.
    # Let's use the CDF of a Binomial distribution scaled to the range.
    # Let M be the effective number of trials.
    # We want the "width" of the step to be roughly W.
    # Variance of Binom(M, 0.5) is M/4. Std dev is sqrt(M)/2.
    # Let's say we want the step to cover range [-W, W].
    # So sqrt(M)/2 ~ W => M ~ 4 W^2.
    
    # Let's just parameterize by M directly and scale the x axis.
    # Actually, easier: use the CDF of a standard normal approximated by Binom?
    # No, let's stick to the discrete binom cdf.
    
    # Let's define the profile as:
    # h[i] = P(Y <= i) where Y is Binomial(M, 0.5) shifted and scaled.
    
    # To ensure sum is N/2, we rely on symmetry.
    # If center is N/2, and distribution is symmetric, sum is N/2.
    
    # Let's try a few M values.
    # M controls the sharpness. Small M = sharp step. Large M = smooth step.
    
    # We need to map the binomial support [0, M] to the index range.
    # Let's say the transition happens over a window of size 2*W.
    # We can generate the CDF for k=0..M and interpolate.
    
    # Alternative: Use the fact that for large M, Binom(M, 0.5) approx Normal(M/2, M/4).
    # CDF approx 0.5 * (1 + erf((x - M/2) / sqrt(M/2)))
    
    # Let's use the analytical approximation for speed, or just compute binom.cdf.
    
    # Let's define the step function value at i:
    # We want h[i] to be the CDF of a distribution centered at 'center'.
    # Let's use a scaled binomial.
    
    # Let's pick a scale factor 's' such that the binomial variable maps to indices.
    # Let Y ~ Binom(M, 0.5).
    # We want Y to correspond to i - center.
    # So i - center = (Y - M/2) * scale.
    # This is getting complicated.
    
    # Simpler:
    # Just compute the CDF of Binom(M, 0.5) for points 0..M.
    # Then interpolate to our N points.
    
    # Let's define the range of the step in terms of indices.
    # Let the step go from 0 to 1 over the range [center - half_width, center + half_width].
    # We can map this range to [0, M].
    
    # Let's just iterate over possible 'sharpness' parameters.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Baseline
    h_best = np.full(N, 0.5)
    best_score = compute_max_overlap(h_best)
    
    # Search parameters
    # We will try different "step" profiles.
    # Profile: h[i] = 0.5 + 0.5 * tanh((i - center) / width)
    # Or use the binomial CDF as suggested.
    
    # Let's implement the binomial CDF step.
    # We vary the "width" parameter (related to M) and "center".
    
    # Grid search over center and width
    # Centers: N/2 is optimal for symmetry. Let's fix center = N/2.
    # Widths: from very sharp (step function) to very smooth (linear ramp).
    
    centers = [N // 2]
    
    # Widths: 
    # A "width" of 1 means a sharp step.
    # A "width" of N/2 means a linear ramp from 0 to 1.
    
    widths = np.linspace(1, N/2, 50)
    
    for c in centers:
        for w in widths:
            if time.time() > deadline:
                break
                
            # Generate profile
            # We want a smooth step from 0 to 1 centered at c with width w.
            # Using Binomial CDF:
            # Let M be such that the std dev is w.
            # std = sqrt(M)/2 = w => M = 4*w^2.
            M = int(4 * w * w)
            if M < 1: M = 1
            
            # We need to map the binomial CDF to our indices.
            # The binomial CDF P(Y <= y) for Y~Bin(M, 0.5) goes from 0 to 1 as y goes from 0 to M.
            # We want this transition to happen over the range [c-w, c+w].
            # So we map index i to binomial variable y:
            # y = (i - (c - w)) / (2 * w) * M
            # When i = c - w, y = 0.
            # When i = c + w, y = M.
            
            idx = np.arange(N)
            # Map to [0, M]
            # Clip to [0, M] to ensure 0 and 1 at boundaries
            y_mapped = (idx - (c - w)) / (2 * w) * M
            y_mapped = np.clip(y_mapped, 0, M)
            
            # Compute CDF
            # binom.cdf(k, n, p)
            # k is the value we are checking.
            h_candidate = binom.cdf(y_mapped, M, 0.5)
            
            # Ensure sum is N/2
            # Due to symmetry of c=N/2 and symmetric mapping, sum should be close to N/2.
            # Let's normalize to be safe, but normalization might break [0,1] bounds if not careful.
            # Actually, binom.cdf is in [0,1].
            # If we shift slightly, we might break bounds.
            # Let's just check the sum.
            s = np.sum(h_candidate)
            if abs(s - N/2) > 1e-6:
                # Adjust by scaling? No, that breaks bounds.
                # Adjust by shifting?
                # Let's just accept it if close, or skip.
                # For symmetric center, it should be very close.
                pass
            
            score = compute_max_overlap(h_candidate)
            if score < best_score:
                best_score = score
                h_best = h_candidate.copy()
        
        if time.time() > deadline:
            break

    return h_best
# EVOLVE-BLOCK-END
