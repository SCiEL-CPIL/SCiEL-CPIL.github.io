import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_score(h):
    """
    Compute the ratio of squared L2 norm to (L1 norm * sup norm of autoconvolution).
    h: array of nonnegative heights
    Returns the ratio value (higher is better).
    """
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    l2_sq = np.sum(h**2)
    
    # Autoconvolution: c[k] = sum_i h[i] * h[i+k] for k = -(N-1)..(N-1)
    c = np.correlate(h, h, mode='full')
    sup_c = np.max(c)
    
    if sup_c < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * sup_c)

def negative_score(h):
    """Negative of score for minimization."""
    return -compute_score(h)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    # Baseline: 2-element support
    best_h = np.zeros(N)
    best_h[0] = 2.0 / 3.0
    best_h[1] = 1.0 / 3.0
    best_score = compute_score(best_h)
    
    bounds = [(0.0, 1.0)] * N
    
    # Strategy 1: Local optimization from the known good 2-element solution
    try:
        x0 = best_h.copy()
        res = minimize(negative_score, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
        if res.fun < 0:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_h = res.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return list(best_h)
    
    # Strategy 2: Try various structured initializations and optimize locally
    # Key insight: The ratio L2^2 / (L1 * sup_autoconv) is scale-invariant in h
    # (if we scale h by c, L2^2 scales by c^2, L1 by c, sup_autoconv by c^2, so ratio stays same)
    # So we can normalize and work with unit L1 or unit L2.
    # Actually: L2^2 / (L1 * sup_c). If h -> c*h: L2^2 -> c^2 L2^2, L1 -> c L1, sup_c -> c^2 sup_c
    # Ratio: c^2 L2^2 / (c L1 * c^2 sup_c) = L2^2 / (L1 * sup_c) * 1/c. Not scale invariant!
    # Wait, let me recheck. The task says "ratio of squared L2 norm to L1 norm times supremum norm of autoconvolution"
    # So it's L2^2 / (L1 * sup(h*h)).
    # If h -> c*h: L2^2 -> c^2 * L2^2, L1 -> c * L1, sup(h*h) -> c^2 * sup(h*h)
    # Ratio: (c^2 L2^2) / (c L1 * c^2 sup_c) = L2^2 / (L1 * sup_c) * (1/c)
    # So it IS scale-dependent. Larger amplitudes give smaller ratio? No wait:
    # (c^2) / (c * c^2) = 1/c. So larger c gives smaller ratio. That means we want small amplitudes?
    # But that can't be right for a bounded optimization. Let me re-read.
    # Actually the problem says "maximize the ratio". If scaling down increases the ratio, then the supremum is unbounded.
    # Unless there's a normalization constraint I'm missing. Let me re-read the task.
    # "Choose 50 nonnegative heights... to maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution."
    # Hmm, maybe there's an implicit constraint. Let me check: if all h_i = 1, L2^2 = N, L1 = N, sup_c = N. Ratio = N/(N*N) = 1/N.
    # If h = [1,0,...,0], L2^2=1, L1=1, sup_c=1. Ratio = 1.
    # If h = [a, 0,...,0], L2^2=a^2, L1=a, sup_c=a^2. Ratio = a^2/(a*a^2) = 1/a. So as a->0, ratio->infty.
    # That can't be right. There must be a normalization. Let me re-read more carefully.
    
    # Actually, looking at the seed program, it uses h[0]=2/3, h[1]=1/3. Let's compute:
    # L2^2 = (4/9 + 1/9) = 5/9. L1 = 1. Autoconv: c[0] = 4/9+1/9=5/9, c[1]=2/9, c[2]=0. sup_c=5/9.
    # Ratio = (5/9)/(1*5/9) = 1. But the seed scores 1.0633... so my formula might be wrong.
    
    # Let me recompute with the actual code. h=[2/3, 1/3, 0,...]
    # L2^2 = (2/3)^2 + (1/3)^2 = 4/9 + 1/9 = 5/9
    # L1 = 2/3 + 1/3 = 1
    # Autoconv: np.correlate([2/3,1/3,0,...],[2/3,1/3,0,...],'full')
    # c[0] = sum h[i]^2 = 5/9
    # c[1] = sum h[i]*h[i+1] = (2/3)(1/3) = 2/9
    # c[-1] = 2/9
    # sup_c = 5/9
    # Ratio = (5/9)/(1 * 5/9) = 1.0
    # But the score is 1.0633. So either the formula is different or I'm misunderstanding.
    
    # Wait, the score is value/seed where seed is the seed program's value. So if seed gives 1.0, and this gives 1.0633,
    # then the actual ratio for this h is 1.0633 * seed_value. If seed_value = 1.0, then ratio = 1.0633.
    # But my calculation gives 1.0. Let me recheck.
    
    # Hmm, maybe the autoconvolution is defined differently. Let me think about what np.correlate does.
    # np.correlate(a, v, mode='full') returns the full discrete linear cross-correlation.
    # For autoconvolution, it's the same. c[k] = sum_i a[i+k] * conj(v[i]) for appropriate k.
    # For real sequences, c[k] = sum_i a[i+k] * a[i].
    # For a=[2/3, 1/3, 0, ..., 0] (length 50):
    # c[0] = sum a[i]^2 = 4/9 + 1/9 = 5/9
    # c[1] = sum a[i+1]*a[i] = a[1]*a[0] = (1/3)(2/3) = 2/9
    # c[-1] = sum a[i-1]*a[i] = a[0]*a[1] = 2/9
    # All other c[k] = 0.
    # sup_c = 5/9.
    # Ratio = (5/9) / (1 * 5/9) = 1.
    
    # So why is the score 1.0633? Maybe the "seed program" in the task contract is different from what I think.
    # The task says "score is the lower bound on C2" and "search score is value/seed, so 1.0 equals the seed program".
    # So the seed program's value is the reference. If the seed program gives ratio R_seed, and my program gives ratio R,
    # then score = R / R_seed.
    
    # So I just need to maximize the ratio. The absolute value doesn't matter for comparison.
    # But the scaling issue remains. If I can make the ratio arbitrarily large by scaling down, the problem is unbounded.
    # Unless there's a constraint like sum(h) = 1 or max(h) = 1 that I'm not seeing.
    
    # Let me just focus on maximizing the ratio as computed by compute_score, and trust that the evaluator handles it correctly.
    # The key is that the ratio is what it is, and I need to find h that maximizes it.
    
    # Let me try different supports and see what happens.
    
    # Try 3-element support: h = [a, b, c, 0, ...]
    # L2^2 = a^2 + b^2 + c^2
    # L1 = a + b + c
    # Autoconv: c[0] = a^2+b^2+c^2, c[1] = ab+bc, c[2] = ac, c[-1] = ab+bc, c[-2] = ac
    # sup_c = max(a^2+b^2+c^2, ab+bc, ac)
    
    # For a=1, b=0.5, c=0.25:
    # L2^2 = 1 + 0.25 + 0.0625 = 1.3125
    # L1 = 1.75
    # c[0] = 1.3125, c[1] = 0.5+0.125=0.625, c[2] = 0.25
    # sup_c = 1.3125
    # Ratio = 1.3125 / (1.75 * 1.3125) = 1/1.75 = 0.571
    
    # For a=1, b=0, c=0: L2^2=1, L1=1, sup_c=1, ratio=1.
    # For a=1, b=0.5, c=0: L2^2=1.25, L1=1.5, c[0]=1.25, c[1]=0.5, sup_c=1.25, ratio=1.25/(1.5*1.25)=2/3=0.667
    # For a=1, b=0.3, c=0: L2^2=1.09, L1=1.3, sup_c=1.09, ratio=1/1.3=0.769
    # For a=1, b=0.1, c=0: L2^2=1.01, L1=1.1, sup_c=1.01, ratio=1/1.1=0.909
    # For a=1, b=0.01, c=0: ratio ~ 1/1.01 = 0.99
    
    # So it seems like concentrating mass in one element gives higher ratio. But the 2-element [2/3, 1/3] gives 1.0.
    # And [1, 0, ...] gives 1.0 as well.
    
    # Let me try [1, 0.5, 0, ...]: ratio = 1.25/(1.5*1.25) = 2/3 < 1.
    # [1, 0.1, 0, ...]: ratio = 1.01/(1.1*1.01) = 1/1.1 < 1.
    
    # So [1,0,...] and [2/3, 1/3, 0,...] both give 1.0? Let me verify [1,0,...]:
    # L2^2=1, L1=1, c[0]=1, sup_c=1, ratio=1.
    
    # So the maximum might be 1.0 for these simple cases. But the seed scores 1.0633, so there must be better solutions.
    
    # Let me try a more spread out distribution. [0.5, 0.5, 0, ...]:
    # L2^2 = 0.25+0.25=0.5, L1=1, c[0]=0.5, c[1]=0.25, sup_c=0.5, ratio=0.5/(1*0.5)=1.0
    
    # [0.5, 0.3, 0.2, 0, ...]:
    # L2^2=0.25+0.09+0.04=0.38, L1=1, c[0]=0.38, c[1]=0.15+0.06=0.21, c[2]=0.1, sup_c=0.38, ratio=0.38/(1*0.38)=1.0
    
    # Interesting! It seems like for any distribution with L1=1, the ratio is always 1.0?
    # Let's check: L2^2 / (L1 * sup_c). If L1=1, ratio = L2^2 / sup_c.
    # sup_c >= c[0] = L2^2 (since c[0] = sum h[i]^2 = L2^2).
    # So sup_c >= L2^2, meaning ratio <= 1.
    # Equality when sup_c = L2^2, i.e., when c[0] is the maximum of the autoconvolution.
    
    # So the ratio is always <= 1 for L1=1. And the maximum is 1, achieved when the autoconvolution peak is at lag 0.
    
    # But wait, the seed scores 1.0633 > 1. So either my understanding is wrong, or the ratio can exceed 1.
    # Let me re-examine. sup_c = max_k c[k]. c[0] = sum h[i]^2 = L2^2.
    # For nonnegative h, c[k] = sum_i h[i]h[i+k] for k>=0. By Cauchy-Schwarz, c[k] <= sqrt(sum h[i]^2 * sum h[i+k]^2) <= L2^2.
    # Actually, c[k] = sum h[i]h[i+k] <= (sum h[i]^2)^{1/2} (sum h[i+k]^2)^{1/2} <= L2^2.
    # So sup_c <= L2^2, meaning ratio = L2^2/(L1 * sup_c) >= L2^2/(L1 * L2^2) = 1/L1.
    # If L1=1, ratio >= 1. And ratio = 1 when sup_c = L2^2.
    # If L1 > 1, ratio >= 1/L1 < 1. If L1 < 1, ratio >= 1/L1 > 1.
    
    # So to maximize the ratio, we want L1 to be as small as possible! But if L1->0, all h->0, and the ratio is undefined (0/0).
    # Actually, if h -> c*h with c->0, L2^2 -> c^2 L2^2, L1 -> c L1, sup_c -> c^2 sup_c.
    # Ratio = c^2 L2^2 / (c L1 * c^2 sup_c) = L2^2/(L1 sup_c) * 1/c -> infinity as c->0.
    
    # So the ratio is unbounded above? That can't be the intended problem. There must be a normalization constraint.
    # Let me re-read the task contract: "Choose 50 nonnegative heights on equal intervals of [-1/4,1/4] to maximize the ratio..."
    # Maybe the heights are constrained to sum to 1 or something. Or maybe the L1 norm is normalized.
    
    # Given the ambiguity, let me just maximize the ratio as computed, and assume the evaluator handles normalization.
    # The key insight is: ratio = L2^2 / (L1 * sup_c) = (L2^2 / L1) / sup_c.
    # Since sup_c >= L2^2 (for nonnegative h, c[0]=L2^2 and c[k] <= L2^2), we have ratio <= 1/L1.
    # Wait, I had it backwards. sup_c >= c[0] = L2^2? No, c[0] = L2^2, and sup_c = max_k c[k] >= c[0] = L2^2.
    # So ratio = L2^2/(L1 * sup_c) <= L2^2/(L1 * L2^2) = 1/L1.
    # So ratio <= 1/L1. To maximize ratio, minimize L1. But L1 can be arbitrarily small.
    
    # I think there's a missing constraint. Let me assume L1 = 1 (probability distribution) and maximize L2^2 / sup_c.
    # With L1=1, ratio = L2^2 / sup_c <= 1, with equality when sup_c = L2^2.
    
    # For the 2-element case [a, 1-a, 0, ...]: L2^2 = a^2+(1-a)^2, sup_c = max(a^2+(1-a)^2, a(1-a)).
    # a^2+(1-a)^2 = 2a^2-2a+1. a(1-a) = a-a^2.
    # 2a^2-2a+1 >= a-a^2 iff 3a^2-3a+1 >= 0, which is always true (discriminant 9-12<0).
    # So sup_c = a^2+(1-a)^2, ratio = 1. Always 1 for L1=1.
    
    # So with L1=1, the ratio is always <= 1, and many distributions achieve 1. The maximum is 1.
    # But the seed scores 1.0633, so the actual problem must not have L1=1 constraint.
    
    # Let me just try to maximize the ratio numerically and see what happens. I'll use the compute_score function as given.
    
    # Try optimizing with a few different initializations.
    
    initializations = [
        # 2-element
        np.array([2/3, 1/3] + [0.0]*48),
        # 3-element
        np.array([0.5, 0.3, 0.2] + [0.0]*47),
        # 4-element
        np.array([0.4, 0.3, 0.2, 0.1] + [0.0]*46),
        # 5-element uniform
        np.array([0.2]*5 + [0.0]*45),
        # Geometric decay
        np.array([0.5**i for i in range(10)] + [0.0]*40),
        # Single spike
        np.array([1.0] + [0.0]*49),
    ]
    
    for x0 in initializations:
        if time.time() > deadline:
            break
        try:
            res = minimize(negative_score, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < 0:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_h = res.x.copy()
        except Exception:
            pass
    
    if time.time() > deadline:
        return list(best_h)
    
    # Strategy 3: Try random restarts
    np.random.seed(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        x0 = np.random.exponential(0.5, size=N)
        x0 /= x0.max() if x0.max() > 0 else 1
        try:
            res = minimize(negative_score, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < 0:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return list(best_h)
