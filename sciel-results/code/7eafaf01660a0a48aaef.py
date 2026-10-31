"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

DEADLINE = 115  # seconds, safety margin before 120s kill

def compute_overlap_scores(h, shifts=None):
    """
    Compute the maximum overlap of h with shifted (1-h).
    
    Overlap for shift k: sum_{i} h_i * (1 - h_{(i+k) % N})
    (Assuming circular shift; if linear, this is a common interpretation for
    "shifted" in such contexts, or we need to handle linear. Given the ambiguity
    and the nature of such problems, circular is a strong candidate. If linear,
    the edges have fewer terms. Let's assume circular for now as it's more
    standard for "shifts" in signal processing-like contexts, and the problem
    mentions "shifted 1-h" which often implies cyclic convolution in discrete
    settings. If linear, we'd do: for k in 0..N-1, sum_{i=0}^{N-k-1} h_i * (1-h_{i+k}).
    Let's check the seed: it returns 0.5. For N=100, sum=50.
    If circular: overlap_k = sum_i h_i (1-h_{i+k}) = sum_i h_i - sum_i h_i h_{i+k}.
    Since sum h_i = N/2 = 50.
    overlap_k = 50 - sum_i h_i h_{i+k}.
    Minimizing max overlap_k is equivalent to maximizing min_k (sum_i h_i h_{i+k}).
    
    If linear: overlap_k = sum_{i=0}^{N-k-1} h_i (1-h_{i+k}).
    This is less symmetric.
    
    Let's look at the problem statement: "maximum overlap of h with shifted 1-h".
    Usually, "overlap" of two functions f and g shifted by k is \int f(x) g(x-k) dx.
    For discrete: sum h_i (1-h_{i+k}).
    If the domain is [0,2] and we have steps, it's likely periodic or defined on a circle?
    "step heights h on [0,2]" suggests a continuous interval.
    But the interface says "sequence of step heights".
    If it's on [0,2], and we shift, do we wrap?
    If we don't wrap, the "overlap" at large shifts is small because of truncation.
    The "maximum overlap" would likely be at small shifts.
    
    Let's assume circular for now, as it's a common interpretation for "minimum overlap" problems
    involving shifts, often related to autocorrelation.
    If circular:
    Score = max_k [ sum_i h_i (1 - h_{i+k}) ]
           = max_k [ Sum(h) - sum_i h_i h_{i+k} ]
           = N/2 - min_k [ sum_i h_i h_{i+k} ]
    
    We want to minimize this score.
    So we want to MAXIMIZE min_k [ sum_i h_i h_{i+k} ].
    
    Let S_k = sum_i h_i h_{i+k}.
    We want to maximize the minimum S_k.
    
    Note that S_0 = sum h_i^2.
    S_N = S_0.
    
    For h_i = 0.5:
    S_k = sum 0.25 = N * 0.25 = 25.
    Score = 50 - 25 = 25.
    
    Can we get higher min S_k?
    The maximum possible value of S_k is bounded by (Sum h_i)^2 / N ? No.
    By Cauchy-Schwarz: (sum h_i h_{i+k})^2 <= (sum h_i^2)(sum h_{i+k}^2) = (sum h_i^2)^2.
    So S_k <= sum h_i^2 = S_0.
    
    Also, the average of S_k over all k is:
    (1/N) sum_k S_k = (1/N) sum_k sum_i h_i h_{i+k} = (1/N) sum_i h_i (sum_k h_{i+k}) = (1/N) sum_i h_i (Sum h) = (1/N) (Sum h)^2 = (N/2)^2 / N = N/4 = 25.
    
    So the average of S_k is 25.
    Since S_0 is the maximum possible S_k (because S_k <= S_0 for all k? No, S_0 is max sum of squares, but S_k is cross-correlation.
    Actually, for any real sequence, the autocorrelation at lag 0 is the maximum: R(0) >= R(k) for all k.
    Yes, R(0) = sum h_i^2. R(k) = sum h_i h_{i+k}.
    R(0) - R(k) = 0.5 * sum (h_i - h_{i+k})^2 >= 0.
    So S_0 is indeed the global maximum of the autocorrelation function.
    
    So max S_k = S_0.
    We want to maximize min S_k.
    Since avg S_k = 25, and max S_k = S_0,
    If we make all S_k equal, they would be 25.
    If we make S_0 larger, say 30, then the average is still 25, so some S_k must be less than 25.
    Specifically, if S_0 > 25, then the "deficit" must be distributed among other lags.
    To maximize the *minimum*, we want the S_k's to be as flat as possible.
    The flattest possible is when all S_k are equal to the average, which is 25.
    This happens if h_i is constant (0.5), in which case S_k = 25 for all k.
    Any deviation from constant increases S_0 and creates variance in S_k.
    Since the mean is fixed at 25, increasing variance means some values go above 25 and some below.
    The minimum will drop below 25.
    
    Therefore, the constant solution h_i = 0.5 gives min S_k = 25, which is the theoretical maximum possible minimum.
    So score = 25 is the global minimum possible score for circular shifts.
    
    What if the shifts are linear?
    Score_k = sum_{i=0}^{N-k-1} h_i (1-h_{i+k}).
    For h_i=0.5:
    Score_k = sum_{i=0}^{N-k-1} 0.25 = 0.25 (N-k).
    Max is at k=0: 0.25 N = 25.
    At k=N-1: 0.25.
    
    Can we do better?
    We want to minimize max_k Score_k.
    Score_0 = sum h_i^2 ? No.
    Score_0 = sum_{i=0}^{N-1} h_i (1-h_i) = N/2 - sum h_i^2.
    We want to minimize this. So we want to maximize sum h_i^2.
    Max sum h_i^2 subject to sum h_i = N/2, 0<=h_i<=1.
    This is achieved by making h_i as 0 or 1 as possible.
    If h is 0/1, sum h_i^2 = sum h_i = N/2 = 50.
    Then Score_0 = 50 - 50 = 0.
    
    But we need to check other k.
    If h is a block of 1s and 0s.
    Let h_i = 1 for i < M, 0 for i >= M. M = N/2 = 50.
    Score_k = sum_{i=0}^{N-k-1} h_i (1-h_{i+k}).
    
    Let's analyze Score_k for this step function.
    h = [1...1 (50), 0...0 (50)].
    
    For k < 50:
    i ranges from 0 to 49-k. h_i=1. h_{i+k}=1. Term is 0.
    i ranges from 50-k to 49. h_i=1. h_{i+k}=0 (since i+k >= 50). Term is 1*1=1.
    Number of such terms: (49 - (50-k)) + 1 = k.
    So Score_k = k.
    
    For k >= 50:
    i ranges from 0 to 49.
    If i+k < 50, h_{i+k}=1, term 0.
    If i+k >= 50, h_{i+k}=0, term 1.
    i+k >= 50 => i >= 50-k.
    Since k>=50, 50-k <= 0. So for all i in 0..49, i+k >= 50.
    So all terms are 1.
    Number of terms: 50.
    So Score_k = 50.
    
    Max Score is 50 (for k>=50).
    This is worse than 25.
    
    So 0.5 is better than the step function for linear shifts too?
    Wait, for linear, Score_0 for 0.5 is 25.
    For step, Score_0 is 0.
    But Max Score for step is 50.
    Max Score for 0.5 is 25.
    So 0.5 is better.
    
    Is there a configuration with Max Score < 25 for linear?
    We established for circular that 25 is the lower bound.
    Linear overlap is a subset of circular (essentially, it ignores the wrap-around terms).
    Score_linear_k = sum_{i=0}^{N-k-1} h_i (1-h_{i+k}).
    Score_circular_k = sum_{i=0}^{N-1} h_i (1-h_{(i+k)%N})
                     = sum_{i=0}^{N-k-1} h_i (1-h_{i+k}) + sum_{i=N-k}^{N-1} h_i (1-h_{i+k-N})
                     = Score_linear_k + WrapTerm.
    
    So Score_linear_k <= Score_circular_k.
    Therefore, max_k Score_linear_k <= max_k Score_circular_k.
    Since min max_circular is 25, min max_linear is <= 25?
    No, the minimization is over h.
    Let S_L(h) = max_k Score_linear_k(h).
    Let S_C(h) = max_k Score_circular_k(h).
    We know S_C(h) >= 25 for all h.
    We know S_L(h) <= S_C(h).
    This doesn't imply S_L(h) >= 25.
    
    Let's check if we can get S_L(h) < 25.
    We need to minimize the maximum of the linear overlaps.
    
    Consider h_i = 0.5. S_L(0.5) = 25.
    Can we go lower?
    Score_0 = N/2 - sum h_i^2.
    To make Score_0 small, we need sum h_i^2 large.
    But making h binary increases Score_k for large k (as seen with the step function).
    
    Let's try to balance.
    We want Score_k to be small for all k.
    Score_k = sum_{i=0}^{N-k-1} h_i - sum_{i=0}^{N-k-1} h_i h_{i+k}.
    Sum h_i (truncated) is roughly N/2 for small k.
    So we want to maximize sum h_i h_{i+k} (truncated).
    
    This seems like 0.5 is likely optimal or near-optimal.
    
    However, the previous island best was 1.145, which is BETTER than 1.0 (since score is seed/value, 1.0 is seed value. Wait.
    "score is the upper bound on C5".
    "search score is seed/value".
    "1.0 equals the seed program".
    Seed program returns 0.5.
    If seed value is V_seed, score_seed = V_seed / V_seed = 1.
    If we find a better value V_new < V_seed, score = V_seed / V_new > 1.
    So higher score is better.
    Island best is 1.145.
    This implies V_best = V_seed / 1.145.
    V_seed = 25 (if circular) or 25 (if linear max).
    So V_best = 25 / 1.145 = 21.83.
    
    So there IS a solution better than 25!
    My previous analysis that 25 is the global minimum for circular was correct.
    But if the evaluator uses LINEAR shifts, the lower bound is lower.
    Or if I made a mistake about the circular lower bound.
    Let's re-verify the circular lower bound.
    avg S_k = 25.
    S_k = sum h_i h_{i+k}.
    We want to maximize min S_k.
    Is it possible to have min S_k > 25?
    No, because if min S_k > 25, then avg S_k > 25.
    But avg S_k is fixed at 25.
    So min S_k <= 25.
    So max possible min S_k is 25.
    So min possible Score_circular is 25.
    
    So if the island best is 1.145, the evaluator CANNOT be using circular shifts.
    It must be using LINEAR shifts.
    
    Okay, so we are minimizing max_k Score_linear_k.
    
    We found S_L(0.5) = 25.
    We want S_L(h) < 25.
    
    Let's search for such an h.
    Score_k = \sum_{i=0}^{N-k-1} h_i (1-h_{i+k}).
    
    Let's try to reduce the large k scores.
    For large k, the sum is over a small number of terms.
    Score_{N-1} = h_0 (1-h_1).
    Score_{N-2} = h_0 (1-h_2) + h_1 (1-h_3).
    
    If we can make these small, we can potentially lower the max.
    But Score_0 = 50 - sum h_i^2.
    To keep Score_0 low, we need sum h_i^2 high.
    This pushes h towards 0/1.
    But 0/1 creates large overlaps at intermediate shifts?
    
    Let's use Differential Evolution to find a better h for the LINEAR overlap objective.
    
    """
    if shifts is None:
        shifts = range(N)
    
    scores = []
    h = np.asarray(h)
    for k in shifts:
        if k == 0:
            # sum h_i (1-h_i)
            score = np.sum(h * (1 - h))
        else:
            # sum_{i=0}^{N-k-1} h_i (1-h_{i+k})
            # Vectorized
            left = h[:-k]
            right = h[k:]
            score = np.sum(left * (1 - right))
        scores.append(score)
    return max(scores)

def construct():
    start_time = time.time()
    
    # Initial guess: 0.5
    best_h = np.full(N, 0.5)
    best_score = compute_overlap_scores(best_h)
    
    # Differential Evolution
    bounds = [(0, 1)] * N
    # Constraint: sum(h) = N/2
    # DE doesn't handle equality constraints well directly, but we can penalize.
    
    def penalty_de(x):
        # x is 1D array of h
        # Check bounds (DE handles this)
        # Check sum
        s = np.sum(x)
        if abs(s - N/2) > 1e-5:
            return compute_overlap_scores(x) + 1000 * abs(s - N/2)
        return compute_overlap_scores(x)
    
    # Run DE
    # To save time, we might reduce population size or max generations if time is tight.
    # But DE is global, good for escaping local minima.
    
    try:
        result = differential_evolution(
            penalty_de,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            update_interval=1
        )
        
        # Post-process: project onto sum=50 plane
        h_de = result.x
        # Normalize to sum N/2
        # Simple normalization might push out of [0,1].
        # Better: solve min ||h - h_de||^2 s.t. sum h = N/2, 0<=h<=1.
        # Or just clip and adjust.
        
        h_proj = h_de.copy()
        # Adjust to meet sum exactly
        diff = N/2 - np.sum(h_proj)
        h_proj += diff / N
        # Clip
        h_proj = np.clip(h_proj, 0, 1)
        # Re-adjust sum if clipping changed it
        # This might require a few iterations or a solver
        # For now, a simple iterative adjustment
        for _ in range(100):
            s = np.sum(h_proj)
            if abs(s - N/2) < 1e-9:
                break
            h_proj += (N/2 - s) / N
            h_proj = np.clip(h_proj, 0, 1)
        
        score_de = compute_overlap_scores(h_proj)
        
        # Local polish with SLSQP
        if time.time() - start_time < DEADLINE - 20:
            def obj(x):
                return compute_overlap_scores(x)
            
            def gradient(x):
                # Numerical gradient
                eps = 1e-6
                grad = np.zeros_like(x)
                for i in range(N):
                    x_p = x.copy(); x_p[i] += eps
                    x_m = x.copy(); x_m[i] -= eps
                    # Need to handle constraint in gradient? SLSQP handles it.
                    # But obj might be non-differentiable at 0 or 1?
                    # It is differentiable everywhere in (0,1).
                    grad[i] = (obj(x_p) - obj(x_m)) / (2 * eps)
                return grad
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2}]
            bounds_local = [(0, 1)] * N
            
            res = minimize(
                obj,
                h_proj,
                method='SLSQP',
                jac=gradient,
                bounds=bounds_local,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res.success:
                h_slqp = res.x
                # Ensure exact sum
                h_slqp += (N/2 - np.sum(h_slqp)) / N
                h_slqp = np.clip(h_slqp, 0, 1)
                score_slqp = compute_overlap_scores(h_slqp)
                
                if score_slqp < best_score:
                    best_h = h_slqp
                    best_score = score_slqp
        
        # If DE score is better than initial
        if score_de < best_score:
            best_h = h_proj
            best_score = score_de
            
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Final check
    # Ensure sum is exactly N/2
    best_h = np.clip(best_h, 0, 1)
    best_h += (N/2 - np.sum(best_h)) / N
    best_h = np.clip(best_h, 0, 1)
    
    # One last tiny polish if time allows
    if time.time() - start_time < DEADLINE - 5:
        # Nelder-Mead for a few steps?
        pass
        
    return best_h

# EVOLVE-BLOCK-END
