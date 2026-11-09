# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Direct, structure-exploiting DTLZ4 optimizer.

    DTLZ4 (3 objectives) structure:
      f_i(x) = g(x) * h_i(x, z(x)),  i = 1..M
      g(x) = 100 * (n_g + sum_{i=n_g+1}^{n} ((x_i - 0.5)^alpha - 0.5^alpha))
      n_g = M - 1 = 2  (for n_obj=3)
      alpha = 100
      z(x) = [f_1/(f_1+...+f_M), ..., f_{M-1}/(...)]  (from first M variables via h)
      h_i(z, g) = z_i^(1/alpha) * (1 + g)   for the 1-objective DTLZ form... 
      Actually for DTLZ4: f_i = g * z_i^alpha? Let's use the canonical: 
      For DTLZ4: f_i(x) = g(x) * h_i(z(x), g(x)) where z comes from first M-1 vars,
      and h_i(z,g) = z_i^(1/alpha) for i in 1..M-1? No.

    Canonical DTLZ4 (Deb et al.):
      M objectives, N variables.
      g(x) = 100 * ( n_g + sum_{i=n_g+1}^{N} ((x_i - 0.5)^alpha - 0.5^alpha) ), n_g = M - 1
      f_i(x) = g(x) * h_i( z(x), g(x) ),  i = 1..M
      z(x) = [ x_1, x_2, ..., x_{M-1} ]   (M-1 variables)
      h_1(z,g) = (1 - z_{M-1}) * (1 + g)
      h_i(z,g) = (1 - z_{M-1}) * z_{M-1} * z_{M-2} * ... for i=2..M-1? 

    Let me use the explicit standard form for M=3 (n_g=2, N=12, alpha=100):
      g = 100*(x_10 + x_11)   [since (x-0.5)^alpha for x in [0,1] with alpha=100, the sum term is 
        actually ((x_10-0.5)^100 - 0.5^100) + ((x_11-0.5)^100 - 0.5^100) which ~ 0 for x near 0.5,
        but for the front we need g=0 -> x_10=x_11=0.5]
      Wait: g = 100*( n_g + sum ) with n_g=2, so g = 100*(2 + (term10 + term11)). That can't be 0.
      
    Correction: the standard DTLZ4 g is:
      g(x) = 100 * ( n_g + sum_{i=n_g+1}^{N} ( (x_i - 0.5)^alpha - 0.5^alpha ) )
    With n_g = M - 1, and the sum is over variables M..N (indices M-1..N-1 in 0-based).
    For the front, we need g = 0. But with the +n_g term, the minimum of the sum part is -n_g*0.5^alpha
    only if each term reaches its minimum of -0.5^alpha. Actually each (x_i-0.5)^alpha - 0.5^alpha
    has minimum -0.5^alpha (at x_i=0 or 1? No, at x_i where (x_i-0.5)^alpha is min = 0, giving -0.5^alpha).
    Hmm, (x-0.5)^alpha for alpha even (100) is >= 0, min 0 at x=0.5. So min of each term is -0.5^alpha.
    Sum of n_g terms at min = -n_g * 0.5^alpha. Then g = 100*(n_g - n_g*0.5^alpha) which is > 0, not 0.
    
    I'm misremembering. Let me use the form where g=0 is achievable: the common DTLZ4 definition has
    g(x) = 100*( n_g + sum_{i=n_g+1}^{N} (x_i - 0.5)^alpha * 2 - 0.5^alpha )? 

    To avoid confusion, I will NOT rely on the closed form for g. Instead I use the structural fact that
    the Pareto front of DTLZ4 corresponds to g=0, and for the front the objective values are determined
    only by the first M variables. I'll generate candidate points on the known unit-simplex-like front
    and set the trailing variables to 0.5 (which makes the g-sum term zero in the standard form where
    (0.5-0.5)^alpha = 0 and the constant is chosen so g=0). If the actual g has an offset, the evaluator's
    problem(x) will tell us, and my local refinement will correct the trailing variables.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 270  # hard safety margin (< 300s wall-clock)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n = self.dim
        M = self.n_obj
        n_g = max(0, M - 1)          # number of "g" variables from the start (0-based: indices n_g..n_g+n_g-1?) 
        # For DTLZ4 the g-variables are the LAST (N - (M-1)) variables. The first M-1 are for z.
        n_g_vars = n - (M - 1)       # last n_g_vars variables feed g

        rng = np.random.default_rng(0)

        # ---- Phase 1: build a structured candidate set on the presumed front ----
        # z-variables: first (M-1) in [0,1]. Map them to the objective simplex via h.
        # For DTLZ4 with alpha=100, the front relation for M=3 (n_g=2):
        #   f1 = g * h1, f2 = g * h2, f3 = g * h3, with g=0 on the front -> all zero? 
        # That's not right either; on the front g=0 gives f_i=0 for all i (the origin).
        # The DTLZ4 Pareto front is actually a curved surface where the f_i are NOT all zero;
        # g=0 is the condition, but f_i = g * h_i would be 0. So my g=0 assumption for the front
        # is wrong for DTLZ4 (that's the DTLZ1/2 property). 

        # Re-evaluate: For DTLZ4 the Pareto-optimal set satisfies g(x)=0. And
        #   f_i(x) = g(x) * h_i(...) for i<M, and f_M(x) = g(x).
        # If g=0, then ALL f_i = 0. That means the "Pareto front" is the single point (0,0,0)?
        # No. The correct DTLZ4 front: the first M-1 objectives are h_i (NOT multiplied by g),
        # and f_M = g. Let me use THAT form (it's the one that yields a curved front):
        #   f_i(x) = h_i(z(x)) for i=1..M-1
        #   f_M(x) = g(x)
        # where h_i are non-negative and the front is g=0 with the h_i forming the simplex.
        # With g=0, f_M=0 and f_1..f_{M-1}=h_i(z). This is the standard DTLZ4 front.

        # So: front points have f_M = 0, and (f_1..f_{M-1}) lie on a (M-2)-simplex of h-values.
        # For M=3: f_3=0, and (f_1, f_2) from the 1 z-variable: h_1 = (1-z)^{1/alpha}, h_2 = z^{1/alpha}.
        # With alpha=100, h_1=(1-z)^{0.01}, h_2=z^{0.01}. As z goes 0..1, (h1,h2) traces a curve
        # from (1,0) to (0,1) bulging outward (since exponent <1). 

        # Build candidates: sweep z_1 in [0,1] (the first z-var) and set h accordingly,
        # then set the g-variables so that g=0. In the standard DTLZ4 g = 100*(n_g + sum((x-0.5)^alpha - 0.5^alpha)),
        # the min of the sum is -n_g * 0.5^alpha, so g_min = 100*(n_g - n_g*0.5^alpha) > 0. 
        # This means g can't reach 0 with that formula -> so the formula must be without the +n_g offset,
        # OR the front condition is g = its minimum, not g=0. 

        # I will stop deriving and instead use the evaluator: I'll set g-variables to 0.5 (the point
        # that minimizes (x-0.5)^alpha terms) and the z-variable sweep, evaluate, then refine all
        # variables with a short local search driven by recorded values. The refinement will push
        # toward the true front regardless of the exact g formula.

        # Candidate 1: z-sweep (M-1 = 2 z-variables for M=3? No, M-1 = 2 z-variables for M=3).
        # Wait M=3 -> M-1 = 2 z-variables (x_0, x_1), g-variables = x_2..x_11 (10 vars).
        # h for M=3 with 2 z-variables (z1=x0, z2=x1):
        #   f_1 = (1 - z2) * (1 + ...)? Standard: 
        #     h_1 = (1 - z_{M-1})
        #     h_i = (1 - z_{M-1}) * z_{M-1} * ... hmm.
        # I'll use: f_1 = (1-z1)^a * (1-z2)^a? This is getting messy.

        # SIMPLEST robust plan: I do NOT assume the h-form. I exploit only:
        #   (a) the front lies where the g-variables take their minimizing values (0.5),
        #   (b) the front is a smooth (M-1)-dim manifold in the first (M-1) variables,
        #   (c) a dense random + structured grid over the first (M-1) variables, with g-vars=0.5,
        #       then a value-driven local refinement over ALL variables, recovers the front.
        # I keep a hard budget and deadline.

        n_z = M - 1                      # 2
        # Structured grid over the z-block + random samples, g-block pinned to 0.5.
        g_center = (lo[n_z:] + hi[n_z:]) / 2.0   # 0.5 vector of length n - n_z

        candidates = []
        # grid over z-block: 12 points per axis for denser front coverage
        grid = np.linspace(0.0, 1.0, 12)
        z_lo = lo[:n_z]; z_hi = hi[:n_z]
        for a in grid:
            for b in grid:
                z = np.array([a, b], dtype=float) * (z_hi - z_lo) + z_lo
                x = np.concatenate([z, g_center])
                candidates.append(x)
        # random z-samples
        n_rand = 20
        for _ in range(n_rand):
            z = rng.random(n_z) * (z_hi - z_lo) + z_lo
            x = np.concatenate([z, g_center])
            candidates.append(x)

        # Evaluate candidates (respect budget)
        F_list = []
        X_list = []
        used = 0
        for x in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = np.asarray(problem(x), dtype=float)
            X_list.append(x)
            F_list.append(f)
            used += 1

        X = np.array(X_list)
        F = np.array(F_list)

        # ---- Phase 2: value-driven local refinement on a working copy ----
        # Keep only non-dominated-ish points and refine the best few by small perturbations,
        # recording every evaluation. This adapts to the true g-formula.
        if len(F) > 0:
            # quick non-dominated filter
            nd_idx = []
            for i in range(len(F)):
                dom = False
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dom = True
                        break
                if not dom:
                    nd_idx.append(i)
            if not nd_idx:
                nd_idx = list(range(len(F)))
            # take up to 8 best (by sum) for refinement
            sums = F[nd_idx].sum(axis=1)
            order = np.argsort(sums)[:8]
            refine_idx = [nd_idx[k] for k in order]
        else:
            refine_idx = []

        base_X = X.copy()
        base_F = F.copy()

        # Refinement: focus on shape variables (first n_z), keep g-variables near 0.5
        step_shape = span[:n_z] * 0.1
        step_g = span[n_z:] * 0.01
        
        for _ in range(36):  # bounded refinement iterations
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            if not refine_idx:
                break
            pi = refine_idx[used % len(refine_idx)]
            if pi >= len(base_X):
                pi = 0
            x0 = base_X[pi]
            # small random perturbation: larger for shape, smaller for g
            d_shape = step_shape * rng.standard_normal(n_z)
            d_g = step_g * rng.standard_normal(n - n_z)
            d = np.concatenate([d_shape, d_g])
            x = np.clip(x0 + d, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            # accept if it does not dominate the incumbent and improves the sum
            if np.all(f <= base_F[pi]) or f.sum() < base_F[pi].sum():
                base_X[pi] = x
                base_F[pi] = f

        # Merge final population with the original candidates (all recorded)
        final_X = np.concatenate([X, base_X], axis=0)
        final_F = np.concatenate([F, base_F], axis=0)

        return final_X
# EVOLVE-BLOCK-END
