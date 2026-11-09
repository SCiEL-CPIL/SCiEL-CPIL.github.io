# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k)*log(k)/k subject to
    M(x) = sum_k f(k)*floor(x/k) <= 1.0001 for all x in [1, 10K],
    where K is the maximum key in the domain.
    
    After evaluation: f(1) is replaced by f(1) - S where S = sum_k f(k)/k,
    so that sum_k f(k)/k = 0.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s kill

    # Best feasible solution found so far
    best_domain = None
    best_values = None
    best_score = -np.inf

    # We'll try various K values. The LP has O(10K) constraints and K variables.
    # For K up to ~2000, this should be tractable with HiGHS.
    
    # Key insight: After the f(1) adjustment, we need sum_k f(k)/k = 0.
    # f(1) = -sum_{k>=2} f(k)/k (since f(1) starts at 0 before adjustment).
    # Then M(x) = f(1)*floor(x/1) + sum_{k>=2} f(k)*floor(x/k)
    #           = f(1)*x + sum_{k>=2} f(k)*floor(x/k)
    #           = -x * sum_{k>=2} f(k)/k + sum_{k>=2} f(k)*floor(x/k)
    #           = sum_{k>=2} f(k) * (floor(x/k) - x/k)
    #           = -sum_{k>=2} f(k) * frac(x/k)
    #
    # So M(x) = -sum_{k=2}^K f(k) * frac(x/k) for all x.
    # We need M(x) <= 1.0001 for all x in [1, 10K].
    # Since frac(x/k) >= 0, we need -f(k) * frac(x/k) contributions to be bounded.
    # Equivalently: sum_{k=2}^K f(k) * frac(x/k) >= -1.0001 for all x.
    # And also M(x) can be negative (no lower bound issue from the constraint, 
    #   only M(x) > 1.0001 is infeasible).
    #
    # Wait, let me re-read: "if any M(x) > 1.0001 the function is infeasible"
    # So the constraint is M(x) <= 1.0001 for all x.
    #
    # M(x) = -sum_{k=2}^K f(k) * frac(x/k)
    # So we need: -sum_{k=2}^K f(k) * frac(x/k) <= 1.0001
    # i.e.: sum_{k=2}^K f(k) * frac(x/k) >= -1.0001
    #
    # Objective: maximize a = -sum_k f(k)*log(k)/k
    # Note: f(1) term is f(1)*log(1)/1 = 0, so a = -sum_{k=2}^K f(k)*log(k)/k
    #
    # To maximize a, we want f(k)*log(k)/k to be as negative as possible,
    # i.e., f(k) should be as negative as possible (since log(k)/k > 0 for k>=2).
    # So we want f(k) = -10 for as many k as possible.
    #
    # But the constraint sum_{k=2}^K f(k) * frac(x/k) >= -1.0001 limits how negative
    # we can make f(k). If f(k) = -10, then the contribution is -10*frac(x/k) <= 0,
    # which makes the LHS more negative, violating the constraint.
    #
    # Actually wait: if f(k) is negative, f(k)*frac(x/k) is negative, making the sum
    # more negative, which violates >= -1.0001. So negative f(k) is constrained.
    #
    # If f(k) is positive, f(k)*frac(x/k) is positive, helping satisfy the constraint.
    # But positive f(k) reduces the objective a (since we want f(k) negative).
    #
    # So the optimal strategy: make f(k) as negative as possible (-10), but add some
    # positive f(k) values to keep the constraint satisfied.
    #
    # Let's set up the LP:
    # Minimize: sum_{k=2}^K f(k)*log(k)/k   (equivalent to maximizing a)
    # Subject to: for all x in [1, 10K]: sum_{k=2}^K f(k) * frac(x/k) >= -1.0001
    #            -10 <= f(k) <= 10
    #
    # Rewrite constraint as: -sum_{k=2}^K f(k) * frac(x/k) <= 1.0001
    # For linprog (min c^T x s.t. A_ub x <= b_ub):
    # A_ub[i, k] = -frac(x_i / k), b_ub[i] = 1.0001

    # Try different K values
    for K in [50, 100, 200, 500, 1000, 2000]:
        if time.time() > deadline - 30:
            break

        ks = np.arange(2, K + 1, dtype=np.int64)
        num_vars = len(ks)
        x_max = 10 * K
        num_x = x_max

        # Objective: minimize sum f(k)*log(k)/k
        c = np.log(ks.astype(np.float64)) / ks.astype(np.float64)

        # Build constraint matrix efficiently
        # A_ub[i, j] = -frac(x_i / ks[j])
        # This is a num_x x num_vars matrix, which can be large.
        # For K=2000, num_x=20000, num_vars=1999 -> 40M entries, ~320MB as float64.
        # Might be too large. Let's be careful.
        
        if num_x * num_vars > 100_000_000:
            # Too large, skip or subsample x
            # Subsample x values
            step = max(1, num_x // 5000)
            xs = np.arange(1, x_max + 1, step, dtype=np.float64)
        else:
            xs = np.arange(1, x_max + 1, dtype=np.float64)
        
        num_x_actual = len(xs)

        # Build A_ub in chunks to manage memory
        # Actually, let's just build it directly if memory allows
        if num_x_actual * num_vars > 200_000_000:
            # Skip this K, too large
            continue

        # Compute frac matrix
        # xs: (num_x, 1), ks: (1, num_vars)
        X = xs[:, np.newaxis].astype(np.float64)
        Ks = ks[np.newaxis, :].astype(np.float64)
        ratio = X / Ks
        frac = ratio - np.floor(ratio)
        A_ub = -frac
        b_ub = np.ones(num_x_actual) * 1.0001

        bounds = [(-10, 10)] * num_vars

        try:
            from scipy.optimize import linprog
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception as e:
            print(f"LP failed for K={K}: {e}")
            continue

        if not result.success:
            print(f"LP infeasible/unsuccessful for K={K}")
            continue

        f_vals = result.x
        # Clip to be safe
        f_vals = np.clip(f_vals, -10, 10)

        # Construct domain and values
        # Domain: [1, 2, 3, ..., K]
        domain = np.arange(1, K + 1, dtype=np.int64)
        # Values: f(1) = 0 initially (will be adjusted by evaluator)
        values = np.zeros(K, dtype=np.float64)
        values[1:] = f_vals  # f(2) through f(K)

        # Compute the objective a = -sum_k f(k)*log(k)/k
        # f(1)*log(1)/1 = 0, so just sum over k>=2
        a = -np.sum(f_vals * np.log(ks.astype(np.float64)) / ks.astype(np.float64))

        # Verify feasibility: check M(x) <= 1.0001 for all x in [1, 10K]
        # M(x) = -sum_{k=2}^K f(k) * frac(x/k)
        # We need to check this for ALL x, not just the sampled ones.
        # If we subsampled x, we need to verify more carefully.
        
        if time.time() > deadline - 20:
            break

        # Full feasibility check
        # M(x) = sum_{k=2}^K f(k) * (floor(x/k) - x/k) = -sum f(k)*frac(x/k)
        # We can compute this efficiently using the fact that floor(x/k) changes slowly.
        
        x_check = np.arange(1, 10 * K + 1, dtype=np.float64)
        
        # For efficiency, compute M(x) in vectorized fashion if possible
        # M(x) = -sum_k f(k) * frac(x/k)
        # = -sum_k f(k) * (x/k - floor(x/k))
        # = -x * sum_k f(k)/k + sum_k f(k) * floor(x/k)
        
        # sum_k f(k)/k for k=2..K
        sum_fk_over_k = np.sum(f_vals / ks.astype(np.float64))
        
        # sum_k f(k) * floor(x/k)
        # floor(x/k) for k=2..K: for each x, floor(x/k) = number of multiples of k <= x
        
        # This is still O(K * 10K) = O(10K^2). For K=2000, that's 40M operations.
        # Let's do it in chunks.
        
        max_M = -np.inf
        chunk_size = 1000
        for i in range(0, len(x_check), chunk_size):
            x_chunk = x_check[i:i+chunk_size]  # (chunk,)
            # For each x in chunk, compute sum_k f(k) * floor(x/k)
            # floor(x/k) for k=2..K
            Xc = x_chunk[:, np.newaxis].astype(np.float64)  # (chunk, 1)
            Ks2 = ks[np.newaxis, :].astype(np.float64)  # (1, K-1)
            floors = np.floor(Xc / Ks2)  # (chunk, K-1)
            M_chunk = -np.sum(f_vals[np.newaxis, :] * floors, axis=1) + x_chunk * sum_fk_over_k
            # Wait: M(x) = sum_k f(k)*(floor(x/k) - x/k) for k>=2
            # = sum_k f(k)*floor(x/k) - x*sum_k f(k)/k
            # So M_chunk = sum_k f(k)*floor(x/k) - x*sum_fk_over_k
            M_chunk = np.sum(f_vals[np.newaxis, :] * floors, axis=1) - x_chunk * sum_fk_over_k
            max_M = max(max_M, np.max(M_chunk))
        
        if max_M > 1.0001:
            print(f"K={K}: infeasible, max_M={max_M}")
            continue

        # Also check that f(1) adjustment doesn't break things
        # f(1) = -sum_k f(k)/k = -sum_fk_over_k (since f(1) was 0)
        # After adjustment, sum_k f(k)/k = 0, so M(x) formula remains the same.
        
        print(f"K={K}: feasible, max_M={max_M}, a={a}")
        
        if a > best_score:
            best_score = a
            best_domain = domain.copy()
            best_values = values.copy()

    # If no LP solution found, return a simple fallback
    if best_domain is None:
        # Simple fallback: f(1)=0, f(2)=-0.5, f(3)=-0.5, etc., small negative values
        K = 10
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = np.zeros(K, dtype=np.float64)
        values[1:] = -0.1  # Small negative values
        
        # Verify feasibility
        sum_fk_over_k = np.sum(values[1:] / np.arange(2, K+1, dtype=np.float64))
        max_M = 0
        for x in range(1, 10*K+1):
            M = 0
            for k in range(2, K+1):
                M += values[k-1] * (x // k - x/k)
            max_M = max(max_M, M)
        
        if max_M > 1.0001:
            # Make values even smaller
            values[1:] = -0.01
            sum_fk_over_k = np.sum(values[1:] / np.arange(2, K+1, dtype=np.float64))
            max_M = 0
            for x in range(1, 10*K+1):
                M = 0
                for k in range(2, K+1):
                    M += values[k-1] * (x // k - x/k)
                max_M = max(max_M, M)
    
    return best_domain, best_values


# EVOLVE-BLOCK-END
