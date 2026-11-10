# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    DEADLINE = 620.0  # safety margin before 630s
    
    # Candidate K values to try
    # We want to maximize K to include more terms, but LP complexity grows with K
    # and the number of constraints (10K).
    # Let's try a few values.
    K_candidates = [5000, 10000, 20000, 50000]
    
    best_domain = None
    best_values = None
    best_score = -np.inf

    for K in K_candidates:
        if time.time() > DEADLINE - 30:
            break
        
        # Variables are f(2), ..., f(K)
        n_vars = K - 1
        if n_vars == 0:
            continue
            
        # Objective: Maximize -sum f(k) log(k) / k
        # linprog minimizes, so we minimize sum f(k) log(k) / k
        # c_i = log(k_i) / k_i
        k_indices = np.arange(2, K + 1)
        c = np.log(k_indices) / k_indices
        
        # Constraints: sum f(k) * C_{x,k} <= 1.0001 for x in [1, 10K]
        # C_{x,k} = floor(x/k) - x/k
        max_x = 10 * K
        x_indices = np.arange(1, max_x + 1)
        
        # Construct A matrix efficiently
        # A[i, j] corresponds to x = x_indices[i], k = k_indices[j]
        # C_{x,k} = floor(x/k) - x/k
        # We can compute this in chunks to avoid memory explosion if K is large
        # For K=50000, max_x=500000, A is 500000 x 49999 ~ 2.5e10 elements. Too big.
        # We need a smarter way or smaller K.
        
        # Let's check memory for K=10000.
        # max_x = 100000. A is 100000 x 9999 ~ 1e9 elements. ~8GB. Still too big for standard RAM?
        # Maybe we can use a sparse matrix?
        # C_{x,k} is non-zero for all x,k. It's dense.
        
        # Alternative: Use a smaller K for the LP, or use a heuristic.
        # The knowledge says 1.069 was achieved. Maybe with a smaller K?
        # Let's try K=2000 first.
        # max_x = 20000. A is 20000 x 1999 ~ 4e7 elements. ~300MB. This is feasible.
        
        # Let's adjust K_candidates to be smaller to ensure feasibility and speed.
        if K > 20000:
            # For larger K, we might just use a heuristic or skip
            # But let's try to solve for smaller K first.
            pass

        # If K is too large for dense LP, we might need to break or use sparse.
        # scipy linprog supports sparse A.
        # Let's use sparse matrices.
        from scipy.sparse import csr_matrix
        
        # We can build the sparse matrix row by row or in blocks.
        # But C_{x,k} is dense for each x (all k <= K contribute).
        # So it's not sparse in the (x,k) grid.
        
        # Let's stick to K <= 5000 for dense LP to be safe and fast.
        # If K > 5000, we might just use the result from K=5000 or a heuristic.
        
        if K > 5000:
            # For now, skip large K in LP due to memory/time
            continue
            
        try:
            # Build A
            # x_indices: (max_x,)
            # k_indices: (K-1,)
            # A[i, j] = floor(x_i / k_j) - x_i / k_j
            
            # Vectorized calculation might be memory heavy.
            # Let's do it in blocks of x.
            A_blocks = []
            block_size = 1000
            for x_start in range(0, max_x, block_size):
                x_end = min(x_start + block_size, max_x)
                x_block = x_indices[x_start:x_end]
                # x_block: (bs,)
                # k_indices: (K-1,)
                # Compute floor(x/k) and x/k
                # Broadcasting: (bs, 1) / (1, K-1)
                x_col = x_block[:, np.newaxis]
                k_row = k_indices[np.newaxis, :]
                
                # C = floor(x/k) - x/k
                C_block = np.floor(x_col / k_row) - x_col / k_row
                A_blocks.append(C_block)
                
            A = np.vstack(A_blocks)
            
            b = np.full(max_x, 1.0001)
            bounds = [(-10, 10)] * n_vars
            
            # Solve LP
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            
            if res.success:
                f_vals = res.x
                # f_vals corresponds to k=2..K
                # f(1) is determined by sum f(k)/k = 0
                # sum_{k=1}^K f(k)/k = 0 => f(1) + sum_{k=2}^K f(k)/k = 0
                # f(1) = -sum_{k=2}^K f(k)/k
                
                sum_fk_over_k = np.sum(f_vals / k_indices)
                f1 = -sum_fk_over_k
                
                # Construct full domain and values
                # Domain: 1, 2, ..., K
                # But we can omit keys where f(k) is 0? 
                # The task says "domain holds distinct positive integers". 
                # Including zeros is allowed but increases m. 
                # It's better to include only non-zero values to keep m small?
                # "returning (domain, values): two 1-D arrays of equal length m >= 1"
                # It doesn't forbid zeros. But smaller m is faster for evaluation?
                # Evaluation cost is m * 10K. So smaller m is better.
                # Let's filter out zeros.
                
                full_f = np.zeros(K + 1)
                full_f[1] = f1
                full_f[2:K+1] = f_vals
                
                non_zero_mask = full_f != 0.0
                domain = np.where(non_zero_mask)[0]
                values = full_f[non_zero_mask]
                
                if len(domain) == 0:
                    continue
                
                # Calculate score a
                # a = -sum f(k) log(k) / k
                # Note: f(1) term is 0 because log(1)=0
                a = -np.sum(values * np.log(domain) / domain)
                
                # Verify feasibility quickly?
                # The LP ensures M(x) <= 1.0001 for the unclipped values.
                # Clipping might change M(x).
                # We should check if clipping violates constraints.
                # Clip values to [-10, 10]
                clipped_values = np.clip(values, -10, 10)
                
                # Recalculate f(1) with clipped values?
                # "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S ..."
                # This implies the evaluator does:
                # 1. Clip values.
                # 2. Calculate S = sum clipped_f(k)/k.
                # 3. f(1) = original_f(1) - S? 
                # No, "f(1) is replaced by f(1) - S ... so that afterwards sum_k f(k)/k = 0".
                # If 1 is in domain, f(1) is adjusted.
                # If we output f(1) calculated from unclipped values, and then evaluator clips and adjusts,
                # the final sum will be 0.
                # But M(x) depends on the final f(k).
                # If clipping changes f(k) for k>1, M(x) changes.
                # Our LP assumed f(k) within [-10, 10]. If res.x is within bounds, clipping doesn't change them.
                # So if bounds are respected, M(x) is valid.
                
                if a > best_score:
                    best_score = a
                    best_domain = domain.copy()
                    best_values = values.copy()
                    
        except Exception as e:
            print(f"LP failed for K={K}: {e}")
            continue

    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback: Mobius function
    def compute_mobius(K):
        mu = np.ones(K + 1, dtype=np.float64)
        is_composite = np.zeros(K + 1, dtype=bool)
        primes = []
        for i in range(2, K + 1):
            if not is_composite[i]:
                primes.append(i)
                mu[i] = -1.0
            for p in primes:
                if i * p > K:
                    break
                is_composite[i * p] = True
                if i % p == 0:
                    mu[i * p] = 0.0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    K = 1000
    mu = compute_mobius(K)
    nonzero_mask = mu[1:K + 1] != 0.0
    domain = np.where(nonzero_mask)[0] + 1
    values = mu[1:K + 1][nonzero_mask]
    
    # Adjust f(1)
    if 1 in domain:
        idx = np.where(domain == 1)[0][0]
        S = np.sum(values / domain)
        values[idx] -= S
    else:
        S = np.sum(values / domain)
        # Insert 1 and -S
        domain = np.insert(domain, 0, 1)
        values = np.insert(values, 0, -S)
        
    return domain, values

# EVOLVE-BLOCK-END
