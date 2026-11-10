# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def prime_number_theorem_partial_function():
    """
    Uses Linear Programming to find optimal coefficients f(k) for a small domain
    that satisfy the feasibility constraint M(x) <= 1 and maximize the objective.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # We will search over different domain sizes N
    # The domain will be {1, 2, ..., N}
    # Variables are f(2), f(3), ..., f(N). f(1) is determined by the others.
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Candidate domain sizes
    N_candidates = [10, 20, 50, 100, 200, 500, 1000]
    
    for N in N_candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
            
        # Variables: x_i corresponds to f(i+1) for i=0..N-1
        # So x_0 = f(2), x_1 = f(3), ..., x_{N-1} = f(N)
        n_vars = N - 1
        
        if n_vars == 0:
            continue
            
        # Objective: Maximize -sum f(k) log(k) / k
        # In linprog, we minimize c @ x.
        # So we minimize sum f(k) log(k) / k
        # c_i = log(i+1) / (i+1)
        ks = np.arange(2, N + 1, dtype=np.float64)
        c = np.log(ks) / ks
        
        # Constraints:
        # 1. M(x) <= 1 for x in [1, 10N]
        # M(x) = f(1)_adj + sum_{k|x, k>1} f(k)
        # f(1)_adj = -sum_{j=2}^N f(j)/j
        # M(x) = -sum_{j=2}^N f(j)/j + sum_{k|x, 2<=k<=N} f(k)
        # M(x) = sum_{j=2}^N f(j) * (I(j|x) - 1/j)
        
        max_x = 10 * N
        # A_ub shape: (max_x, n_vars)
        # We need to build A_ub efficiently.
        # For each variable j (from 2 to N), the coefficient for constraint x is:
        # 1 if j divides x, else 0, minus 1/j.
        
        # We can build this by iterating over j and adding to rows.
        # Or use sparse matrix construction.
        
        A_data = []
        A_rows = []
        A_cols = []
        
        # Precompute divisors or just check divisibility?
        # For N=1000, max_x=10000.
        # Loop j from 2 to N.
        # For each j, it contributes to x = j, 2j, 3j... up to max_x.
        # Coefficient is 1 - 1/j.
        # For x not divisible by j, coefficient is -1/j.
        
        # This is dense if we do it naively.
        # Let's use the structure:
        # M(x) = sum_{j=2}^N f(j) * (delta_{j|x} - 1/j)
        
        # We can construct A_ub as:
        # A_ub[x, j] = 1 if (x % j == 0) else 0, then subtract 1/j from all entries in column j.
        
        # Using sparse matrices
        # First part: Divisibility matrix D where D[x, j] = 1 if j|x
        # Second part: Vector v_j = 1/j. A_ub = D - v (broadcasted)
        
        # Let's build D first.
        D_data = []
        D_rows = []
        D_cols = []
        
        for j in range(2, N + 1):
            col_idx = j - 2  # index in x array
            # x values divisible by j: j, 2j, 3j...
            for x in range(j, max_x + 1, j):
                row_idx = x - 1 # M(x) for x=1..max_x, but M(1) doesn't involve j>1
                # Actually M(x) is defined for x in [1, 10K].
                # If x < 2, no j>1 divides it.
                # So for x=1, M(1) = f(1)_adj.
                # f(1)_adj = -sum f(j)/j.
                # M(1) = -sum f(j)/j <= 1.
                
                # Let's handle x=1 separately or include it in the loop logic.
                # If x=1, no j>1 divides 1. So D[1, j] = 0.
                # Coefficient is 0 - 1/j = -1/j.
                
                if x >= 2:
                    D_data.append(1.0)
                    D_rows.append(x - 1)
                    D_cols.append(col_idx)
                    
        # Build D sparse
        D = csr_matrix((D_data, (D_rows, D_cols)), shape=(max_x, n_vars))
        
        # Subtract 1/j from each column j
        # col_j corresponds to variable j (2..N)
        # value to subtract is 1/j
        inv_j = 1.0 / ks # array of 1/j for j=2..N
        # We need to subtract inv_j[i] from every entry in column i.
        # In sparse matrix, this means adding -inv_j[i] to all rows in column i.
        
        # It is easier to construct A_ub directly in COO format.
        # A_ub[row, col] = D[row, col] - 1/ks[col]
        
        # Let's just iterate and build A_ub directly to avoid dense intermediate.
        A_data = []
        A_rows = []
        A_cols = []
        
        for j in range(2, N + 1):
            col_idx = j - 2
            coeff_div = 1.0 - 1.0/j
            coeff_non_div = -1.0/j
            
            # Rows where j divides x
            div_rows = []
            for x in range(j, max_x + 1, j):
                div_rows.append(x - 1)
                
            # Rows where j does not divide x
            # This is all rows 0..max_x-1 except div_rows.
            # Instead of iterating non-div rows, let's just fill the whole column with coeff_non_div
            # and then overwrite div_rows with coeff_div.
            
            # This will create duplicates in COO if we're not careful, or we can use LIL.
            # Let's use LIL for construction.
            pass
            
        # Alternative: Build A_ub using LIL
        A_lil = np.zeros((max_x, n_vars)) # Dense might be too big for N=1000? 10000x999 ~ 80MB. OK.
        # For N=5000, 50000x4999 ~ 2GB. Too big.
        # So we must use sparse.
        
        # Let's stick to COO with careful appending.
        # For each column j:
        # 1. Add (row, col, -1/j) for all rows 0..max_x-1? No, that's dense.
        
        # Let's use the property:
        # A_ub = D - V
        # Where V is a matrix with V[row, col] = 1/ks[col].
        # We can compute D (sparse) and then subtract V (sparse, diagonal-like but repeated).
        
        # Let's just compute D as sparse.
        D_data = []
        D_rows = []
        D_cols = []
        
        for j in range(2, N + 1):
            col_idx = j - 2
            for x in range(j, max_x + 1, j):
                D_data.append(1.0)
                D_rows.append(x - 1)
                D_cols.append(col_idx)
                
        D = csr_matrix((D_data, (D_rows, D_cols)), shape=(max_x, n_vars))
        
        # Now construct V. V has same shape.
        # V[row, col] = 1/ks[col]
        # We can create a sparse matrix V by taking the column sums? No.
        # Just create a dense array of 1/ks and broadcast?
        # Let's create a sparse matrix V by adding entries.
        # Since V is "dense" in the sense that every column has non-zero values everywhere,
        # it's better to keep A_ub in a format that allows column operations.
        
        # Actually, we can just compute the final coefficients in COO.
        # For each column j:
        #   Coeff is -1/j for all rows.
        #   Coeff is 1 - 1/j for rows where j|x.
        
        A_data = []
        A_rows = []
        A_cols = []
        
        for j in range(2, N + 1):
            col_idx = j - 2
            base_val = -1.0 / j
            div_val = 1.0 - 1.0 / j
            
            # Add base_val for all rows
            # This creates max_x entries per column. Total N * max_x entries.
            # For N=100, max_x=1000 -> 100,000 entries. OK.
            # For N=1000, max_x=10000 -> 10,000,000 entries. OK.
            # For N=5000, max_x=50000 -> 250,000,000 entries. Might be slow/memory heavy.
            
            # Let's limit N to 1000 for safety in LP construction.
            if N > 1000:
                continue
                
            for x_idx in range(max_x):
                A_data.append(base_val)
                A_rows.append(x_idx)
                A_cols.append(col_idx)
                
            # Overwrite with div_val for divisible rows
            # We can just append and let csr_matrix sum them? No, we need to replace.
            # Better: Build two sparse matrices and add.
            # M1: Base matrix (dense-like columns)
            # M2: Divisibility matrix (sparse)
            # A_ub = M1 + M2 - D (since D has 1s, we want 1-1/j instead of -1/j + 1)
            # Wait.
            # Base: -1/j
            # Div: +1
            # Sum: 1 - 1/j. Correct.
            
            # So A_ub = M_base + D
            # M_base has -1/j in every entry of column j.
            
            pass # We'll construct M_base and D separately and add them.

        # Construct M_base
        # M_base is sparse. Each column j has value -1/j in all rows.
        mb_data = []
        mb_rows = []
        mb_cols = []
        
        for j in range(2, N + 1):
            col_idx = j - 2
            val = -1.0 / j
            for x_idx in range(max_x):
                mb_data.append(val)
                mb_rows.append(x_idx)
                mb_cols.append(col_idx)
                
        M_base = csr_matrix((mb_data, (mb_rows, mb_cols)), shape=(max_x, n_vars))
        
        # Construct D (Divisibility)
        d_data = []
        d_rows = []
        d_cols = []
        
        for j in range(2, N + 1):
            col_idx = j - 2
            for x in range(j, max_x + 1, j):
                d_data.append(1.0)
                d_rows.append(x - 1)
                d_cols.append(col_idx)
                
        D = csr_matrix((d_data, (d_rows, d_cols)), shape=(max_x, n_vars))
        
        A_ub = M_base + D
        
        # Constraints A_ub @ x <= 1
        b_ub = np.ones(max_x)
        
        # Bounds for x (which are f(k) for k=2..N)
        # Clipping is [-10, 10]
        bounds = [(-10, 10)] * n_vars
        
        # Additional constraint on f(1)_adj
        # f(1)_adj = -sum f(j)/j
        # -10 <= f(1)_adj <= 10
        # -10 <= -sum f(j)/j <= 10
        # 10 >= sum f(j)/j >= -10
        # sum f(j)/j <= 10
        # -sum f(j)/j <= 10  => sum f(j)/j >= -10
        
        # We add these to A_ub and b_ub.
        # Row 1: sum f(j)/j <= 10
        # Row 2: -sum f(j)/j <= 10
        
        row1 = 1.0 / ks
        row2 = -1.0 / ks
        
        A_ub = A_ub.tocsr()
        # Append rows
        A_extra = np.vstack([row1, row2])
        b_extra = np.array([10.0, 10.0])
        
        A_ub = np.vstack([A_ub.toarray(), A_extra]) # Converting to dense might be ok for N<=1000
        b_ub = np.append(b_ub, b_extra)
        
        # Solve LP
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            continue
            
        if not res.success:
            continue
            
        x_opt = res.x
        
        # Reconstruct f(k)
        # f(1)_adj = -sum f(j)/j
        f1_adj = -np.sum(x_opt / ks)
        
        # Check clipping on f(1)_adj (should be handled by LP, but verify)
        if f1_adj < -10 or f1_adj > 10:
            continue
            
        # Build domain and values
        # Domain: 1, 2, ..., N
        # Values: f(1)_adj, x_opt[0], ..., x_opt[N-2]
        
        domain = np.arange(1, N + 1, dtype=np.int64)
        values = np.empty(N, dtype=np.float64)
        values[0] = f1_adj
        values[1:] = x_opt
        
        # Verify feasibility manually (safety check)
        # M(x) = sum f(k) floor(x/k)
        # We used linear approximation M(x) = sum_{k|x} f(k).
        # Wait. The task defines M(x) = sum_k f(k) * floor(x/k).
        # My LP constraint was M(x) = sum_{k|x} f(k).
        # These are DIFFERENT.
        # floor(x/k) is 1 if k<=x and k|x? No.
        # floor(x/k) counts how many multiples of k are <= x.
        # sum_k f(k) floor(x/k) = sum_k f(k) * (number of multiples of k <= x)
        # = sum_{m=1}^x ( sum_{k|m} f(k) )
        # Let S(m) = sum_{k|m} f(k).
        # Then M(x) = sum_{m=1}^x S(m).
        
        # My LP constraint was S(x) <= 1? Or M(x) <= 1?
        # The task says: "if any M(x) > 1.0001".
        # So we need sum_{m=1}^x S(m) <= 1.
        
        # My LP constrained S(x) <= 1 (since I built A_ub based on S(x) = sum_{k|x} f(k)).
        # If S(x) <= 1 for all x, then M(x) = sum S(m) can be up to x.
        # This is a HUGE error in the LP formulation.
        
        # Correction:
        # We need M(x) = sum_{m=1}^x S(m) <= 1.
        # This implies S(m) must be small on average.
        # If S(m) <= 0 for most m, M(x) stays low.
        
        # Let's reformulate the LP constraints.
        # Variables f(j).
        # S(m) = sum_{j|m} f(j).
        # M(x) = sum_{m=1}^x S(m) = sum_{m=1}^x sum_{j|m} f(j) = sum_{j=1}^N f(j) * floor(x/j).
        
        # We need sum_{j=1}^N f(j) * floor(x/j) <= 1 for all x in [1, 10N].
        
        # This is a linear constraint on f(j).
        # A_ub[x, j] = floor(x/j).
        
        # Let's rebuild A_ub with floor(x/j).
        
        pass # Restarting LP construction with correct constraints

    # Since the first attempt had a logical flaw in the constraint definition, 
    # and N=1000 dense matrix is 10000x1000 = 10M floats = 80MB, which is fine.
    # Let's implement the correct LP.
    
    for N in [20, 50, 100, 200]:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
            
        n_vars = N - 1
        if n_vars == 0: continue
        
        ks = np.arange(2, N + 1, dtype=np.float64)
        c = np.log(ks) / ks
        
        max_x = 10 * N
        
        # Build A_ub: A_ub[x, j] = floor((x+1) / (j+1)) ? 
        # Indexing: 
        # x in [1, max_x]
        # j in [2, N]
        # f(1) is separate.
        
        # M(x) = f(1)_adj * floor(x/1) + sum_{j=2}^N f(j) * floor(x/j)
        # f(1)_adj = -sum_{j=2}^N f(j)/j
        # M(x) = -floor(x) * sum_{j=2}^N f(j)/j + sum_{j=2}^N f(j) * floor(x/j)
        # M(x) = sum_{j=2}^N f(j) * ( floor(x/j) - floor(x)/j )
        
        # Constraint: M(x) <= 1
        
        A_data = []
        A_rows = []
        A_cols = []
        
        for x in range(1, max_x + 1):
            row_idx = x - 1
            floor_x = x # floor(x/1)
            for j in range(2, N + 1):
                col_idx = j - 2
                coeff = np.floor(x / j) - floor_x / j
                if coeff != 0:
                    A_data.append(coeff)
                    A_rows.append(row_idx)
                    A_cols.append(col_idx)
                    
        A_ub = csr_matrix((A_data, (A_rows, A_cols)), shape=(max_x, n_vars))
        b_ub = np.ones(max_x)
        
        # Clipping constraints for f(1)_adj
        # f(1)_adj = -sum f(j)/j
        # -10 <= -sum f(j)/j <= 10
        # sum f(j)/j <= 10
        # -sum f(j)/j <= 10
        
        row1 = 1.0 / ks
        row2 = -1.0 / ks
        
        A_ub_dense = A_ub.toarray()
        A_extra = np.vstack([row1, row2])
        A_ub_final = np.vstack([A_ub_dense, A_extra])
        b_ub_final = np.append(b_ub, [10.0, 10.0])
        
        bounds = [(-10, 10)] * n_vars
        
        try:
            res = linprog(c, A_ub=A_ub_final, b_ub=b_ub_final, bounds=bounds, method='highs')
            if not res.success:
                continue
                
            x_opt = res.x
            f1_adj = -np.sum(x_opt / ks)
            
            if f1_adj < -10 or f1_adj > 10:
                continue
                
            domain = np.arange(1, N + 1, dtype=np.int64)
            values = np.empty(N, dtype=np.float64)
            values[0] = f1_adj
            values[1:] = x_opt
            
            # Quick feasibility check
            # Compute M(x) for all x
            # M(x) = sum f(k) floor(x/k)
            # We can compute this efficiently.
            g = np.zeros(max_x + 1)
            for k, v in zip(domain, values):
                if v != 0:
                    g[k::k] += v
            M = np.cumsum(g)
            if np.max(M[1:]) > 1.0001:
                continue
                
            # Calculate score
            k_float = domain.astype(np.float64)
            score = -np.sum(values * np.log(k_float) / k_float)
            
            if score > best_score:
                best_score = score
                best_domain = domain
                best_values = values
                
        except Exception:
            continue
            
    if best_domain is None:
        # Fallback
        domain = np.array([1, 2], dtype=np.int64)
        values = np.array([0.0, 0.0], dtype=np.float64)
        # Ensure S=0, M(x)=0 <= 1. Score 0.
        return domain, values
        
    return best_domain, best_values

# EVOLVE-BLOCK-END
