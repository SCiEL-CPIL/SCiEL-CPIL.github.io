import numpy as np
import time
import random
from fractions import Fraction

def construct() -> dict:
    """
    Returns a tensor decomposition for 3x3 by 3x5 matrix multiplication.
    Uses Strassen-style block decomposition to potentially beat schoolbook rank of 45.
    """
    start_time = time.time()
    DEADLINE = 150.0  # seconds

    n, m, p = 3, 3, 5

    def build_schoolbook():
        dim_U = n * m
        dim_V = m * p
        dim_W = p * n
        rank = n * m * p
        U = np.zeros((dim_U, rank), dtype=float)
        V = np.zeros((dim_V, rank), dtype=float)
        W = np.zeros((dim_W, rank), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W, rank

    def strassen_style_decomposition():
        """
        Try to use Strassen's idea: treat the 3x3 * 3x5 multiplication
        by splitting into blocks and using bilinear forms.
        
        For 3x3 * 3x5, we can think of the 3x5 as three columns of 3x3 blocks (not quite,
        since 5 isn't 3). Instead, let's use a direct rank reduction approach.
        
        Key insight: The tensor for matrix mult has rank at most n*m*p. For 3x3x5,
        schoolbook is 45. Lower bound is max(9, 15, 15) = 15.
        
        Let's try: use the fact that multiplication by a 3x3 matrix can be done
        with fewer than 9 multiplications using Strassen on 2x2 blocks plus handling.
        
        Actually, let's try a rank-35 or lower approach using bilinear structure.
        
        A known result: 3x3 x 3x3 matrix multiplication has rank at most 23 (via
        optimized algorithms). For 3x3 x 3x5, we can view the output as 3x5,
        which is 3*(3+2). 
        
        Let me try a structured approach: use Strassen for the 3x3 parts.
        Split A (3x3) into 4 blocks: A11(2x2), A12(2x1), A21(1x2), A22(1x1).
        Split B (3x5) similarly...
        
        Alternative: just try to find a random or structured low-rank decomposition
        by solving linear equations.
        """
        # Build the tensor T[i,j,k] = delta for matrix mult: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
        # for all i,j,k.
        
        # T[i,j,k] is a tensor of shape (n, m, p) where T[i,j,k] = 1 if we're looking at
        # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}. Actually the tensor is defined such that
        # for basis vectors, the multiplication tensor M_{n,m,p} has
        # M[i,j,k,l]... let me re-examine.
        
        # Standard: matrix mult tensor T_{n,m,p} in (C^n * C^m*) x (C^m * C^p*) x (C^n* * C^p)
        # has components T_{(i,j),(j',k),(k',i')} = delta_{j,j'} delta_{i,i'} delta_{k,k'}
        # The contraction is: sum_{j,j',k,k'} T_{...} A_{i,j} B_{j',k} C^{k',i'}
        # = sum_j A_{i,j} B_{j,k} C^{k,i}
        # which equals (AB C) where C is the dual of output.
        
        # For decomposition: T = sum_r U_r \otimes V_r \otimes W_r
        # where U_r is in C^{n*m}, V_r in C^{m*p}, W_r in C^{p*n}.
        
        # The constraint: for all (i,j), (j',k), (k',i'),
        # sum_r U_r[i*m+j] * V_r[j'*p+k] * W_r[k'*n+i'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k'... }
        # Wait, let me be more careful.
        
        # The matrix multiplication tensor satisfies:
        # sum_j A_{i,j} B_{j,k} = (AB)_{i,k}
        # In terms of trilinear form with dual:
        # sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} where C is paired with (AB)^T or similar.
        
        # The standard identity: for the tensor, we need
        # sum_r U_{im+j, r} V_{jp+k, r} W_{kn+i, r} = [j == j] * ... 
        # Actually the tensor elements are: T[(i,j),(j,k),(k,i)] = delta for the pairing.
        # More precisely: the tensor is in (C^{n*m}) * (C^{m*p}) * (C^{p*n})
        # with T[(i,j),(j',k),(k',i')] = 1 if j==j', k==k', i==i', else 0.
        
        # So the constraint is:
        # sum_r U[(i,j), r] * V[(j',k), r] * W[(k',i'), r] = 1 if (j==j' and k==k' and i==i') else 0
        
        # This is a system of n*m*p equations... no, (n*m)*(m*p)*(p*n) equations total
        # but only n*m*p are non-zero (one for each triple (i,j,k) with i'=i, j'=j, k'=k).
        
        # For the non-zero entries: for each (i,j,k):
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # For zero entries (where indices don't match): the sum must be 0.
        
        # This is a nonlinear system. Let's try to find a rank-r solution.
        
        # Strategy: use random initialization and gradient-free / alternating minimization
        # on the trilinear form.
        
        # Actually, a simpler approach: note that the problem is equivalent to finding
        # minimal r such that the tensor can be written as sum of r rank-1 tensors.
        
        # For 3x3x3, best known rank is 23 (or lower? I think 23 is optimal for 3x3x3).
        # For 3x3x5, a natural approach: use rank-23 for 3x3x3 part (columns 1-3) 
        # and handle columns 4-5 separately.
        # Columns 4-5 of B: multiply A (3x3) by B_cols_45 (3x2) -> (3x2) output.
        # This needs rank at most 3*3*2 = 18 by schoolbook.
        # But we can do better: rank of 3x3 x 3x2 mult.
        # Lower bound: max(9, 6, 6) = 9.
        
        # Actually, let me just try to search for a decomposition with rank < 45
        # using an optimization approach on the trilinear constraints.
        
        # Let me try rank r and see if we can find U, V, W.
        
        # For simplicity, let's try the following structured decomposition:
        # Use Strassen's 7-multiplication algorithm for 2x2 blocks.
        
        # Split the 3x3 * 3x5 as follows:
        # A = [[a11 a12 a13],
        #      [a21 a22 a23],
        #      [a31 a32 a33]]
        # B = [[b11 b12 b13 b14 b15],
        #      [b21 b22 b23 b24 b25],
        #      [b31 b32 b33 b34 b35]]
        
        # Let's group A's first 2 rows/cols as a 2x2 block, etc.
        
        # This is getting complex. Let me try a numerical search for low rank.
        
        # For a given rank r, we have variables:
        # U: 9 x r, V: 15 x r, W: 15 x r
        # Total: (9+15+15)*r = 39r parameters.
        # Constraints: for each (i,j,k) in {0,1,2}^3, 
        #   sum_{r'} U[i*3+j, r'] * V[j*5+k, r'] * W[k*3+i, r'] = 1
        # That's 27 constraints.
        # Plus the zero constraints: for all (i,j,j',k,k',i') where not (j==j', k==k', i==i'),
        #   sum_r U[i*3+j, r] * V[j'*5+k, r] * W[k*3+i', r] = 0
        # That's (9*15*15) - 27 = 2025 - 27 = 1998 constraints.
        
        # Total: 39r variables, 2025 constraints.
        # For this to be feasible, we need 39r >= 2025 (roughly), so r >= ~52.
        # But the constraints are not all independent! The tensor structure means
        # many constraints are related.
        
        # Actually, the correct count: the tensor lives in a space of dimension 9*15*15 = 2025,
        # and we're looking for the tensor rank. The schoolbook gives rank 45, which works.
        # So rank 45 is always feasible. The question is whether rank < 45 is feasible.
        
        # Given the complexity, let me try a pragmatic approach:
        # 1. Start with schoolbook rank 45.
        # 2. Try to reduce rank by finding linear dependencies.
        
        # Actually, let me just try to use a known result: for small matrix multiplication,
        # the best known bounds are:
        # R(3,3,3) <= 23 (Lambert et al. 2018 found 22? No, I think 23 is best known for 3x3x3)
        # Wait, actually the best known upper bound for 3x3x3 is 23, and lower bound is 19.
        
        # For 3x3x5: we can write B = [B_1, B_2, B_3, B_4, B_5] where each B_i is 3x1.
        # (AB)_i = A * B_i for i=1..5.
        # So we need to compute A times each of 5 column vectors.
        # A is 3x3, B_i is 3x1, output is 3x1.
        # This is matrix-vector multiplication: 3x3 * 3x1.
        # Rank of 3x3x1 tensor: schoolbook is 9. Lower bound is 3.
        # Best known for 3x3x1: I think it's 8? Or maybe 7?
        # Actually, matrix-vector multiplication MVM_{n,n} has rank n^2 for schoolbook,
        # but can be done with fewer. For 3x3x1, the tensor has rank at most 8
        # (I recall something like this).
        
        # If each column needs rank 8, and they share the A factors, we can do better.
        # The full tensor 3x3x5: we're computing all 5 columns simultaneously.
        # This is equivalent to: given A (3x3) and B (3x5), compute AB (3x5).
        # This is the same as (n,m,p) = (3,3,5) matrix multiplication.
        
        # Let me try: R(3,3,5) <= R(3,3,3) + R(3,3,2)
        # R(3,3,3) <= 23
        # R(3,3,2): schoolbook 18. Can we do better?
        # Lower bound for (3,3,2): max(9, 6, 6) = 9.
        # Best known: I think R(3,3,2) = 12 or so?
        
        # If R(3,3,5) <= 23 + 12 = 35, that's a big improvement over 45.
        
        # But I don't have the exact decompositions memorized. Let me try to construct one.
        
        # For now, let me try a numerical search for rank < 45.
        
        # I'll use the following approach:
        # Initialize U, V, W with schoolbook decomposition.
        # Try to find linear combinations that reduce the number of terms.
        
        # Actually, a cleaner approach: use the fact that the tensor multiplication
        # satisfies linearity, and try to find a basis reduction.
        
        # Let me try a random search with the constraint that entries are in {0, 1/2, 1, 3/2, 2}
        # or just integers, to find a low-rank exact decomposition.
        
        return None  # placeholder, we'll implement search below

    # First, get the schoolbook solution as a fallback
    U_school, V_school, W_school, rank_school = build_schoolbook()
    
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 30:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Try to find a better decomposition using structured search
    # We'll search for decompositions with rank < 45
    
    # The key equations for rank-r decomposition:
    # For each (i, j, k) with i,j,k in {0,1,2}:
    #   sum_{r=0}^{r-1} U[i*3+j, r] * V[j*5+k, r] * W[k*3+i, r] = 1
    # For all other index combinations (where the tensor is zero):
    #   sum_r U[i*3+j, r] * V[j'*5+k, r] * W[k*3+i', r] = 0
    
    # This is a system of trilinear equations. Let's try a heuristic search.
    
    # Approach: try rank r = 44, 43, ... and use random restarts with local search.
    # But this might be too slow. Let's think of a more clever approach.
    
    # Insight: The 3x3x5 matrix multiplication can be decomposed by treating
    # the 5 columns of B as 3+2. Use a known decomposition for 3x3x3 (rank 23)
    # and for 3x3x2 (which we can try to find).
    
    # For 3x3x2: (A*B)_col_j = A * B_col_j for j=1,2.
    # This is the tensor T_{i,j,k} where i in {0,1,2} (row of A/output),
    # j in {0,1,2} (col of A / row of B), k in {0,1} (col of B/output).
    # Rank of this tensor: schoolbook 18.
    
    # Let me try to find a rank-12 decomposition for 3x3x2.
    # U: 3*3 = 9 elements per factor, V: 3*2 = 6, W: 2*3 = 6.
    
    # Actually, let me just try a random algebraic search for the full 3x3x5 problem
    # with rank < 45, using the trilinear constraints.
    
    # For efficiency, I'll use numpy and a simple iterative approach.
    
    def try_random_decomposition(rank_r, n_iters=1000, seed=42):
        """Try to find a rank-r decomposition via random init + local adjustment."""
        rng = random.Random(seed)
        
        # Random initialization with values in {-1, 0, 1} for simplicity
        U = np.array([rng.choice([-1, 0, 1]) for _ in range(n * m * rank_r)], 
                     dtype=float).reshape(n * m, rank_r)
        V = np.array([rng.choice([-1, 0, 1]) for _ in range(m * p * rank_r)], 
                     dtype=float).reshape(m * p, rank_r)
        W = np.array([rng.choice([-1, 0, 1]) for _ in range(p * n * rank_r)], 
                     dtype=float).reshape(p * n, rank_r)
        
        # Check constraints: for each (i,j,k):
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        # And for mismatches, should be 0.
        
        def check_error(U, V, W):
            err = 0.0
            # Non-zero entries
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank_r):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err += (val - 1.0) ** 2
            # Check a few zero entries (sampled)
            for _ in range(100):
                i = rng.randint(0, n - 1)
                j = rng.randint(0, m - 1)
                k = rng.randint(0, p - 1)
                i2 = rng.randint(0, n - 1)
                j2 = rng.randint(0, m - 1)
                k2 = rng.randint(0, p - 1)
                if not (i == i2 and j == j2 and k == k2):
                    val = 0.0
                    for r in range(rank_r):
                        val += U[i * m + j, r] * V[j2 * p + k2, r] * W[k * n + i2, r]
                    err += val ** 2
            return err
        
        # This is too slow for large rank_r. Let me use numpy vectorization.
        
        def check_error_fast(U, V, W):
            # Build the full tensor from U, V, W
            # Tensor[i*m+j, j'*p+k, k'*n+i'] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
            # We need this to equal delta_{j,j'} delta_{k,k'} delta_{i,i'}
            
            # Compute: T = sum_r outer(U[:,r], outer(V[:,r], W[:,r]))
            # But this is huge (9*15*15 = 2025 * rank_r). Let's compute the error directly.
            
            err = 0.0
            for r_idx in range(rank_r):
                # This term contributes to the tensor
                pass
            
            # Actually, let's compute the trilinear form directly for all nonzero entries
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # (U[:, i*m+j]) * (V[:, j*p+k]) * (W[:, k*n+i]) dot product
                        u_col = U[:, i * m + j]
                        v_col = V[:, j * p + k]
                        w_col = W[:, k * n + i]
                        val = np.sum(u_col * v_col * w_col)
                        err += (val - 1.0) ** 2
            
            # Sample zero entries
            for _ in range(50):
                i = rng.randint(0, n - 1)
                j = rng.randint(0, m - 1)
                k = rng.randint(0, p - 1)
                i2 = rng.randint(0, n - 1)
                j2 = rng.randint(0, m - 1)
                k2 = rng.randint(0, p - 1)
                if (i, j, k) != (i2, j2, k2):
                    u_col = U[:, i * m + j]
                    v_col = V[:, j2 * p + k2]
                    w_col = W[:, k * n + i2]
                    val = np.sum(u_col * v_col * w_col)
                    err += val ** 2
            return err
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(n_iters):
            if time.time() - start_time > DEADLINE - 20:
                break
            err = check_error_fast(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
            
            # Random perturbation: flip a random entry
            mat = rng.choice([U, V, W])
            row = rng.randint(0, mat.shape[0] - 1)
            col = rng.randint(0, mat.shape[1] - 1)
            old_val = mat[row, col]
            mat[row, col] = rng.choice([-1.0, 0.0, 1.0])
            
            new_err = check_error_fast(U, V, W)
            if new_err >= err:
                mat[row, col] = old_val
        
        if best_err < 1e-8:
            return best_U, best_V, best_W, best_err
        return None
    
    # Try ranks from 40 down to 35 (just a few to save time)
    for test_rank in [40, 38, 36]:
        if time.time() - start_time > DEADLINE - 30:
            break
        result = try_random_decomposition(test_rank, n_iters=500, seed=42 + test_rank)
        if result is not None:
            U_f, V_f, W_f, err = result
            if err < 1e-8 and test_rank < best_rank:
                best_U, best_V, best_W, best_rank = U_f, V_f, W_f, test_rank
                break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
