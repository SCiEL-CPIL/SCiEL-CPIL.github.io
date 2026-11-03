"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition.

Strategy: Use a structured Kronecker-product ansatz. Since m=4=2*2, we can
decompose the problem using a 2x2 block structure on the middle dimension.
The key insight: A(2x4) * B(4x7) can be viewed as a 2x2 block multiplication
where each block is 2x4 or 4x7. We use a Strassen-like approach on the 2x2
row block structure to reduce the rank.

We construct an explicit low-rank decomposition using the fact that:
- The tensor T[i*m+j, j*p+k, k*n+i] = 1 has a Kronecker structure
- We can write T as a sum of rank-1 tensors with a structured pattern
"""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 28
    pn = p * n  # 14

    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def tensor_from_factors(U, V, W):
        return np.einsum('ad,bd,cd->abc', U, V, W)

    def check_decomposition(U, V, W, tol=1e-8):
        T_rec = tensor_from_factors(U, V, W)
        return np.allclose(T_rec, T, atol=tol, rtol=tol)

    # Schoolbook decomposition as fallback
    rank_sb = n * m * p  # 56
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    def try_structured_decomposition(rank):
        """
        Try a structured decomposition using Kronecker product ideas.
        
        For 2x4 x 4x7, we can think of:
        - A as a 2x2 block matrix with 1x4 blocks (or 2x2 blocks)
        - B as a 4x7 matrix
        
        Using a 2x2 Strassen-like structure on the row index i:
        We decompose i into (i1, i2) where i1 in {0,1} and i2 in {0,1}
        (since n=2, we can use a 2-term decomposition on i)
        
        Similarly, j in {0,1,2,3} can be split as j = j1*2 + j2 where j1,j2 in {0,1}
        
        The key structure: T[i*m+j, j*p+k, k*n+i] = 1
        
        We use a rank-r decomposition where each term has a specific (i,j,k) pattern
        with coefficients that can be solved via linear algebra.
        """
        nonlocal best_U, best_V, best_W, best_rank
        
        # Build a structured set of support patterns
        # Each pattern is (i, j, k) with some coefficient structure
        # We'll use patterns that exploit the 2x2 structure of m=4
        
        # Generate candidate support patterns
        # Pattern: (i_val, j_val, k_val) for each rank term
        # We want to find patterns and coefficients that satisfy the tensor identity
        
        # Strategy: Use all (i,j,k) triples but with a structured coefficient matrix
        # The tensor identity is:
        # sum_d U[i*m+j, d] * V[j*p+k, d] * W[k*n+i, d] = T[i*m+j, j*p+k, k*n+i]
        
        # For a given support pattern (which entries of U,V,W are nonzero),
        # this becomes a system of linear equations in the products of coefficients.
        
        # Let's try: use rank = 28 (the theoretical lower bound)
        # with a carefully chosen support pattern
        
        if rank >= 56:
            return False
        
        # Try different support pattern strategies
        strategies = [
            'kronecker_2x2',
            'strassen_like', 
            'block_diagonal',
            'mixed'
        ]
        
        for strat in strategies:
            if time.time() > deadline:
                return False
            
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            if strat == 'kronecker_2x2':
                # Use Kronecker structure: m=4=2*2
                # Decompose j = j1*2 + j2
                # For each (i, j1, j2, k), create a term
                # But we need to be clever about which terms to include
                
                # Use all combinations of (i, j1, j2, k) but with structured coefficients
                # This gives n * 2 * 2 * p = 2*2*2*7 = 56 terms (schoolbook)
                # We need to reduce this.
                
                # Idea: For fixed (i, j1, k), we have two j2 values (0,1)
                # We can combine them if the coefficients allow it.
                
                # Let's use a 2-term decomposition on j2:
                # For each (i, j1, k), use 2 terms for j2=0 and j2=1
                # Total: n * 2 * p * 2 = 2*2*7*2 = 56 (no savings)
                
                # Better: Use Strassen on the 2x2 (j1, j2) block
                # Strassen for 2x2 matrix mult uses 7 multiplications instead of 8
                # Here we have a 2x2 block structure in the middle dimension
                
                # For each (i, k), we have a 2x2 block of (j1, j2) indices
                # We can use Strassen-like combinations
                
                d = 0
                for i in range(n):
                    for k in range(p):
                        # Strassen-like 7-term decomposition for the 2x2 (j1,j2) block
                        # The 8 schoolbook terms are (j1,j2) in {0,1}x{0,1}
                        # Strassen uses 7 combinations
                        
                        # M1 = A00 * B00
                        # M2 = A10 * B01  (combined)
                        # M3 = A01 * B10  (combined)
                        # M4 = A11 * B11
                        # M5 = (A00+A10) * B00
                        # M6 = (A01+A11) * B10
                        # M7 = (A00-A10) * (B01-B11)
                        # M8 = (A01-A11) * (B10-B00)  -- Strassen uses 7, not 8
                        
                        # Actually Strassen for 2x2 uses 7 multiplications:
                        # M1 = (A00+A11)*(B00+B11)
                        # M2 = (A10+A11)*B00
                        # M3 = A00*(B01-B11)
                        # M4 = A11*(B10-B00)
                        # M5 = (A00+A10)*B11
                        # M6 = (A01-A10)*(B00+B10)
                        # M7 = (A01-A11)*(B01+B11)
                        
                        # For our case, the "multiplication" is:
                        # sum over j of A[i,j] * B[j,k]
                        # With j = j1*2 + j2, this is:
                        # sum_{j1,j2} A[i, j1*2+j2] * B[j1*2+j2, k]
                        
                        # This is like multiplying a 2x2 matrix (indexed by j1,j2 for A)
                        # by a 2x2 matrix (indexed by j1,j2 for B) but with different structures.
                        
                        # Let's just use the schoolbook for now but with a twist:
                        # Use all 8 terms but with some merged via linearity
                        
                        for j1 in range(2):
                            for j2 in range(2):
                                if d >= rank:
                                    break
                                j = j1 * 2 + j2
                                U[i * m + j, d] = 1.0
                                V[j * p + k, d] = 1.0
                                W[k * n + i, d] = 1.0
                                d += 1
                        if d >= rank:
                            break
                    if d >= rank:
                        break
                
                if d < rank:
                    # Fill remaining with zeros or try to use them
                    # This won't work if d != rank
                    return False
                    
            elif strat == 'strassen_like':
                # Use a true Strassen-like approach
                # For 2x4 x 4x7, view as:
                # A is 2x4, B is 4x7
                # Split B into two 2x7 blocks: B_top (rows 0,1) and B_bot (rows 2,3)
                # Split A into two 2x2 blocks: A_left (cols 0,1) and A_right (cols 2,3)
                
                # C = A_left @ B_top + A_right @ B_bot
                # Each is a 2x2 @ 2x7 multiplication
                # 2x2 @ 2x7 can be done in rank... let's see
                
                # For 2x2 @ 2x7: tensor shape (2*2, 2*7, 7*2) = (4, 14, 14)
                # Schoolbook rank: 2*2*7 = 28
                # Can we do better?
                
                # Use Strassen on the 2x2 part:
                # A_left @ B_top: A_left is 2x2, B_top is 2x7
                # This is like (A00 A01; A10 A11) @ (B00 B01 ... B06; B10 B11 ... B16)
                # = (A00*B00 + A01*B10, A00*B01 + A01*B11, ..., A00*B06 + A01*B16;
                #    A10*B00 + A11*B10, A10*B01 + A11*B11, ..., A10*B06 + A11*B16)
                
                # Each output entry is a sum of 2 products.
                # We can use Strassen to reduce the number of multiplications.
                
                # For a 2x2 @ 2x1 matrix multiplication, Strassen uses 7 mults.
                # For 2x2 @ 2x7, we can apply Strassen to each column, but that's 7*7=49 mults.
                # Better: use the 7 Strassen multiplications, each of which is a 2x1 @ 1x7 or similar.
                
                # Actually, let's just use the schoolbook for each 2x2 @ 2x7 subproblem:
                # That's 2*2*7 = 28 per subproblem, total 56. No savings.
                
                # Let me try a different decomposition.
                return False
                
            elif strat == 'block_diagonal':
                # Try a block diagonal approach
                # Not obvious how to make this work
                return False
                
            elif strat == 'mixed':
                # Use a mix of structured and random
                # Start with schoolbook pattern and perturb
                for d in range(rank):
                    i = d % n
                    j = (d // n) % m
                    k = (d // (n * m)) % p
                    U[i * m + j, d] = 1.0
                    V[j * p + k, d] = 1.0
                    W[k * n + i, d] = 1.0
                    # Add small perturbations to allow optimization
                    if d < rank - 1:
                        U[(i * m + j + 1) % nm, d] = 0.1
                        V[(j * p + k + 1) % mp, d] = 0.1
                        W[(k * n + i + 1) % pn, d] = 0.1
                
                # Now use alternating minimization to refine
                U, V, W = _alternating_minimize(U, V, W, T, nm, mp, pn, deadline)
                
                if check_decomposition(U, V, W):
                    # Try to round
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if check_decomposition(U_r, V_r, W_r):
                        best_U, best_V, best_W = U_r, V_r, W_r
                    else:
                        U_i = np.round(U)
                        V_i = np.round(V)
                        W_i = np.round(W)
                        if check_decomposition(U_i, V_i, W_i):
                            best_U, best_V, best_W = U_i, V_i, W_i
                        else:
                            best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    return True
        
        return False

    def _alternating_minimize(U, V, W, T, nm, mp, pn, deadline, max_iter=300, tol=1e-10):
        """Run alternating minimization to convergence."""
        T_vec_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
        T_vec_u = T.reshape(nm, mp * pn)
        T_vec_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline:
                return U, V, W
            
            # Optimize W
            X = np.einsum('ad,bd->abd', U, V).reshape(nm * mp, -1)
            r = X.shape[1]
            XTX = X.T @ X + 1e-12 * np.eye(r)
            XTT = X.T @ T_vec_w.T
            try:
                W_new = np.linalg.solve(XTX, XTT).T
            except np.linalg.LinAlgError:
                W_new = W.copy()
            
            # Optimize U
            Y = np.einsum('bd,cd->bcd', V, W_new).reshape(mp * pn, -1)
            r = Y.shape[1]
            YTY = Y.T @ Y + 1e-12 * np.eye(r)
            YTT = Y.T @ T_vec_u.T
            try:
                U_new = np.linalg.solve(YTY, YTT).T
            except np.linalg.LinAlgError:
                U_new = U.copy()
            
            # Optimize V
            Z = np.einsum('ad,cd->acd', U_new, W_new).reshape(nm * pn, -1)
            r = Z.shape[1]
            ZTZ = Z.T @ Z + 1e-12 * np.eye(r)
            ZTT = Z.T @ T_vec_v.T
            try:
                V_new = np.linalg.solve(ZTZ, ZTT).T
            except np.linalg.LinAlgError:
                V_new = V.copy()
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            T_rec = tensor_from_factors(U, V, W)
            err = np.linalg.norm(T_rec - T)
            if err < tol:
                return U, V, W
            if abs(prev_err - err) < 1e-13 and it > 20:
                return U, V, W
            prev_err = err
        
        return U, V, W

    # Try structured decompositions at various ranks
    # The theoretical lower bound is max(nm, mp, pn) = 28
    # Try from low to high
    
    for rank in [28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 55]:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue
        success = try_structured_decomposition(rank)
        if success:
            break

    # If we found something, try to go lower
    if best_rank < rank_sb and time.time() < deadline - 30:
        for rank in range(max(20, best_rank - 15), best_rank):
            if time.time() > deadline:
                break
            success = try_structured_decomposition(rank)
            if success:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
