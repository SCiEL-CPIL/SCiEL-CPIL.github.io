import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 3, 3, 5
    
    # Target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
                
    def validate(U, V, W):
        """Check if U, V, W exactly decompose T"""
        try:
            R = np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.allclose(R, T, atol=1e-9):
                return True
        except Exception:
            pass
        return False

    def schoolbook_decomposition():
        """Return the trivial rank 45 decomposition"""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, column] = 1
                    V[j*p + k, column] = 1
                    W[k*n + i, column] = 1
                    column += 1
                    
        return U, V, W

    def try_rank_41():
        """
        Attempt to construct a rank 41 decomposition.
        Uses a rank 23 decomposition for 3x3x3 (first 3 columns)
        and schoolbook 18 terms for 3x3x2 (last 2 columns).
        """
        if time.time() > deadline - 15:
            return None
        
        # Known rank-23 decomposition for 3x3x3 matrix multiplication.
        # This uses the result that r(3,3,3) = 23.
        # We'll use a concrete construction based on the known 23-term formula.
        # The 23 terms are given as triples (u, v, w) of 9-vectors.
        
        # I'll use a verified 23-term decomposition.
        # Reference: The 23-term decomposition exists and can be constructed.
        # For practical purposes, let's use a known explicit set.
        
        # Actually, let me use a simpler approach: 
        # Use Strassen's 7 terms for each of 3 overlapping 2x2x2 blocks,
        # but with careful deduplication. The key insight is that the 3x3x3
        # tensor can be decomposed in 23 terms.
        
        # Let me use a different strategy: use the known fact that
        # r(3,3,3) <= 23 and construct it via the "border rank" approach
        # or use a precomputed set of 23 terms.
        
        # For reliability, let me try a construction that I can verify:
        # Use 3 copies of Strassen (7 each = 21) for the 3 off-diagonal 2x2 blocks
        # plus 2 more terms to handle the remaining structure.
        
        # Actually, the cleanest approach: use the known 23-term decomposition
        # from the literature. Here's a concrete set:
        
        terms_333 = []
        
        # Helper to create a zero vector of length 9
        def zero9():
            return [0.0]*9
        
        # The 23 terms for 3x3x3.
        # I'll use the construction from Bini et al. / Landsberg.
        # For now, let me use a simpler verified approach:
        # 
        # Strategy: Decompose 3x3x3 as follows:
        # - 3 diagonal terms (i=i'=k, j free) -> actually these are part of the structure
        # - Use 20 off-diagonal terms from 3 Strassen-like decompositions
        #
        # But to be safe, let me just use 24 terms (3+3*7) and see if it validates.
        # If it does, rank is 24+18=42, which is still better than 45.
        
        for j in range(3):
            idx = [i for i in range(3) if i != j]
            i1, i2 = idx
            
            # 7 Strassen-like terms for the 2x2x2 subproblem on indices {i1,i2} x {j} x {i1,i2}
            # Wait, the subproblem is A[i1:i2, j] * B[j, i1:i2] -> C[i1:i2, i1:i2]
            # This is a 2x1 * 1x2 = 2x2 problem, which has rank 4 (schoolbook) or can be done in 4.
            # Actually for 2x1*1x2, the tensor is sum_{i in {i1,i2}} e_i ⊗ e_j ⊗ e_i
            # This has rank 2 (just 2 terms).
            
            # So for each j, we need 2 terms for the "column" of A times "row" of B.
            # That's 3*2 = 6 terms for the off-diagonal parts.
            
            # Plus the 3 diagonal terms where i=k and j is arbitrary... 
            # No, the full 3x3x3 tensor is sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
            
            # Let me just do schoolbook for 3x3x3: 27 terms.
            # Then 27 + 18 = 45. That's the same as before.
            
            pass
        
        # OK, let me take a different, more reliable approach.
        # I'll use the schoolbook for the first 3 columns (27 terms) and
        # try to find a better decomposition for the last 2 columns.
        # 
        # Actually, the simplest improvement: use schoolbook for all 5 columns = 45.
        # To beat this, I need a non-trivial decomposition.
        #
        # Let me try: use the 3x3x3 schoolbook (27) + 3x3x2 schoolbook (18) = 45.
        # No improvement.
        #
        # Alternative: Can I share terms between the two parts?
        # The tensor for 3x3x5 is:
        # T = sum_{i,j,k=0}^{4} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
        # = sum_{i,j,k=0}^{2} ... + sum_{i,j} sum_{k=3}^{4} ...
        #
        # The first part is the 3x3x3 tensor (27 terms in schoolbook).
        # The second part is 3x3x2 (18 terms in schoolbook).
        #
        # For the 3x3x2 part, the tensor is:
        # sum_{i,j} sum_{k=3}^{4} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
        # = sum_{i,j} [e_{i*m+j} ⊗ e_{j*p+3} ⊗ e_{3*n+i} + e_{i*m+j} ⊗ e_{j*p+4} ⊗ e_{4*n+i}]
        #
        # This is 18 rank-1 tensors. Can we do better?
        # The 3x3x2 tensor has rank at most 18, and likely exactly 18
        # since it's a "generic" enough tensor.
        #
        # For the 3x3x3 part, rank is 23 (known result).
        # So total rank could be 23 + 18 = 41.
        #
        # Let me use a precomputed 23-term decomposition for 3x3x3.
        # I'll use the one from the paper "The rank of the matrix multiplication tensor"
        # or construct it using known methods.
        
        # For now, let me just return None and use schoolbook.
        # The key issue is that I don't have a verified 23-term decomposition
        # that I can hardcode reliably.
        
        return None

    best_result = None
    
    # Try rank 41/42 construction
    result = try_rank_41()
    if result is not None:
        best_result = result
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook_decomposition()
        best_result = (U, V, W)
        
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
