import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 110.0  # Safety margin under 120s

    n, m, p = 2, 2, 10

    # Build rank-35 Strassen block-diagonal decomposition
    # Strassen terms: (a_vec, b_vec, c_vec) each of length 4
    # a_vec: coefficients on A = [A00, A01, A10, A11] (row-major, index i*m+j)
    # b_vec: coefficients on B block = [B00, B01, B10, B11]
    # c_vec: coefficients on C block = [C00, C10, C01, C11] (W[k*n+i] convention)
    #
    # Standard Strassen (c_vec order: C00, C10, C01, C11):
    # M1: a=[1,0,0,1] b=[1,0,0,1] c=[1,0,0,1]
    # M2: a=[0,0,1,1] b=[1,0,0,0] c=[0,1,0,-1]
    # M3: a=[1,0,0,0] b=[0,1,0,-1] c=[0,0,1,1]
    # M4: a=[0,0,0,1] b=[-1,0,1,0] c=[1,1,0,0]
    # M5: a=[1,1,0,0] b=[0,0,0,1] c=[-1,0,1,0]
    # M6: a=[-1,0,1,0] b=[1,1,0,0] c=[0,0,0,1]
    # M7: a=[0,1,0,-1] b=[0,0,1,1] c=[1,0,0,0]

    strassen = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,0,1,1]),
        ([0,0,0,1], [-1,0,1,0], [1,1,0,0]),
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]

    def build_rank35():
        U = np.zeros((n*m, 35))
        V = np.zeros((m*p, 35))
        W = np.zeros((p*n, 35))
        col = 0
        for k in range(5):
            for av, bv, cv in strassen:
                for i in range(4):
                    U[i, col] = av[i]
                V[2*k, col] = bv[0]
                V[2*k+1, col] = bv[1]
                V[10+2*k, col] = bv[2]
                V[10+2*k+1, col] = bv[3]
                W[4*k, col] = cv[0]
                W[4*k+1, col] = cv[1]
                W[4*k+2, col] = cv[2]
                W[4*k+3, col] = cv[3]
                col += 1
        return U, V, W

    def verify(U, V, W):
        """Check that the decomposition is exact: for all entries (i,k) of C,
        sum_r U[i,r]*W[r,k] == sum_j A[i,j]*B[j,k] for all j.
        Actually verify the tensor identity:
        For each C[i,k] (row i of C, col k), the bilinear form
        sum_r U[i*m*? ...]
        
        The tensor T has indices (i, j, k) where i indexes rows of A (0..n-1),
        j indexes the m intermediate dim (0..m-1) AND columns of B, k indexes cols of B/C.
        
        Wait, let me re-derive. C = A @ B where A is n x m, B is m x p.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        The bilinear decomposition: C[i,k] = sum_r U[i,r] * prod_r * V[k,r]
        where U has shape (n, rank)? No...
        
        Actually the task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
        
        Hmm, that's unusual. Let me re-read.
        
        "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        
        So U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank.
        
        The tensor T for matrix mult has indices (i, j, k) with i in [0,n), j in [0,m), k in [0,p).
        T[i,j,k] = delta_{i,k} * delta_{j,j}? No...
        
        C[i,k] = sum_j A[i,j] B[j,k]
        
        The tensor is T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]?
        That doesn't work dimensionally.
        
        Actually, the standard bilinear decomposition: the multiplication tensor
        tau: (A space) x (B space) -> (C space)
        has entries tau_{i,j,k} where the bilinear form is:
        C = sum_{i,j,k} tau_{i,j,k} A_{i,j} B_{j,k} ... no that's not right either.
        
        Let me think again. A is n x m = (n*m) entries. B is m x p = (m*p) entries.
        C is n x p... wait, but the task says W has shape (p*n, rank).
        
        Hmm, p*n = 10*2 = 20 = n*p. So W is indexed over (k, i) pairs? With W[k*n + i].
        
        So the output space C is indexed by (k, i) where k is column (0..p-1) and i is row (0..n-1)?
        But C[i,k] = (A@B)[i,k], so row i, col k. The index in W is k*n + i, meaning it's column-major:
        for fixed k (column), i varies (rows). So W index = k*n + i corresponds to C[i, k].
        
        So: U indices: (i, j) for A entries, flattened as i*m + j (n*m entries)
              V indices: (j, k) for B entries, flattened as j*p + k (m*p entries)  
              W indices: (k, i) for C entries, flattened as k*n + i (p*n entries)
        
        And the tensor identity is:
        C[i,k] = sum_{r} U[i*m+j?] ... 
        
        Wait, that's not how bilinear decompositions work. Let me reconsider.
        
        In a rank-r bilinear decomposition of matrix multiplication:
        C[i,k] = sum_{r=1}^{r} prod_r * U_{i,?,r} * V_{?,k,r}
        
        No, the standard form is:
        We have r products: M_r = sum_i A_{i, sigma_r(i)} * sum_j B_{tau_r(j), j}
        Then C[i,k] = sum_r coeff_{r,i,k} * M_r
        
        So: M_r = (sum over A-indices with coeffs) * (sum over B-indices with coeffs)
        
        In the task's notation:
        U has shape (n*m, rank): U[a_idx, r] = coefficient of A entry a_idx in product r
        V has shape (m*p, rank): V[b_idx, r] = coefficient of B entry b_idx in product r
        W has shape (p*n, rank): W[c_idx, r] = coefficient for product r in output entry c_idx
        
        And: C[i,k] = sum_r W[k*n+i, r] * (sum_a U[a,r] * A[flat(a)]) * (sum_b V[b,r] * B[flat(b)])
        
        Wait, that gives a product of two scalars times W, which is trilinear, not bilinear in the right way.
        
        Actually no. The bilinear map is: (A, B) -> C where C is bilinear in A and B.
        A rank-r decomposition: C[i,k] = sum_r (sum_a U[a,r] A[a]) * (sum_b V[b,r] B[b]) * W[c,r]
        
        That's: C = sum_r W_c,r * (U_a,r * A_a) * (V_b,r * B_b)
        
        For this to equal sum_j A[i,j] B[j,k], we need:
        For each (i,k), sum_r W[k*n+i, r] * (sum_a U[a,r] A[a]) * (sum_b V[b,r] B[b]) = sum_j A[i,j] B[j,k]
        
        The left side is sum_r W_c,r * (U A)_r * (V B)_r where (U A)_r = sum_a U[a,r] A[a].
        
        This is a sum over r of products, which is bilinear in A and B (since each term is product of a linear form in A and a linear form in B).
        
        So the verification is: for random A, B, compute C_direct = A @ B, and
        C_decomp where C_decomp[i,k] = sum_r W[k*n+i, r] * dot(U[:,r], A_flat) * dot(V[:,r], B_flat).
        
        They should be equal."""
        # A_flat: shape (n*m,), B_flat: shape (m*p,)
        # C_decomp[i,k] = sum_r W[k*n+i, r] * (U[:,r] @ A_flat) * (V[:,r] @ B_flat)
        # = sum_r W[k*n+i, r] * (U.T @ A_flat)[r] * (V.T @ B_flat)[r]
        
        n, m, p = 2, 2, 10
        # Generate random test matrices
        rng = np.random.RandomState(42)
        for _ in range(3):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_direct = A @ B  # shape (n, p)
            
            A_flat = A.flatten()  # (n*m,)
            B_flat = B.flatten()  # (m*p,)
            
            UA = U.T @ A_flat  # (rank,)
            VB = V.T @ B_flat  # (rank,)
            
            # C_decomp[i,k] = sum_r W[k*n+i, r] * UA[r] * VB[r]
            # W shape (p*n, rank), index k*n+i
            WV = W * np.outer(np.ones(p*n), UA * VB)  # (p*n, rank) * (rank,) broadcast
            # Actually: WV[c_idx, r] = W[c_idx, r] * UA[r] * VB[r]
            WV = W * (UA * VB)[np.newaxis, :]  # (p*n, rank)
            C_decomp_flat = WV.sum(axis=1)  # (p*n,)
            
            # Compare: C_decomp_flat[k*n + i] should equal C_direct[i, k]
            C_decomp = np.zeros((n, p))
            for k in range(p):
                for i in range(n):
                    C_decomp[i, k] = C_decomp_flat[k*n + i]
            
            if not np.allclose(C_direct, C_decomp, atol=1e-10):
                return False
        return True

    def to_list(M):
        return M.tolist()

    # Start with rank-35
    U35, V35, W35 = build_rank35()
    
    # Try to find rank-34 or rank-33 via random search
    # Strategy: the rank-35 has a block-diagonal structure.
    # Each block of 7 terms operates on 4 A-entries, 4 B-entries, 4 C-entries.
    # To reduce rank, we'd need to find linear dependencies across blocks.
    # But since different blocks use disjoint B columns and C columns, 
    # direct linear dependencies are unlikely.
    
    # Alternative: try random perturbations of the Strassen structure
    # and check if a lower-rank exact decomposition exists.
    
    # For now, attempt a quick randomized search for rank 34
    # using a different approach: try to express one Strassen term
    # as a combination that shares computation with another block.
    
    # Given the block-diagonal nature, rank 35 is likely optimal.
    # Return rank 35.
    
    best_rank = 35
    best_U, best_V, best_W = U35, V35, W35
    
    if verify(best_U, best_V, best_W):
        # Confirmed correct
        pass
    else:
        # Should not happen, but fallback
        raise RuntimeError("Rank-35 verification failed")
    
    elapsed = time.time() - start
    # If we have time, try a brief search for rank 34
    if time.time() < deadline - 30:
        # Try: can we combine two Strassen terms from different blocks
        # by finding a shared linear combination?
        # This is unlikely to work due to disjoint support, but try briefly.
        
        # Actually, let's try a different structural approach:
        # Treat B as 2x5 matrix of 2-vectors and use a 2x2 matrix mult
        # over a ring of dimension 5? This is the "ring-based" approach.
        # But implementing this correctly in the required format is complex
        # and unlikely to reduce rank below 35 for these small dimensions.
        
        # Try randomized rank-34 search: perturb U, V, W and check
        # if the bilinear identity still holds.
        # This is a very constrained search space, unlikely to succeed.
        
        pass  # Skip: the structural argument strongly suggests 35 is optimal
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
