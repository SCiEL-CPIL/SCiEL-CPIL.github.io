import numpy as np
import random

DIMENSIONS = [3, 3, 8]


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    N, M, P = n * m, m * p, p * n

    # Build the matrix multiplication tensor T[i,j,k] = 1
    T = np.ones((n, m, p))

    def matmul_matrix(rank, U, V, W):
        """Compute matmul_matrix using factors, return shape (N*M, P*N)"""
        # U: (N, rank), V: (M, rank), W: (P, rank)
        # matmul_matrix[r, (i,k)] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Actually we need to compute the full tensor contraction
        # For each (i,k), value = sum over (j,r) U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is equivalent to: for each i,k: sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # = sum_r W[k*n+i,r] * sum_j U[i*m+j,r]*V[j*p+k,r]
        # Let's compute via: for each r, outer product structure
        # Actually let's just do it directly
        result = np.zeros((M, N))  # result[j, (i,k)]? No...
        # Let me think differently.
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # The matmul matrix maps input (A⊗B) to output C
        # Input index: (j, i, k) -> A[i,j]*B[j,k], total M*N dimensions? 
        # Actually input is A (n*m) ⊗ B (m*p) = N*M dimensions
        # Output is C (p*n) = P dimensions
        # matmul_matrix: (P, N*M) where matmul_matrix[(k*n+i), (j, i, k)]... 
        # Hmm, let me use the standard formulation.
        # The matmul tensor T has indices (i, j, k) with T[i,j,k]=1
        # The matmul matrix is the "mode-1 unfolding" or similar
        # Let's just verify: does sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == T[i,j,k]?
        # We'll compute the reconstructed tensor and compare
        
        # Reconstruct tensor: T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    T_recon[i, j, k] = val
        return T_recon

    def error(rank, U, V, W):
        T_recon = matmul_matrix(rank, U, V, W)
        return np.sum((T_recon - T) ** 2)

    best_rank = None
    best_factors = None

    # Try various ranks with ALS
    for target_rank in range(20, 40):
        found = False
        for attempt in range(20):
            random.seed(attempt)
            np.random.seed(attempt)
            
            # Random initialization
            U = np.random.randn(N, target_rank) * 0.1
            V = np.random.randn(M, target_rank) * 0.1
            W = np.random.randn(P, target_rank) * 0.1
            
            for it in range(2000):
                # Update U
                for r in range(target_rank):
                    # U[:,r] should minimize ||T - sum_r' U[:,r']*V[:,r']*W[:,r']||^2
                    # This is a linear least squares: for each (i,j), T[i,j,k] = U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] + rest
                    # Fix all except U[:,r]: T[i,j,k] - sum_{r'!=r} U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r'] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                    # For each (i,j): let a_ij = V[j, r] (depends on j) ... this is getting complex
                    # Use gradient-based ALS:
                    # d/dU[i*m+j, r] of error = 2 * sum_k (T_recon[i,j,k] - T[i,j,k]) * V[j*p+k, r] * W[k*n+i, r]
                    # But T_recon depends on U too. Use one-step approximation.
                    pass
                
                # Simpler: use coordinate descent on the tensor
                # For each r, solve for U[:,r], V[:,r], W[:,r] given others fixed
                # This is the standard ALS for tensor decomposition
        
                # Let's use a simpler approach: gradient descent with random restarts
                break
        
        # Alternative: use the known structure
        # Since T[i,j,k] = 1 for all valid indices, we can try specific structured decompositions
    
    # Known good decomposition: rank 31
    # Construction:
    # Split into blocks based on middle index j and the (i,k) pattern
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term with sparse dicts mapping index->value"""
        U_terms.append(u_dict)
        V_terms.append(v_dict)
        W_terms.append(w_dict)
    
    # Strategy for rank 31:
    # For j=0,1: use Strassen on 2x2 (i=0,1) x 2x8 (k=0..7)
    # That's 4 blocks of 2x2x2 Strassen = 28 terms
    # For j=0,1 and i=2: A[2,j]*B[j,k] for all k -> but we can do this with 2 terms using the structure
    # Actually for i=2, j in {0,1}, k in 0..7: T[2,j,k]=1
    # We need: sum_r U[2*m+j, r]*V[j*p+k, r]*W[k*n+2, r] = 1
    # If we set W[k*n+2, r] = 1 for all k and one r, and U[2*m+j, r]=1, V[j*p+k, r]=1... 
    # But V[j*p+k, r] depends on k, so we can't set it to constant unless we have separate terms per k.
    # Alternatively: for fixed j, the terms T[2,j,k]=1 for all k can be handled by:
    # U[2*m+j, r] = 1, V[j*p+k, r] = 1 for all k, W[k*n+2, r] = 1 for all k
    # But that gives product = 1*1*1 = 1. Yes! One term per j (for i=2).
    # Similarly for j=2, i in {0,1}: one term per i.
    # And j=2, i=2: one term.
    
    # Wait, but V[j*p+k, r] = 1 for all k means V[j*p+0, r]=V[j*p+1, r]=...=V[j*p+7, r]=1
    # And W[k*n+2, r] = 1 for all k means W[0*n+2, r]=W[1*n+2, r]=...=W[7*n+2, r]=1
    # This is valid! Each such "block" of T[i,j,:]=1 or T[:,j,k] etc. can be one rank-1 term.
    
    # Let's verify: term r has U[2*m+0, r]=1, V[0*p+k, r]=1 for k=0..7, W[k*n+2, r]=1 for k=0..7
    # Then for T[2,0,k]: contribution = U[2*3+0, r]*V[0*8+k, r]*W[k*3+2, r] = 1*1*1 = 1. Correct!
    # For other (i',j',k'): if i'!=2 or j'!=0, then either U or V or W is 0, so no contribution. Correct!
    
    # So the decomposition:
    # 28 Strassen terms for i in {0,1}, j in {0,1}, k in {0..7} (4 blocks of 2x2x2)
    # 1 term for i=2, j=0 (covers all k)
    # 1 term for i=2, j=1 (covers all k)  
    # 1 term for i=0, j=2 (covers all k)
    # 1 term for i=1, j=2 (covers all k)
    # 1 term for i=2, j=2 (covers all k)
    # Total: 28 + 5 = 33? Let me recount.
    
    # Actually for the Strassen blocks, each 2x2x2 block handles i in {0,1}, j in {0,1}, k in {block}
    # Block 1: k in {0,1}, Block 2: k in {2,3}, Block 3: k in {4,5}, Block 4: k in {6,7}
    # Each block is 2x2x2, rank 7 via Strassen. Total 28.
    # These cover T[i,j,k] for i in {0,1}, j in {0,1}, all k.
    
    # Remaining: T[i,j,k]=1 where i=2 or j=2 (or both)
    # i=2, j=0: T[2,0,k]=1 for all k -> 1 term
    # i=2, j=1: T[2,1,k]=1 for all k -> 1 term
    # i=0, j=2: T[0,2,k]=1 for all k -> 1 term
    # i=1, j=2: T[1,2,k]=1 for all k -> 1 term
    # i=2, j=2: T[2,2,k]=1 for all k -> 1 term
    # Total remaining: 5 terms
    # Grand total: 28 + 5 = 33
    
    # But can we do better? The 5 "boundary" terms might be reducible.
    # T[2,0,k]=1, T[2,1,k]=1, T[0,2,k]=1, T[1,2,k]=1, T[2,2,k]=1
    # These are all "full lines" in k. Each is rank 1 as shown.
    # Can any two be combined? They have different (i,j) patterns so they're independent.
    # So 33 seems to be the floor with this approach.
    
    # Let's try to see if we can reduce further by mixing.
    # Actually, let's just implement the 33-rank decomposition and see if we can optimize.
    
    # Build Strassen terms for the 2x2x2 blocks
    def add_strassen_block(i0, j0, k0):
        """Add 7 Strassen terms for T[i0:i0+2, j0:j0+2, k0:k0+2] = 1"""
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        # m1 = (a+d)(e+h): U has (i0,j0) and (i1,j1), V has (j0,k0) and (j1,k1), W has (k0,i0) and (k1,i1)
        add_term(
            {i0*m+j0: 1, i1*m+j1: 1},
            {j0*p+k0: 1, j1*p+k1: 1},
            {k0*n+i0: 1, k1*n+i1: 1}
        )
        # m2 = (c+d)e: U has (i1,j0) and (i1,j1), V has (j0,k0), W has (k0,i1) and (k1,i1) with -1
        add_term(
            {i1*m+j0: 1, i1*m+j1: 1},
            {j0*p+k0: 1},
            {k0*n+i1: 1, k1*n+i1: -1}
        )
        # m3 = a(f-h): U has (i0,j0), V has (j0,k1) and (j1,k1) with -1, W has (k1,i0) and (k1,i1)
        add_term(
            {i0*m+j0: 1},
            {j0*p+k1: 1, j1*p+k1: -1},
            {k1*n+i0: 1, k1*n+i1: 1}
        )
        # m4 = d(g-e): U has (i1,j1), V has (j1,k0) and (j0,k0) with -1, W has (k0,i0) and (k0,i1)
        add_term(
            {i1*m+j1: 1},
            {j1*p+k0: 1, j0*p+k0: -1},
            {k0*n+i0: 1, k0*n+i1: 1}
        )
        # m5 = (a+b)h: U has (i0,j0) and (i0,j1), V has (j1,k1), W has (k0,i0) with -1 and (k1,i0)
        add_term(
            {i0*m+j0: 1, i0*m+j1: 1},
            {j1*p+k1: 1},
            {k0*n+i0: -1, k1*n+i0: 1}
        )
        # m6 = (c-a)(e+f): U has (i1,j0) and (i0,j0) with -1, V has (j0,k0) and (j0,k1), W has (k1,i1)
        add_term(
            {i1*m+j0: 1, i0*m+j0: -1},
            {j0*p+k0: 1, j0*p+k1: 1},
            {k1*n+i1: 1}
        )
        # m7 = (b-d)(g+h): U has (i0,j1) and (i1,j1) with -1, V has (j1,k0) and (j1,k1), W has (k0,i0)
        add_term(
            {i0*m+j1: 1, i1*m+j1: -1},
            {j1*p+k0: 1, j1*p+k1: 1},
            {k0*n+i0: 1}
        )
    
    # Add 4 Strassen blocks for k=0-1, 2-3, 4-5, 6-7
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 0, 6)
    
    # Add 5 boundary terms
    # i=2, j=0
    w_dict = {}
    for k in range(p):
        w_dict[k*n+2] = 1
    v_dict = {}
    for k in range(p):
        v_dict[0*p+k] = 1
    add_term({2*m+0: 1}, v_dict, w_dict)
    
    # i=2, j=1
    v_dict = {}
    for k in range(p):
        v_dict[1*p+k] = 1
    add_term({2*m+1: 1}, v_dict, w_dict.copy())
    
    # i=0, j=2
    w_dict = {}
    for k in range(p):
        w_dict[k*n+0] = 1
    v_dict = {}
    for k in range(p):
        v_dict[2*p+k] = 1
    add_term({0*m+2: 1}, v_dict, w_dict)
    
    # i=1, j=2
    w_dict = {}
    for k in range(p):
        w_dict[k*n+1] = 1
    add_term({1*m+2: 1}, v_dict.copy(), w_dict)
    
    # i=2, j=2
    w_dict = {}
    for k in range(p):
        w_dict[k*n+2] = 1
    v_dict = {}
    for k in range(p):
        v_dict[2*p+k] = 1
    add_term({2*m+2: 1}, v_dict, w_dict)
    
    rank = len(W_terms)
    
    # Build matrices
    U = np.zeros((N, rank))
    V = np.zeros((M, rank))
    W = np.zeros((P, rank))
    
    for r in range(rank):
        for idx, val in U_terms[r].items():
            U[idx, r] = val
        for idx, val in V_terms[r].items():
            V[idx, r] = val
        for idx, val in W_terms[r].items():
            W[idx, r] = val
    
    # Verify
    T_recon = matmul_matrix(rank, U, V, W)
    err = np.sum((T_recon - T) ** 2)
    
    if err > 1e-6:
        # Fallback to naive rank 72
        rank = 72
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1
        T_recon = matmul_matrix(rank, U, V, W)
        err = np.sum((T_recon - T) ** 2)
    
    # Clean floats
    def clean(val):
        if abs(val - round(val)) < 1e-9:
            return int(round(val))
        return val
    
    U_list = [[clean(x) for x in row] for row in U.tolist()]
    V_list = [[clean(x) for x in row] for row in V.tolist()]
    W_list = [[clean(x) for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
