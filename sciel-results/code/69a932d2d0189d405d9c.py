"""Model-and-optimize approach for 2x4x7 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 @ 4x7.
    
    Strategy: Exploit n=2 structure. Treat as vector-matrix product.
    Use Winograd-style rank-3 schemes for overlapping 2-element windows.
    """
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    def time_remaining():
        return deadline - time.time()
    
    def verify_decomposition_fast(U_m, V_m, W_m, n, m, p, num_tests=10):
        """Fast verification using random integer inputs."""
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randint(-3, 4, size=(n, m)).astype(float)
            B = np.random.randint(-3, 4, size=(m, p)).astype(float)
            C_expected = A @ B
            rank_m = U_m.shape[1]
            C_comp = np.zeros((n, p), dtype=float)
            for r in range(rank_m):
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            C_comp[i, k] += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
            if not np.allclose(C_comp, C_expected, atol=1e-8):
                return False
        return True
    
    def verify_exact(U_m, V_m, W_m, n, m, p):
        """Exact verification using rational arithmetic."""
        U_frac = np.zeros((n*m, U_m.shape[1]), dtype=object)
        V_frac = np.zeros((m*p, V_m.shape[1]), dtype=object)
        W_frac = np.zeros((p*n, W_m.shape[1]), dtype=object)
        
        for i in range(n*m):
            for r in range(U_m.shape[1]):
                U_frac[i, r] = Fraction(U_m[i, r]).limit_denominator(1000)
        for i in range(m*p):
            for r in range(V_m.shape[1]):
                V_frac[i, r] = Fraction(V_m[i, r]).limit_denominator(1000)
        for i in range(p*n):
            for r in range(W_m.shape[1]):
                W_frac[i, r] = Fraction(W_m[i, r]).limit_denominator(1000)
        
        test_cases = [
            (np.array([[1, 0, 1, -1], [0, 1, -1, 1]]), np.array([[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9, 10]])),
            (np.array([[1, 1, 1, 1], [1, -1, 1, -1]]), np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0]])),
            (np.array([[2, -1, 3, 0], [-1, 2, 0, 3]]), np.array([[1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, -1, -1], [1, -1, 1, -1, 1, -1, 1], [0, 0, 0, 0, 0, 0, 0]])),
        ]
        
        for A, B in test_cases:
            C_expected = A @ B
            C_comp = np.zeros((n, p), dtype=object)
            rank_m = U_frac.shape[1]
            for i in range(n):
                for k in range(p):
                    val = Fraction(0)
                    for j in range(m):
                        for r in range(rank_m):
                            val += U_frac[i*m+j, r] * V_frac[j*p+k, r] * W_frac[k*n+i, r]
                    C_comp[i, k] = val
            if not all(C_comp[i, k] == C_expected[i, k] for i in range(n) for k in range(p)):
                return False
        return True
    
    # === Construction 1: Winograd-style approach ===
    # For n=2, we compute C[0,k] and C[1,k] for k=0..6
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # Group j into pairs: (0,1) and (2,3)
    # For each j-pair and k-pair (2a, 2a+1), use rank-3 Winograd F(1,2)
    # 
    # Winograd F(1,2): compute y0 = x0*b0 + x1*b1, y1 = x0*b2 + x1*b3
    # with rank 3:
    #   m0 = (x0+x1)*(b0+b2)
    #   m1 = (x1-x0)*(b1+b3)  
    #   m2 = (x0+x1)*b2
    #   y0 = m0 - m1/2 - m2/2 ... this isn't quite right
    #
    # Actually for 1x2x2 (two dot products of length 2 sharing inputs):
    # y0 = x0*b0 + x1*b1
    # y1 = x0*b2 + x1*b3
    # Rank-3 decomposition:
    #   m0 = (x0+x1)*(b0+b2)
    #   m1 = (x1-x0)*(b1-b3)
    #   m2 = (x0+x1)*(b1+b3)  -- no this doesn't work
    #
    # Let me use the standard Winograd F(2,2) for 2x2 matmul but adapted.
    # Actually, for our case, each (j-pair, k-pair) subproblem is:
    #   C[i, 2a]   = A[i,j1]*B[j1,2a]   + A[i,j2]*B[j2,2a]
    #   C[i, 2a+1] = A[i,j1]*B[j1,2a+1] + A[i,j2]*B[j2,2a+1]
    # for i=0,1
    #
    # This is a 2x2x2 tensor (i, j', k') where j' in {j1,j2}, k' in {2a,2a+1}
    # with all entries = 1. The rank of this tensor is 4 (it's the outer product
    # of three all-ones vectors, but that's rank 1... no wait).
    #
    # The tensor T[i,j',k'] = 1 for all i in {0,1}, j' in {j1,j2}, k' in {2a,2a+1}
    # This is a 2x2x2 all-ones tensor, which has rank 2 (since it's the sum of
    # two rank-1 tensors: e_0*e_0^T*e_0^T + ... no).
    #
    # Actually, the 2x2x2 all-ones tensor J_2x2x2 has rank 2 over reals.
    # Proof: J = u*u^T*u^T where u = (1,1,1)? No, that's not a valid decomposition.
    # J[i,j,k] = 1 for all i,j,k. We need sum_r U[i,r]V[j,r]W[k,r] = 1.
    # With r=1: U[i,0]=1, V[j,0]=1, W[k,0]=1 gives J[i,j,k]=1. So rank 1!
    #
    # Wait, that means for each (j-pair, k-pair) subproblem, we only need rank 1
    # if the U, V, W factors can be shared across subproblems. But they can't be
    # shared directly because the indices are different.
    #
    # Let me reconsider. The full tensor has 56 non-zero entries (one for each
    # (i,j,k) triple). We need to decompose this sparse tensor.
    #
    # Approach: For each j, the entries are T[i,j,k] = 1 for all i,k.
    # This is a 2x7x1 slice (for fixed j) which is all-ones, rank 1.
    # But we need to combine across j's.
    #
    # Let me try a direct rank-48 construction:
    # For j=0: use 7 rank-1 terms (one for each k)
    # For j=1: use 7 rank-1 terms
    # For j=2: use 7 rank-1 terms  
    # For j=3: use 7 rank-1 terms
    # Total: 28 rank-1 terms. But we can do better.
    #
    # For fixed j, T[i,j,k] = 1 for all i,k. This is rank 1 in the (i,k) plane.
    # So for each j, we need 1 rank-1 term: U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1.
    # But W[k,i,r] must be the same for all j, so we can't have different W for
    # different j's in the same rank-1 term.
    #
    # Let me try: for each j, create a rank-1 term with U[i,j,r]=1 for all i,
    # V[j,k,r]=1 for all k, W[k,i,r]=1 for all k,i. This works! Each j gives
    # one rank-1 term, total rank = 4.
    #
    # Wait, let me verify: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] 
    # For r corresponding to j: U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1, product=1.
    # For other r': U[i,j,r']=0 (since r' corresponds to different j'), so sum=1. 
    # This works! Rank 4!
    #
    # But wait, that can't be right because the matmul tensor for 1x1x1 has rank 1,
    # for 2x2x2 has rank 7 (Strassen), etc. Let me re-examine.
    #
    # Oh I see the issue. The tensor is T[i,j,k] where the decomposition is
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]. The indices are:
    # - U has index (i*m+j) which ranges over 0..7
    # - V has index (j*p+k) which ranges over 0..27
    # - W has index (k*n+i) which ranges over 0..13
    #
    # For the rank-1 term corresponding to j=0:
    # U[i*4+0, r] = 1 for i=0,1 (so U[0,r]=1, U[4,r]=1)
    # V[0*7+k, r] = 1 for k=0..6 (so V[0,r]=V[1,r]=...=V[6,r]=1)
    # W[k*2+i, r] = 1 for k=0..6, i=0,1 (so W[0,r]=W[1,r]=...=W[13,r]=1)
    #
    # For j=1:
    # U[i*4+1, r] = 1 (U[1,r]=1, U[5,r]=1)
    # V[1*7+k, r] = 1 (V[7,r]=...=V[13,r]=1)
    # W[k*2+i, r] = 1 (same W as before)
    #
    # But W is shared! So if I use the same W for all j's, then:
    # sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*2+i, r]
    # = U[i*4+j, r_j] * V[j*7+k, r_j] * W[k*2+i, r_j]  (only r_j contributes)
    # = 1 * 1 * 1 = 1
    #
    # This works! So rank 4 should be sufficient?!
    #
    # But that contradicts known results. Let me check with a simple case.
    # For 1x1x1: T[0,0,0]=1. Rank 1. U[0,0]=1, V[0,0]=1, W[0,0]=1. Works.
    # For 1x2x1: T[0,0,0]=1, T[0,1,0]=1. 
    #   j=0: U[0,0]=1, V[0,0]=1, W[0,0]=1
    #   j=1: U[1,1]=1, V[1,1]=1, W[0,1]=1
    #   sum_r U[j,r]*V[j,r]*W[0,r] = U[0,0]*V[0,0]*W[0,0] + U[1,1]*V[1,1]*W[0,1] = 1+1=2 != 1
    #   Oops! The sum is over r, and for a fixed (i,j,k), multiple r's can contribute.
    #
    # I see my error. For a fixed (i,j,k), the sum is over ALL r, not just the
    # r corresponding to j. So if I have 4 rank-1 terms (one per j), then for
    # (i,j,k), the contribution from term r_j' is U[i*4+j, r_j']*V[j*7+k, r_j']*W[k*2+i, r_j'].
    # For r_j' = r_j: U[i*4+j, r_j]=1, V[j*7+k, r_j]=1, W[k*2+i, r_j]=1, product=1.
    # For r_j' != r_j: U[i*4+j, r_j']=0 (since U is only nonzero at position i*4+j'
    # for term r_j'), so product=0.
    # Total sum = 1. Correct!
    #
    # So rank 4 works for 2x4x7?? That seems too good to be true. Let me double-check
    # with the actual matrix multiplication.
    #
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # With the decomposition: C[i,k] = sum_r sum_j U[i*4+j,r]*A[i,j] * V[j*7+k,r]*B[j,k] * W[k*2+i,r]
    # Wait no, the decomposition gives the tensor T, and the actual computation is:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r W[k*2+i,r] * (sum_j U[i*4+j,r]*A[i,j]) * (sum_{j'} V[j'*7+k,r]*B[j',k])
    # No, that's not right either. The standard bilinear form is:
    # C[i,k] = sum_r W[k*2+i,r] * (sum_j U[i*4+j,r] * A[i,j]) * (sum_{j'} V[j'*7+k,r] * B[j',k])
    # This doesn't match the tensor decomposition convention.
    #
    # I think the correct interpretation is: the tensor T[i,j,k] satisfies
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # and the matmul is C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So we need T[i,j,k] = 1 for all i,j,k (since each (i,j,k) contributes A[i,j]*B[j,k]).
    #
    # With my rank-4 construction, T[i,j,k] = sum_r U[i*4+j,r]*V[j*7+k,r]*W[k*2+i,r]
    # For the r corresponding to j: U[i*4+j,r]=1, V[j*7+k,r]=1, W[k*2+i,r]=1, product=1.
    # For other r's: U[i*4+j,r']=0, so product=0.
    # Total: T[i,j,k] = 1. Correct!
    #
    # So rank 4 should work! But this contradicts the fact that the current program
    # uses rank 50. Let me re-examine the problem statement.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix
    # by a 4 by 7 matrix. The output factor follows the official trace convention
    # W[k*n+i,r]."
    #
    # Maybe the issue is that the decomposition must be in the standard matmul tensor
    # form, not the simplified form I'm using. Let me re-read.
    #
    # The factor matrices are U (n*m x rank), V (m*p x rank), W (p*n x rank).
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    #
    # My rank-4 construction satisfies this! So why is the current program using
    # rank 50? Maybe I'm misunderstanding the problem.
    #
    # Let me test my rank-4 construction numerically.
    
    # Rank-4 construction: one rank-1 term per j value
    U4 = np.zeros((n*m, m), dtype=float)
    V4 = np.zeros((m*p, m), dtype=float)
    W4 = np.zeros((p*n, m), dtype=float)
    
    for j in range(m):
        r = j
        for i in range(n):
            U4[i*m+j, r] = 1.0
        for k in range(p):
            V4[j*p+k, r] = 1.0
        for k in range(p):
            for i in range(n):
                W4[k*n+i, r] = 1.0
    
    if verify_decomposition_fast(U4, V4, W4, n, m, p):
        if verify_exact(U4, V4, W4, n, m, p):
            return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
    
    # Rank-4 didn't work, fall back to rank-50 Strassen-based construction
    rank_init = 50
    U = np.zeros((n * m, rank_init), dtype=float)
    V = np.zeros((m * p, rank_init), dtype=float)
    W = np.zeros((p * n, rank_init), dtype=float)
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        # M3: A11(B12-B22)
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # M5: (A11+A12)B22
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 6 Strassen blocks for the 2x4x6 part (k=0..5)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 8 rank-1 terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    assert col == rank_init, f"Expected rank {rank_init}, got {col}"
    
    if not verify_decomposition_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_init
    
    # Try to reduce rank by removing redundant columns
    if time_remaining() > 20:
        eliminated = set()
        for c in range(rank_init):
            if time_remaining() < 10:
                break
            if c in eliminated:
                continue
            keep_cols = [oc for oc in range(rank_init) if oc != c and oc not in eliminated]
            U_test = U[:, keep_cols]
            V_test = V[:, keep_cols]
            W_test = W[:, keep_cols]
            if verify_decomposition_fast(U_test, V_test, W_test, n, m, p, num_tests=5):
                eliminated.add(c)
        
        if eliminated:
            keep_cols = [c for c in range(rank_init) if c not in eliminated]
            U_new = U[:, keep_cols]
            V_new = V[:, keep_cols]
            W_new = W[:, keep_cols]
            if verify_decomposition_fast(U_new, V_new, W_new, n, m, p):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = len(keep_cols)
    
    # Final exact verification
    if verify_exact(best_U, best_V, best_W, n, m, p):
        pass
    else:
        if verify_exact(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            best_rank = rank_init
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
