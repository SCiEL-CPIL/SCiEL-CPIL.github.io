"""Matrix multiplication 2x3x3 - rank minimization via structured search."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_tensor(n, m, p):
    """Build the tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _rank17_decomposition(n, m, p):
    """Return the rank-17 decomposition using Strassen for 2x2 block + direct for rest."""
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    U17 = np.zeros((n_a, 17))
    V17 = np.zeros((n_b, 17))
    W17 = np.zeros((n_c, 17))
    
    # Strassen for the 2x2 block (rows 0,1 of A, cols 0,1 of B -> C cols 0,1)
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B00=0, B01=1, B10=3, B11=4
    # C indices: C00=0, C10=1, C01=2, C11=3 (2x3 matrix, row-major: [C00,C01,C02,C10,C11,C12])
    # Wait, C is 2x3, so C[i,j] for i in {0,1}, j in {0,1,2}
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[1,0]=3, C[1,1]=4, C[1,2]=5
    
    # Let me redo. The output C is p*n = 3*2 = 6 elements.
    # C[k*n+i, r] where k is column of B (0..p-1=0..2), i is row of A (0..n-1=0..1)
    # So C index = k*n + i = k*2 + i
    # C[0*2+0]=C[0]=C00, C[0*2+1]=C[1]=C10, C[1*2+0]=C[2]=C01, C[1*2+1]=C[3]=C11, C[2*2+0]=C[4]=C02, C[2*2+1]=C[5]=C12
    
    # So:
    # C00 = index 0
    # C10 = index 1
    # C01 = index 2
    # C11 = index 3
    # C02 = index 4
    # C12 = index 5
    
    # Strassen for the 2x2 block (rows 0,1 of A, cols 0,1 of B -> C cols 0,1):
    # M1 = (A00+A11)(B00+B11) = (A[0]+A[4])(B[0]+B[4])
    # M2 = (A10+A11)B00 = (A[3]+A[4])B[0]
    # M3 = A00(B10-B00) = A[0](B[3]-B[0])
    # M4 = A11(B01-B10) = A[4](B[1]-B[3])
    # M5 = (A00+A01)B11 = (A[0]+A[1])B[4]
    # M6 = (A10-A00)(B00+B10) = (A[3]-A[0])(B[0]+B[3])
    # M7 = (A01-A11)(B01+B11) = (A[1]-A[4])(B[1]+B[4])
    
    # C00 = M1+M4-M5+M7 -> index 0
    # C10 = M2+M4 -> index 1
    # C01 = M3+M5 -> index 2
    # C11 = M1-M2+M3+M7 -> index 3
    
    # For each Strassen term, we have U factors (coefficients on A indices), V factors (coefficients on B indices), W factors (coefficients on C indices)
    
    # M1: U = {0:1, 4:1}, V = {0:1, 4:1}, W = {0:1, 3:1}
    U17[0,0]=1; U17[4,0]=1
    V17[0,0]=1; V17[4,0]=1
    W17[0,0]=1; W17[3,0]=1
    
    # M2: U = {3:1, 4:1}, V = {0:1}, W = {1:1, 3:-1}
    U17[3,1]=1; U17[4,1]=1
    V17[0,1]=1
    W17[1,1]=1; W17[3,1]=-1
    
    # M3: U = {0:1}, V = {3:1, 0:-1}, W = {2:1, 3:1}
    U17[0,2]=1
    V17[3,2]=1; V17[0,2]=-1
    W17[2,2]=1; W17[3,2]=1
    
    # M4: U = {4:1}, V = {1:1, 3:-1}, W = {0:1, 1:1}
    U17[4,3]=1
    V17[1,3]=1; V17[3,3]=-1
    W17[0,3]=1; W17[1,3]=1
    
    # M5: U = {0:1, 1:1}, V = {4:1}, W = {0:-1, 2:1}
    U17[0,4]=1; U17[1,4]=1
    V17[4,4]=1
    W17[0,4]=-1; W17[2,4]=1
    
    # M6: U = {3:1, 0:-1}, V = {0:1, 3:1}, W = {1:1}
    U17[3,5]=1; U17[0,5]=-1
    V17[0,5]=1; V17[3,5]=1
    W17[1,5]=1
    
    # M7: U = {1:1, 4:-1}, V = {1:1, 4:1}, W = {0:1, 3:1}
    U17[1,6]=1; U17[4,6]=-1
    V17[1,6]=1; V17[4,6]=1
    W17[0,6]=1; W17[3,6]=1
    
    # Part 2: Direct for B col 2 (indices 2,5,8)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2] = A[0]*B[2]+A[1]*B[5]+A[2]*B[8] -> index 4
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2] = A[3]*B[2]+A[4]*B[5]+A[5]*B[8] -> index 5
    
    # 6 terms for col 2:
    # A[0]*B[2] -> C[0,2]
    U17[0,7]=1; V17[2,7]=1; W17[4,7]=1
    # A[1]*B[5] -> C[0,2]
    U17[1,8]=1; V17[5,8]=1; W17[4,8]=1
    # A[2]*B[8] -> C[0,2]
    U17[2,9]=1; V17[8,9]=1; W17[4,9]=1
    # A[3]*B[2] -> C[1,2]
    U17[3,10]=1; V17[2,10]=1; W17[5,10]=1
    # A[4]*B[5] -> C[1,2]
    U17[4,11]=1; V17[5,11]=1; W17[5,11]=1
    # A[5]*B[8] -> C[1,2]
    U17[5,12]=1; V17[8,12]=1; W17[5,12]=1
    
    # But wait, that's 7+6=13 terms. I need 17. Let me recount.
    # The 2x2 block uses A cols 0,1 (indices 0,1,3,4) and B cols 0,1 (indices 0,1,3,4)
    # But A has 3 columns! A[0,2]=index 2 and A[1,2]=index 5 are not in the 2x2 block.
    # Similarly B has 3 rows, so B[2,0]=6, B[2,1]=7, B[2,2]=8 are not in the 2x2 block.
    
    # C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k]
    # For k=0,1 (the 2x2 block part), we need j=0,1,2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # So the 2x2 block only covers j=0,1. We need additional terms for j=2:
    # A[0,2]*B[2,0] -> C[0,0] (index 0)
    # A[0,2]*B[2,1] -> C[0,1] (index 2)
    # A[1,2]*B[2,0] -> C[1,0] (index 1)
    # A[1,2]*B[2,1] -> C[1,1] (index 3)
    
    # That's 4 more terms, giving 7+4+6 = 17.
    
    # A[0,2]*B[2,0] -> C[0,0]
    U17[2,13]=1; V17[6,13]=1; W17[0,13]=1
    # A[0,2]*B[2,1] -> C[0,1]
    U17[2,14]=1; V17[7,14]=1; W17[2,14]=1
    # A[1,2]*B[2,0] -> C[1,0]
    U17[5,15]=1; V17[6,15]=1; W17[1,15]=1
    # A[1,2]*B[2,1] -> C[1,1]
    U17[5,16]=1; V17[7,16]=1; W17[3,16]=1
    
    return U17, V17, W17

def _try_rank16_via_hosvd(n, m, p, deadline):
    """
    Try to find a rank-16 decomposition using HOSVD-inspired approach.
    The idea: use the structure of the tensor to find a low-rank approximation
    that might be exact with integer/half-integer entries.
    """
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    T = _build_tensor(n, m, p)
    
    # Flatten T to a matrix and use SVD to get a rank-16 approximation
    # T is 6x9x6, flatten to 6x54
    T_flat = T.reshape(6, 54)
    
    # SVD
    U_svd, s, Vt = np.linalg.svd(T_flat, full_matrices=False)
    
    # Try rank-16 approximation
    if len(s) < 16:
        return None
    
    T_approx = (U_svd[:, :16] * s[:16]) @ Vt[:16, :]
    T_approx = T_approx.reshape(6, 9, 6)
    
    # Check if this is close to the original tensor
    if not np.allclose(T_approx, T, atol=1e-6):
        return None
    
    # Now try to find a CP decomposition of T_approx with rank 16
    # This is a non-convex optimization problem, so we use a heuristic approach
    
    # Initialize with random factors
    np.random.seed(42)
    U = np.random.randn(n_a, 16)
    V = np.random.randn(n_b, 16)
    W = np.random.randn(n_c, 16)
    
    # Normalize
    for r in range(16):
        norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
        if norm > 1e-10:
            scale = norm ** (1/3)
            U[:, r] /= scale
            V[:, r] /= scale
            W[:, r] /= scale
    
    # Alternating least squares (ALS)
    for iteration in range(100):
        if time.time() > deadline - 10:
            break
        
        # Update U
        for r in range(16):
            # Residual for rank r
            T_r = T.copy()
            for r2 in range(16):
                if r2 != r:
                    T_r -= np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
            
            # Solve for U[:, r]
            # T_r[a, b, c] = U[a, r] * V[b, r] * W[c, r]
            # For each a: sum_{b,c} T_r[a, b, c] * V[b, r] * W[c, r] = U[a, r] * sum_{b,c} V[b, r]^2 * W[c, r]^2
            
            V_r = V[:, r]
            W_r = W[:, r]
            denom = np.sum(V_r ** 2) * np.sum(W_r ** 2)
            if abs(denom) < 1e-10:
                continue
            U[:, r] = np.sum(T_r * np.outer(V_r, W_r), axis=(1, 2)) / denom
        
        # Update V
        for r in range(16):
            T_r = T.copy()
            for r2 in range(16):
                if r2 != r:
                    T_r -= np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
            
            U_r = U[:, r]
            W_r = W[:, r]
            denom = np.sum(U_r ** 2) * np.sum(W_r ** 2)
            if abs(denom) < 1e-10:
                continue
            V[:, r] = np.sum(T_r * np.outer(U_r, W_r), axis=(0, 2)) / denom
        
        # Update W
        for r in range(16):
            T_r = T.copy()
            for r2 in range(16):
                if r2 != r:
                    T_r -= np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
            
            U_r = U[:, r]
            V_r = V[:, r]
            denom = np.sum(U_r ** 2) * np.sum(V_r ** 2)
            if abs(denom) < 1e-10:
                continue
            W[:, r] = np.sum(T_r * np.outer(U_r, V_r), axis=(0, 1)) / denom
    
    # Check if we got an exact decomposition
    if _verify_decomposition(U, V, W, T):
        # Try to round to integers or half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, T):
            return U_rounded, V_rounded, W_rounded
    
    return None

def _try_rank16_via_structured_search(n, m, p, deadline):
    """
    Try a structured search for rank-16 decomposition.
    Key insight: The tensor has a specific structure that might allow
    a rank-16 decomposition with a particular pattern.
    """
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    T = _build_tensor(n, m, p)
    
    # The tensor T has non-zero entries at:
    # T[i*m+j, j*p+k, k*n+i] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    
    # Let's list all non-zero entries:
    # i=0, j=0, k=0: T[0, 0, 0]
    # i=0, j=0, k=1: T[0, 1, 2]
    # i=0, j=0, k=2: T[0, 2, 4]
    # i=0, j=1, k=0: T[1, 3, 0]
    # i=0, j=1, k=1: T[1, 4, 2]
    # i=0, j=1, k=2: T[1, 5, 4]
    # i=0, j=2, k=0: T[2, 6, 0]
    # i=0, j=2, k=1: T[2, 7, 2]
    # i=0, j=2, k=2: T[2, 8, 4]
    # i=1, j=0, k=0: T[3, 0, 1]
    # i=1, j=0, k=1: T[3, 1, 3]
    # i=1, j=0, k=2: T[3, 2, 5]
    # i=1, j=1, k=0: T[4, 3, 1]
    # i=1, j=1, k=1: T[4, 4, 3]
    # i=1, j=1, k=2: T[4, 5, 5]
    # i=1, j=2, k=0: T[5, 6, 1]
    # i=1, j=2, k=1: T[5, 7, 3]
    # i=1, j=2, k=2: T[5, 8, 5]
    
    # So there are 18 non-zero entries.
    
    # For a rank-16 decomposition, we need 16 terms.
    # Each term is a rank-1 tensor: u ⊗ v ⊗ w
    # The sum of these 16 terms must equal T.
    
    # This is a system of 18 equations (one for each non-zero entry) with 16*6 + 16*9 + 16*6 = 96 + 144 + 96 = 336 unknowns.
    # But we also have the constraint that the off-diagonal entries must be zero.
    
    # Actually, the tensor is 6x9x6 = 324 entries, so we have 324 equations.
    # 18 are non-zero, 306 are zero.
    
    # This is a highly underdetermined system, so there should be many solutions.
    # The question is whether there's a solution with integer or half-integer entries.
    
    # Let's try a random search with a structured initialization.
    
    # Initialize with the rank-17 decomposition and try to merge two terms.
    U17, V17, W17 = _rank17_decomposition(n, m, p)
    
    # Try to find two terms in the rank-17 decomposition that can be merged
    # into a single term with integer/half-integer coefficients.
    
    # For each pair of terms (r1, r2), check if U[:, r1] + U[:, r2], V[:, r1] + V[:, r2], W[:, r1] + W[:, r2]
    # can form a valid rank-1 term.
    
    # Actually, this approach doesn't directly work because merging two terms
    # doesn't generally produce a valid rank-1 term.
    
    # Instead, let's try a different approach: use the fact that the tensor
    # has a specific structure and try to find a rank-16 decomposition
    # with a particular pattern.
    
    # Pattern: The tensor can be viewed as a sum of 18 rank-1 terms (one for each
    # non-zero entry). We want to find a way to combine these 18 terms into
    # 16 terms.
    
    # This is equivalent to finding a 2-term cancellation: two terms that
    # can be combined into a single term.
    
    # For example, if we have:
    # u1 ⊗ v1 ⊗ w1 + u2 ⊗ v2 ⊗ w2
    # and we can find u, v, w such that:
    # u ⊗ v ⊗ w = u1 ⊗ v1 ⊗ w1 + u2 ⊗ v2 ⊗ w2
    # then we can replace the two terms with one.
    
    # This is possible if u1, u2 are linearly dependent, v1, v2 are linearly dependent,
    # and w1, w2 are linearly dependent.
    
    # In the rank-17 decomposition, let's check if any two terms have
    # linearly dependent U, V, or W vectors.
    
    # Actually, this is getting too complex. Let's just try a random search.
    
    np.random.seed(123)
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random factors
        U = np.random.randn(n_a, 16)
        V = np.random.randn(n_b, 16)
        W = np.random.randn(n_c, 16)
        
        # Normalize
        for r in range(16):
            norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
            if norm > 1e-10:
                scale = norm ** (1/3)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        # ALS
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for r in range(16):
                T_r = T.copy()
                for r2 in range(16):
                    if r2 != r:
                        T_r -= np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
                
                V_r = V[:, r]
                W_r = W[:, r]
                denom = np.sum(V_r ** 2) * np.sum(W_r ** 2)
                if abs(denom) < 1e-10:
                    continue
                U[:, r] = np.sum(T_r * np.outer(V_r, W_r), axis=(1, 2)) / denom
            
            # Update V
            for r in range(16):
                T_r = T.copy()
                for r2 in range(16):
                    if r2 != r:
                        T_r -= np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
                
                U_r = U[:, r]
                W_r = W[:, r]
                denom = np.sum(U_r ** 2) * np.sum(W_r ** 2)
                if abs(denom) < 1e-10:
                    continue
                V[:, r] = np.sum(T_r * np.outer(U_r, W_r), axis=(0, 2)) / denom
            
            # Update W
            for r in range(16):
                T_r = T.copy()
                for r2 in range(16):
                    if r2 != r:
                        T_r -= np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
                
                U_r = U[:, r]
                V_r = V[:, r]
                denom = np.sum(U_r ** 2) * np.sum(V_r ** 2)
                if abs(denom) < 1e-10:
                    continue
                W[:, r] = np.sum(T_r * np.outer(U_r, V_r), axis=(0, 1)) / denom
        
        # Check if we got an exact decomposition
        if _verify_decomposition(U, V, W, T):
            # Try to round to integers or half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, T):
                return U_rounded, V_rounded, W_rounded
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # Build the rank-17 decomposition
    U17, V17, W17 = _rank17_decomposition(n, m, p)
    
    # Verify rank-17
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U17, V17, W17, T):
        # Fallback to schoolbook rank 18
        U_fb = np.zeros((n_a, 18))
        V_fb = np.zeros((n_b, 18))
        W_fb = np.zeros((n_c, 18))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to find a rank-16 decomposition using HOSVD
    if time.time() < deadline - 30:
        result16 = _try_rank16_via_hosvd(n, m, p, deadline)
        if result16 is not None:
            U16, V16, W16 = result16
            if _verify_decomposition(U16, V16, W16, T):
                return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Try a structured search for rank-16
    if time.time() < deadline - 15:
        result16 = _try_rank16_via_structured_search(n, m, p, deadline)
        if result16 is not None:
            U16, V16, W16 = result16
            if _verify_decomposition(U16, V16, W16, T):
                return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Return the rank-17 decomposition
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
