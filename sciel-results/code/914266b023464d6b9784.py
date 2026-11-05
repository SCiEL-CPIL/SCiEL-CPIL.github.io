import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Target: Rank 85
    # Structure:
    # 1. Strassen 4x4x4 block for indices i,j,k in 0..3 (Rank 49)
    # 2. Schoolbook for i in 0..3, j in 0..3, k=4 (Rank 16)
    # 3. Schoolbook for i in 0..3, j=4, k in 0..3 (Rank 16)
    # 4. Schoolbook for i in 0..3, j=4, k=4 (Rank 4)
    # Total Rank = 49 + 16 + 16 + 4 = 85
    
    rank_total = 85
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)
    
    # --- Part 1: Strassen 4x4x4 ---
    # Base Strassen 2x2x2 terms
    # u, v, w are length-4 vectors representing 2x2 matrices
    # Indexing: u[i*2+j]
    s2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    def tensor_product_vec(v1, v2):
        # v1, v2 are length 4 (2x2 matrices)
        # Result is length 16 (4x4 matrix)
        # idx = (i1*2+i2)*4 + (j1*2+j2)
        result = [0.0] * 16
        for i1 in range(2):
            for j1 in range(2):
                val1 = v1[i1*2+j1]
                if val1 == 0: continue
                for i2 in range(2):
                    for j2 in range(2):
                        val2 = v2[i2*2+j2]
                        if val2 == 0: continue
                        idx = (i1*2+i2)*4 + (j1*2+j2)
                        result[idx] = val1 * val2
        return result

    r_idx = 0
    # Generate 49 terms for 4x4x4
    for t1 in s2:
        for t2 in s2:
            u_vec = tensor_product_vec(t1[0], t2[0])
            v_vec = tensor_product_vec(t1[1], t2[1])
            w_vec = tensor_product_vec(t1[2], t2[2])
            
            # Map 4x4 indices to 5x5 indices (top-left corner)
            # U[i*5+j] for i,j in 0..3
            for i in range(4):
                for j in range(4):
                    U[i*5+j, r_idx] = u_vec[i*4+j]
            
            # V[j*5+k] for j,k in 0..3
            for j in range(4):
                for k in range(4):
                    V[j*5+k, r_idx] = v_vec[j*4+k]
                    
            # W[k*5+i] for k,i in 0..3
            for k in range(4):
                for i in range(4):
                    W[k*5+i, r_idx] = w_vec[k*4+i]
            
            r_idx += 1
            
    # --- Part 2: Schoolbook for k=4, i,j in 0..3 ---
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            U[i*5+j, r_idx] = 1.0
            V[j*5+4, r_idx] = 1.0
            W[4*5+i, r_idx] = 1.0
            r_idx += 1

    # --- Part 3: Schoolbook for j=4, i,k in 0..3 ---
    # C[i, k] = A[i, 4] B[4, k]
    for i in range(4):
        for k in range(4):
            U[i*5+4, r_idx] = 1.0
            V[4*5+k, r_idx] = 1.0
            W[k*5+i, r_idx] = 1.0
            r_idx += 1

    # --- Part 4: Schoolbook for j=4, k=4, i in 0..3 ---
    # C[i, 4] = A[i, 4] B[4, 4]
    for i in range(4):
        U[i*5+4, r_idx] = 1.0
        V[4*5+4, r_idx] = 1.0
        W[4*5+i, r_idx] = 1.0
        r_idx += 1
        
    # Verification
    def verify(U, V, W):
        rank = U.shape[1]
        # Check all entries of the 4x5 output matrix
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Tensor equation: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == delta_{i,k} (if we view A,B as variables)
        # Actually, we just need to check that the decomposition reproduces the multiplication tensor.
        # The tensor M_{i,j,k} is 1 if the term corresponds to the (i,j) of A, (j,k) of B, (i,k) of C.
        # Wait, the standard definition:
        # M = sum_{i,j,k} e_{i,j}^U \otimes e_{j,k}^V \otimes e_{i,k}^W
        # Our decomposition T = sum_r u_r \otimes v_r \otimes w_r
        # We need T_{i,j,k} = 1 for all i,j,k? No.
        # The multiplication tensor is defined such that C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # This is linear in A and B.
        # The tensor T has indices (i,j) for U, (j,k) for V, (i,k) for W?
        # Let's check the indices in the code:
        # U index: i*m + j  (i in 0..n-1, j in 0..m-1)
        # V index: j*p + k  (j in 0..m-1, k in 0..p-1)
        # W index: k*n + i  (k in 0..p-1, i in 0..n-1)
        # The contraction is sum_{j,r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # This should equal 1 for all i, k?
        # Let's test with schoolbook:
        # Term r corresponds to (i0, j0, k0).
        # U[i*m+j, r] is 1 if i=i0, j=j0.
        # V[j*p+k, r] is 1 if j=j0, k=k0.
        # W[k*n+i, r] is 1 if k=k0, i=i0.
        # Sum over j, r:
        # Non-zero only if there exists r such that:
        # 1. i=i0, j=j0 (from U)
        # 2. j=j0, k=k0 (from V)
        # 3. k=k0, i=i0 (from W)
        # This requires i=i0, k=k0, and j=j0.
        # So for a fixed i, k, the sum is non-zero only if i=i0 and k=k0 for some term r.
        # But wait, the multiplication is C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # The tensor T_{i,j,k} should be 1?
        # No, the tensor is defined on the space of A, B, C.
        # The "multiplication tensor" M is such that M(A, B) = C.
        # In index notation, C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
        # This can be written as C_{i,k} = sum_{i',j,k'} M_{i',j,k'} A_{i',j} B_{j,k'} ? No.
        # The standard tensor for matrix multiplication is T_{i,j,k,l,m,n} ... no.
        # The rank of matrix multiplication is the rank of the tensor T_{i,j,k} where:
        # T_{i,j,k} = 1 if the index j is the "inner" index?
        # Let's look at the verify function in the seed:
        # s = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # if abs(s - 1.0) > 1e-9: return False
        # This checks if s == 1 for ALL i, j, k.
        # Is that correct?
        # Let's re-read the tensor definition.
        # If A is n x m, B is m x p.
        # A_{i,j}, B_{j,k}, C_{i,k}.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # This is a bilinear map.
        # The tensor T is in U^* \otimes V^* \otimes W^*.
        # T(u, v, w) = w(C) if u(A)=1, v(B)=1?
        # Usually, the tensor M is defined as sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}.
        # Here e_{i,j} is the basis for A, e_{j,k} for B, e_{i,k} for C.
        # Let's check the contraction in the seed's verify:
        # It sums over j and r.
        # Term: U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        # If we plug in A = e_{i0, j0} (1 at i0,j0), B = e_{j1, k1} (1 at j1,k1).
        # Then C should be e_{i0, k1} if j0 == j1, else 0.
        # Let's see what the sum gives for a specific i, k.
        # The verify function checks if the sum is 1 for ALL i, j, k?
        # That would imply that for ANY i, j, k, the coefficient is 1.
        # That is NOT the matrix multiplication tensor.
        # The matrix multiplication tensor has non-zero entries only for specific (i,j,k) combinations?
        # No, the tensor M_{i,j,k} in the decomposition sum_r u_r \otimes v_r \otimes w_r
        # corresponds to the coefficient of A_{i,j} B_{j,k} C_{i,k}^*.
        # For matrix multiplication, the coefficient is 1 for ALL i, j, k.
        # Because C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # So the term A_{i,j} B_{j,k} appears with coefficient 1 in the expression for C_{i,k}.
        # Thus, the tensor T_{i,j,k} = 1 for all valid i, j, k.
        
        # So the verify function in the seed is correct: it checks if the reconstructed tensor
        # has value 1 at every (i, j, k) entry.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(rank):
                        s += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to Schoolbook if construction failed (should not happen)
        rank_school = n * m * p
        U_fb = np.zeros((n * m, rank_school), dtype=float)
        V_fb = np.zeros((m * p, rank_school), dtype=float)
        W_fb = np.zeros((p * n, rank_school), dtype=float)
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, r] = 1
                    V_fb[j*p+k, r] = 1
                    W_fb[k*n+i, r] = 1
                    r += 1
                    
        return {'U': U_fb.tolist(), 'V': V_fb.tolist(), 'W': W_fb.tolist()}

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

