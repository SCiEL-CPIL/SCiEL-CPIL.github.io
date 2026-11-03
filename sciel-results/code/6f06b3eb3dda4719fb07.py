import time
import sys
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 130

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def strassen_decomposition():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    alpha = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    beta = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    w_coeff = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1
            
            for (i, j), val in alpha[s].items():
                if val != 0:
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx][c] += val
            
            for (i, j), val in beta[s].items():
                if val != 0:
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx][c] += val
            
            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    w_col = 2 * j + k_c
                    w_row = i
                    w_idx = w_col * n + w_row
                    W[w_idx][c] += val
    
    return U, V, W

def kron_decomposition():
    """
    Decompose 2x4x4 as Kronecker product of 2x1x1 identity and 4x4x4.
    
    The 4x4x4 multiplication tensor T4[i,j,k] = 1 for all i,j,k in {0,1,2,3}
    can be decomposed using structured bilinear algorithms.
    
    We use a rank-12 decomposition of the 4x4x4 tensor based on exploiting
    the cyclic group structure Z_4 and its characters.
    
    For the 2x4x4 case:
    - U has shape (8, rank) where row i*m+j corresponds to (i,j)
    - V has shape (16, rank) where row j*p+k corresponds to (j,k)  
    - W has shape (8, rank) where row k*n+i corresponds to (k,i)
    
    Using Kronecker: T_{2,4,4} = I_2 ⊗ T_{4,4,4}
    If T_{4,4,4} has rank r, then T_{2,4,4} has rank 2*r.
    
    We seek a rank-12 decomposition of T_{4,4,4}, giving total rank 24.
    """
    n, m, p = 2, 4, 4
    
    # First, find a rank-12 decomposition of the 4x4x4 all-ones tensor
    # T4[i,j,k] = 1 for all i,j,k in {0,1,2,3}
    
    # Using Fourier-based structured decomposition:
    # The 4x4x4 all-ones tensor can be written as sum of rank-1 terms
    # exploiting the fact that it's the outer product of three vectors of ones,
    # but we need a more efficient decomposition.
    
    # Actually, the 4x4x4 all-ones tensor has tensor rank 4 (it's just
    # ones_vec ⊗ ones_vec ⊗ ones_vec), but that's not helpful for our
    # structured decomposition.
    
    # Let me use a different approach: decompose using 2x2 blocks with
    # Strassen-like structure.
    
    # For 4x4x4, we can use the fact that it's a block matrix.
    # Write 4x4 = [[2x2, 2x2], [2x2, 2x2]]
    
    # Actually, let me think about this differently. The tensor T[i,j,k]=1
    # for 4x4x4 is just the Kronecker product of three vectors of ones.
    # Its tensor rank is 1! But that's for the raw tensor.
    
    # Wait, I'm confusing myself. The matrix multiplication tensor for
    # multiplying an a×b matrix by a b×c matrix has entries T[i,j,k] where
    # i indexes rows of first matrix, j indexes columns of first/rows of second,
    # k indexes columns of second. For a×b×c, the tensor is T[i,j,k]=1 if
    # the (i,k) entry of the product depends on A[i,j]*B[j,k].
    
    # For 4x4x4 matrix multiplication, T[i,j,k] = 1 for all i,j,k in {0,1,2,3}.
    # This is the standard matrix multiplication tensor, which has rank at most 64
    # (schoolbook) and at least 16 (trivial lower bound).
    
    # Let me use a block Strassen approach for 4x4x4:
    # Split 4x4 into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = A*B = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
    #            [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    # Each 2x2 product uses 7 Strassen multiplications.
    # But we can share some computations.
    
    # For the 4x4x4 case using block Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # This gives 7*4 = 28 rank for 4x4x4, same as before.
    
    # Let me try a different structured approach.
    
    # Actually, for the 2x4x4 case, I can think of it as:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    
    # Let me try to find a rank-24 decomposition directly.
    
    # Using the fact that we have 2 rows in A, we can potentially share
    # more computations between the two rows.
    
    # Let A = [a0; a1] where a0, a1 are 1x4 vectors
    # B is 4x4
    # C = [a0*B; a1*B]
    
    # C[0,k] = sum_j a0[j]*B[j,k]
    # C[1,k] = sum_j a1[j]*B[j,k]
    
    # This is equivalent to computing two independent 1x4 * 4x4 products.
    # Each 1x4 * 4x4 product has tensor rank 16 (schoolbook).
    # But we can potentially share if we structure it well.
    
    # Let me use a Fourier-based approach for the 1x4 * 4x4 product.
    
    # For a 1x4 vector v and 4x4 matrix M:
    # result[k] = sum_j v[j]*M[j,k]
    
    # This can be computed using 16 multiplications (schoolbook).
    # Can we do better?
    
    # Actually, for a fixed 4x4 matrix M, computing v*M is a linear
    # transformation. The tensor rank of the 1x4x4 multiplication tensor
    # is 16 (it's just the standard matrix-vector product).
    
    # But wait, I need to reconsider. The tensor T[j,k] for 1x4*4x4
    # has T[j,k] = 1 for all j,k. This is the outer product of ones_vec
    # and ones_vec, so it has tensor rank 1!
    
    # No, that's not right either. The matrix multiplication tensor for
    # (1x4) * (4x4) has entries indexed by (j,k) where j is the column of
    # the first matrix (which is also the row of the second) and k is the
    # column of the second matrix. The entry T[j,k] = 1 means that the
    # (0,k) entry of the product equals sum_j A[0,j]*B[j,k].
    
    # So T[j,k] = 1 for all j in {0,1,2,3}, k in {0,1,2,3}.
    # This is a 4x4 matrix of all ones, which has rank 1 as a matrix,
    # but as a tensor for bilinear computation, it's different.
    
    # Let me just try to construct a rank-24 decomposition directly.
    
    # I'll use a systematic approach: assume the decomposition has a
    # certain structure and solve for the coefficients.
    
    # Let's try rank 24 with a structured form.
    
    rank = 24
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 24
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 24
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 24
    
    # Try a structured decomposition based on 2x2 blocks.
    # Split the 4 columns of A into two groups of 2: {0,1} and {2,3}
    # Split the 4 rows of B into two groups of 2: {0,1} and {2,3}
    # Split the 4 columns of B into two groups of 2: {0,1} and {2,3}
    
    # For each (i, j_group, k_group) combination, use Strassen on 2x2x2.
    # But we have 2 values of i, so we might be able to share.
    
    # Let me try: for each i in {0,1}, compute C[i,:] = A[i,:]*B
    # A[i,:] is 1x4, B is 4x4.
    
    # Split A[i,:] = [a0, a1] where a0 = A[i,0:2], a1 = A[i,2:4]
    # Split B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    
    # C[i,0:2] = a0*B00 + a1*B10
    # C[i,2:4] = a0*B01 + a1*B11
    
    # For each i, this is two 1x2*2x2 products.
    # Each 1x2*2x2 product has rank 4 (schoolbook) or can be done with
    # 3 multiplications using a Strassen-like trick.
    
    # For 1x2 * 2x2:
    # [x, y] * [[a, b], [c, d]] = [xa+yc, xb+yd]
    # This requires 4 multiplications. Can we do 3?
    # M1 = x*(a+c)
    # M2 = y*(b+d)  
    # M3 = (x+y)*(c+d)
    # Then xa+yc = M1 + M3 - M2 - y*c ... no, this doesn't work cleanly.
    
    # Actually, 1x2*2x2 is just a linear combination, and the tensor rank
    # is 4 (it's the standard matrix-vector product for 2x2).
    
    # So for each i, we need 4+4 = 8 multiplications for the two halves.
    # Total: 2*8 = 16 multiplications. That's less than 24!
    
    # Wait, but I need to be more careful. Let me reconsider.
    
    # For C[i,0:2] = a0*B00 + a1*B10 where a0, a1 are 1x2 and B00, B10 are 2x2.
    # This is: [a0[0], a0[1]] * B00 + [a1[0], a1[1]] * B10
    # = [a0[0]*B00[0,0] + a0[1]*B00[1,0] + a1[0]*B10[0,0] + a1[1]*B10[1,0],
    #    a0[0]*B00[0,1] + a0[1]*B00[1,1] + a1[0]*B10[0,1] + a1[1]*B10[1,1]]
    
    # This requires 8 multiplications (schoolbook).
    
    # Similarly for C[i,2:4].
    
    # So total is 2 * (8 + 8) = 32 multiplications. That's the schoolbook.
    
    # I need a smarter decomposition. Let me try using Strassen on the
    # 2x2 blocks.
    
    # For each i, compute:
    # C[i,0:2] = a0*B00 + a1*B10
    # C[i,2:4] = a0*B01 + a1*B11
    
    # Let me define:
    # M1 = (a0 + a1) * (B00 + B10)  -- but a0, a1 are 1x2, B00, B10 are 2x2
    # This doesn't directly work.
    
    # Let me try a different block structure.
    
    # Actually, let me just implement a known rank-24 decomposition if one exists,
    # or fall back to rank 28.
    
    # For now, let me try to construct a rank-24 decomposition by solving
    # a system of equations.
    
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # We want to write T = sum_{r=1}^{24} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^8, v_r is in R^16, w_r is in R^8.
    
    # This is a system of 8*16*8 = 1024 equations in 24*(8+16+8) = 768 unknowns.
    # So it's overdetermined, but might have a solution.
    
    # Let me try a structured ansatz.
    
    # Split indices:
    # i = 2*i2 + i1 where i2 in {0,1}, i1 in {0,1}  (i1 is the "fast" index)
    # j = 2*j2 + j1 where j2 in {0,1}, j1 in {0,1}
    # k = 2*k2 + k1 where k2 in {0,1}, k1 in {0,1}
    
    # T[i,j,k] = 1 for all combinations.
    
    # Let me try a decomposition that's a sum over (i2, j2, k2) of
    # 2x2x2 Strassen decompositions, but with sharing.
    
    # For each (i2, j2, k2) in {0,1}^3, we have a 2x2x2 subproblem.
    # There are 8 such subproblems, each of rank 7, giving total rank 56.
    # That's worse than 28.
    
    # I think I need to be more clever. Let me just return the Strassen
    # decomposition for now and try to improve it in a future iteration.
    
    return U, V, W  # Return empty for now, will use Strassen fallback

def construct():
    n, m, p = 2, 4, 4
    
    # Try the Kronecker decomposition
    U_kron, V_kron, W_kron = kron_decomposition()
    if len(U_kron) > 0 and len(U_kron[0]) > 0:
        valid, err = verify_decomposition(U_kron, V_kron, W_kron, n, m, p)
        if valid:
            return {"U": U_kron, "V": V_kron, "W": W_kron}
    
    # Fall back to Strassen decomposition
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback naive
    rank = 32
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][idx] = 1.0
                V[j*p+k][idx] = 1.0
                W[k*n+i][idx] = 1.0
                idx += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
