"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    The tensor for C = A B where A is 2x2 and B is 2x10 can be viewed as
    5 independent 2x2x2 multiplication problems, one for each pair of columns in B.
    Specifically, let B = [B_0, B_1, B_2, B_3, B_4] where each B_i is a 2x2 matrix.
    Then C = A B = [A B_0, A B_1, A B_2, A B_3, A B_4].
    
    The rank of the 2x2x2 matrix multiplication tensor is 7 (Strassen's algorithm).
    Since the 5 blocks are independent, the total rank is the sum of the ranks of the blocks.
    Total Rank = 5 * 7 = 35.
    
    The interface requires:
    U: shape (n*m, rank) = (4, 35)
    V: shape (m*p, rank) = (20, 35)
    W: shape (p*n, rank) = (20, 35)
    
    Indices:
    U index: i*m + j  (i in 0..1, j in 0..1) -> 0,1,2,3
    V index: j*p + k  (j in 0..1, k in 0..9) -> 0..19
    W index: k*n + i  (k in 0..9, i in 0..1) -> 0..19 (Official trace convention)
    
    Strassen's 7 products for multiplying 2x2 matrices M1 and M2:
    M1 = [[a, b], [c, d]], M2 = [[e, f], [g, h]]
    P1 = (a+d)(e+h)
    P2 = (c+d)e
    P3 = a(f-h)
    P4 = d(g-e)
    P5 = (a+b)h
    P6 = (c-a)(e+f)
    P7 = (b-d)(g+h)
    
    Result M = M1 M2:
    M[0,0] = P1 + P4 - P5 + P7
    M[0,1] = P3 + P5
    M[1,0] = P2 + P4
    M[1,1] = P1 - P2 + P3 + P6
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen coefficients for the 7 products
    strassen_A = [
        [1, 0, 0, 1],   # P1: (a+d)
        [0, 0, 1, 1],   # P2: (c+d)
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: (a+b)
        [-1, 0, 1, 0],  # P6: (c-a)
        [0, 1, 0, -1]   # P7: (b-d)
    ]
    
    strassen_B = [
        [1, 0, 0, 1],   # P1: (e+h)
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: (f-h)
        [-1, 0, 1, 0],  # P4: (g-e)
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: (e+f)
        [0, 0, 1, 1]    # P7: (g+h)
    ]
    
    # W contributions for each Strassen product to the local 2x2 C matrix
    # local indices: 0->C00, 1->C01, 2->C10, 3->C11
    w_contrib = [
        [1,  0,  0,  1],  # P1: C00 +1, C11 +1
        [0,  0,  1, -1],  # P2: C10 +1, C11 -1
        [0,  1,  0,  1],  # P3: C01 +1, C11 +1
        [1,  0,  1,  0],  # P4: C00 +1, C10 +1
        [-1, 1,  0,  0],  # P5: C00 -1, C01 +1
        [0,  0,  0,  1],  # P6: C11 +1
        [1,  0,  0,  0]   # P7: C00 +1
    ]
    
    col = 0
    for b in range(5):
        for r in range(7):
            # U: same for all blocks
            for i in range(4):
                U[i][col] = strassen_A[r][i]
            
            # V: specific to block b
            # B indices: e=B[0,2b], f=B[0,2b+1], g=B[1,2b], h=B[1,2b+1]
            V[2*b][col] = strassen_B[r][0]
            V[2*b+1][col] = strassen_B[r][1]
            V[10+2*b][col] = strassen_B[r][2]
            V[10+2*b+1][col] = strassen_B[r][3]
            
            # W: specific to block b
            # W index = k_global * n + i = (2b + k_local)*2 + i
            W[4*b][col] = w_contrib[r][0]      # C00: k_local=0, i=0
            W[4*b+2][col] = w_contrib[r][1]    # C01: k_local=1, i=0
            W[4*b+1][col] = w_contrib[r][2]    # C10: k_local=0, i=1
            W[4*b+3][col] = w_contrib[r][3]    # C11: k_local=1, i=1
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

def _verify_decomposition(result, n=2, m=2, p=10, tol=1e-10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    # C index: k*n + i
    # We need: C[k*n+i] = sum_{i',j'} A[i',j'] * B[j',k]
    # In tensor form: T[i*m+j, j*p+k, k*n+i] = delta (the multiplication tensor)
    
    # Check: for each r, the term is U[:,r] outer V[:,r] outer W[:,r]
    # Sum over r should give the multiplication tensor
    
    # The multiplication tensor M[i*m+j, j'*p+k, k*n+i] = 1 if i==i' and j==j' else 0
    # Wait, let me think more carefully.
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor index: A has indices (i,j) -> flat i*m+j
    # B has indices (j,k) -> flat j*p+k
    # C has indices (i,k) -> flat k*n+i (trace convention)
    # The tensor T[a_idx, b_idx, c_idx] = 1 iff a_idx=i*m+j, b_idx=j*p+k, c_idx=k*n+i
    # i.e., T[(i,j),(j,k),(k,i)] = 1 (with consistent j and k)
    
    # Compute the reconstructed tensor
    # T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # For efficiency, we can check on random inputs
    np.random.seed(42)
    for trial in range(20):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # U[:,r] is n*m vector, V[:,r] is m*p vector, W[:,r] is p*n vector
            u_vec = U[:, r]  # shape (n*m,)
            v_vec = V[:, r]  # shape (m*p,)
            w_vec = W[:, r]  # shape (p*n,)
            
            # Reshape u_vec to (n, m), v_vec to (m, p)
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            
            # The contribution to C from rank-r term:
            # C_contrib[i,k] = sum_j u_mat[i,j] * v_mat[j,k] * w_coeff
            # where w_coeff relates to w_vec[k*n+i]
            # Actually, the term is: sum_{i,j,k} u_mat[i,j] * v_mat[j,k] * w_vec[k*n+i] * A[i,j] * B[j,k]
            # Wait, that's not right either. Let me reconsider.
            
            # The decomposition says:
            # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j' B[j',k] * V[j'*p+k, r]) * W[k*n+i, r]
            # Hmm, that doesn't look right dimensionally. Let me re-derive.
            
            pass
    
    # Simpler verification: build the full tensor and compare
    # T_reconstructed[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Expected: T_expected[(i*m+j), (j*p+k), (k*n+i)] = 1
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_reconstructed = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    diff = np.max(np.abs(T_reconstructed - T_expected))
    return diff < tol

if __name__ == "__main__":
    result = construct()
    valid = _verify_decomposition(result)
    print(f"Rank: {result['U'].shape[1]}")
    print(f"Valid: {valid}")
# EVOLVE-BLOCK-END
