import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Known lower bound for matmul tensor rank R(n,m,p) >= max(n*m, m*p, n*p) = max(8,24,12) = 24
    # But that's for standard matmul. Here we have a specific structure.
    # The tensor T[i,j,k] = 1 for all i,j,k is NOT the matmul tensor.
    # Wait - the task says "multiplying a 2 by 4 matrix by a 4 by 6 matrix".
    # The matmul tensor for A(n,m) @ B(m,p) -> C(n,p) has T[i,j,k] = 1 if the (i,j) element of A
    # and (j,k) element of B contribute to C[i,k]. So T[i,j,k] = 1 for all valid i,j,k.
    # Actually the standard matmul tensor is T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # So yes, it's the all-ones tensor of shape (2,4,6).
    
    # The rank of the all-ones tensor of shape (n,m,p) is max(n,m,p) = 6? No.
    # Actually, the all-ones tensor of shape (n,m,p) has rank min(n*m, m*p, n*p) in general? No.
    # Let me think again. The all-ones tensor J_{n,m,p} where J[i,j,k]=1.
    # J can be written as sum_{r=1}^{min(n,m,p)} e_{i,r} * f_{j,r} * g_{k,r} where...
    # Actually J[i,j,k] = sum_r u_r[i] v_r[j] w_r[k]. 
    # One decomposition: use r=1..n, u_r[i]=delta_{ir}, v_r[j]=1, w_r[k]=1. That gives rank n.
    # Similarly rank m or rank p. So rank of all-ones (n,m,p) tensor is min(n,m,p).
    # For (2,4,6), that's 2!
    
    # Wait, but the decomposition format requires:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For the all-ones tensor: T[i,j,k] = 1 for all i,j,k.
    # Try rank 1: U[i*m+j,0] = a_ij, V[j*p+k,0] = b_jk, W[k*n+i,0] = c_ki
    # Need sum_r a_ij b_jk c_ki = 1 for all i,j,k.
    # With rank 1: a_ij b_jk c_ki = 1 for all i,j,k.
    # Fix j: a_ij b_jk c_ki = 1 for all i,k. This means a_ij * c_ki = 1/b_jk for all i,k.
    # So a_ij * c_ki must not depend on i or k... but it does depend on i and k.
    # a_ij * c_ki = 1/b_jk for all i,k.
    # Set i=i0: a_i0j * c_ki0 = 1/b_jk for all k. So c_ki0 = 1/(b_jk * a_i0j) for all j,k.
    # This is getting complicated. Let me just try small ranks.
    
    # Actually, for rank 2 (since n=2):
    # r=0,1
    # T[i,j,k] = sum_{r=0}^{1} U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Let's try: U[i*m+j,r] = 1 for all i,j,r; V[j*p+k,r] = 1 for all j,k,r; W[k*n+i,r] = 1/2 for all k,i,r.
    # Then T[i,j,k] = sum_r 1*1*(1/2) = 2*(1/2) = 1. Yes!
    
    # So rank 2 works! Let me verify.
    # U has shape (2*4, 2) = (8, 2), all ones.
    # V has shape (4*6, 2) = (24, 2), all ones.
    # W has shape (6*2, 2) = (12, 2), all 1/2.
    # T[i,j,k] = sum_{r=0}^{1} U[4i+j, r] * V[6j+k, r] * W[2k+i, r]
    #          = sum_{r=0}^{1} 1 * 1 * (1/2) = 1. Correct!
    
    # But wait - can we do rank 1?
    # T[i,j,k] = U[4i+j,0] * V[6j+k,0] * W[2k+i,0] = 1 for all i,j,k.
    # This means U[4i+j,0] * V[6j+k,0] * W[2k+i,0] = 1.
    # Fix j,k: U[4i+j,0] * W[2k+i,0] = 1/V[6j+k,0] for all i.
    # Let f_jk = 1/V[6j+k,0]. Then U[4i+j,0] * W[2k+i,0] = f_jk for all i.
    # Fix i=0: U[j,0] * W[2k,0] = f_jk for all j,k.
    # Fix i=1: U[4+j,0] * W[2k+1,0] = f_jk for all j,k.
    # From i=0: W[2k,0] = f_jk / U[j,0] for all j. So f_jk / U[j,0] is independent of j.
    # f_jk = U[j,0] * W[2k,0]. But also f_jk is independent of j? No, f_jk depends on j.
    # From i=0: for fixed k, f_jk / U[j,0] = W[2k,0] which is independent of j.
    # So f_jk = U[j,0] * W[2k,0].
    # From i=1: f_jk = U[4+j,0] * W[2k+1,0].
    # So U[j,0] * W[2k,0] = U[4+j,0] * W[2k+1,0] for all j,k.
    # This means U[j,0]/U[4+j,0] = W[2k+1,0]/W[2k,0] for all j,k.
    # So U[j,0]/U[4+j,0] is constant (independent of j) and W[2k+1,0]/W[2k,0] is constant (independent of k).
    # Let U[j,0] = c * U[4+j,0] for all j, and W[2k+1,0] = d * W[2k,0] for all k.
    # And c = d.
    # Also need f_jk = 1/V[6j+k,0] > 0 (assuming positive entries).
    # f_jk = U[j,0] * W[2k,0].
    # Also T[0,0,0] = U[0,0]*V[0,0]*W[0,0] = 1.
    # T[0,0,1] = U[0,0]*V[1,0]*W[1,0] = 1.
    # T[0,1,0] = U[1,0]*V[0,0]*W[0,0] = 1.
    # From T[0,0,0]: U[0,0]*V[0,0]*W[0,0] = 1.
    # From T[0,0,1]: U[0,0]*V[1,0]*W[1,0] = 1.
    # W[1,0] = d*W[0,0].
    # So U[0,0]*V[1,0]*d*W[0,0] = 1. And U[0,0]*V[0,0]*W[0,0] = 1.
    # So V[1,0]*d = V[0,0]. V[1,0] = V[0,0]/d.
    # From T[0,1,0]: U[1,0]*V[0,0]*W[0,0] = 1. And U[0,0]*V[0,0]*W[0,0] = 1.
    # So U[1,0] = U[0,0].
    # Similarly T[1,0,0] = U[4,0]*V[0,0]*W[0,0] = 1. And U[0,0]*V[0,0]*W[0,0] = 1.
    # So U[4,0] = U[0,0]. But U[j,0] = c*U[4+j,0], so U[0,0] = c*U[4,0] = c*U[0,0], so c=1.
    # So c=d=1. U[j,0] = U[4+j,0] for all j. W[2k,0] = W[2k+1,0] for all k.
    # f_jk = U[j,0]*W[2k,0] = U[j,0]*W[0,0] (since W is constant in k... wait W[2k,0]=W[0,0] for all k? No.)
    # W[2k,0] = W[0,0] for even indices, W[2k+1,0] = W[1,0] = W[0,0]. So W is all the same value w0.
    # U[j,0] = U[4+j,0], so U[0,0]=U[4,0], U[1,0]=U[5,0], etc. Let U[j,0] = u_j for j=0..3, U[4+j,0] = u_j.
    # f_jk = u_j * w0. V[6j+k,0] = 1/(u_j * w0).
    # Check T[i,j,k]: U[4i+j,0]*V[6j+k,0]*W[2k+i,0] = u_j * (1/(u_j*w0)) * w0 = 1. Yes!
    # So rank 1 works!
    
    # U[4i+j, 0] = u_j where u_0=u_1=u_2=u_3 (since U[j,0]=U[4+j,0] just means the top half equals bottom half)
    # Actually U has 8 rows: rows 0-3 are for i=0, rows 4-7 are for i=1.
    # U[0,0]=u_0, U[1,0]=u_1, U[2,0]=u_2, U[3,0]=u_3
    # U[4,0]=u_0, U[5,0]=u_1, U[6,0]=u_2, U[7,0]=u_3
    # V[6j+k,0] = 1/(u_j * w0)
    # W[2k+i,0] = w0
    
    # Simplest: u_j = 1 for all j, w0 = 1.
    # U all 1s (8x1), V all 1s (24x1), W all 1s (12x1).
    # T[i,j,k] = 1*1*1 = 1. 
    
    # Rank 1 is the minimum possible (can't be 0 for a non-zero tensor).
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        rank = U_np.shape[1]
        tensor = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u_mat = U_np[:, r].reshape(n, m)
            v_mat = V_np[:, r].reshape(m, p)
            w_mat = W_np[:, r].reshape(p, n)
            prod = u_mat[:, :, np.newaxis] * v_mat[np.newaxis, :, :] * w_mat.T[:, np.newaxis, :]
            tensor += prod
        expected = np.ones((n, m, p), dtype=np.float64)
        return np.allclose(tensor, expected, atol=1e-8)

    def construct_rank_1():
        rank = 1
        U = [[1] * rank for _ in range(n * m)]
        V = [[1] * rank for _ in range(m * p)]
        W = [[1] * rank for _ in range(p * n)]
        return {"U": U, "V": V, "W": W}

    result = construct_rank_1()
    if verify(result["U"], result["V"], result["W"]):
        return result
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
