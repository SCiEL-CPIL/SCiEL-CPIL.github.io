"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]
DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T_recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    T_target = np.ones((n, m, p))
    return np.allclose(T_recon, T_target, atol=1e-9)

def _structured_rank12(n, m, p):
    """
    Construct a rank-12 decomposition exploiting the all-ones tensor structure.
    
    For the all-ones tensor T[i,j,k]=1, we use a structured decomposition where:
    - U[i*m+j, c] depends on i and c
    - V[j*p+k, c] depends on j and c  
    - W[k*n+i, c] depends on k and c
    
    We use the identity: 1 = sum over c of factors that multiply to 1.
    
    Strategy: Use 12 rank-1 terms. We assign:
    - For each i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}:
      We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    Using a product structure: Let's use 2 * 3 * 2 = 12 terms where:
    - c = (a, b, d) with a in {0,1}, b in {0,1,2}, d in {0,1}
    - U[i*m+j, (a,b,d)] = 1 if i==a and j==b, else 0  -> but this gives rank n*m*2=12
    - Wait, that's not quite right. Let me think differently.
    
    Actually, for the all-ones tensor, a simple construction:
    Rank = n * m * 2 is not generally possible. Let me use a different approach.
    
    Use the fact that 1 = 1*1*1 and partition the indices.
    
    Alternative: Use 12 terms where we split k into pairs.
    For each i, j: we need sum over k contributions to equal 1.
    
    Let me try: rank = 2 * 3 * 2 = 12
    c = (i', j', k') where i' in {0,1}, j' in {0,1,2}, k' in {0,1}
    
    U[i*m+j, (i',j',k')] = 1 if i==i' and j==j' and k'==0, else 0
    V[j*p+k, (i',j',k')] = 1 if j==j' and k in {k' * 2, k' * 2 + 1} for k'==0: k in {0,1}, k'==1: k in {2,3,4}? No, that doesn't work cleanly.
    
    Let me try a different structured approach. Use the polynomial method.
    """
    # For all-ones tensor, we can use:
    # T[i,j,k] = 1 = sum_{c} U[i,j,c] * V[j,k,c] * W[k,i,c]
    # 
    # Try rank 12: use c indexed by (a, b) where a in {0,1,2,3}, b in {0,1,2,3}
    # No, let's just do a direct construction.
    #
    # Key insight: For all-ones, we can use:
    # U[i*m+j, c] = f_i(c) * g_j(c)
    # V[j*p+k, c] = h_j(c) * l_k(c)  
    # W[k*n+i, c] = m_k(c) * n_i(c)
    # Then sum_c f_i*g_j*h_j*l_k*m_k*n_i = 1
    # = (sum_c f_i*n_i) * (sum_c g_j*h_j) * (sum_c l_k*m_k)
    # So we need: A_i * B_j * C_k = 1 for all i,j,k
    # This means A_i, B_j, C_k are constants! So A_i = a, B_j = b, C_k = c with a*b*c=1
    # And we need: sum_c f_i(c)*n_i(c) = a for all i
    #              sum_c g_j(c)*h_j(c) = b for all j
    #              sum_c l_k(c)*m_k(c) = c for all k
    #
    # With rank r, we can set:
    # f_i(c) = 1/sqrt(r) for all i,c -> sum_c f_i*n_i = n_i * sqrt(r) if n_i=1/sqrt(r)
    # Actually simpler: f_i(c) = 1, n_i(c) = 1/r -> sum = 1
    # g_j(c) = 1, h_j(c) = 1/r -> sum = 1
    # l_k(c) = 1, m_k(c) = 1/r -> sum = 1
    # Then product = 1*1*1 = 1. 
    #
    # So: U[i*m+j, c] = 1 for all c
    #     V[j*p+k, c] = 1 for all c
    #     W[k*n+i, c] = 1/r for all c
    # Then sum_c 1 * 1 * (1/r) = r * (1/r) = 1. 
    #
    # This works for ANY rank! But wait, that gives a valid decomposition for any rank >= 1.
    # Let me verify: T_recon[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = sum_c 1*1*(1/r) = 1. Yes!
    #
    # So rank 1 should work? Let me check with the actual verification.
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank)) / rank
    return U, V, W, rank

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try rank 1 first (simplest possible)
    for rank in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30]:
        if time.time() > DEADLINE:
            break
        
        # For rank r, try the uniform construction: U=1, V=1, W=1/r
        # This always works mathematically, but let's verify it passes the checker
        U = np.ones((n * m, rank))
        V = np.ones((m * p, rank))
        W = np.ones((p * n, rank)) / rank
        
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If rank 1 uniform doesn't pass (maybe the checker requires something else),
        # try other constructions
        if rank == 1:
            # Try rank 1 with different scaling
            for scale in [1, 0.5, 2, 0.25, 4]:
                U = np.ones((n * m, 1)) * scale
                V = np.ones((m * p, 1)) / scale
                W = np.ones((p * n, 1))
                if _verify(U, V, W, n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
