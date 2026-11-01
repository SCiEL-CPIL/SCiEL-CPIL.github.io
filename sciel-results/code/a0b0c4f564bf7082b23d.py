"""Matrix multiplication 2x2x11 rank-minimal decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x2x11 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Fallback: schoolbook
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try a structured approach: 
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [0,2)x[0,2)x[0,11)
    # We want T = sum_r u_r v_r w_r where u_r in R^4, v_r in R^22, w_r in R^22
    
    # Key idea: Use a "factorization" where we group the k-index.
    # For each pair (i,j), we need sum_k... no, each (i,j,k) is separate.
    
    # Actually, let's think about it as: for fixed (i,j), the vector over k is all 1s.
    # So T[i,j,:] = (1,1,...,1) for each (i,j).
    
    # A rank-1 term u v w contributes to T[i,j,k] by u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    # Let's try a construction where we use p "main" terms plus some "correction" terms.
    
    # Construction attempt: 
    # Use p terms where term r handles the k=r slice primarily.
    # Term r: u_r has 1s at positions (i*m+j) for all i,j; v_r has 1 at position (j*p+r) for all j; 
    #          w_r has 1 at position (r*n+i) for all i.
    # But this gives sum over r of u[i*m+j]*v[j*p+r]*w[r*n+i] which is not 1.
    
    # Let me try a different structured approach.
    # For the 2x2xp case, there's a known result that rank is 2p for p>=3 (I think).
    # Let's try to construct rank 2p = 22.
    
    # Rank 22 construction attempt:
    # We need 22 terms. Let's try to make each term "efficient".
    
    # Idea: For each k in 0..10, use 2 terms.
    # Term (k,0): contributes to entries where the k-index is k
    # Term (k,1): also contributes to entries where the k-index is k
    
    # For term (k,a) where a in {0,1}:
    # u_{k,a}[i*m+j] = alpha_{i,j,a}
    # v_{k,a}[j*p+k] = beta_{j,a}  (only nonzero at k-position for each j)
    # v_{k,a}[j*p+k'] = 0 for k' != k
    # w_{k,a}[k*n+i] = gamma_{i,a}  (only nonzero at k-row for each i)
    # w_{k,a}[k'*n+i] = 0 for k' != k
    
    # Then T[i,j,k] = sum_a alpha_{i,j,a} * beta_{j,a} * gamma_{i,a} = 1
    # And T[i,j,k'] = 0 for k' != k (automatically satisfied)
    
    # So we need: for each (i,j), sum_{a=0,1} alpha_{i,j,a} * beta_{j,a} * gamma_{i,a} = 1
    # This is a system of 4 equations (for 4 (i,j) pairs) with variables alpha, beta, gamma.
    
    # Let's try: alpha_{i,j,a} = 1, beta_{j,a} = b_{j,a}, gamma_{i,a} = g_{i,a}
    # Then sum_a b_{j,a} * g_{i,a} = 1 for all i,j.
    # This is a 2x2 matrix B (rows j, cols a) times G^T (rows i, cols a) = ones matrix.
    # B is 2x2, G is 2x2. We need B * G^T = [[1,1],[1,1]].
    # Let B = [[b00, b01],[b10, b11]], G = [[g00, g01],[g10, g11]]
    # B*G^T = [[b00*g00+b01*g10, b00*g01+b01*g11],
    #           [b10*g00+b11*g10, b10*g01+b11*g11]] = [[1,1],[1,1]]
    # Simple solution: b00=1, b01=0, b10=0, b11=1, g00=1, g01=1, g10=1, g11=1
    # Check: B*G^T = [[1*1+0*1, 1*1+0*1],[0*1+1*1, 0*1+1*1]] = [[1,1],[1,1]] ✓
    
    # So with alpha=1, this works! Rank = 2*p = 22.
    
    rank = 2 * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each k, two terms: a=0 and a=1
    for k in range(p):
        for a in range(2):
            r = k * 2 + a  # rank index
            
            # u_{k,a}[i*m+j] = 1 for all i,j
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = 1
            
            # v_{k,a}[j*p+k] = b_{j,a}, 0 elsewhere
            # b_{j,a}: from above, b_{0,0}=1, b_{0,1}=0, b_{1,0}=0, b_{1,1}=1
            b_val = 1 if (j == a) else 0
            for j in range(m):
                V[j * p + k][r] = b_val
                # Other k' positions are 0 (already initialized)
            
            # w_{k,a}[k*n+i] = g_{i,a}, 0 elsewhere
            # g_{i,a}: from above, g_{0,0}=1, g_{0,1}=1, g_{1,0}=1, g_{1,1}=1
            g_val = 1  # all g's are 1
            for i in range(n):
                W[k * n + i][r] = g_val
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return schoolbook()
# EVOLVE-BLOCK-END
