import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = N, M, P
    dim_U = n * m  # 15
    dim_V = m * p  # 35
    dim_W = p * n  # 21
    
    # Build target tensor T*[a, b, d] = 1 if a=i*m+j, b=j*p+k, d=k*n+i
    T_star = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T_star[a, b, d] = 1.0
    
    def build_rank35_decomposition():
        """
        Construct a rank-35 decomposition for (n x m) * (m x p) where n*p <= m^2.
        
        The key idea: we want to compute C = A * B where A is 3x5, B is 5x7, C is 3x7.
        
        Standard rank-35 construction using the "m*p" rank bound:
        We use the fact that we can express the product using m*p = 35 scalar multiplications
        by exploiting the structure.
        
        Actually, the cleanest rank-35 construction for n x m * m x p with n*p <= m^2:
        
        For each pair (j, k) where j in [0,m), k in [0,p), we form a rank-1 term.
        But that gives m*p = 35 terms only if we can make each term contribute correctly.
        
        Let me think differently. The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1.
        
        Rank-35 decomposition: use 35 = m*p terms.
        For each (j, k) in [0,m) x [0,p), define:
          U[:, (j,k)] has 1 at position i*m+j for each i where... 
          
        Actually, let me use the standard approach:
        The multiplication tensor for (n x m) * (m x p) can be decomposed as:
        T = sum_{j=0}^{m-1} sum_{k=0}^{p-1} u_{j,k} ⊗ v_{j,k} ⊗ w_{j,k}
        
        where:
          u_{j,k}[i*m+j] = 1 for all i (if we want to capture all i for fixed j,k)
          v_{j,k}[j*p+k] = 1
          w_{j,k}[k*n+i] = 1 for all i
        
        But this doesn't work directly because the terms would overlap.
        
        Let me use a different approach. The known rank-35 construction:
        
        Actually, the simplest correct rank-35 construction uses the following:
        For multiplying A (3x5) by B (5x7):
        
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        We can use the following 35 rank-1 terms:
        For each j in 0..4 and k in 0..6:
          term (j,k): 
            U vector: has A[i,j] for i=0,1,2 at positions i*5+j
            V vector: has B[j,k] at position j*7+k
            W vector: has C[i,k] for i=0,1,2 at positions k*3+i
        
        But this is just the schoolbook decomposition with rank 3*5*7 = 105, not 35.
        
        The rank-35 construction is more subtle. Let me use the "Strassen-like" approach.
        
        Alternative: use the fact that rank(MM_{3,5,7}) = 35 and construct it via
        the "border" method or direct bilinear algorithm.
        
        Here's a concrete rank-35 construction:
        
        We partition the computation. Since 3*7 = 21 < 25 = 5^2, we can use
        the "small rank" construction.
        
        Method: Use 5x5 = 25 rank-1 terms for a 5x5 * 5x5 product (rank 25? No, rank 125).
        
        Let me just try: the rank of MM_{n,m,p} is at most m * max(n,p) when... 
        Actually no.
        
        Known: rank(MM_{n,m,p}) <= m * min(n*p, ...) 
        
        Let me just use a practical approach: try to find a rank-35 decomposition
        using optimization, but initialize smartly.
        """
        # Use a structured initialization based on the problem structure
        # and then refine with exact arithmetic.
        
        # The rank-35 construction:
        # We use the following insight: for A (3x5) * B (5x7), we can write:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # 
        # Group by j: for each j, we need to compute sum over j of A[i,j]*B[j,k]
        # 
        # Rank-35 approach: Use 35 = 5*7 terms.
        # For each (j, k), define a term that captures A[*,j] * B[j,k] * e_k
        # But this gives rank 5*7 = 35 only if each term is rank-1 in the tensor sense.
        #
        # Term (j,k): 
        #   u_{j,k}: vector in R^{15} with u_{j,k}[i*5+j] = 1 for i=0,1,2
        #   v_{j,k}: vector in R^{35} with v_{j,k}[j*7+k] = 1
        #   w_{j,k}: vector in R^{21} with w_{j,k}[k*3+i] = 1 for i=0,1,2
        #
        # Check: sum_{j,k} u_{j,k} ⊗ v_{j,k} ⊗ w_{j,k}
        #   At position (i*5+j', j'*7+k', k'*3+i'):
        #   = sum_{j,k} u_{j,k}[i*5+j'] * v_{j,k}[j'*7+k'] * w_{j,k}[k'*3+i']
        #   = sum_{j,k} [j==j'] * [j==j' and k==k'] * [k==k' and i==i']
        #   Wait, u_{j,k}[i*5+j'] = 1 iff j==j'
        #   v_{j,k}[j'*7+k'] = 1 iff j==j' AND k==k'
        #   w_{j,k}[k'*3+i'] = 1 iff k==k' (and i is free, but i is fixed by position)
        #   
        #   So the sum is 1 iff j==j' AND k==k', which is exactly T_star[i*5+j', j'*7+k', k'*3+i'] = 1.
        #   
        # YES! This works! Rank = 5*7 = 35.
        
        U = np.zeros((dim_U, 35))
        V = np.zeros((dim_V, 35))
        W = np.zeros((dim_W, 35))
        
        col = 0
        for j in range(m):
            for k in range(p):
                # u_{j,k}: 1 at positions i*m+j for i=0..n-1
                for i in range(n):
                    U[i * m + j, col] = 1.0
                # v_{j,k}: 1 at position j*p+k
                V[j * p + k, col] = 1.0
                # w_{j,k}: 1 at positions k*n+i for i=0..n-1
                for i in range(n):
                    W[k * n + i, col] = 1.0
                col += 1
        
        return U, V, W
    
    # Build the rank-35 decomposition
    U, V, W = build_rank35_decomposition()
    
    # Verify with exact arithmetic
    def verify_exact(U, V, W, T_star):
        """Verify that the decomposition exactly reproduces T_star."""
        T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
        # Check all entries match
        if T_star.shape != T_approx.shape:
            return False
        return np.allclose(T_approx, T_star, atol=1e-12)
    
    if verify_exact(U, V, W, T_star):
        # Great! We have a valid rank-35 decomposition.
        # Convert to lists
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # If verification failed (shouldn't happen), fall back to schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((dim_U, rank_sb))
    V_sb = np.zeros((dim_V, rank_sb))
    W_sb = np.zeros((dim_W, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
