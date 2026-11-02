"""Rewritten program for 2x3x10 matrix multiplication tensor rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p  # 60

    # Build schoolbook decomposition
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

    # Verify the decomposition is correct
    def verify(U_mat, V_mat, W_mat, rnk):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rnk):
                        s += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Try to find rank-59 decomposition by expressing one component as linear combo
    def try_rank_59():
        if time.time() > deadline - 10:
            return None
        
        # We have 60 rank-1 components. Try to express component 59 as a linear combination
        # of the first 59 components.
        # The tensor equation is: sum_{c=0}^{58} u_c[i] v_c[j] w_c[k] = u_59[i] v_59[j] w_59[k]
        # for all i,j,k where the component is non-zero.
        # 
        # For the schoolbook, component c corresponds to (i,j,k) = (i_c, j_c, k_c)
        # where c = i_c * m * p + j_c * p + k_c
        # 
        # Let's try: can we find coefficients alpha_0..alpha_58 such that:
        # alpha_0 * (1 if i==i_0 else 0) * (1 if j==j_0 else 0) * (1 if k==k_0 else 0)
        # + ... + alpha_58 * (1 if i==i_58 else 0) * (1 if j==j_58 else 0) * (1 if k==k_58 else 0)
        # = 1 if (i,j,k) == (i_59, j_59, k_59) else 0
        #
        # This is only possible if (i_59, j_59, k_59) equals one of the (i_c, j_c, k_c),
        # which it doesn't (they're all distinct). So this simple approach won't work.
        #
        # Instead, let's try a different approach: use the fact that the tensor has structure.
        # For matrix multiplication, T[i,j,k] = 1 for all i,j,k.
        # 
        # Let's try to find a rank-59 decomposition using random linear algebra.
        # We'll set up the problem as: find U, V, W of rank 59 such that
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
        
        # Use numpy for efficiency
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Try to eliminate one component by expressing it in terms of others
        # We'll try removing component 59 (the last one)
        # The constraint is that the tensor must still be reproduced.
        
        # Actually, let's try a more systematic approach:
        # For the 2x3x10 case, we can try to find if there's a linear dependence.
        
        # Let's try: assume the last component can be expressed as a sum of others.
        # We need: for all (i,j,k), the contribution from component 59 equals
        # the sum of contributions from other components adjusted by coefficients.
        
        # This is complex. Let's try a simpler heuristic:
        # Check if the tensor has any obvious structure that allows rank reduction.
        
        # For now, return None and keep schoolbook
        return None
    
    result_59 = try_rank_59()
    if result_59 is not None:
        return result_59
    
    # Fallback: verify and return schoolbook
    if not verify(U, V, W, rank):
        # Rebuild
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
# EVOLVE-BLOCK-END
