"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = time.time() + 140

    # Schoolbook fallback: rank = n*m*p = 128
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Strategy: Split B (4x8) into two 4x4 blocks B1, B2.
    # C = A*B1 + A*B2.
    # If we have a rank R44 decomposition for 4x4*4x4 -> 4x4,
    # we can construct a rank 2*R44 decomposition for 4x4*4x8.
    # Schoolbook for 4x4x4 is 64, so schoolbook for 4x4x8 is 128.
    # We need R44 < 64 to improve.
    
    n4, m4, p4 = 4, 4, 4
    
    def verify_444(U, V, W, rank, tol=1e-9):
        for i in range(n4):
            for j in range(m4):
                for k in range(p4):
                    val = np.dot(U[i*m4+j, :], V[j*p4+k, :] * W[k*p4+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_448(U, V, W, rank, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Try to find a low-rank 4x4x4 decomposition using randomized SVD-like approach
    # The tensor T[i,j,k] = 1 for all i,j,k in 4x4x4.
    # We want to write T = sum_t U_t * V_t * W_t where each is a rank-1 tensor.
    # This is equivalent to finding U, V, W matrices such that
    # sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1
    
    # Use a structured approach: exploit the symmetry of the 4x4x4 all-ones tensor.
    # The all-ones tensor has a known low-rank decomposition.
    # For an n x n x n all-ones tensor, rank is n (using diagonal-like decomposition).
    # Actually, let's think more carefully.
    
    # The tensor T_{ijk} = 1 can be written as:
    # T_{ijk} = sum_{t=1}^{4} delta_{i,t} * delta_{j,t} * delta_{k,t} ... no that gives 1 only when i=j=k.
    
    # Better: T_{ijk} = sum_{t=1}^{4} (1/sqrt(4)) * (1/sqrt(4)) * (1/sqrt(4)) * ... no.
    
    # Actually, the simplest: T_{ijk} = 1 = (1/4) * sum_{t=1}^{4} 1 * 1 * 1 * ... 
    # No, that's not right either. Let me think again.
    
    # We need: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all i,j,k.
    
    # One approach: Use the fact that the all-ones tensor of size 4x4x4 has rank 4.
    # Proof: Let U[i,j,t] = 1 if i == t else 0, V[j,k,t] = 1 if j == t else 0, W[k,i,t] = 1 if k == t else 0.
    # Then sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1 if there exists t such that i=t, j=t, k=t, i.e., i=j=k.
    # That gives 1 only when i=j=k, not always.
    
    # Let's try: U[i,j,t] = 1/sqrt(4) for all i,j,t. Then sum_t (1/4)*(1/4)*(1/4) = 4*(1/64) = 1/16. Not 1.
    
    # Actually, let U[i,j,t] = a, V[j,k,t] = b, W[k,i,t] = c for all indices.
    # Then sum_t a*b*c = R*a*b*c = 1, so we need R*a*b*c = 1.
    # We can set a = b = c = (1/R)^(1/3). This works for any R!
    # But we want minimal R.
    
    # Wait, this gives a valid decomposition for ANY rank R >= 1!
    # U[i,j,t] = (1/R)^(1/3), V[j,k,t] = (1/R)^(1/3), W[k,i,t] = (1/R)^(1/3)
    # sum_t (1/R)^(1/3) * (1/R)^(1/3) * (1/R)^(1/3) = R * (1/R) = 1. Yes!
    
    # So the rank-1 decomposition works! But wait, that would mean the rank is 1 for any all-ones tensor.
    # Let me verify: T_{ijk} = sum_{t=1}^{R} U_{ij,t} V_{jk,t} W_{ki,t}
    # If U_{ij,t} = V_{jk,t} = W_{ki,t} = R^{-1/3} for all i,j,k,t,
    # then T_{ijk} = sum_{t=1}^R R^{-1/3} * R^{-1/3} * R^{-1/3} = R * R^{-1} = 1. Correct!
    
    # So for 4x4x4, rank 1 suffices! Then for 4x4x8, rank 2 suffices!
    
    # Let's construct this:
    R44 = 1
    R448 = 2 * R44  # = 2
    
    # 4x4x4 decomposition with rank 1:
    alpha = 1.0  # since R=1, alpha = 1
    U44 = np.ones((n4*m4, R44)) * alpha
    V44 = np.ones((m4*p4, R44)) * alpha
    W44 = np.ones((p4*n4, R44)) * alpha
    
    # Verify 4x4x4
    if verify_444(U44, V44, W44, R44):
        # Construct 4x4x8 decomposition
        U_448 = np.zeros((n * m, R448))
        V_448 = np.zeros((m * p, R448))
        W_448 = np.zeros((p * n, R448))
        
        # Term 1 (B1, k=0..3):
        U_448[:, 0] = U44[:, 0]
        for j in range(m4):
            for k in range(4):
                V_448[j*8+k, 0] = V44[j*4+k, 0]
            for k in range(4):
                W_448[k*4+i, 0] = W44[k*4+i, 0]
        
        # Wait, I need to be more careful. Let me redo.
        # For t in 0..R44-1 (just t=0 here):
        # U_448[i*4+j, 0] = U44[i*4+j, 0]
        # V_448[j*8+k, 0] = V44[j*4+k, 0] for k in 0..3 (B1)
        # V_448[j*8+k, 0] = 0 for k in 4..7
        # W_448[k*4+i, 0] = W44[k*4+i, 0] for k in 0..3
        # W_448[k*4+i, 0] = 0 for k in 4..7
        
        # For t in R44..2R44-1 (just t=1 here):
        # U_448[i*4+j, 1] = U44[i*4+j, 0]
        # V_448[j*8+k, 1] = 0 for k in 0..3
        # V_448[j*8+k, 1] = V44[j*4+(k-4), 0] for k in 4..7 (B2)
        # W_448[k*4+i, 1] = 0 for k in 0..3
        # W_448[k*4+i, 1] = W44[(k-4)*4+i, 0] for k in 4..7
        
        for i in range(4):
            for j in range(4):
                U_448[i*4+j, 0] = U44[i*4+j, 0]
                U_448[i*4+j, 1] = U44[i*4+j, 0]
        
        for j in range(4):
            for k in range(4):
                V_448[j*8+k, 0] = V44[j*4+k, 0]
                V_448[j*8+(k+4), 1] = V44[j*4+k, 0]
        
        for k in range(4):
            for i in range(4):
                W_448[k*4+i, 0] = W44[k*4+i, 0]
                W_448[(k+4)*4+i, 1] = W44[k*4+i, 0]
        
        if verify_448(U_448, V_448, W_448, R448):
            best_U, best_V, best_W = U_448, V_448, W_448
            best_rank = R448

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
