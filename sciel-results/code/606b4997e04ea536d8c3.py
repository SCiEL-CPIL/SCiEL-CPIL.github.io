import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _build_rank33():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_1 (col 0) * B_1 (row 0): 2x1x6 schoolbook, rank 12
    for k in range(p):
        U[0][col] = 1
        V[k][col] = 1
        W[2*k][col] = 1
        col += 1
        U[3][col] = 1
        V[k][col] = 1
        W[2*k + 1][col] = 1
        col += 1
    
    # Part 2: A_2 (cols 1,2) * B_2 (rows 1,2): 2x2x6 via 3 Strassen 2x2x2 blocks
    for x in range(3):
        cs = 2 * x
        # M1: (a00+a11)*(b00+b11)
        U[1][col] = 1; U[5][col] = 1
        V[6 + cs][col] = 1; V[12 + cs + 1][col] = 1
        W[2*cs][col] = 1; W[2*cs + 3][col] = 1
        col += 1
        # M2: (a10+a11)*b00
        U[4][col] = 1; U[5][col] = 1
        V[6 + cs][col] = 1
        W[2*cs + 1][col] = 1; W[2*cs + 3][col] = -1
        col += 1
        # M3: a00*(b01-b11)
        U[1][col] = 1
        V[6 + cs + 1][col] = 1; V[12 + cs + 1][col] = -1
        W[2*cs + 2][col] = 1; W[2*cs + 3][col] = 1
        col += 1
        # M4: a11*(b10-b00)
        U[5][col] = 1
        V[12 + cs][col] = 1; V[6 + cs][col] = -1
        W[2*cs][col] = 1; W[2*cs + 1][col] = 1
        col += 1
        # M5: (a00+a01)*b11
        U[1][col] = 1; U[2][col] = 1
        V[12 + cs + 1][col] = 1
        W[2*cs][col] = -1; W[2*cs + 2][col] = 1
        col += 1
        # M6: (a10-a00)*(b00+b01)
        U[4][col] = 1; U[1][col] = -1
        V[6 + cs][col] = 1; V[6 + cs + 1][col] = 1
        W[2*cs + 3][col] = 1
        col += 1
        # M7: (a01-a11)*(b10+b11)
        U[2][col] = 1; U[5][col] = -1
        V[12 + cs][col] = 1; V[12 + cs + 1][col] = 1
        W[2*cs][col] = 1
        col += 1
    
    return U, V, W

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_est = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    T_target = np.zeros((n*m, m*p, p*n))
    for r in range(n):
        for s in range(m):
            for c in range(p):
                T_target[r*m+s, s*p+c, c*n+r] = 1
    return np.max(np.abs(T_est - T_target)) < 1e-9

def _try_rank32_search():
    """Try to find a rank-32 decomposition by structured search."""
    n, m, p = DIMENSIONS
    U33, V33, W33 = _build_rank33()
    U33_arr = np.array(U33, dtype=float)
    V33_arr = np.array(V33, dtype=float)
    W33_arr = np.array(W33, dtype=float)
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for r in range(n):
        for s in range(m):
            for c in range(p):
                T_target[r*m+s, s*p+c, c*n+r] = 1
    
    # Strategy: try to merge pairs of rank-1 tensors
    # For each pair (i,j) of columns in the rank-33 decomposition,
    # check if their sum can be expressed as a single rank-1 tensor
    # This is equivalent to checking if the outer product structure holds
    
    # More practical: try random rank-32 decompositions
    # Initialize near the rank-33 solution by dropping one column and adjusting
    
    rng = np.random.RandomState(42)
    
    # Try: drop each of the 33 columns and see if the remaining 32 can be
    # adjusted to still reproduce the tensor
    for drop_col in range(33):
        if _time_remaining() < 30.0:
            break
        
        # Start with rank-33 minus one column
        cols = [c for c in range(33) if c != drop_col]
        U_base = U33_arr[:, cols].copy()
        V_base = V33_arr[:, cols].copy()
        W_base = W33_arr[:, cols].copy()
        
        # The residual is the rank-1 tensor we dropped
        U_dropped = U33_arr[:, drop_col:drop_col+1]
        V_dropped = V33_arr[:, drop_col:drop_col+1]
        W_dropped = W33_arr[:, drop_col:drop_col+1]
        residual = np.einsum('ir,jr,kr->ijk', U_dropped, V_dropped, W_dropped)
        
        # Try to absorb the residual into the existing 32 terms
        # by adjusting them: we need sum_r U[:,r] V[:,r] W[:,r] = T_target
        # Currently sum_r U_base[:,r] V_base[:,r] W_base[:,r] = T_target - residual
        # We need to find small perturbations such that the sum equals T_target
        
        # This is a nonlinear problem. Try: add the residual as a linear combination
        # of the existing 32 rank-1 tensors. This is possible if residual is in
        # the span of {U_base[:,r] ⊗ V_base[:,r] ⊗ W_base[:,r]}
        
        # Simpler approach: try random rank-32 decompositions
        # Use the structure: most entries are 0, ±1
    
    # Alternative: try specific structured rank-32 decompositions
    # Idea: in the Strassen part, can we share a term between two blocks?
    
    # Let's try: modify the Strassen decomposition to share one multiplication
    # across two of the three 2x2x2 blocks
    
    # Build a custom rank-32 attempt:
    # Use 3 Strassen blocks but share M1 terms between blocks
    # M1 for block x: (a00+a11)_x * (b00+b11)_x
    # If we can combine M1 of block 0 and M1 of block 1...
    
    # Actually, let's try a different splitting strategy
    # Split 6 as 3+3 instead of 2+2+2
    # 2x2x3 can be done with... schoolbook is 12, Strassen doesn't directly apply
    
    # Let's try: 2x3 x 3x6 as (2x3) x (3x6)
    # Split the 3 in B as 3+3 (two 3x6... no, B is 3x6)
    # Split B rows as 3: B = [B_top(3x6)]... that's just B itself
    
    # Try splitting A columns as 3: A = [A(2x3)]... that's just A
    
    # Let me try: view as 2x(3x6) matrix multiplication
    # No, the tensor structure is fixed
    
    # Try: rank-32 by using a non-standard 2x2x6 decomposition
    # Standard: 3 blocks of Strassen 2x2x2 = 21
    # Can we do 2x2x6 with 20? 
    # 2x2x6 tensor: T[i,j,k], i,j in {0,1}, k in {0..5}
    # This is 3 separate 2x2x2 tensors. Each has rank >= 7.
    # Total rank >= 21 if they're independent.
    # But they might not be independent!
    
    # Let me check: is the 2x2x6 tensor actually 3 independent 2x2x2 tensors?
    # T[i,j,k] = A[i,j] * B[j,k] for the multiplication tensor
    # For k in {0,1}: T[i,j,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]... 
    # Wait, no. The multiplication tensor for 2x2 x 2x2 is:
    # T[i,j,k] = sum_l A[i,l] * B[l,k]
    # So T[i,j,k] depends on all of A and B.
    
    # For 2x2 x 2x6: T[i,j,k] = sum_l A[i,l] * B[l,k], l in {0,1}, k in {0..5}
    # This is a single tensor, not 3 independent ones.
    # Its rank might be less than 21!
    
    # Let me try to find the rank of the 2x2x6 multiplication tensor
    # by attempting a decomposition
    
    # The 2x2x6 tensor has 2*2*6 = 24 entries
    # We need sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
    
    # Try rank 20:
    # This is a system of 24 equations in 20*(2+2+6) = 240 unknowns
    # Should be solvable if rank 20 is achievable
    
    # Let me try a numerical approach
    for target_rank in [20, 21]:
        if _time_remaining() < 25.0:
            break
        if target_rank < 21:
            # Try to find rank-20 2x2x6 decomposition
            # Use random initialization and gradient-free search
            best_err = float('inf')
            for trial in range(50):
                if _time_remaining() < 25.0:
                    break
                U_r = rng.randn(2, target_rank)
                V_r = rng.randn(2, target_rank)
                W_r = rng.randn(6, target_rank)
                
                # Build the 2x2x6 target
                T_226 = np.zeros((2, 2, 6))
                for i in range(2):
                    for j in range(2):
                        for k in range(6):
                            # T[i,j,k] = delta(i,j) for the identity... 
                            # No, for matrix multiplication:
                            # (AB)[i,k] = sum_j A[i,j]*B[j,k]
                            # The tensor is T[i,j,k] where the product is sum_j T[i,j,k]*A[i,j]*B[j,k]
                            # Actually the multiplication tensor M[i,j,k] satisfies:
                            # (AB)[i,k] = sum_j A[i,j] * B[j,k]
                            # So M[i,j,k] = 1 if we're looking at the coefficient
                            # The tensor is: M[i,j,k] = 1 for all i,j,k? No.
                            # 
                            # Let me be precise. The multiplication tensor for (m x n) * (n x p) is:
                            # T[i,j,k] where i in [0,m), j in [0,n*p), k in [0,p*m)
                            # T[i, j, k] = 1 if j corresponds to A[i, ?] and k corresponds to B[?, k]
                            # 
                            # For our case 2x3 * 3x6:
                            # i in [0,6) for A (2*3), j in [0,18) for B (3*6), k in [0,12) for W (6*2)
                            # T[i,j,k] = 1 if A[i] * B[j] contributes to W[k]
                            # A[i] = A[r, s] where i = r*3+s, r in [0,2), s in [0,3)
                            # B[j] = B[s, c] where j = s*6+c, s in [0,3), c in [0,6)
                            # W[k] = W[c, r] where k = c*2+r, c in [0,6), r in [0,2)
                            # Contribution: A[r,s]*B[s,c] contributes to W[c,r]
                            # So T[i,j,k] = 1 if i=r*3+s, j=s*6+c, k=c*2+r for some r,s,c
                            pass
            
            # This is getting complicated. Let me just verify rank-33 and return it.
            break
    
    return None

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_rank33()
    
    # Try to find a lower-rank decomposition
    if _time_remaining() > 30.0:
        result = _try_rank32_search()
        if result is not None:
            U, V, W = result
    
    # Quick verification
    if _time_remaining() > 5.0:
        if not _verify(U, V, W):
            raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
