import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    deadline = time.time() + 140.0
    
    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomposition(U, V, W):
        """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T (up to tolerance)"""
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=1e-9)
    
    def to_lists(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Start with the known rank-42 block-Strassen decomposition
    def make_rank42():
        U = np.zeros((n*m, 42))
        V = np.zeros((m*p, 42))
        W = np.zeros((p*n, 42))
        
        # Strassen 7-term decomposition for 2x2 matrix multiplication
        # For matrices X (2x2) and Y (2x2), product Z = XY:
        # P1 = (X00 + X11) * (Y00 + Y11)
        # P2 = (X10 + X11) * Y00
        # P3 = X00 * (Y01 - Y11)
        # P4 = X11 * (Y10 - Y00)
        # P5 = (X00 + X01) * Y11
        # P6 = (X10 - X00) * (Y00 + Y01)
        # P7 = (X01 - X11) * (Y10 + Y11)
        # Z00 = P1 + P4 - P5 + P7
        # Z01 = P3 + P5
        # Z10 = P2 + P4
        # Z11 = P1 - P2 + P3 + P6
        
        U_strassen = [
            {(0, 0): 1, (1, 1): 1},      # P1
            {(1, 0): 1, (1, 1): 1},      # P2
            {(0, 0): 1},                  # P3
            {(1, 1): 1},                  # P4
            {(0, 0): 1, (0, 1): 1},      # P5
            {(1, 0): 1, (0, 0): -1},     # P6
            {(0, 1): 1, (1, 1): -1}      # P7
        ]
        
        V_strassen = [
            {(0, 0): 1, (1, 1): 1},      # P1
            {(0, 0): 1},                  # P2
            {(0, 1): 1, (1, 1): -1},     # P3
            {(1, 0): 1, (0, 0): -1},     # P4
            {(1, 1): 1},                  # P5
            {(0, 0): 1, (0, 1): 1},      # P6
            {(1, 0): 1, (1, 1): 1}       # P7
        ]
        
        W_strassen = [
            {(0, 0): 1, (1, 1): 1},      # P1: Z00(+1), Z11(+1)
            {(0, 1): 1, (1, 1): -1},     # P2: Z10(+1), Z11(-1)
            {(1, 0): 1, (1, 1): 1},      # P3: Z01(+1), Z11(+1)
            {(0, 0): 1, (0, 1): 1},      # P4: Z00(+1), Z10(+1)
            {(0, 0): -1, (1, 0): 1},     # P5: Z00(-1), Z01(+1)
            {(1, 1): 1},                 # P6: Z11(+1)
            {(0, 0): 1}                  # P7: Z00(+1)
        ]
        
        col = 0
        j_groups = [(0, 1), (2, 3)]
        k_groups = [(0, 1), (2, 3), (4, 5)]
        
        for jg in j_groups:
            j_base = jg[0]
            for kg in k_groups:
                k_base = kg[0]
                for s in range(7):
                    for (i_loc, j_loc), coeff in U_strassen[s].items():
                        i = i_loc
                        j = j_base + j_loc
                        u_idx = i * m + j
                        U[u_idx, col] += coeff
                    
                    for (j_loc, k_loc), coeff in V_strassen[s].items():
                        j = j_base + j_loc
                        k = k_base + k_loc
                        v_idx = j * p + k
                        V[v_idx, col] += coeff
                    
                    for (k_loc, i_loc), coeff in W_strassen[s].items():
                        k = k_base + k_loc
                        i = i_loc
                        w_idx = k * n + i
                        W[w_idx, col] += coeff
                    
                    col += 1
        
        return U, V, W
    
    U42, V42, W42 = make_rank42()
    
    # Try to find a lower rank decomposition
    def try_reduce(rank, max_trials=20):
        """Try to find a rank-r decomposition using iterative refinement"""
        if time.time() > deadline - 20:
            return None
        
        T_flat = T.flatten()
        
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                return None
            
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            best_err = float('inf')
            
            for it in range(500):
                if time.time() > deadline - 20:
                    break
                
                # Fix U and V, solve for W
                # For each (i,j): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
                # For fixed (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
                
                # Build linear system for W
                # Unknowns: W[k*n+i, r] for all k,i,r
                # Equation (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
                
                # This is linear in W. For each (i,j,k), we have one equation.
                # Total: n*m*p equations in p*n*rank unknowns
                
                # Coefficient matrix A: (n*m*p, p*n*rank)
                # A[(i,j,k), (k',i',r)] = U[i*m+j, r] * V[j*p+k', r] if k==k' else 0
                # Wait, that's not right. Let me reconsider.
                
                # For equation (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
                # The unknown is W[k*n+i, r] for r=0..rank-1
                # So for fixed (i,j,k), the coefficient of W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
                
                # So we can solve for each (i,j,k) independently? No, because W[k*n+i, r] appears in multiple equations.
                
                # Actually, for fixed k and i, W[k*n+i, r] appears in equations for all j.
                # So let's group by (k,i): for each (k,i), we have m equations (one for each j):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i] for j=0..m-1
                
                # This is a linear system in W[k*n+i, :] (a vector of length rank)
                # Coefficient matrix: (m, rank) with A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # RHS: b[j] = T[i*m+j, j*p+k, k*n+i]
                
                # Solve: A * w = b where w = W[k*n+i, :]
                
                solved = True
                for k in range(p):
                    for i in range(n):
                        # Build A: (m, rank)
                        A_sub = np.zeros((m, rank))
                        b_sub = np.zeros(m)
                        for j in range(m):
                            for r in range(rank):
                                A_sub[j, r] = U[i*m+j, r] * V[j*p+k, r]
                            b_sub[j] = T[i*m+j, j*p+k, k*n+i]
                        
                        # Solve A_sub * w = b_sub
                        try:
                            w_sol, residuals, rank_sub, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                            W[k*n+i, :] = w_sol
                        except:
                            solved = False
                            break
                    if not solved:
                        break
                
                if not solved:
                    break
                
                # Check error
                reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.linalg.norm(reconstructed - T)
                
                if err < best_err:
                    best_err = err
                    if err < 1e-9:
                        return U.copy(), V.copy(), W.copy()
                
                # If error is not decreasing, break early
                if it > 50 and err > best_err * 0.99:
                    break
            
            if best_err < 1e-9:
                return U.copy(), V.copy(), W.copy()
        
        return None
    
    # Try ranks from 41 down to 30
    best_U, best_V, best_W = None, None, None
    best_rank = 42
    
    for rank in range(41, 29, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_reduce(rank, max_trials=10)
        if result is not None:
            U_r, V_r, W_r = result
            if check_decomposition(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = rank
                break
    
    if best_U is not None:
        return to_lists(best_U, best_V, best_W)
    else:
        return to_lists(U42, V42, W42)
