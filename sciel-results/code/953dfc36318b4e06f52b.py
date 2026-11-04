import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Build the tensor T[i,j,k] = 1 if k == j (A[i,j]*B[j,k])
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if k == j else 0.0
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m + j, r] * Va[j*p + k, r] * Wa[k*n + i, r]
                if abs(s - expected) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    random.seed(42)
    
    # Start with the known rank-33 solution
    best_U, best_V, best_W = _make_rank33(n, m, p)
    best_rank = 33
    
    # Try to find lower rank solutions via random search
    # The tensor has n*m*p = 36 entries, each being 0 or 1
    # We need rank-r decomposition
    
    # Try various ranks from 32 down
    for target_rank in range(32, 20, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            # Random initialization with small values
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Gradient descent on the reconstruction error
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 20:
                    break
                
                # Compute gradients
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # dT/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
                # dT/dV[j*p+k, r] = U[i*m+j, r] * W[k*n+i, r]
                # dT/dW[k*n+i, r] = U[i*m+j, r] * V[j*p+k, r]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            expected = 1.0 if k == j else 0.0
                            s = 0.0
                            for r in range(target_rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err = s - expected
                            for r in range(target_rank):
                                dU[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                                dV[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                                dW[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                if step % 50 == 0:
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                expected = 1.0 if k == j else 0.0
                                s = 0.0
                                for r in range(target_rank):
                                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                max_err = max(max_err, abs(s - expected))
                    if max_err < 1e-6:
                        # Found a solution, try to clean up to integer/half-integer
                        U_clean, V_clean, W_clean = _clean_solution(U, V, W, n, m, p)
                        if U_clean is not None:
                            Uc = [[round(x) if abs(x - round(x)) < 1e-6 else 2*round(x/2) for x in row] for row in U_clean]
                            Vc = [[round(x) if abs(x - round(x)) < 1e-6 else 2*round(x/2) for x in row] for row in V_clean]
                            Wc = [[round(x) if abs(x - round(x)) < 1e-6 else 2*round(x/2) for x in row] for row in W_clean]
                            if _verify(Uc, Vc, Wc, n, m, p):
                                return {"U": Uc, "V": Vc, "W": Wc}
    
    # Return the best known solution
    Uc = [[int(x) for x in row] for row in best_U]
    Vc = [[int(x) for x in row] for row in best_V]
    Wc = [[int(x) for x in row] for row in best_W]
    return {"U": Uc, "V": Vc, "W": Wc}

def _make_rank33(n, m, p):
    """Create the known rank-33 decomposition."""
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_1 (col 0) * B_1 (row 0)
    for k in range(p):
        U[0][col] = 1
        V[k][col] = 1
        W[2*k][col] = 1
        col += 1
        
        U[3][col] = 1
        V[k][col] = 1
        W[2*k + 1][col] = 1
        col += 1
    
    # Part 2: A_2 (cols 1,2) * B_2 (rows 1,2) split into 3 blocks of 2x2
    for x in range(3):
        c_start = 2 * x
        
        # Strassen M1
        U[1][col] = 1; U[5][col] = 1
        V[6 + c_start][col] = 1; V[12 + c_start + 1][col] = 1
        W[2*c_start][col] = 1
        W[2*c_start + 3][col] = 1
        col += 1
        
        # Strassen M2
        U[4][col] = 1; U[5][col] = 1
        V[6 + c_start][col] = 1
        W[2*c_start + 1][col] = 1
        W[2*c_start + 3][col] = -1
        col += 1
        
        # Strassen M3
        U[1][col] = 1
        V[6 + c_start + 1][col] = 1
        V[12 + c_start + 1][col] = -1
        W[2*c_start + 2][col] = 1
        W[2*c_start + 3][col] = 1
        col += 1
        
        # Strassen M4
        U[5][col] = 1
        V[12 + c_start][col] = 1
        V[6 + c_start][col] = -1
        W[2*c_start][col] = 1
        W[2*c_start + 1][col] = 1
        col += 1
        
        # Strassen M5
        U[1][col] = 1; U[2][col] = 1
        V[12 + c_start + 1][col] = 1
        W[2*c_start][col] = -1
        W[2*c_start + 2][col] = 1
        col += 1
        
        # Strassen M6
        U[4][col] = 1; U[1][col] = -1
        V[6 + c_start][col] = 1
        V[6 + c_start + 1][col] = 1
        W[2*c_start + 3][col] = 1
        col += 1
        
        # Strassen M7
        U[2][col] = 1; U[5][col] = -1
        V[12 + c_start][col] = 1
        V[12 + c_start + 1][col] = 1
        W[2*c_start][col] = 1
        col += 1
    
    return U, V, W

def _clean_solution(U, V, W, n, m, p):
    """Try to clean a numerical solution to integer/half-integer values."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Round to nearest half-integer
    U_clean = np.round(Ua * 2) / 2
    V_clean = np.round(Va * 2) / 2
    W_clean = np.round(Wa * 2) / 2
    
    # Verify
    rank = U_clean.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if k == j else 0.0
                s = 0.0
                for r in range(rank):
                    s += U_clean[i*m+j, r] * V_clean[j*p+k, r] * W_clean[k*n+i, r]
                if abs(s - expected) > 1e-6:
                    return None
    
    return U_clean, V_clean, W_clean
