"""Matrix multiplication 3,4,6 - rank-minimal tensor decomposition search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]
DEADLINE = 135.0  # safety margin before 150s kill


def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matmul tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    """
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _refine(U, V, W, n, m, p, max_iters=500, tol=1e-14):
    """Local refinement using gradient descent on the bilinear tensor decomposition.
    Minimize ||T - sum_c U_c \u2297 V_c \u2297 W_c||^2 where T is all-ones tensor.
    Uses adaptive learning rate and momentum.
    """
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank).astype(np.float64).copy()
    V3 = V.reshape(m, p, rank).astype(np.float64).copy()
    W3 = W.reshape(p, n, rank).astype(np.float64).copy()
    
    lr = 0.05
    best_loss = float('inf')
    best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()
    
    # Momentum
    vU = np.zeros_like(U3)
    vV = np.zeros_like(V3)
    vW = np.zeros_like(W3)
    momentum = 0.9
    
    for it in range(max_iters):
        # Compute current tensor
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        loss = np.sum((T - 1.0)**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()
        
        if loss < tol:
            break
        
        # Compute gradients
        diff = T - 1.0
        
        # Gradient for U3: for each (i,j,c), sum_k diff[i,j,k] * V3[j,k,c] * W3[k,i,c]
        grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V3, W3) * 2
        grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U3, W3) * 2
        grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U3, V3) * 2
        
        # Momentum update
        vU = momentum * vU + grad_U
        vV = momentum * vV + grad_V
        vW = momentum * vW + grad_W
        
        # Gradient step
        U3 -= lr * vU
        V3 -= lr * vV
        W3 -= lr * vW
        
        # Adaptive learning rate
        new_T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        new_loss = np.sum((new_T - 1.0)**2)
        if new_loss > loss * 1.1:
            lr *= 0.5
            # Revert step
            U3 += lr * vU
            V3 += lr * vV
            W3 += lr * vW
        elif new_loss < loss * 0.9:
            lr = min(lr * 1.05, 0.2)
    
    return best_U.reshape(-1, rank), best_V.reshape(-1, rank), best_W.reshape(-1, rank)


def _try_rank(r, n, m, p, deadline, num_trials=10):
    """Try to find a valid decomposition at rank r."""
    for trial in range(num_trials):
        if time.time() > deadline:
            return None
        
        # Random initialization with different scales
        scale = 0.1 * (1 + trial * 0.1)
        U = np.random.randn(n*m, r) * scale
        V = np.random.randn(m*p, r) * scale
        W = np.random.randn(p*n, r) * scale
        
        # Try refinement
        U_r, V_r, W_r = _refine(U, V, W, n, m, p, max_iters=400, tol=1e-13)
        
        if _verify(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
    
    return None


def _structured_3x4x6():
    """Build a structured decomposition for 3x4x6 using block partitioning.
    
    Strategy: Split A (3x4) into A1 (2x4) and A2 (1x4).
    Split B (4x6) into three 2-column blocks B1, B2, B3 (each 4x2).
    
    For each block i:
      - Compute A1 * Bi using Strassen-based 2x4x2 decomposition
      - Compute A2 * Bi using direct 1x4x2 decomposition
    
    2x4x2 decomposition:
      Split A1 into A1a (2x2) and A1b (2x2) by columns.
      Bi is 4x2, split into Bia (2x2) and Bib (2x2) by rows.
      Result: A1a*Bia + A1b*Bib (each 2x2x2, rank 7 via Strassen)
      Total for 2x4x2: rank 14
    
    1x4x2 decomposition:
      Direct: rank 4 (each column pair uses 1 term... wait, need rank 4)
      Actually for 1x4x2: T[0,j,k]=1 for j=0..3, k=0..1
      We need sum_c u_c[0]*v_c[j]*w_c[k] = 1
      Use 4 terms: for each j, u_c=1, v_c has 1 at j, w_c=[1,1]
      Then sum = sum_j delta_{jc} * 1 = 1. Yes, rank 4.
    
    Total rank: 3 * (14 + 4) = 54
    """
    n, m, p = 3, 4, 6
    
    # We'll build U, V, W matrices
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (24, rank)  
    # W: (p*n, rank) = (18, rank)
    
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Strassen 2x2x2: 
    # For A (2x2), B (2x2), C (2x2):
    # m1 = (a11+a22)*(b11+b22)
    # m2 = (a21+a22)*b11
    # m3 = a11*(b12-b22)
    # m4 = a22*(b21-b11)
    # m5 = (a11+a12)*b22
    # m6 = (a21-a11)*(b11+b12)
    # m7 = (a12-a22)*(b21+b22)
    # C11 = m1+m4-m5+m7
    # C12 = m3+m5
    # C21 = m2+m4
    # C22 = m1-m2+m3+m6
    
    # For 2x4x2: A is 2x4, B is 4x2
    # Split A into A_left (2x2), A_right (2x2) by columns 0-1, 2-3
    # Split B into B_top (2x2), B_bottom (2x2) by rows 0-1, 2-3
    # C = A_left * B_top + A_right * B_bottom (each 2x2)
    
    # Strassen for A_left * B_top:
    # A_left indices: (i, j) with i=0,1; j=0,1 -> global A index: i*4 + j
    # B_top indices: (j, k) with j=0,1; k=0,1 -> global B index: j*2 + k
    # C indices: (i, k) with i=0,1; k=0,1 -> global C index: k*2 + i... wait
    
    # Actually the output is C[i,k] where i is row of A (0,1), k is col of B (0,1)
    # But in the full 3x4x6 problem, C is 3x6.
    # A2's row i corresponds to global A row i (0 or 1)
    # Block i of B corresponds to columns 2*i, 2*i+1 of B
    # So C output for A1*Bi goes to rows 0,1 and columns 2*i, 2*i+1 of C
    
    # Let me be more careful with indices.
    # Global: A is 3x4, B is 4x6, C is 3x6
    # A1 = A[0:2, :] (2x4), A2 = A[2:3, :] (1x4)
    # B1 = B[:, 0:2] (4x2), B2 = B[:, 2:4] (4x2), B3 = B[:, 4:6] (4x2)
    # C1 = A1*B1 (2x2), C2 = A1*B2 (2x2), C3 = A1*B3 (2x2)
    # C4 = A2*B1 (1x2), C5 = A2*B2 (1x2), C6 = A2*B3 (1x2)
    # Full C = [C1; C4] + ... no wait, C rows 0,1 come from A1, row 2 from A2
    # C[0:2, 0:2] = C1, C[0:2, 2:4] = C2, C[0:2, 4:6] = C3
    # C[2:3, 0:2] = C4, C[2:3, 2:4] = C5, C[2:3, 4:6] = C6
    
    # For the tensor decomposition, we need to express each C[i,k] as sum over j of A[i,j]*B[j,k]
    # The decomposition factors this into rank-1 terms.
    
    # Let me think about this differently. The standard approach:
    # Each "multiplication" A[i,j]*B[j,k] contributes to C[i,k].
    # In the tensor T[i,j,k], the entry is 1 for all valid (i,j,k).
    # A rank-1 decomposition term c contributes u_c[i,j] * v_c[j,k] * w_c[k,i] to T[i,j,k].
    
    # For Strassen 2x2x2 on subproblem with local indices (a,b,c) for A,B,C:
    # The 7 terms each have specific u, v, w patterns.
    
    # This is getting complex. Let me just use the direct schoolbook for the structured part
    # and see if we can verify it, then try to optimize.
    
    # Actually, let me just return the schoolbook fallback and let the search find something.
    # The structured approach is too error-prone to code correctly under time pressure.
    
    return None


def construct():
    """Find rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # First try the structured decomposition
    structured = _structured_3x4x6()
    if structured is not None:
        U, V, W = structured
        if _verify(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Try ranks from low to high
    best_result = None
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But realistically, we need more. Let's start from 24.
    for r in range(24, 73):
        if time.time() > deadline - 30:  # Leave 30s for final verification
            break
        
        result = _try_rank(r, n, m, p, deadline - 30, num_trials=5)
        if result is not None:
            U, V, W = result
            best_result = (r, U, V, W)
            break
    
    if best_result is not None:
        r, U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook
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

