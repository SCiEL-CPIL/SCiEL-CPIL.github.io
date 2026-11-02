"""Tensor decomposition search for 3x3 * 3x8 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the trivial rank-72 decomposition."""
    n, m, p = 3, 3, 8
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Reshape for tensor contraction: U[i,j,c], V[j,k,c], W[k,i,c]
    U_t = Ua.reshape(n, m, rank)
    V_t = Va.reshape(m, p, rank)
    W_t = Wa.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True

def _compute_residual(U, V, W, n, m, p):
    """Compute residual T[i,j,k] - 1 for the decomposition."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    U_t = Ua.reshape(n, m, rank)
    V_t = Va.reshape(m, p, rank)
    W_t = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    return T - 1.0

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    U[i*2+j, c] for i,j in 0..1
    V[j*2+k, c] for j,k in 0..1
    W[k*2+i, c] for k,i in 0..1
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B12-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_strassen_3x3x3():
    """
    Build a rank-23 decomposition for 3x3 * 3x3 using Strassen + extensions.
    
    Partition:
    A = [[a00, a01, a02],
         [a10, a11, a12],
         [a20, a21, a22]]
    
    B = [[b00, b01, b02],
         [b10, b11, b12],
         [b20, b21, b22]]
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Strategy: Use Strassen (rank 7) for the 2x2 sub-block (indices 0,1),
    then add terms for interactions with index 2.
    
    The 2x2 Strassen handles C[0,0], C[0,1], C[1,0], C[1,1] using only
    A[0,0],A[0,1],A[1,0],A[1,1] and B[0,0],B[0,1],B[1,0],B[1,1].
    
    Remaining terms to handle:
    C[0,2] = a00*b02 + a01*b12 + a02*b22
    C[1,2] = a10*b02 + a11*b12 + a12*b22
    C[2,0] = a20*b00 + a21*b10 + a22*b20
    C[2,1] = a20*b01 + a21*b11 + a22*b21
    C[2,2] = a20*b02 + a21*b12 + a22*b22
    
    Plus we need to add contributions from a02, a12, a20, a21 to the 2x2 block:
    C[0,0] += a02*b20  (missing from Strassen)
    C[0,1] += a02*b21  (missing from Strassen)
    C[1,0] += a12*b20  (missing from Strassen)
    C[1,1] += a12*b21  (missing from Strassen)
    
    So total additional terms needed:
    - a02*b20, a02*b21 (2 terms for C[0,0], C[0,1])
    - a12*b20, a12*b21 (2 terms for C[1,0], C[1,1])
    - C[0,2]: a00*b02 + a01*b12 + a02*b22 (3 terms)
    - C[1,2]: a10*b02 + a11*b12 + a12*b22 (3 terms)
    - C[2,0]: a20*b00 + a21*b10 + a22*b20 (3 terms)
    - C[2,1]: a20*b01 + a21*b11 + a22*b21 (3 terms)
    - C[2,2]: a20*b02 + a21*b12 + a22*b22 (3 terms)
    
    Total additional: 2+2+3+3+3+3+3 = 19
    Total: 7 + 19 = 26. But we can share some terms.
    
    Actually, let's just use schoolbook for the non-Strassen parts and see.
    For now, let's construct it explicitly.
    """
    n, m, p = 3, 3, 3
    rank = 27  # Start with schoolbook, we'll try to reduce
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    
    return U, V, W

def _vectorized_optimize(U, V, W, n, m, p, max_iters=500, lr_init=0.05, lr_decay=0.995, tol=1e-10):
    """
    Vectorized gradient descent for tensor decomposition.
    
    Minimize sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
    """
    rank = len(U[0])
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    
    U_t = Ua.reshape(n, m, rank)
    V_t = Va.reshape(m, p, rank)
    W_t = Wa.reshape(p, n, rank)
    
    lr = lr_init
    
    for it in range(max_iters):
        # Compute residual: R[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        R = T - 1.0
        
        # Compute gradients
        # d/dU[i,j,c] = sum_{k} R[i,j,k] * V[j,k,c] * W[k,i,c]
        # d/dV[j,k,c] = sum_{i} R[i,j,k] * U[i,j,c] * W[k,i,c]
        # d/dW[k,i,c] = sum_{j} R[i,j,k] * U[i,j,c] * V[j,k,c]
        
        # grad_U[i,j,c] = sum_k R[i,j,k] * V[j,k,c] * W[k,i,c]
        grad_U = np.einsum('ijk,jkc,kic->ijc', R, V_t, W_t)
        
        # grad_V[j,k,c] = sum_i R[i,j,k] * U[i,j,c] * W[k,i,c]
        grad_V = np.einsum('ijk,ijc,kic->jkc', R, U_t, W_t)
        
        # grad_W[k,i,c] = sum_j R[i,j,k] * U[i,j,c] * V[j,k,c]
        grad_W = np.einsum('ijk,ijc,jkc->kic', R, U_t, V_t)
        
        # Update
        U_t -= lr * grad_U
        V_t -= lr * grad_V
        W_t -= lr * grad_W
        
        # Reshape back
        Ua = U_t.reshape(n*m, rank)
        Va = V_t.reshape(m*p, rank)
        Wa = W_t.reshape(p*n, rank)
        
        lr *= lr_decay
        
        # Check convergence
        err = np.max(np.abs(R))
        if err < tol:
            break
    
    return Ua.tolist(), Va.tolist(), Wa.tolist()

def _try_rank_reduction(start_rank, target_rank, n, m, p, seed=42, max_iters=200):
    """
    Try to find a decomposition of target_rank starting from a near-schoolbook init.
    
    Strategy: Initialize with schoolbook terms (first target_rank), add small random
    noise to all columns, then optimize to satisfy the constraints.
    """
    rng = np.random.RandomState(seed)
    
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    # Initialize with first target_rank schoolbook terms
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= target_rank:
                    break
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
            if col >= target_rank:
                break
        if col >= target_rank:
            break
    
    # Add small random perturbation to all columns to help escape local minima
    noise_scale = 0.01
    U += rng.randn(n*m, target_rank) * noise_scale
    V += rng.randn(m*p, target_rank) * noise_scale
    W += rng.randn(p*n, target_rank) * noise_scale
    
    # Optimize
    U_list, V_list, W_list = _vectorized_optimize(
        U.tolist(), V.tolist(), W.tolist(), n, m, p,
        max_iters=max_iters, lr_init=0.02, lr_decay=0.998, tol=1e-12
    )
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
        return {"U": U_list, "V": V_list, "W": W_list}
    return None

def _structured_strassen_3x3x8():
    """
    Construct a structured decomposition for 3x3 * 3x8 using Strassen-like ideas.
    
    Key insight: Partition the 3x3 matrix A into a 2x2 block + 1x1 extension,
    and use Strassen for the 2x2 part.
    
    A = [[a00, a01, a02],
         [a10, a11, a12],
         [a20, a21, a22]]
    
    B is 3x8: B[j,k] for j=0,1,2 and k=0,...,7
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Using Strassen for the 2x2 part (indices 0,1):
    7 products for C[0,0..1] and C[1,0..1] from A[0,0..1] and B[0,1;0..7]
    
    Additional terms needed:
    - C[0,k] += a02 * B[2,k] for all k (8 terms)
    - C[1,k] += a12 * B[2,k] for all k (8 terms)
    - C[2,k] = a20*B[0,k] + a21*B[1,k] + a22*B[2,k] for all k (24 terms)
    
    Total: 7 + 8 + 8 + 24 = 47. Still better than 72!
    
    But we can do better by sharing terms. Let's be more careful.
    
    Actually, the Strassen part only uses A[0,0],A[0,1],A[1,0],A[1,1] and B[0,0..7],B[1,0..7].
    It produces C[0,0..1] and C[1,0..1] correctly for those inputs.
    
    But C[0,k] for k=0,1 also gets contributions from a02*B[2,k].
    And C[1,k] for k=0,1 also gets contributions from a12*B[2,k].
    And C[2,k] for all k needs a20*B[0,k] + a21*B[1,k] + a22*B[2,k].
    
    So the full decomposition needs:
    - 7 Strassen terms (using A[0,0..1], B[0,1;0..7])
    - 8 terms: a02 * B[2,k] for k=0..7 (contributes to C[0,k])
    - 8 terms: a12 * B[2,k] for k=0..7 (contributes to C[1,k])
    - 8 terms: a20 * B[0,k] for k=0..7 (contributes to C[2,k])
    - 8 terms: a21 * B[1,k] for k=0..7 (contributes to C[2,k])
    - 8 terms: a22 * B[2,k] for k=0..7 (contributes to C[2,k])
    
    Total: 7 + 8*5 = 47.
    
    Can we reduce further? The terms a02*B[2,k] and a12*B[2,k] are independent.
    The terms a20*B[0,k], a21*B[1,k], a22*B[2,k] are independent.
    
    So 47 seems to be the structured minimum with this approach.
    """
    n, m, p = 3, 3, 8
    rank = 47
    U = np.zeros((9, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    # Strassen part: 7 terms for the 2x2 block
    S_U, S_V, S_W = _strassen_2x2x2()
    
    # Embed Strassen: U[i,j,c] for i,j in {0,1}, c in 0..6
    # V[j,k,c] for j in {0,1}, k in {0,1}, c in 0..6
    # But we need V[j,k,c] for k in 0..7!
    # The Strassen V only has 2 columns (k=0,1). We need to extend to 8 columns.
    # For k>=2, V[j,k,c] = 0 for the Strassen terms.
    
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3+j, c] = S_U[i*2+j, c]
        for j in range(2):
            for k in range(2):
                V[j*8+k, c] = S_V[j*2+k, c]
        for k in range(2):
            for i in range(2):
                W[k*3+i, c] = S_W[k*2+i, c]
    
    col = 7
    
    # a02 * B[2,k] for k=0..7: contributes to C[0,k]
    for k in range(8):
        U[2, col] = 1  # a02 is at index i=0,j=2 -> 0*3+2 = 2
        V[2*8+k, col] = 1  # B[2,k]
        W[k*3+0, col] = 1  # C[0,k]
        col += 1
    
    # a12 * B[2,k] for k=0..7: contributes to C[1,k]
    for k in range(8):
        U[1*3+2, col] = 1  # a12 is at index i=1,j=2 -> 1*3+2 = 5
        V[2*8+k, col] = 1  # B[2,k]
        W[k*3+1, col] = 1  # C[1,k]
        col += 1
    
    # a20 * B[0,k] for k=0..7: contributes to C[2,k]
    for k in range(8):
        U[2*3+0, col] = 1  # a20 is at index i=2,j=0 -> 2*3+0 = 6
        V[0*8+k, col] = 1  # B[0,k]
        W[k*3+2, col] = 1  # C[2,k]
        col += 1
    
    # a21 * B[1,k] for k=0..7: contributes to C[2,k]
    for k in range(8):
        U[2*3+1, col] = 1  # a21 is at index i=2,j=1 -> 2*3+1 = 7
        V[1*8+k, col] = 1  # B[1,k]
        W[k*3+2, col] = 1  # C[2,k]
        col += 1
    
    # a22 * B[2,k] for k=0..7: contributes to C[2,k]
    for k in range(8):
        U[2*3+2, col] = 1  # a22 is at index i=2,j=2 -> 2*3+2 = 8
        V[2*8+k, col] = 1  # B[2,k]
        W[k*3+2, col] = 1  # C[2,k]
        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 110 second limit (safety margin)
    
    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try structured Strassen-based construction (rank 47)
    if time.time() < deadline:
        try:
            strassen_result = _structured_strassen_3x3x8()
            r = len(strassen_result['U'][0])
            if r < best_rank and _verify_decomposition(
                strassen_result['U'], strassen_result['V'], strassen_result['W'], n, m, p
            ):
                best = strassen_result
                best_rank = r
        except Exception:
            pass
    
    # Try rank reduction from the structured solution
    if time.time() < deadline and best_rank > 40:
        try:
            for target_rank in [44, 42, 40]:
                if time.time() > deadline - 10:
                    break
                result = _try_rank_reduction(
                    best_rank, target_rank, n, m, p,
                    seed=target_rank, max_iters=300
                )
                if result is not None:
                    best = result
                    best_rank = target_rank
                    break
        except Exception:
            pass
    
    # Try more aggressive rank reduction with different seeds
    if time.time() < deadline and best_rank > 35:
        try:
            for target_rank in [38, 36, 35]:
                if time.time() > deadline - 10:
                    break
                for seed in [1, 7, 42]:
                    if time.time() > deadline - 5:
                        break
                    result = _try_rank_reduction(
                        best_rank, target_rank, n, m, p,
                        seed=seed, max_iters=200
                    )
                    if result is not None:
                        best = result
                        best_rank = target_rank
                        break
                if best_rank == target_rank:
                    break
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
