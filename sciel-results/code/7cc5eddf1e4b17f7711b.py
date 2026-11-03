"""Matrix multiplication 2x4x4 rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 4]


def _build_matrix_multiplication_tensor():
    """Build the 3-way tensor T[i, j, k] for matmul: A(n x m) * B(m x p) -> C(n x p).
    
    n=2, m=4, p=4.
    U index: i*m + j  (0..7)  -> n*m = 8
    V index: j*p + k  (0..15) -> m*p = 16
    W index: k*n + i  (0..7)  -> p*n = 8
    
    T[(i*m+j), (j*p+k), (k*n+i)] = 1 when all indices are consistent, 0 otherwise.
    """
    n, m, p = DIMENSIONS
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that U * V^T (element-wise contracted) reconstructs the tensor via W."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    T = _build_matrix_multiplication_tensor()
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    T_est = np.zeros((8, 16, 8), dtype=float)
    for r in range(rank):
        T_est += np.einsum('r,s,t->rst', U_np[:, r], V_np[:, r], W_np[:, r])
    if not np.allclose(T_est, T, atol=tol):
        return False
    return True


def _build_rank28_solution():
    """Build the known rank-28 solution using 4 Strassen-like blocks."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # M1: (X0+X3)(Y0+Y3)(Z0+Z3)
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2: (X2+X3)Y0(Z2-Z3)
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # M3: X0(Y1-Y3)(Z1+Z3)
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # M4: X3(Y2-Y0)(Z0+Z2)
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # M5: (X0+X1)Y3(-Z0+Z1)
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # M6: (X2-X0)(Y0+Y1)Z3
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # M7: (X1-X3)(Y2+Y3)Z0
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Block 1: i=0, j=0, k in {0,1}
    X1 = [0, 1, 4, 5]    # U indices for (i=0,j=0),(i=0,j=1),(i=1,j=0),(i=1,j=1)
    Y1 = [0, 1, 4, 5]    # V indices for (j=0,k=0),(j=0,k=1),(j=1,k=0),(j=1,k=1)
    Z1 = [0, 2, 1, 3]    # W indices for (k=0,i=0),(k=1,i=0),(k=0,i=1),(k=1,i=1)
    add_block(0, X1, Y1, Z1)

    # Block 2: i=0, j=2, k in {2,3}
    X2 = [2, 3, 6, 7]    # U indices
    Y2 = [8, 9, 12, 13]  # V indices for (j=2,k=0),(j=2,k=1),(j=3,k=0),(j=3,k=1)
    Z2 = [4, 6, 5, 7]    # W indices for (k=2,i=0),(k=3,i=0),(k=2,i=1),(k=3,i=1)
    add_block(7, X2, Y2, Z2)

    # Block 3: i=1, j=0, k in {0,1}
    X3 = [0, 1, 4, 5]    # Same U indices as block 1
    Y3 = [2, 3, 6, 7]    # V indices for (j=0,k=2),(j=0,k=3),(j=1,k=2),(j=1,k=3)
    Z3 = [4, 6, 5, 7]    # W indices for (k=2,i=0),(k=3,i=0),(k=2,i=1),(k=3,i=1)
    add_block(14, X3, Y3, Z3)

    # Block 4: i=1, j=2, k in {2,3}
    X4 = [2, 3, 6, 7]    # Same U indices as block 2
    Y4 = [10, 11, 14, 15]  # V indices for (j=2,k=2),(j=2,k=3),(j=3,k=2),(j=3,k=3)
    Z4 = [4, 6, 5, 7]    # W indices
    add_block(21, X4, Y4, Z4)

    return U, V, W


def _try_reduce_rank(U, V, W, target_rank, deadline):
    """Try to reduce the rank from current to target_rank by finding linear dependencies."""
    current_rank = len(U[0])
    if target_rank >= current_rank:
        return None
    
    # Convert to numpy arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor
    T = np.zeros((8, 16, 8), dtype=float)
    for r in range(current_rank):
        T += np.einsum('r,s,t->rst', U_np[:, r], V_np[:, r], W_np[:, r])
    
    # Try to find a rank-target_rank decomposition by random projection
    # This is a heuristic: randomly project the factor matrices to lower dimension
    # and check if the tensor is still preserved
    
    best_result = None
    attempts = 0
    max_attempts = 50000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Random linear transformation to reduce rank
        # Project U, V, W to target_rank dimensions
        P_U = np.random.randn(current_rank, target_rank)
        P_V = np.random.randn(current_rank, target_rank)
        P_W = np.random.randn(current_rank, target_rank)
        
        # Scale to maintain norm
        P_U /= np.linalg.norm(P_U) * np.sqrt(current_rank / target_rank)
        P_V /= np.linalg.norm(P_V) * np.sqrt(current_rank / target_rank)
        P_W /= np.linalg.norm(P_W) * np.sqrt(current_rank / target_rank)
        
        U_new = U_np @ P_U
        V_new = V_np @ P_V
        W_new = W_np @ P_W
        
        # Check if the tensor is preserved
        T_new = np.zeros((8, 16, 8), dtype=float)
        for r in range(target_rank):
            T_new += np.einsum('r,s,t->rst', U_new[:, r], V_new[:, r], W_new[:, r])
        
        if np.allclose(T_new, T, atol=1e-8):
            # Convert to list format
            U_list = [[float(x) for x in row] for row in U_new]
            V_list = [[float(x) for x in row] for row in V_new]
            W_list = [[float(x) for x in row] for row in W_new]
            return U_list, V_list, W_list
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Start with the known rank-28 solution
    best_U, best_V, best_W = _build_rank28_solution()
    best_rank = len(best_U[0])
    
    # Verify the rank-28 solution
    if not _verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook if verification fails
        best_rank = n * m * p
        best_U = [[0] * best_rank for _ in range(n * m)]
        best_V = [[0] * best_rank for _ in range(m * p)]
        best_W = [[0] * best_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j][col] = 1
                    best_V[j * p + k][col] = 1
                    best_W[k * n + i][col] = 1
                    col += 1
    
    # Set deadline: 130 seconds (10s margin before 140s kill)
    start_time = time.time()
    deadline = start_time + 130
    
    # Try to reduce rank from 28 down to 24
    for target_rank in range(best_rank - 1, 23, -1):
        if time.time() > deadline:
            break
        
        result = _try_reduce_rank(best_U, best_V, best_W, target_rank, deadline)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            # Continue trying to reduce further
        else:
            # If we can't reduce to this rank, try the next lower rank
            # But don't spend too much time on one rank
            if time.time() > deadline:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
