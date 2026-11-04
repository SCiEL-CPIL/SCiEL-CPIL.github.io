import numpy as np
import time
from fractions import Fraction
import itertools

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    def is_valid_exact(U, V, W, rank):
        """Exact validation using Fractions"""
        T = np.zeros((n, m, p), dtype=object)
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[i, j, k] != 1:
                        return False
        return True

    def is_valid_numeric(U, V, W, rank, atol=1e-8):
        """Fast numeric validation"""
        U_arr = U.astype(float)
        V_arr = V.astype(float)
        W_arr = W.astype(float)
        
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U_arr[:, r].reshape(n, m)
            v = V_arr[:, r].reshape(m, p)
            w = W_arr[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return np.allclose(Rec, 1.0, atol=atol)

    def compute_error(U, V, W, rank):
        """Compute reconstruction error"""
        U_arr = U.astype(float)
        V_arr = V.astype(float)
        W_arr = W.astype(float)
        
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U_arr[:, r].reshape(n, m)
            v = V_arr[:, r].reshape(m, p)
            w = W_arr[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return np.sum((Rec - 1.0) ** 2)

    def gradient_descent(U, V, W, rank, max_steps=500, lr=0.01, tol=1e-10):
        """Perform gradient descent to minimize reconstruction error"""
        U = U.astype(float).copy()
        V = V.astype(float).copy()
        W = W.astype(float).copy()
        
        prev_error = compute_error(U, V, W, rank)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if prev_error < tol:
                break
            
            # Compute gradients numerically
            eps = 1e-6
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            # Gradient for U
            for idx in range(U.size):
                i, j = np.unravel_index(idx, U.shape)
                U_plus = U.copy()
                U_plus[i, j] += eps
                err_plus = compute_error(U_plus, V, W, rank)
                U_minus = U.copy()
                U_minus[i, j] -= eps
                err_minus = compute_error(U_minus, V, W, rank)
                grad_U[i, j] = (err_plus - err_minus) / (2 * eps)
            
            # Gradient for V
            for idx in range(V.size):
                i, j = np.unravel_index(idx, V.shape)
                V_plus = V.copy()
                V_plus[i, j] += eps
                err_plus = compute_error(U, V_plus, W, rank)
                V_minus = V.copy()
                V_minus[i, j] -= eps
                err_minus = compute_error(U, V_minus, W, rank)
                grad_V[i, j] = (err_plus - err_minus) / (2 * eps)
            
            # Gradient for W
            for idx in range(W.size):
                i, j = np.unravel_index(idx, W.shape)
                W_plus = W.copy()
                W_plus[i, j] += eps
                err_plus = compute_error(U, V, W_plus, rank)
                W_minus = W.copy()
                W_minus[i, j] -= eps
                err_minus = compute_error(U, V, W_minus, rank)
                grad_W[i, j] = (err_plus - err_minus) / (2 * eps)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            new_error = compute_error(U, V, W, rank)
            if abs(prev_error - new_error) < 1e-12:
                break
            prev_error = new_error
        
        return U, V, W, prev_error

    def try_optimization(rank, init_U=None, init_V=None, init_W=None, num_restarts=5):
        """Try to find a valid decomposition of given rank using optimization"""
        if time.time() > deadline:
            return None
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            if init_U is not None and restart == 0:
                U = init_U.astype(float).copy()
                V = init_V.astype(float).copy()
                W = init_W.astype(float).copy()
            else:
                # Random initialization
                np.random.seed(1000 + rank * 100 + restart)
                U = np.random.randn(n * m, rank) * 0.5
                V = np.random.randn(m * p, rank) * 0.5
                W = np.random.randn(p * n, rank) * 0.5
            
            # Run gradient descent
            U, V, W, err = gradient_descent(U, V, W, rank, max_steps=200, lr=0.05)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Try rounding to half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            if is_valid_numeric(U_round, V_round, W_round, rank, atol=1e-6):
                # Verify exactly
                U_frac = np.zeros((n * m, rank), dtype=object)
                V_frac = np.zeros((m * p, rank), dtype=object)
                W_frac = np.zeros((p * n, rank), dtype=object)
                for r in range(rank):
                    for idx in range(n * m):
                        U_frac[idx, r] = Fraction(int(U_round[idx, r] * 2)) / 2
                    for idx in range(m * p):
                        V_frac[idx, r] = Fraction(int(V_round[idx, r] * 2)) / 2
                    for idx in range(p * n):
                        W_frac[idx, r] = Fraction(int(W_round[idx, r] * 2)) / 2
                
                if is_valid_exact(U_frac, V_frac, W_frac, rank):
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
            
            # Also try integer rounding
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            if is_valid_numeric(U_int, V_int, W_int, rank, atol=1e-6):
                U_frac = np.zeros((n * m, rank), dtype=object)
                V_frac = np.zeros((m * p, rank), dtype=object)
                W_frac = np.zeros((p * n, rank), dtype=object)
                for r in range(rank):
                    for idx in range(n * m):
                        U_frac[idx, r] = Fraction(int(U_int[idx, r]))
                    for idx in range(m * p):
                        V_frac[idx, r] = Fraction(int(V_int[idx, r]))
                    for idx in range(p * n):
                        W_frac[idx, r] = Fraction(int(W_int[idx, r]))
                
                if is_valid_exact(U_frac, V_frac, W_frac, rank):
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        return None

    def strassen_3x3_init():
        """Create a Strassen-like initialization for the 3x3x3 block"""
        rank = 7
        U = np.zeros((9, rank))
        V = np.zeros((15, rank))
        W = np.zeros((15, rank))
        
        # Strassen's 7 products for 3x3 (adapted from 2x2)
        # Using block structure where we treat 3x3 as having 2x2 blocks
        # with the third row/column handled separately
        
        # For simplicity, use a known rank-7 pattern
        # P1 = A[0,0] * (B[0,0] + B[1,1])
        # P2 = (A[0,0] + A[1,1]) * B[1,1]
        # P3 = A[0,1] * B[1,0]
        # P4 = A[1,0] * B[0,1]
        # P5 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        # P6 = (A[0,0] - A[1,1]) * (B[0,1] + B[1,1])
        # P7 = (A[0,1] + A[1,1]) * (B[1,0] - B[0,0])
        
        # This is complex for 3x3. Let's use a simpler initialization
        # that the optimizer can work with.
        
        # Initialize with random values that have some structure
        np.random.seed(42)
        U[:, :7] = np.random.randn(9, 7) * 0.5
        V[:, :7] = np.random.randn(15, 7) * 0.5
        W[:, :7] = np.random.randn(15, 7) * 0.5
        
        # Zero out V and W entries for k=3,4 in the first 7 columns
        # V indices: j*5+k, so for k=3: j*5+3 for j in 0,1,2 -> indices 3, 8, 13
        # For k=4: j*5+4 for j in 0,1,2 -> indices 4, 9, 14
        V[[3, 8, 13, 4, 9, 14], :7] = 0
        
        # W indices: k*3+i, so for k=3: 3*3+i = 9,10,11
        # For k=4: 4*3+i = 12,13,14
        W[[9, 10, 11, 12, 13, 14], :7] = 0
        
        return U, V, W

    def search_optimization():
        """Search for lower rank decompositions using optimization"""
        nonlocal best_solution, best_rank
        search_start = time.time()
        
        # Try rank 25 first (Strassen + schoolbook hybrid)
        for rank in [25, 24, 23, 22, 21, 20, 19, 18, 17, 16]:
            if time.time() > deadline - 10:
                break
            if rank >= best_rank:
                continue
            
            # Try with Strassen-like initialization
            if rank == 25:
                U_init, V_init, W_init = strassen_3x3_init()
                # Pad to rank 25
                U_full = np.zeros((9, 25))
                V_full = np.zeros((15, 25))
                W_full = np.zeros((15, 25))
                U_full[:, :7] = U_init
                V_full[:, :7] = V_init
                W_full[:, :7] = W_init
                
                # Schoolbook for k=3,4
                col = 7
                for i in range(3):
                    for j in range(3):
                        for k in [3, 4]:
                            U_full[i * 3 + j, col] = 1
                            V_full[j * 5 + k, col] = 1
                            W_full[k * 3 + i, col] = 1
                            col += 1
                
                result = try_optimization(25, U_full, V_full, W_full, num_restarts=3)
                if result is not None:
                    best_solution = result
                    best_rank = 25
                    return True
            
            # Try random initialization for other ranks
            result = try_optimization(rank, num_restarts=3)
            if result is not None:
                best_solution = result
                best_rank = rank
                return True
        
        return False

    # Try to find a better solution using optimization
    search_optimization()

    def to_lists(mat):
        """Convert matrix of Fractions to list of lists of floats"""
        result = []
        for i in range(mat.shape[0]):
            row = []
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if isinstance(val, Fraction):
                    row.append(float(val))
                else:
                    row.append(float(val))
            result.append(row)
        return result

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
