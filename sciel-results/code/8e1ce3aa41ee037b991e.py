"""Program that attempts to find a lower-rank exact tensor decomposition
for 3x4x6 matrix multiplication using a sparse alternating projection
algorithm over integer-valued factor matrices.

Strategy:
  - Start from the schoolbook (rank 72) decomposition.
  - Iteratively try to reduce rank by one at a time.
  - At each rank, run a sparse alternating projection:
      * Project U onto integer lattice given fixed V, W
      * Project V onto integer lattice given fixed U, W
      * Project W onto integer lattice given fixed U, V
    where "projection" means finding the nearest integer solution to a
    linear system derived from the tensor equations.
  - Use the sparse support of the matrix multiplication tensor to keep
    the linear systems small.
  - If a valid integer decomposition is found at a lower rank, return it.
  - Otherwise, fall back to the schoolbook rank.

The tensor T for C = A B with A (n x m), B (m x p), C (n x p) satisfies:
  For all i in [0,n), j in [0,m), k in [0,p):
    sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1

This is a system of n*m*p = 72 equations in the entries of U, V, W.
"""
import numpy as np
import random
import sys
import time

DIMENSIONS = [3, 4, 6]


def build_tensor_equations(n, m, p, rank):
    """Build the linear system for a fixed rank.
    
    We have n*m*p equations, each of the form:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    For a fixed set of two matrices, this is linear in the third.
    """
    nmp = n * m * p
    # Each equation is indexed by (i, j, k)
    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_indices.append((i, j, k))
    return eq_indices


def project_integer_linear(A, b, x0, max_iter=100):
    """Find nearest integer solution to A x = b, starting from x0.
    
    Uses a simple approach: solve least squares, round, then refine.
    A is (num_eq x num_vars), b is (num_eq,), x0 is (num_vars,).
    
    Returns the integer solution or None if not found.
    """
    num_vars = len(x0)
    
    # Solve least squares
    try:
        x_ls, residuals, rank_, sv = np.linalg.lstsq(A, b, rcond=None)
    except:
        return None
    
    # Round to nearest integer
    x_int = np.round(x_ls).astype(int)
    
    # Check if it's an exact solution
    if np.allclose(A @ x_int, b, atol=1e-9):
        return x_int
    
    # Try local search around the rounded solution
    best_x = x_int.copy()
    best_err = np.linalg.norm(A @ best_x - b)
    
    for _ in range(max_iter):
        # Pick a random variable to perturb
        idx = random.randint(0, num_vars - 1)
        old_val = best_x[idx]
        for delta in [-2, -1, 1, 2]:
            new_val = old_val + delta
            if new_val == old_val:
                continue
            best_x[idx] = new_val
            err = np.linalg.norm(A @ best_x - b)
            if err < best_err:
                best_err = err
                if best_err < 1e-9:
                    best_x[idx] = new_val
                    return best_x
            else:
                best_x[idx] = old_val
    
    if best_err < 1e-9:
        return best_x
    return None


def verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def try_find_decomposition(n, m, p, target_rank, max_time=60):
    """Try to find an exact integer decomposition of the given rank."""
    start_time = time.time()
    
    # Initialize with random integer values in [-2, 2]
    U = np.random.randint(-2, 3, size=(n * m, target_rank))
    V = np.random.randint(-2, 3, size=(m * p, target_rank))
    W = np.random.randint(-2, 3, size=(p * n, target_rank))
    
    # Also try starting from a perturbed schoolbook-like initialization
    # For the first n*m*p columns of a schoolbook, but we only have target_rank columns
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    max_iters = 500
    for iteration in range(max_iters):
        if time.time() - start_time > max_time:
            break
        
        # Project U: for each (i,j), fix V and W, solve for U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                # Equations: for each k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is linear in U[i*m+j, :]
                # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                A = np.zeros((p, target_rank))
                b = np.ones(p)
                for k in range(p):
                    for r in range(target_rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                
                x0 = U[i * m + j, :]
                x_int = project_integer_linear(A, b, x0, max_iter=20)
                if x_int is not None:
                    U[i * m + j, :] = x_int
        
        # Project V: for each (j,k), fix U and W, solve for V[j*p+k, :]
        for j in range(m):
            for k in range(p):
                # Equations: for each i, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                A = np.zeros((n, target_rank))
                b = np.ones(n)
                for i in range(n):
                    for r in range(target_rank):
                        A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                
                x0 = V[j * p + k, :]
                x_int = project_integer_linear(A, b, x0, max_iter=20)
                if x_int is not None:
                    V[j * p + k, :] = x_int
        
        # Project W: for each (k,i), fix U and V, solve for W[k*n+i, :]
        for k in range(p):
            for i in range(n):
                # Equations: for each j, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                A = np.zeros((m, target_rank))
                b = np.ones(m)
                for j in range(m):
                    for r in range(target_rank):
                        A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                
                x0 = W[k * n + i, :]
                x_int = project_integer_linear(A, b, x0, max_iter=20)
                if x_int is not None:
                    W[k * n + i, :] = x_int
        
        # Verify
        if verify_decomposition(U, V, W, n, m, p, target_rank):
            return U.copy(), V.copy(), W.copy()
        
        # Track best
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(target_rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += abs(s - 1.0)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Occasionally randomize to escape local minima
        if iteration % 50 == 0 and iteration > 0:
            if random.random() < 0.3:
                idx = random.randint(0, n * m - 1)
                col = random.randint(0, target_rank - 1)
                U[idx, col] += random.choice([-1, 1])
            if random.random() < 0.3:
                idx = random.randint(0, m * p - 1)
                col = random.randint(0, target_rank - 1)
                V[idx, col] += random.choice([-1, 1])
            if random.random() < 0.3:
                idx = random.randint(0, p * n - 1)
                col = random.randint(0, target_rank - 1)
                W[idx, col] += random.choice([-1, 1])
    
    return None


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Try to find a lower rank decomposition
    # Start from schoolbook - 1 and go down
    # Limit the search to save time
    max_search_rank = schoolbook_rank - 1
    min_search_rank = max(1, schoolbook_rank - 10)  # Only try a few below
    
    for target_rank in range(max_search_rank, min_search_rank - 1, -1):
        result = try_find_decomposition(n, m, p, target_rank, max_time=15)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    rank = schoolbook_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
