"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and compare with 1 for all (i,j,k).
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # For each (i,j,k), compute the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                val = np.dot(np.dot(u_val, v_val), w_val)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_random_search(rank, n, m, p, time_budget, rng):
    """Try to find a decomposition of given rank using random search + local optimization."""
    start = time.time()
    
    # We need U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Strategy: initialize with small random values, then use gradient-like updates
    # on the constraint equations.
    
    num_constraints = n * m * p
    
    def init_matrices():
        # Initialize with small random values
        U = np.random.uniform(-1, 1, (n * m, rank))
        V = np.random.uniform(-1, 1, (m * p, rank))
        W = np.random.uniform(-1, 1, (p * n, rank))
        return U, V, W
    
    def compute_residuals(U, V, W):
        """Compute residuals: for each (i,j,k), residual = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1"""
        residuals = np.zeros(num_constraints)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i * m + j, :]
                    v_row = V[j * p + k, :]
                    w_row = W[k * n + i, :]
                    # sum_r u_row[r] * v_row[r] * w_row[r]
                    prod = u_row * v_row * w_row
                    residuals[idx] = np.sum(prod) - 1.0
                    idx += 1
        return residuals
    
    def compute_grads(U, V, W):
        """Compute gradients of sum of squared residuals w.r.t. U, V, W."""
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # For each constraint (i,j,k): f = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
        # df/dU[i*m+j,r] = V[j*p+k,r]*W[k*n+i,r]
        # df/dV[j*p+k,r] = U[i*m+j,r]*W[k*n+i,r]
        # df/dW[k*n+i,r] = U[i*m+j,r]*V[j*p+k,r]
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    u_row = U[u_idx, :]
                    v_row = V[v_idx, :]
                    w_row = W[w_idx, :]
                    
                    prod = u_row * v_row * w_row
                    f = np.sum(prod) - 1.0
                    
                    # Gradient of f^2 is 2*f * grad_f
                    factor = 2.0 * f
                    
                    grad_U[u_idx, :] += factor * (v_row * w_row)
                    grad_V[v_idx, :] += factor * (u_row * w_row)
                    grad_W[w_idx, :] += factor * (u_row * v_row)
                    
                    idx += 1
        
        return grad_U, grad_V, grad_W
    
    best_residual = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(20):
        U, V, W = init_matrices()
        
        # Adaptive learning rate
        lr = 0.01
        for step in range(5000):
            if time.time() - start > time_budget:
                break
            
            residuals = compute_residuals(U, V, W)
            sq_res = np.sum(residuals ** 2)
            
            if sq_res < best_residual:
                best_residual = sq_res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if sq_res < 1e-14:
                break
            
            grad_U, grad_V, grad_W = compute_grads(U, V, W)
            
            # Adaptive learning rate
            max_grad = max(np.max(np.abs(grad_U)), np.max(np.abs(grad_V)), np.max(np.abs(grad_W)))
            if max_grad > 0:
                lr = max(1e-6, min(0.1, 0.01 / max_grad))
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        if time.time() - start > time_budget:
            break
    
    if best_U is not None and best_residual < 1e-10:
        # Convert to list of lists
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return U_list, V_list, W_list
    
    return None

def _try_structure_based(rank, n, m, p, time_budget, rng):
    """Try structure-based approaches for specific ranks."""
    # For 3x3x4, we know the minimum rank is at most 3*3*4=36 (schoolbook)
    # Known results: for 3x3x3, min rank is 23 (Landsberg/Strassen)
    # For 3x3x4, we can try ranks from 24 to 35
    
    # One approach: try to find a decomposition where some entries are 0, ±1, ±1/2
    # This is a combinatorial search.
    
    # For now, try a heuristic: start from schoolbook and try to merge terms
    # by finding linear dependencies.
    
    # Actually, let's try a different approach: use the fact that for small matrices,
    # we can set up a system of polynomial equations and try to solve them.
    
    # For rank r, we have:
    # U: 9 x r, V: 12 x r, W: 12 x r
    # Constraints: 36 equations (one for each i,j,k)
    # Unknowns: 9*r + 12*r + 12*r = 33*r
    
    # For r = 24: 792 unknowns, 36 equations - very underdetermined
    # The issue is finding integer/half-integer solutions.
    
    # Let's try a greedy approach: build the decomposition term by term.
    
    start = time.time()
    
    # Initialize with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # Greedy: for each term r, choose U[:,r], V[:,r], W[:,r] to satisfy as many
    # remaining constraints as possible.
    
    # This is complex. Let's try a simpler approach: use the schoolbook and
    # try to reduce rank by finding linear combinations.
    
    # Alternative: for specific small ranks, try random initialization with
    # constrained values (0, ±1, ±1/2) and local search.
    
    for trial in range(50):
        if time.time() - start > time_budget:
            break
        
        # Random initialization with values from {-1, -0.5, 0, 0.5, 1}
        vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        U = np.random.choice(vals, size=(n * m, rank))
        V = np.random.choice(vals, size=(m * p, rank))
        W = np.random.choice(vals, size=(p * n, rank))
        
        # Check if it works
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-9):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds: for 3x3x3, rank >= 23. For 3x3x4, rank >= ?
    # Let's try ranks from 24 to best_rank-1
    
    rng = random.Random(42)
    
    for rank in range(24, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try random search
        result = _try_random_search(rank, n, m, p, time_budget=10, rng=rng)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-9):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        
        if time.time() > deadline - 5:
            break
    
    # If we still have time, try structure-based for the best rank found
    if time.time() < deadline - 10 and best_rank > 24:
        result = _try_structure_based(best_rank - 1, n, m, p, time_budget=30, rng=rng)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-9):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = best_rank - 1
    
    return best_result
# EVOLVE-BLOCK-END
