"""Find rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the tensor: T[(i,j), (j,k), (k,i)] = 1 for all valid triples
    # U index: i*m+j, V index: j*p+k, W index: k*n+i
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Precompute all (i,j,k) triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                triples.append((u_idx, v_idx, w_idx))
    
    # Try ranks from low to high
    # Lower bound: at least 1, but realistically for this size, maybe 15-25
    # Start from a reasonable guess and go down
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for (u_idx, v_idx, w_idx) in triples:
            s = 0.0
            for c in range(rank):
                s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
            if abs(s - 1.0) > 1e-9:
                return False
        return True
    
    def to_half_int(x):
        """Round to nearest half-integer."""
        return round(x * 2) / 2
    
    def is_half_int(x, tol=1e-9):
        return abs(x * 2 - round(x * 2)) < tol
    
    # Try to find decomposition for a given rank
    def try_rank(rank, max_attempts=50):
        """Try to find a rank-r decomposition using random initialization."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                return None
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Scale to make products roughly order 1
            U /= np.sqrt(rank)
            V /= np.sqrt(rank)
            W /= np.sqrt(rank)
            
            # Use gradient-free search: try to minimize error
            # Error = sum over all (u,v,w) of (sum_c U[u,c]*V[v,c]*W[w,c] - 1)^2
            # This is a non-convex problem, so use random restarts
            
            # Simple approach: random perturbation with acceptance
            best_error = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Compute current error
                error = 0.0
                for (u_idx, v_idx, w_idx) in triples:
                    s = 0.0
                    for c in range(rank):
                        s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    error += (s - 1.0) ** 2
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_error < 1e-12:
                    break
                
                # Random perturbation
                lr = 0.1 * (1 - step / 200)
                U += lr * np.random.randn(nm, rank)
                V += lr * np.random.randn(mp, rank)
                W += lr * np.random.randn(pn, rank)
            
            # Try to round to half-integers
            if best_error < 1e-6:
                U_rounded = np.round(best_U * 2) / 2
                V_rounded = np.round(best_V * 2) / 2
                W_rounded = np.round(best_W * 2) / 2
                
                # Verify rounded solution
                valid = True
                for (u_idx, v_idx, w_idx) in triples:
                    s = 0.0
                    for c in range(rank):
                        s += U_rounded[u_idx, c] * V_rounded[v_idx, c] * W_rounded[w_idx, c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                
                if valid:
                    # Convert to list of lists
                    U_list = [[float(U_rounded[i][c]) for c in range(rank)] for i in range(nm)]
                    V_list = [[float(V_rounded[i][c]) for c in range(rank)] for i in range(mp)]
                    W_list = [[float(W_rounded[i][c]) for c in range(rank)] for i in range(pn)]
                    return (U_list, V_list, W_list)
        
        return None
    
    # Try ranks from low to high
    # Theoretical lower bound is at least max(n, m, p) = 5, but realistically higher
    # For 2x3x3x5, let's try from 15 upward
    for rank in range(15, 31):
        if time.time() > deadline - 30:
            break
        
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END
