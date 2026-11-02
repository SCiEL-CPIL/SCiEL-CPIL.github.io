"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need to check that for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # And for all (i,j,k) that don't correspond to valid matrix mult terms, it should be 0
    # Actually, the tensor for matrix multiplication is T[i,j,k] = delta[i,k] * 1 (for the product)
    # Wait, let me reconsider. The standard tensor decomposition for matrix multiplication M_{n,p}^{m}
    # is defined such that: C[i,k] = sum_{j,c} A[i,j] * B[j,k] * T[i,j,k]
    # where T[i,j,k] is the structure tensor. For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] (or similar indexing)
    
    # Let's verify by checking a random instance
    # Actually, the simplest verification: for all i in range(n), j in range(m), k in range(p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer."""
    if isinstance(x, Fraction):
        return x.denominator in (1, 2)
    # Check if 2*x is an integer
    fx = float(x)
    return abs(fx * 2 - round(fx * 2)) < 1e-9

def _snap_to_half_integer(x):
    """Snap a value to the nearest half-integer."""
    return Fraction(round(float(x) * 2), 2)

def _try_rank(rank, n, m, p, time_limit=10):
    """Try to find a rank-`rank` decomposition using random initialization and refinement."""
    start_time = time.time()
    
    # Initialize with random half-integer values
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n*m):
        for c in range(rank):
            U[i][c] = random.choice([-1, 0, 1])
    for j in range(m*p):
        for c in range(rank):
            V[j][c] = random.choice([-1, 0, 1])
    for k in range(p*n):
        for c in range(rank):
            W[k][c] = random.choice([-1, 0, 1])
    
    # Verify the initial decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return U, V, W
    
    # Try to refine using gradient descent on a relaxed problem
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Define the loss: we want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    # Loss = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    
    def loss_and_grad(U, V, W):
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute the current value
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err = val - 1.0
                    loss += err * err
                    
                    for c in range(rank):
                        grad_U[i*m+j, c] += 2 * err * V[j*p+k, c] * W[k*n+i, c]
                        grad_V[j*p+k, c] += 2 * err * U[i*m+j, c] * W[k*n+i, c]
                        grad_W[k*n+i, c] += 2 * err * U[i*m+j, c] * V[j*p+k, c]
        
        return loss, grad_U, grad_V, grad_W
    
    # Simple gradient descent with projection to half-integers
    lr = 0.01
    for step in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        loss, gU, gV, gW = loss_and_grad(U_np, V_np, W_np)
        
        if loss < 1e-6:
            # Snap to half-integers and verify
            U_snap = [[_snap_to_half_integer(U_np[i,c]) for c in range(rank)] for i in range(n*m)]
            V_snap = [[_snap_to_half_integer(V_np[j,c]) for c in range(rank)] for j in range(m*p)]
            W_snap = [[_snap_to_half_integer(W_np[k,c]) for c in range(rank)] for k in range(p*n)]
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
            break
        
        U_np -= lr * gU
        V_np -= lr * gV
        W_np -= lr * gW
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    deadline = time.time() + 120  # Leave safety margin
    for rank in range(schoolbook_rank - 1, max(10, schoolbook_rank - 20), -1):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank(rank, n, m, p, time_limit=min(10, deadline - time.time() - 2))
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
