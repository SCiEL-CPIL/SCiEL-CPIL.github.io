"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i, j, k), the sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    # But actually, the decomposition is:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] = 1 (the multiplication tensor)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Leave 10s safety margin
    
    # Start with schoolbook decomposition
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try to reduce rank by random perturbations and verification
    # Strategy: Try various ranks from low to high, use random initialization + local search
    np.random.seed(42)
    
    # Known lower bound for 2x4x6 matrix multiplication rank
    # The rank is at least max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # But actually the tensor rank is the key quantity
    
    # Let's try a systematic approach:
    # For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # The bilinear complexity (tensor rank) is what we're minimizing
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound: max(n*m, m*p, p*n) / something, but let's just try from 24 up
    
    min_possible = max(n*m, m*p, p*n)
    
    for target_rank in range(min_possible, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 50 if target_rank < 40 else 20
        attempts = 0
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            U_r = np.zeros((n*m, target_rank))
            V_r = np.zeros((m*p, target_rank))
            W_r = np.zeros((p*n, target_rank))
            
            # Random initialization
            U_r[:] = np.random.randn(n*m, target_rank) * 0.1
            V_r[:] = np.random.randn(m*p, target_rank) * 0.1
            W_r[:] = np.random.randn(p*n, target_rank) * 0.1
            
            # Alternating minimization / gradient-free local search
            # We want sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
            
            # Use a simple iterative approach: fix two, solve for third
            # Actually, let's use a different approach: treat this as a system of equations
            
            # For each (i,j,k), we have one equation. Total: n*m*p equations
            # Unknowns: target_rank * (n*m + m*p + p*n)
            # This is an underdetermined system if target_rank is large enough
            
            # Let's use a greedy/iterative approach:
            # For each column c, choose U[:,c], V[:,c], W[:,c] to satisfy some equations
            
            # Simpler: use random search with local refinement
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Pick a random (i,j,k)
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                
                current_sum = 0.0
                for c in range(target_rank):
                    current_sum += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                
                error = current_sum - 1.0
                if abs(error) < 1e-10:
                    continue
                
                # Adjust one random component
                c = np.random.randint(target_rank)
                choice = np.random.randint(3)
                if choice == 0:
                    U_r[i*m+j, c] -= error * 0.1 / (V_r[j*p+k, c] * W_r[k*n+i, c] + 1e-10)
                elif choice == 1:
                    V_r[j*p+k, c] -= error * 0.1 / (U_r[i*m+j, c] * W_r[k*n+i, c] + 1e-10)
                else:
                    W_r[k*n+i, c] -= error * 0.1 / (U_r[i*m+j, c] * V_r[j*p+k, c] + 1e-10)
            
            # Verify
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-6):
                # Found a valid decomposition
                best_rank = target_rank
                best_U = U_r.tolist()
                best_V = V_r.tolist()
                best_W = W_r.tolist()
                break
        
        if best_rank == target_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
