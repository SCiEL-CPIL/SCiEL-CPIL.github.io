"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
    
    def matmul_tensor(U, V, W):
        """Reconstruct tensor from factors U, V, W."""
        return U @ np.einsum('ab,cb->ac', V.T, W.T)
    
    def verify_exact(U, V, W):
        """Check if the decomposition is exact."""
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # We can check this by flattening
        rank = U.shape[1]
        # Use einsum for efficiency
        reconstructed = np.einsum('ir, jr, kr -> ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=1e-10)
    
    # Start with schoolbook
    U = np.zeros((n*m, rank_school), dtype=np.int64)
    V = np.zeros((m*p, rank_school), dtype=np.int64)
    W = np.zeros((p*n, rank_school), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    deadline = time.time() + 120
    current_rank = rank_school
    rng = np.random.default_rng(42)
    
    # Try to reduce rank by removing linearly dependent columns
    # First, try greedy removal
    def try_remove_columns(U, V, W, T):
        """Try to remove columns that don't contribute to the tensor."""
        rank = U.shape[1]
        keep = list(range(rank))
        
        for col in range(rank):
            if time.time() > deadline - 5:
                break
            if col not in keep:
                continue
            # Check if removing this column still gives exact reconstruction
            test_keep = [k for k in keep if k != col]
            if not test_keep:
                continue
            U_test = U[:, test_keep]
            V_test = V[:, test_keep]
            W_test = W[:, test_keep]
            reconstructed = np.einsum('ir, jr, kr -> ijk', U_test, V_test, W_test)
            if np.allclose(reconstructed, T, atol=1e-10):
                keep = test_keep
                U = U[:, keep]
                V = V[:, keep]
                W = W[:, keep]
        
        return U, V, W
    
    # Apply greedy removal
    U, V, W = try_remove_columns(U, V, W, T)
    current_rank = U.shape[1]
    if current_rank < best_rank and verify_exact(U, V, W):
        best_rank = current_rank
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
    
    # Try random perturbations to find lower rank decompositions
    # Strategy: try rank = best_rank - 1, best_rank - 2, etc.
    for target_rank in range(best_rank - 1, max(best_rank - 10, 14), -1):
        if time.time() > deadline - 10:
            break
        
        success = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random perturbations around schoolbook
            U_rand = np.zeros((n*m, target_rank), dtype=np.float64)
            V_rand = np.zeros((m*p, target_rank), dtype=np.float64)
            W_rand = np.zeros((p*n, target_rank), dtype=np.float64)
            
            # Copy some columns from schoolbook
            cols_to_copy = min(target_rank, best_rank)
            for r in range(cols_to_copy):
                U_rand[:, r] = best_U[:, r]
                V_rand[:, r] = best_V[:, r]
                W_rand[:, r] = best_W[:, r]
            
            # Add small random noise
            noise_level = 0.1
            for r in range(target_rank):
                U_rand[:, r] += rng.normal(0, noise_level, size=n*m)
                V_rand[:, r] += rng.normal(0, noise_level, size=m*p)
                W_rand[:, r] += rng.normal(0, noise_level, size=p*n)
            
            # Try to find exact solution via local search
            # We want T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # This is a system of equations. Let's try iterative refinement.
            
            # Flatten the tensor equation
            # For each (i,j,k) with T[i,j,k] = 1, we need sum_r U[i,r]*V[j,r]*W[k,r] = 1
            # For each (i,j,k) with T[i,j,k] = 0, we need sum_r U[i,r]*V[j,r]*W[k,r] = 0
            
            # Simple approach: try to match the non-zero entries
            # Use least squares to minimize the error
            error = 0.0
            for iter in range(20):
                if time.time() > deadline - 10:
                    break
                reconstructed = np.einsum('ir, jr, kr -> ijk', U_rand, V_rand, W_rand)
                error = np.sum((reconstructed - T)**2)
                if error < 1e-8:
                    break
                
                # Gradient descent step
                # dU[i,r] = sum_{j,k} (T[i,j,k] - recon[i,j,k]) * V[j,r] * W[k,r]
                # dV[j,r] = sum_{i,k} (T[i,j,k] - recon[i,j,k]) * U[i,r] * W[k,r]
                # dW[k,r] = sum_{i,j} (T[i,j,k] - recon[i,j,k]) * U[i,r] * V[j,r]
                
                recon = np.einsum('ir, jr, kr -> ijk', U_rand, V_rand, W_rand)
                diff = T - recon
                
                learning_rate = 0.01
                U_rand += learning_rate * np.einsum('ijk,jr,kr->ir', diff, V_rand, W_rand)
                V_rand += learning_rate * np.einsum('ijk,ir,kr->jr', diff, U_rand, W_rand)
                W_rand += learning_rate * np.einsum('ijk,ir,jr->kr', diff, U_rand, V_rand)
            
            # Check if we found an exact solution
            if verify_exact(U_rand, V_rand, W_rand):
                # Round to nearest integers or half-integers
                U_round = np.round(U_rand * 2) / 2
                V_round = np.round(V_rand * 2) / 2
                W_round = np.round(W_rand * 2) / 2
                if verify_exact(U_round, V_round, W_round):
                    best_rank = target_rank
                    best_U = U_round.astype(np.int64) * 2  # Store as doubled integers
                    best_V = V_round.astype(np.int64) * 2
                    best_W = W_round.astype(np.int64) * 2
                    success = True
                    break
        
        if success:
            break
    
    # Convert back to list of lists with proper values
    # best_U, best_V, best_W are stored as doubled integers if half-integers were found
    # Otherwise they're regular integers
    
    if best_U.dtype == np.int64:
        # Check if we need to divide by 2 (half-integers)
        # If all values are even, they're integers; otherwise half-integers
        # For simplicity, return as floats if needed
        U_out = [[float(x) for x in row] for row in best_U]
        V_out = [[float(x) for x in row] for row in best_V]
        W_out = [[float(x) for x in row] for row in best_W]
    else:
        U_out = [[float(x) for x in row] for row in best_U]
        V_out = [[float(x) for x in row] for row in best_V]
        W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
