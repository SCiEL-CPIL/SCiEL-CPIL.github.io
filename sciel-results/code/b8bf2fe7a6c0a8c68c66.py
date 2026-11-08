"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_sb[i * m + j, c] = 1
                V_sb[j * p + k, c] = 1
                W_sb[k * n + i, c] = 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    deadline = time.time() + 140
    
    # Try structured decompositions with reduced rank
    # For matrix multiplication, we can try various rank values
    # Known lower bounds: for 4x4 * 4x9, we can try ranks below 144
    
    # Strategy: Use a structured ansatz where we group terms
    # Try rank = n*m*p - savings where we exploit linearity
    
    # First, try a simple structured approach:
    # Group by shared indices to reduce rank
    
    # For each (i,j) pair, we need p terms for the k summation
    # But we can share structure across different (i,j) pairs
    
    # Try a bilinear decomposition with shared factors
    # Rank = n * m * p is the trivial bound
    # We can try: for each k, use a rank-1 term per (i,j), but share across k
    
    # Actually, let's try a different approach:
    # Use the fact that matrix multiplication is a tensor
    # and try to find a low-rank decomposition numerically
    
    # For small matrices, we can try random initializations and optimize
    
    # Try various rank values from schoolbook downward
    target_ranks = list(range(max(n*m + m*p + p*n, 1), schoolbook_rank + 1, -1))
    # Sort by most promising (lower is better, but we need feasibility)
    # Start with ranks that are likely feasible
    
    # A reasonable lower bound: max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But that's too optimistic. Let's try a range.
    
    # Actually, for 4x4 * 4x9, the bilinear complexity is at most 144
    # Known results suggest it might be lower. Let's try a search.
    
    # Use a structured ansatz: 
    # U has rows indexed by (i,j), V by (j,k), W by (k,i)
    # We want sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r] = 1 for all i,j,k
    
    # Try rank values in a strategic order
    candidate_ranks = []
    # Start near schoolbook and go down, but skip too-optimistic values
    # The minimum possible rank is at least max(n*m, m*p, p*n) = 36
    # But realistically, for 4x4*4x9, it's probably between 60-100
    
    # Let's try a range of ranks
    for r in range(100, 35, -1):  # Try from 100 down to 36
        if time.time() > deadline - 20:
            break
        if r >= best_rank:
            continue
        
        # Try to find a decomposition of rank r
        success = _try_rank(n, m, p, r, deadline - 10)
        if success:
            best_rank = r
            best_U, best_V, best_W = success
            break  # Found a lower rank, return it
    
    # Convert to lists for the interface
    U_list = [[float(best_U[i, j]) for j in range(best_rank)] for i in range(n * m)]
    V_list = [[float(best_V[i, j]) for j in range(best_rank)] for i in range(m * p)]
    W_list = [[float(best_W[i, j]) for j in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank-rank decomposition using structured initialization."""
    
    # Initialize with a structured pattern
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Use a cyclic/structured initialization
    np.random.seed(42)
    
    # Try multiple initializations
    for trial in range(3):
        if time.time() > deadline:
            return None
        
        # Structured initialization
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Fill with a structured pattern
        for c in range(rank):
            # Assign each column to a specific (i,j,k) pattern
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            
            if i * m + j < n * m:
                U[i * m + j, c] = 1
            if j * p + k < m * p:
                V[j * p + k, c] = 1
            if k * n + i < p * n:
                W[k * n + i, c] = 1
        
        # Verify
        if _verify_decomposition(n, m, p, U, V, W):
            return (U, V, W)
        
        # Try random perturbation
        U += np.random.randn(n * m, rank) * 0.1
        V += np.random.randn(m * p, rank) * 0.1
        W += np.random.randn(p * n, rank) * 0.1
        
        # Simple iterative refinement
        for it in range(50):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    target = np.zeros(rank)
                    for k in range(p):
                        target += V[j * p + k] * W[k * n + i]
                    # Minimize ||U[idx] - target||
                    # This is not quite right; let me think differently
            
            # Actually, the constraint is:
            # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a system of equations. Let's use a different approach.
            # Fix V and W, solve for U, etc.
            
            # Update U: for each (i,j), we need
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is a linear system in U[i*m+j, :]
            
            # Build the system
            A = np.zeros((p, rank))
            b = np.ones(p)
            for k in range(p):
                A[k, :] = V[j * p + k] * W[k * n + i]
            
            # Least squares
            try:
                U_new_idx, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                U[idx, :] = U_new_idx
            except:
                pass
        
        if _verify_decomposition(n, m, p, U, V, W):
            return (U, V, W)
    
    return None

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True
# EVOLVE-BLOCK-END
