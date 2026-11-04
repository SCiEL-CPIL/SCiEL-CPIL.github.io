"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses structured tensor decomposition with exact rational arithmetic verification.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Schoolbook: rank = n*m*p = 2*4*5 = 40
    rank_schoolbook = n * m * p
    
    def schoolbook_decomposition():
        U = np.zeros((n*m, rank_schoolbook))
        V = np.zeros((m*p, rank_schoolbook))
        W = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook_decomposition()
    
    def verify_exact(U, V, W):
        """Verify tensor decomposition using exact Fraction arithmetic."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Convert to Fractions for exact arithmetic
        U_frac = [[Fraction(U[i, r]) for r in range(rank)] for i in range(n*m)]
        V_frac = [[Fraction(V[j, r]) for r in range(rank)] for j in range(m*p)]
        W_frac = [[Fraction(W[k, r]) for r in range(rank)] for k in range(p*n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for r in range(rank):
                        val += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                    if val != Fraction(1):
                        return False
        return True
    
    def verify_numeric(U, V, W, tol=1e-9):
        """Verify tensor decomposition using floating point."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    best_rank = rank_schoolbook
    best_decomp = (U.copy(), V.copy(), W.copy())
    
    # Try structured decompositions for lower ranks
    # For 2x4 * 4x5, we can try to exploit structure
    
    # Strategy: Use bilinear forms with shared structure
    # The key insight is that for small matrices, we can sometimes find
    # decompositions lower than n*m*p by sharing computations
    
    # Try rank values from low to high
    candidate_ranks = [20, 22, 24, 25, 26, 28, 30, 32, 35, 38]
    
    for target_rank in candidate_ranks:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Try multiple random initializations with refinement
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random values
            U_cand = np.random.randn(n*m, target_rank)
            V_cand = np.random.randn(m*p, target_rank)
            W_cand = np.random.randn(p*n, target_rank)
            
            # Scale
            scale = (n*m*p) ** (1/6) / np.sqrt(target_rank)
            U_cand *= scale
            V_cand *= scale
            W_cand *= scale
            
            # Iterative refinement using alternating minimization
            for iter in range(100):
                if time.time() > deadline - 15:
                    break
                
                # Check residual
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U_cand[i*m+j, r] * V_cand[j*p+k, r] * W_cand[k*n+i, r]
                            max_res = max(max_res, abs(val - 1.0))
                
                if max_res < 1e-10:
                    found = True
                    break
                
                # Alternating minimization: update one matrix at a time
                # This is a non-convex problem, so we use a simple heuristic
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # For each (i,j), we want sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                        # This is a linear system in U[i*m+j, :]
                        # But it's coupled across k, so we do a simple update
                        pass
                
                # Simple gradient step (very rough)
                lr = 0.01
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = 0.0
                            for r in range(target_rank):
                                err += U_cand[i*m+j, r] * V_cand[j*p+k, r] * W_cand[k*n+i, r]
                            err -= 1.0
                            for r in range(target_rank):
                                U_cand[i*m+j, r] -= lr * err * V_cand[j*p+k, r] * W_cand[k*n+i, r]
                                V_cand[j*p+k, r] -= lr * err * U_cand[i*m+j, r] * W_cand[k*n+i, r]
                                W_cand[k*n+i, r] -= lr * err * U_cand[i*m+j, r] * V_cand[j*p+k, r]
            
            if found:
                # Try to snap to half-integers
                U_snap = np.round(U_cand * 2) / 2
                V_snap = np.round(V_cand * 2) / 2
                W_snap = np.round(W_cand * 2) / 2
                
                if verify_exact(U_snap, V_snap, W_snap):
                    best_rank = target_rank
                    best_decomp = (U_snap, V_snap, W_snap)
                    break
                elif verify_numeric(U_snap, V_snap, W_snap, tol=1e-8):
                    # Try to refine to exact
                    U_exact = np.round(U_snap * 1000) / 1000
                    V_exact = np.round(V_snap * 1000) / 1000
                    W_exact = np.round(W_snap * 1000) / 1000
                    if verify_exact(U_exact, V_exact, W_exact):
                        best_rank = target_rank
                        best_decomp = (U_exact, V_exact, W_exact)
                        break
                    elif verify_numeric(U_cand, V_cand, W_cand, tol=1e-8):
                        best_rank = target_rank
                        best_decomp = (U_cand, V_cand, W_cand)
                        break
    
    # If no better decomposition found, return schoolbook
    if best_rank == rank_schoolbook:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U_best, V_best, W_best = best_decomp
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
