<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank_school
    
    def verify_exact(U, V, W, rank):
        """Check if U, V, W (list of lists of floats) correctly decompose the tensor with half-integer entries."""
        # First, check if entries are close to half-integers
        # Then verify with Fractions
        
        # Convert to Fractions for exact check
        U_frac = [[Fraction(round(x * 2) / 2) for x in row] for row in U]
        V_frac = [[Fraction(round(x * 2) / 2) for x in row] for row in V]
        W_frac = [[Fraction(round(x * 2) / 2) for x in row] for row in W]
        
        # Check if they are exactly half-integers (numerator is integer, denominator 1 or 2)
        # Actually, just checking the product sum is sufficient if we assume the rounding was correct.
        # But we must ensure the inputs to the sum are the rounded values.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def als_decompose(rank, seed):
        """Run ALS to find a rank-rank decomposition."""
        np.random.seed(seed)
        
        # Initialize with structured random values
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to prevent overflow/underflow
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        max_iters = 2000
        tol = 1e-10
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                return None, False
            
            # Update U
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        u, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        new_U[i*m+j, :] = u
                    except:
                        return None, False
            U = new_U
            
            # Update V
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    N = np.zeros((n, rank))
                    for i in range(n):
                        N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, residuals, rank_N, sv = np.linalg.lstsq(N, np.ones(n), rcond=None)
                        new_V[j*p+k, :] = v
                    except:
                        return None, False
            V = new_V
            
            # Update W
            new_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    P = np.zeros((m, rank))
                    for j in range(m):
                        P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w, residuals, rank_P, sv = np.linalg.lstsq(P, np.ones(m), rcond=None)
                        new_W[k*n+i, :] = w
                    except:
                        return None, False
            W = new_W
            
            # More frequent convergence check
            if it % 20 == 19:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            err += (val - 1.0)**2
                if err < 1e-10:
                    break
                elif err < 1e-8 and it > 100:
                    # Polish pass with tighter tolerance
                    for _ in range(50):
                        if time.time() > deadline - 3:
                            break
                        new_U2 = np.zeros_like(U)
                        for i in range(n):
                            for j in range(m):
                                M2 = np.zeros((p, rank))
                                for k in range(p):
                                    M2[k, :] = V[j*p+k, :] * W[k*n+i, :]
                                try:
                                    u2, _, _, _ = np.linalg.lstsq(M2, np.ones(p), rcond=None)
                                    new_U2[i*m+j, :] = u2
                                except:
                                    break
                        U = new_U2
                        new_V2 = np.zeros_like(V)
                        for j in range(m):
                            for k in range(p):
                                N2 = np.zeros((n, rank))
                                for i in range(n):
                                    N2[i, :] = U[i*m+j, :] * W[k*n+i, :]
                                try:
                                    v2, _, _, _ = np.linalg.lstsq(N2, np.ones(n), rcond=None)
                                    new_V2[j*p+k, :] = v2
                                except:
                                    break
                        V = new_V2
                        new_W2 = np.zeros_like(W)
                        for k in range(p):
                            for i in range(n):
                                P2 = np.zeros((m, rank))
                                for j in range(m):
                                    P2[j, :] = U[i*m+j, :] * V[j*p+k, :]
                                try:
                                    w2, _, _, _ = np.linalg.lstsq(P2, np.ones(m), rcond=None)
                                    new_W2[k*n+i, :] = w2
                                except:
                                    break
                        W = new_W2
                    break
                     
        # Check final error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err += (val - 1.0)**2
        if err > 1e-6:
            return None, False
            
        # Snap to half-integers with multiple strategies
        best_result = None
        
        # Strategy 1: Direct rounding
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        if verify_exact(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank):
            return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist()), True
        
        # Strategy 2: Try small perturbations around rounded values
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            U_pert = U_snap + np.random.randn(*U_snap.shape) * 0.01
            V_pert = V_snap + np.random.randn(*V_snap.shape) * 0.01
            W_pert = W_snap + np.random.randn(*W_snap.shape) * 0.01
            U_snap2 = np.round(U_pert * 2) / 2
            V_snap2 = np.round(V_pert * 2) / 2
            W_snap2 = np.round(W_pert * 2) / 2
            if verify_exact(U_snap2.tolist(), V_snap2.tolist(), W_snap2.tolist(), rank):
                return (U_snap2.tolist(), V_snap2.tolist(), W_snap2.tolist()), True
        
        # Strategy 3: Try different rounding modes
        for mode in ['floor', 'ceil']:
            if mode == 'floor':
                U_snap = np.floor(U * 2) / 2
                V_snap = np.floor(V * 2) / 2
                W_snap = np.floor(W * 2) / 2
            else:
                U_snap = np.ceil(U * 2) / 2
                V_snap = np.ceil(V * 2) / 2
                W_snap = np.ceil(W * 2) / 2
            if verify_exact(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank):
                return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist()), True
        
        return None, False

    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # Try to find lower rank decompositions
    # Lower bound is n*p = 12, but likely higher.
    # Try a comprehensive range of ranks
    candidate_ranks = list(range(60, 11, -1))
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 5:
            break
        
        # Adaptive seed count based on time remaining
        time_remaining = deadline - time.time()
        if time_remaining < 10:
            break
        max_seeds = min(50, max(10, int(time_remaining / 2)))
        
        found = False
        for seed in range(max_seeds):
            if time.time() > deadline - 3:
                break
            
            result, success = als_decompose(target_rank, seed)
            if success:
                best_U, best_V, best_W, best_rank = result, result[1], result[2], target_rank
                found = True
                break
        
        # Continue searching for lower ranks even if we found one
        # We want the minimal rank
    
    # If we didn't find anything better, return schoolbook
    if best_rank == rank_school:
        U, V, W, _ = make_schoolbook()
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank_school
    
    def verify_exact(U, V, W, rank):
        """Check if U, V, W (list of lists of floats) correctly decompose the tensor with half-integer entries."""
        # Convert to Fractions for exact check
        U_frac = [[Fraction(round(x * 2)) / 2 for x in row] for row in U]
        V_frac = [[Fraction(round(x * 2)) / 2 for x in row] for row in V]
        W_frac = [[Fraction(round(x * 2)) / 2 for x in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def als_decompose(rank, seed):
        """Run ALS to find a rank-rank decomposition."""
        np.random.seed(seed)
        
        # Initialize with structured random values
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to prevent overflow/underflow
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        max_iters = 3000
        tol = 1e-12
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                return None, False
            
            # Update U
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        u, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        new_U[i*m+j, :] = u
                    except:
                        return None, False
            U = new_U
            
            # Update V
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    N = np.zeros((n, rank))
                    for i in range(n):
                        N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, residuals, rank_N, sv = np.linalg.lstsq(N, np.ones(n), rcond=None)
                        new_V[j*p+k, :] = v
                    except:
                        return None, False
            V = new_V
            
            # Update W
            new_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    P = np.zeros((m, rank))
                    for j in range(m):
                        P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w, residuals, rank_P, sv = np.linalg.lstsq(P, np.ones(m), rcond=None)
                        new_W[k*n+i, :] = w
                    except:
                        return None, False
            W = new_W
            
            # Convergence check every 10 iterations
            if it % 10 == 9:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            err += (val - 1.0)**2
                if err < 1e-12:
                    break
                elif err < 1e-10 and it > 50:
                    # Polish pass with tighter tolerance
                    for _ in range(100):
                        if time.time() > deadline - 3:
                            break
                        new_U2 = np.zeros_like(U)
                        for i in range(n):
                            for j in range(m):
                                M2 = np.zeros((p, rank))
                                for k in range(p):
                                    M2[k, :] = V[j*p+k, :] * W[k*n+i, :]
                                try:
                                    u2, _, _, _ = np.linalg.lstsq(M2, np.ones(p), rcond=None)
                                    new_U2[i*m+j, :] = u2
                                except:
                                    break
                        U = new_U2
                        new_V2 = np.zeros_like(V)
                        for j in range(m):
                            for k in range(p):
                                N2 = np.zeros((n, rank))
                                for i in range(n):
                                    N2[i, :] = U[i*m+j, :] * W[k*n+i, :]
                                try:
                                    v2, _, _, _ = np.linalg.lstsq(N2, np.ones(n), rcond=None)
                                    new_V2[j*p+k, :] = v2
                                except:
                                    break
                        V = new_V2
                        new_W2 = np.zeros_like(W)
                        for k in range(p):
                            for i in range(n):
                                P2 = np.zeros((m, rank))
                                for j in range(m):
                                    P2[j, :] = U[i*m+j, :] * V[j*p+k, :]
                                try:
                                    w2, _, _, _ = np.linalg.lstsq(P2, np.ones(m), rcond=None)
                                    new_W2[k*n+i, :] = w2
                                except:
                                    break
                        W = new_W2
                    break
                     
        # Check final error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err += (val - 1.0)**2
        if err > 1e-8:
            return None, False
            
        # Snap to half-integers with multiple strategies
        best_result = None
        
        # Strategy 1: Direct rounding
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        if verify_exact(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank):
            return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist()), True
        
        # Strategy 2: Try small perturbations around rounded values
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            U_pert = U_snap + np.random.randn(*U_snap.shape) * 0.005
            V_pert = V_snap + np.random.randn(*V_snap.shape) * 0.005
            W_pert = W_snap + np.random.randn(*W_snap.shape) * 0.005
            U_snap2 = np.round(U_pert * 2) / 2
            V_snap2 = np.round(V_pert * 2) / 2
            W_snap2 = np.round(W_pert * 2) / 2
            if verify_exact(U_snap2.tolist(), V_snap2.tolist(), W_snap2.tolist(), rank):
                return (U_snap2.tolist(), V_snap2.tolist(), W_snap2.tolist()), True
        
        # Strategy 3: Try different rounding modes
        for mode in ['floor', 'ceil']:
            if mode == 'floor':
                U_snap = np.floor(U * 2) / 2
                V_snap = np.floor(V * 2) / 2
                W_snap = np.floor(W * 2) / 2
            else:
                U_snap = np.ceil(U * 2) / 2
                V_snap = np.ceil(V * 2) / 2
                W_snap = np.ceil(W * 2) / 2
            if verify_exact(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank):
                return (U_snap.tolist(), V_snap.tolist(), W_snap.tolist()), True
        
        return None, False

    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # Try to find lower rank decompositions
    # Lower bound is n*p = 12, but likely higher.
    # Try a comprehensive range of ranks
    candidate_ranks = list(range(60, 11, -1))
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 5:
            break
        
        # Adaptive seed count based on time remaining
        time_remaining = deadline - time.time()
        if time_remaining < 10:
            break
        max_seeds = min(100, max(20, int(time_remaining / 1.5)))
        
        found = False
        for seed in range(max_seeds):
            if time.time() > deadline - 3:
                break
            
            result, success = als_decompose(target_rank, seed)
            if success:
                best_U, best_V, best_W, best_rank = result, result[1], result[2], target_rank
                found = True
                break
        
        # Continue searching for lower ranks even if we found one
        # We want the minimal rank
    
    # If we didn't find anything better, return schoolbook
    if best_rank == rank_school:
        U, V, W, _ = make_schoolbook()
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
