"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Uses structured search with alternating minimization and rational verification.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    nm, mp, pn = n*m, m*p, p*n  # 6, 12, 8
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    def validate(U, V, W, rank):
        """Check if U, V, W with given rank exactly decompose the multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries: T[i*m+j, j*p+k, k*n+i] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                    if abs(val - 1) > 1e-9:
                        return False
        
        # Check that T[a,b,c] = 0 when (a,b,c) doesn't correspond to valid multiplication
        for a in range(nm):
            i, j = divmod(a, m)
            for b in range(mp):
                j2, k = divmod(b, p)
                if j2 != j:
                    # Should be 0 for all c
                    for c in range(pn):
                        val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if abs(val) > 1e-9:
                            return False
                else:
                    valid_c = k * n + i
                    for c in range(pn):
                        val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if c == valid_c:
                            if abs(val - 1) > 1e-9:
                                return False
                        else:
                            if abs(val) > 1e-9:
                                return False
        return True

    # Schoolbook solution (rank 24)
    rank_sb = nm * mp  # Actually it's n*m*p = 24
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_solution = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    # Try to find lower rank solutions using alternating minimization
    # Key idea: fix W, solve for U and V; or use structured initialization
    
    random.seed(12345)
    
    for target_rank in range(12, 24):
        if time.time() > deadline - 15:
            break
        
        found = False
        num_attempts = 20 if target_rank <= 16 else 10
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Alternating minimization: fix two, solve for third
            # The constraint is bilinear in each pair
            # For fixed V and W: U[a,r] appears in equations for (a,b,c)
            # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            # For fixed a: sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c]
            # This is linear in U[a,:] for each (b,c)
            
            for iteration in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each a, solve linear system
                for a in range(nm):
                    i, j = divmod(a, m)
                    # Equations: for each (b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                    # Build system: M[a,b,c,r] = V[b,r]*W[c,r], rhs = T[a,b,c]
                    # Only need equations where T[a,b,c] is potentially nonzero
                    # T[a,b,c] = 1 iff b=j*p+k for some k and c=k*n+i
                    # Otherwise T[a,b,c] = 0
                    # So we have equations for all (b,c) in range(mp) x range(pn)
                    # But many are 0, which is fine - they're still valid equations
                    
                    # Build the linear system for U[a,:]
                    # Number of equations: mp * pn = 96, unknowns: target_rank
                    # This is overdetermined if target_rank < 96
                    M = np.zeros((mp * pn, target_rank))
                    rhs = np.zeros(mp * pn)
                    idx = 0
                    for b in range(mp):
                        j2, k = divmod(b, p)
                        for c in range(pn):
                            # What is T[a,b,c]?
                            # a = i*m+j, b = j2*p+k, c = c
                            # T = 1 iff j2==j and c == k*n+i
                            if j2 == j:
                                k_from_b = k
                                valid_c = k_from_b * n + i
                                if c == valid_c:
                                    rhs[idx] = 1.0
                            # else rhs[idx] = 0
                            M[idx, :] = V[b, :] * W[c, :]
                            idx += 1
                    
                    # Solve least squares: M @ U[a,:] = rhs
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                        U[a, :] = sol
                    except:
                        pass
                
                # Update V: for each b, solve linear system
                for b in range(mp):
                    j2, k = divmod(b, p)
                    M = np.zeros((nm * pn, target_rank))
                    rhs = np.zeros(nm * pn)
                    idx = 0
                    for a in range(nm):
                        i, j = divmod(a, m)
                        for c in range(pn):
                            if j == j2:
                                valid_c = k * n + i
                                if c == valid_c:
                                    rhs[idx] = 1.0
                            M[idx, :] = U[a, :] * W[c, :]
                            idx += 1
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                        V[b, :] = sol
                    except:
                        pass
                
                # Update W: for each c, solve linear system
                for c in range(pn):
                    M = np.zeros((nm * mp, target_rank))
                    rhs = np.zeros(nm * mp)
                    idx = 0
                    for a in range(nm):
                        i, j = divmod(a, m)
                        for b in range(mp):
                            j2, k = divmod(b, p)
                            if j == j2:
                                valid_c = k * n + i
                                if c == valid_c:
                                    rhs[idx] = 1.0
                            M[idx, :] = U[a, :] * V[b, :]
                            idx += 1
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                        W[c, :] = sol
                    except:
                        pass
                
                # Check if converged
                if validate(U.tolist(), V.tolist(), W.tolist(), target_rank):
                    # Try to convert to rational
                    found = True
                    break
            
            if found:
                # Convert to fractions for exactness
                U_frac = [[Fraction(U[i,j]).limit_denominator(1000) for j in range(target_rank)] for i in range(nm)]
                V_frac = [[Fraction(V[i,j]).limit_denominator(1000) for j in range(target_rank)] for i in range(mp)]
                W_frac = [[Fraction(W[i,j]).limit_denominator(1000) for j in range(target_rank)] for i in range(pn)]
                
                # Verify with fractions
                U_f = np.array(U_frac, dtype=float)
                V_f = np.array(V_frac, dtype=float)
                W_f = np.array(W_frac, dtype=float)
                if validate(U_f.tolist(), V_f.tolist(), W_f.tolist(), target_rank):
                    best_solution = {"U": U_frac, "V": V_frac, "W": W_frac}
                    best_rank = target_rank
                    break
        
        if best_rank == target_rank:
            break
    
    return best_solution
# EVOLVE-BLOCK-END
