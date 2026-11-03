"""Combined program for the suite task; uses schoolbook with structured low-rank search."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 150s kill
    n, m, p = 3, 5, 7
    
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the 3x5x7 matrix multiplication tensor.
        U: (15, r), V: (35, r), W: (21, r)
        Check: for all i in [0,3), j in [0,5), k in [0,7):
            sum_t U[i*5+j, t] * V[j*7+k, t] * W[k*3+i, t] == 1
        For all other (a, b, c) triples where a = i*5+j, b = j'*7+k, c = k'*3+i
        with j' != j or k' != k (i.e., inconsistent indices), the sum must be 0.
        """
        r = len(U[0]) if len(U) > 0 else 0
        # Check all 105 valid positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(r):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if s != 1:
                        return False
        # Check invalid positions: (a, b, c) where a = i*m+j, b = j2*p+k2, c = k3*n+i3
        # but not (j2 == j and k3 == k for the same i)
        # More precisely: for each triple (a, b, c), extract i from a (i = a // m, j = a % m),
        # extract j2 from b (j2 = b // p, k2 = b % p), extract i3 from c (i3 = c // n, k3 = c % n)
        # Valid iff i == i3 and j == j2 and k == k2
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k2 = b % p
                    i3 = c // n
                    k3 = c % n
                    if i != i3 or j != j2 or k2 != k3:
                        s = 0
                        for t in range(r):
                            s += U[a][t] * V[b][t] * W[c][t]
                        if s != 0:
                            return False
        return True
    
    def try_low_rank(target_rank, max_attempts=200):
        """Try to find a rank-target_rank decomposition using random U,V and solving for W."""
        dim_u = n * m  # 15
        dim_v = m * p  # 35
        dim_w = p * n  # 21
        r = target_rank
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random U, V with entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(dim_u, r))
            V = np.random.randint(-1, 2, size=(dim_v, r))
            
            # For each valid (i, j, k), we need:
            #   sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
            # This is a system of 105 equations in 21*r unknowns (W entries).
            # But W is shared across equations: W[k*n+i, t] appears in equation (i,j,k)
            # for each j. So for fixed (i, k), W[k*n+i, :] appears in m=5 equations.
            
            # Let's set up: for each (i, k), we have 5 equations (one per j):
            #   sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1  for j = 0..4
            # This is a linear system in W[k*n+i, 0..r-1] (r unknowns) with 5 equations.
            # But we also need the "zero" constraints for inconsistent triples.
            
            # Actually, let's build the full linear system.
            # Unknowns: W[c, t] for c in 0..20, t in 0..r-1. Total 21*r unknowns.
            # Equations:
            # 1. For each (i, j, k): sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
            #    This gives 105 equations.
            # 2. For each inconsistent (a, b, c): sum_t U[a, t] * V[b, t] * W[c, t] = 0
            #    This gives many more equations.
            
            # The system is linear in W. Let's build it.
            # Number of unknowns: dim_w * r = 21 * r
            num_unknowns = dim_w * r
            
            # Build matrix A and vector b for the linear system A * w_vec = b
            # w_vec is W flattened: W[0,0], W[0,1], ..., W[0,r-1], W[1,0], ...
            
            # We'll use the 105 "equals 1" equations first, then check zeros.
            # Actually, let's use all equations. But that might be too many.
            # Let's just use the 105 "equals 1" equations and then verify.
            
            # Build 105 x (21*r) system
            rows = []
            rhs = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = [Fraction(0)] * num_unknowns
                        for t in range(r):
                            coeff = U[i * m + j, t] * V[j * p + k, t]
                            if coeff != 0:
                                idx = (k * n + i) * r + t
                                row[idx] = Fraction(coeff)
                        rows.append(row)
                        rhs.append(Fraction(1))
            
            # Solve the linear system using Gaussian elimination (exact)
            # This is 105 equations, 21*r unknowns. For r < 5, 21*r < 105, so underdetermined.
            # We need to find ANY solution. Let's use a simple approach:
            # Pick r*21 free variables and solve for the rest? That's complex.
            
            # Alternative: just check if there's a solution by seeing if the system is consistent.
            # But we need an actual W. Let's try a different approach:
            # For each (i, k), the 5 equations (over j) give a 5 x r system for W[k*n+i, :].
            # If r >= 5, we can solve it (if consistent).
            
            if r < 5:
                continue
            
            W = np.zeros((dim_w, r), dtype=object)
            consistent = True
            for i in range(n):
                for k in range(p):
                    # Build 5 x r system: A_jk * W[k*n+i, :] = 1 for j=0..4
                    A_sub = np.zeros((m, r), dtype=object)
                    for j in range(m):
                        for t in range(r):
                            A_sub[j, t] = Fraction(U[i * m + j, t] * V[j * p + k, t])
                    
                    # Solve A_sub * x = [1,1,1,1,1]^T where x = W[k*n+i, :]
                    # Use numpy with float for speed, then check
                    A_float = np.array([[float(v) for v in row] for row in A_sub], dtype=np.float64)
                    b_float = np.ones(m, dtype=np.float64)
                    
                    try:
                        # Least squares solution
                        x, residuals, rank, sv = np.linalg.lstsq(A_float, b_float, rcond=None)
                        # Check if the solution is close to integer/half-integer
                        # and if it satisfies the equations
                        if np.allclose(A_float @ x, b_float, atol=1e-6):
                            # Round to nearest half-integer
                            x_rounded = np.round(x * 2) / 2
                            if np.allclose(A_float @ x_rounded, b_float, atol=1e-6):
                                W[k * n + i, :] = [Fraction(v) for v in x_rounded]
                            else:
                                consistent = False
                                break
                        else:
                            consistent = False
                            break
                    except:
                        consistent = False
                        break
                
                if not consistent:
                    break
            
            if not consistent:
                continue
            
            # Verify the full decomposition
            U_list = [[int(v) for v in row] for row in U]
            V_list = [[int(v) for v in row] for row in V]
            W_list = [[float(v) for v in row] for row in W]
            
            if verify(U_list, V_list, W_list):
                return {'U': U_list, 'V': V_list, 'W': W_list}
        
        return None
    
    # Try to find a lower-rank decomposition
    best_result = None
    for target_rank in range(35, 105):
        if time.time() > deadline - 5:
            break
        result = try_low_rank(target_rank, max_attempts=50)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    return schoolbook()
# EVOLVE-BLOCK-END
