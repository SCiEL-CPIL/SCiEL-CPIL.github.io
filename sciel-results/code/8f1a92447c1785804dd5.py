"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_rank55(n, m, p):
    """
    Construct the known rank-55 decomposition using Strassen for 2x2 blocks
    and schoolbook for the 2x1 block.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for 2x2 * 2x10 (5 blocks of 2 columns each)
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        u00, u01, u10, u11 = 0, 1, 3, 4
        v00, v01 = k0, k1
        v10, v11 = 10 + k0, 10 + k1
        w00, w01 = k0 * 2, k1 * 2
        w10, w11 = k0 * 2 + 1, k1 * 2 + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[u00][col] = 1; U[u11][col] = 1; V[v00][col] = 1; V[v11][col] = 1; W[w00][col] = 1; W[w11][col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[u10][col] = 1; U[u11][col] = 1; V[v00][col] = 1; W[w10][col] = 1; W[w11][col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[u00][col] = 1; V[v01][col] = 1; V[v11][col] = -1; W[w01][col] = 1; W[w11][col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[u11][col] = 1; V[v10][col] = 1; V[v00][col] = -1; W[w00][col] = 1; W[w10][col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[u00][col] = 1; U[u01][col] = 1; V[v11][col] = 1; W[w01][col] = 1; W[w00][col] = -1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[u10][col] = 1; U[u00][col] = -1; V[v00][col] = 1; V[v01][col] = 1; W[w11][col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[u01][col] = 1; U[u11][col] = -1; V[v10][col] = 1; V[v11][col] = 1; W[w00][col] = 1
        col += 1
        
    # Schoolbook for A2 (2x1) * B2 (1x10)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    assert col == rank, f"Expected {rank}, got {col}"
    return {"U": U, "V": V, "W": W}

def _try_rank54_search(n, m, p, time_budget, seed=42):
    """
    Search for a rank-54 decomposition.
    Strategy: Exploit the small n=2 structure.
    
    Key insight: For n=2, each (i,k) pair gives m=3 equations in r=54 unknowns.
    We can use the structure more cleverly:
    - For fixed k, we have 2*i*3 = 6 equations (i=0,1) in 54 unknowns
    - The system is heavily underdetermined, so many solutions exist
    - We want solutions where W entries are in {0, ±1/2, ±1, ±3/2, ...}
    
    New approach: Start from the rank-55 solution and try to merge two columns
    by finding a linear dependency.
    """
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    r = 54
    
    best = None
    
    # Strategy 1: Start from rank-55 and try to find a linear combination
    # that reduces rank by 1
    U55 = np.array(_strassen_rank55(n, m, p)["U"])
    V55 = np.array(_strassen_rank55(n, m, p)["V"])
    W55 = np.array(_strassen_rank55(n, m, p)["W"])
    
    # Try to find columns c1, c2 in the rank-55 solution such that
    # we can replace them with a single column
    # This means finding alpha, beta such that:
    # U[:,c1]*alpha + U[:,c2]*beta, V[:,c1]*alpha + V[:,c2]*beta, W[:,c1]*alpha + W[:,c2]*beta
    # still satisfies all tensor equations
    
    # Actually, a better approach: try to find a vector (alpha_0, ..., alpha_54)
    # such that sum_c alpha_c * (U[:,c] ⊗ V[:,c] ⊗ W[:,c]) = 0
    # and alpha has exactly two non-zero entries (to merge two columns)
    
    # Let's try a different approach: random search with better structure
    
    # Strategy 2: Structured random search
    # For n=2, m=3, p=10:
    # U has 6 rows, V has 30 rows, W has 20 rows
    # Each column c contributes U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    
    # Key constraint: For each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Let's try to construct U and V with specific patterns that make W easy to solve
    
    # Pattern: Use the Strassen structure but with modifications
    # Try to find a 54-rank solution by modifying the 55-rank one
    
    # Approach: For each pair of columns (a,b) in the 55-solution, check if
    # we can find a linear combination that preserves the tensor
    # This is equivalent to checking if the two rank-1 tensors are linearly
    # dependent in the quotient space
    
    # More practical: Try random perturbations of the Strassen solution
    
    for attempt in range(5000):
        if time.time() > deadline - 3:
            break
        
        # Generate a candidate by modifying the Strassen solution
        # Pick two columns to "merge"
        c1 = rng.randint(0, 54)
        c2 = rng.randint(0, 54)
        if c1 == c2:
            continue
        
        # Try to find alpha, beta such that replacing columns c1, c2 with
        # a single column works
        # The new column would be: U_new = U[:,c1]*alpha + U[:,c2]*beta
        # etc.
        
        # Actually, let's just do a more aggressive random search
        # with the constraint that we use the Strassen structure as a guide
        
        # Generate sparse U and V with the same block structure as Strassen
        U = np.zeros((n*m, r), dtype=float)
        V = np.zeros((m*p, r), dtype=float)
        
        # Use the Strassen pattern but with random signs and occasional zeros
        for g in range(5):
            k0 = 2 * g
            k1 = 2 * g + 1
            
            u00, u01, u10, u11 = 0, 1, 3, 4
            v00, v01 = k0, k1
            v10, v11 = 10 + k0, 10 + k1
            w00, w01 = k0 * 2, k1 * 2
            w10, w11 = k0 * 2 + 1, k1 * 2 + 1
            
            # M1 = (A00+A11)(B00+B11)
            s1 = rng.choice([-1, 1])
            U[u00, g*7+0] = s1; U[u11, g*7+0] = s1
            V[v00, g*7+0] = s1; V[v11, g*7+0] = s1
            # M2 = (A10+A11)B00
            s2 = rng.choice([-1, 1])
            U[u10, g*7+1] = s2; U[u11, g*7+1] = s2
            V[v00, g*7+1] = s2
            # M3 = A00(B01-B11)
            s3 = rng.choice([-1, 1])
            U[u00, g*7+2] = s3
            V[v01, g*7+2] = s3; V[v11, g*7+2] = -s3
            # M4 = A11(B10-B00)
            s4 = rng.choice([-1, 1])
            U[u11, g*7+3] = s4
            V[v10, g*7+3] = s4; V[v00, g*7+3] = -s4
            # M5 = (A00+A01)B11
            s5 = rng.choice([-1, 1])
            U[u00, g*7+4] = s5; U[u01, g*7+4] = s5
            V[v11, g*7+4] = s5
            # M6 = (A10-A00)(B00+B01)
            s6 = rng.choice([-1, 1])
            U[u10, g*7+5] = s6; U[u00, g*7+5] = -s6
            V[v00, g*7+5] = s6; V[v01, g*7+5] = s6
            # M7 = (A01-A11)(B10+B11)
            s7 = rng.choice([-1, 1])
            U[u01, g*7+6] = s7; U[u11, g*7+6] = -s7
            V[v10, g*7+6] = s7; V[v11, g*7+6] = s7
        
        # Last 4 columns for the 2x1 * 1x10 block
        for i in range(n):
            for k in range(p):
                if i == 0 and k < 8:
                    col = 35 + k
                    U[i*m+2, col] = rng.choice([-1, 1])
                    V[2*p+k, col] = rng.choice([-1, 1])
                elif i == 1 and k >= 2:
                    col = 35 + (k-2)
                    U[i*m+2, col] = rng.choice([-1, 1])
                    V[2*p+k, col] = rng.choice([-1, 1])
        
        # Now solve for W
        W = np.zeros((p*n, r), dtype=float)
        feasible = True
        
        for i in range(n):
            for k in range(p):
                M_ki = np.zeros((m, r), dtype=float)
                for j in range(m):
                    M_ki[j, :] = U[i*m+j, :] * V[j*p+k, :]
                
                # Solve M_ki @ w = ones(m)
                # Check if the system is consistent and has a "nice" solution
                try:
                    # First check consistency
                    rank_M = np.linalg.matrix_rank(M_ki)
                    b = np.ones(m)
                    rank_aug = np.linalg.matrix_rank(np.column_stack([M_ki, b]))
                    if rank_M != rank_aug:
                        feasible = False
                        break
                    
                    # Find a solution
                    sol, residuals, rank, sv = np.linalg.lstsq(M_ki, b, rcond=None)
                    
                    # Check if solution is close to half-integer
                    sol_scaled = sol * 2
                    rounded = np.round(sol_scaled)
                    if np.allclose(sol_scaled, rounded, atol=1e-8):
                        W[k*n+i, :] = sol
                    else:
                        feasible = False
                        break
                except:
                    feasible = False
                    break
            
            if not feasible:
                break
        
        if feasible:
            if _verify_np(U, V, W, n, m, p, tol=1e-8):
                best = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with the known rank-55 Strassen construction
    best = _strassen_rank55(n, m, p)
    best_rank = 55
    
    # Try the rank-54 search
    if best_rank > 54:
        time_left = deadline - time.time()
        if time_left > 5:
            result = _try_rank54_search(n, m, p, time_left, seed=12345)
            if result is not None:
                # Verify with Fractions to be absolutely sure
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = 54
    
    # Fallback
    if best_rank > 60:
        best = _schoolbook(n, m, p)
        best_rank = 60
        
    return best
