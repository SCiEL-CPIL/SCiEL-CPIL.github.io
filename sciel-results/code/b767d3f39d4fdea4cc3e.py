"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Use a "Fix U, V, Solve W" approach.
    2. For a given rank R, generate random sparse U and V.
    3. For each output index c, solve the linear system for W[:, c].
    4. If a solution exists and entries are integer/half-integer, verify and return.
    5. Fall back to schoolbook (rank 36) if no solution found in time.
    """
    start_time = time.time()
    deadline = start_time + 130  # 130s budget, leaving 20s margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    nm, mp, pn = n * m, m * p, p * n  # 9, 12, 12
    
    # Precompute the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, r):
        """Check if U, V, W with rank r satisfy all constraints."""
        # Convert to numpy for speed
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Compute tensor
        # T_comp[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We can do this efficiently by iterating over r or using einsum
        # T_comp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        # But for small dimensions, explicit loops or vectorized ops are fine.
        
        # Let's use a loop over r to accumulate
        T_comp = np.zeros((nm, mp, pn), dtype=float)
        for r in range(r):
            T_comp += U_np[:, r, None, None] * V_np[None, :, r, None] * W_np[None, None, :, r]
            
        # Check against T
        if not np.allclose(T_comp, T, atol=1e-9):
            return False
        
        # Check if entries are integer or half-integer
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if isinstance(val, Fraction):
                        if val.denominator not in [1, 2]:
                            return False
                    else:
                        # If it's a float, check if it's close to int or half-int
                        v = float(val)
                        if abs(v - round(2*v)) > 1e-9:
                            return False
        return True

    def try_rank(R, max_attempts=500):
        """
        Try to find a rank-R decomposition by fixing U, V and solving for W.
        """
        if time.time() > deadline:
            return None
            
        t_start = time.time()
        
        # We want U (9xR), V (12xR), W (12xR)
        # Linear system for W_c:
        # For each c, we have 108 equations (9*12) and R unknowns.
        # A_c is 108 x R. A_c[(a*b + b), r] = U[a, r] * V[b, r]
        # b_c is 108 x 1. b_c[(a*b + b)] = T[a, b, c]
        
        for attempt in range(max_attempts):
            if time.time() > deadline or (time.time() - t_start) > 20: # 20s per rank attempt
                break
                
            # Generate random sparse U and V
            # Use values from {-1, 0, 1}
            # Density: start with low density, maybe increase if R is small?
            # For R=36, density 1/36 is expected.
            # Let's use density 1/max(3, R/10)
            density = 1.0 / max(3, R / 10)
            
            U = np.zeros((nm, R), dtype=int)
            V = np.zeros((mp, R), dtype=int)
            
            for r in range(R):
                # Random non-zero positions
                # Ensure each column has some structure?
                # Just random sparsity
                for a in range(nm):
                    if random.random() < density:
                        U[a, r] = random.choice([-1, 1])
                for b in range(mp):
                    if random.random() < density:
                        V[b, r] = random.choice([-1, 1])
                        
            # Check if U and V are valid (non-zero columns)
            if np.all(U == 0) or np.all(V == 0):
                continue
                
            # Try to solve for W
            W = np.zeros((pn, R), dtype=float)
            valid_solution = True
            
            for c in range(pn):
                # Build A_c
                # Rows: (a, b) -> index a*mp + b
                # Cols: r
                A_c = np.zeros((nm * mp, R), dtype=float)
                b_c = np.zeros(nm * mp, dtype=float)
                
                for a in range(nm):
                    for b in range(mp):
                        idx = a * mp + b
                        # A_c[idx, r] = U[a, r] * V[b, r]
                        A_c[idx, :] = U[:, r] * V[b, r] # Wait, U[:, r] is vector of a, V[b, r] is scalar
                        # Correct: A_c[idx, r] = U[a, r] * V[b, r]
                        # This loop is slow. Let's vectorize.
                        pass
                
                # Vectorized construction of A_c
                # A_c is outer product of U and V flattened?
                # U is (nm, R), V is (mp, R)
                # We want A_c[idx, r] = U[a, r] * V[b, r]
                # This is not a standard outer product because the r index is shared.
                # It's element-wise product of U and V broadcast?
                # Let's construct it properly.
                # A_c[:, r] = vec(U[:, r] * V[:, r]) ? No.
                # A_c[(a,b), r] = U[a, r] V[b, r].
                # This is the Kronecker product of U[:, r] and V[:, r] flattened?
                # No, it's just the element-wise product of the column vectors U[:, r] and V[:, r] 
                # expanded to all pairs (a,b).
                # Actually, for a fixed r, the vector of coefficients for W_c[r] is 
                # U[:, r] * V[:, r] ? No.
                # The equation is sum_r (U[a, r] V[b, r]) W[c, r] = T[a, b, c].
                # So for fixed c, the coefficient of W[c, r] in equation (a,b) is U[a, r] V[b, r].
                # So A_c[:, r] is the vector of length nm*mp where element (a,b) is U[a, r] V[b, r].
                
                # Let's build A_c efficiently
                # A_c[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                A_c = np.zeros((nm * mp, R), dtype=float)
                for r in range(R):
                    A_c[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
                # b_c is just T[:, :, c].flatten()
                b_c = T[:, :, c].flatten()
                
                # Solve A_c w_c = b_c
                # Use least squares
                try:
                    w_c, residuals, rank, sv = np.linalg.lstsq(A_c, b_c, rcond=None)
                    # Check residual
                    if residuals is not None and len(residuals) > 0:
                        if residuals[0] > 1e-8:
                            valid_solution = False
                            break
                    else:
                        # Check norm of residual
                        res_norm = np.linalg.norm(A_c @ w_c - b_c)
                        if res_norm > 1e-8:
                            valid_solution = False
                            break
                            
                    W[c, :] = w_c
                except np.linalg.LinAlgError:
                    valid_solution = False
                    break
                    
            if not valid_solution:
                continue
                
            # Check if W entries are close to integer or half-integer
            # And U, V are integers (they are)
            W_int = np.round(W * 2) / 2.0
            if not np.allclose(W, W_int, atol=1e-8):
                continue
                
            # Construct final matrices
            U_final = [[int(U[a, r]) for r in range(R)] for a in range(nm)]
            V_final = [[int(V[b, r]) for r in range(R)] for b in range(mp)]
            W_final = [[Fraction(int(W_int[c, r] * 2), 2) for r in range(R)] for c in range(pn)]
            
            # Final verification
            if verify(U_final, V_final, W_final, R):
                return {"U": U_final, "V": V_final, "W": W_final}
                
        return None

    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 22 to 35
    # We can try a few specific ranks that are likely to have solutions
    # Rank 36 is schoolbook.
    # Try 35, 34, ... down to 22.
    # To save time, we can try a subset or allocate time per rank.
    
    ranks_to_try = list(range(35, 21, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        result = try_rank(r, max_attempts=100)
        if result is not None:
            best_result = result
            best_rank = r
            break
            
    return best_result
# EVOLVE-BLOCK-END
