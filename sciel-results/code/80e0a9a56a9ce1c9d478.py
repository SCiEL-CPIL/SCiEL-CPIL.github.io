"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    
    We use a structured search approach:
    1. Start with the known rank-16 decomposition (if it exists for these dimensions)
    2. Try to find rank-15, rank-14, etc. using randomized/local search
    3. Verify exactness using rational arithmetic
    
    For 2x3 * 3x4, the schoolbook rank is 24.
    Theoretical lower bound: max(2*3, 3*4, 2*4) = max(6, 12, 8) = 12
    Known results suggest rank might be around 15-18 for such dimensions.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = 2, 3, 4  # A is n x m, B is m x p, C = A*B is n x p
    
    # The tensor T_{i,j,k} corresponds to C[i,k] = sum_j A[i,j] * B[j,k]
    # In the factorization: C[i,k] = sum_r U[i*m + j? ...]
    # Wait, let's re-read the interface:
    # U: (n*m, rank) -> indexed by [i*m + j]
    # V: (m*p, rank) -> indexed by [j*p + k]  
    # W: (p*n, rank) -> indexed by [k*n + i] (trace convention W[k*n+i, r])
    #
    # The decomposition requires:
    # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ... 
    # Actually, the tensor multiplication identity is:
    # sum_j A[i,j] * B[j,k] = C[i,k]
    # Using factors: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])?
    # No, the standard tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_r U[i,j,r] * A[i,j] * V[j,k,r] ... 
    # 
    # Let me re-read the task description more carefully.
    # "U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)"
    # The standard formulation: 
    #   C[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r]
    # But this doesn't involve W directly in the product...
    # 
    # Actually, looking at the seed program:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1  
    # W[k*n+i][column] = 1
    # 
    # The verification should be: for each column r,
    # sum_{i,j,k} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should reconstruct...
    # 
    # Let me think about this differently. The matrix multiplication tensor has indices (i, j, k)
    # where i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 if this term appears in C[i,k] = sum_j A[i,j]B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k).
    #
    # A rank-1 tensor has the form u[i,j] * v[j,k] * w[k,i]... no that's not right either.
    # 
    # Standard tensor decomposition for matrix mult:
    # The multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 for all i,j,k.
    # A rank-r decomposition: M[i,j,k] = sum_{r=1}^r U_r[i,j] * V_r[j,k] * W_r[k,i]
    # where U_r is n x m, V_r is m x p, W_r is p x n.
    #
    # In the flattened interface:
    # U[i*m+j, r] = U_r[i,j]
    # V[j*p+k, r] = V_r[j,k]
    # W[k*n+i, r] = W_r[k,i]
    #
    # So we need: for all i,j,k: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1
    #
    # This is a system of n*m*p = 24 equations in 3*r variables (well, r*(n*m + m*p + p*n) variables).
    # But the equations are nonlinear (products of three terms).
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        r = len(U[0])
        # For each (i,j,k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_exact(U, V, W, n, m, p):
        """Verify with exact arithmetic using fractions."""
        from fractions import Fraction
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r_idx in range(r):
                        # Convert to Fraction
                        u_val = U[i*m+j][r_idx]
                        v_val = V[j*p+k][r_idx]
                        w_val = W[k*n+i][r_idx]
                        # Handle both float and Fraction
                        if isinstance(u_val, float):
                            u_frac = Fraction(str(u_val))
                        else:
                            u_frac = Fraction(u_val)
                        if isinstance(v_val, float):
                            v_frac = Fraction(str(v_val))
                        else:
                            v_frac = Fraction(v_val)
                        if isinstance(w_val, float):
                            w_frac = Fraction(str(w_val))
                        else:
                            w_frac = Fraction(w_val)
                        s += u_frac * v_frac * w_frac
                    if s != Fraction(1):
                        return False
        return True
    
    # Start with schoolbook rank-24 solution
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank solutions
    # Strategy: use randomized search with local optimization
    # For a given rank r, we have r columns. Each column (i,j,k) pattern must satisfy
    # the constraint that the sum over r of products equals 1.
    
    # Let's try a structured approach: assume entries are from {0, 1, -1, 1/2, -1/2, 2, -2}
    # and search for valid decompositions.
    
    def try_rank(r, max_iterations=10000):
        """Try to find a rank-r decomposition using randomized search."""
        if time.time() > deadline - 10:
            return None
        
        # Use a smarter initialization based on known patterns
        # For 2x3x4, try various structures
        
        # Approach: start with random sparse matrices and use gradient-like updates
        # But since we need exact solutions, let's try a different approach.
        
        # Try to build up from rank 1 components
        # Each rank-1 component is defined by three matrices U_r (2x3), V_r (3x4), W_r (4x2)
        # such that sum_r U_r[i,j]*V_r[j,k]*W_r[k,i] = 1 for all i,j,k
        
        # Let's try a structured search with small integer/half-integer values
        candidate_values = [0, 1, -1, 0.5, -0.5, 2, -2]
        
        # Random restarts
        for restart in range(50):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random sparse matrices
            U = [[random.choice(candidate_values) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice(candidate_values) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice(candidate_values) for _ in range(r)] for _ in range(p*n)]
            
            # Local search: try to improve the fit
            best_err = float('inf')
            for it in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r_idx in range(r):
                                s += U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                            err += (s - 1.0) ** 2
                
                if err < 1e-12:
                    # Check exactness
                    if verify(U, V, W, n, m, p):
                        return U, V, W
                    if verify_exact(U, V, W, n, m, p):
                        return U, V, W
                
                # Try a random perturbation
                choice = random.randint(0, 2)
                if choice == 0:
                    idx = random.randint(0, n*m-1)
                    col = random.randint(0, r-1)
                    U[idx][col] = random.choice(candidate_values)
                elif choice == 1:
                    idx = random.randint(0, m*p-1)
                    col = random.randint(0, r-1)
                    V[idx][col] = random.choice(candidate_values)
                else:
                    idx = random.randint(0, p*n-1)
                    col = random.randint(0, r-1)
                    W[idx][col] = random.choice(candidate_values)
        
        return None
    
    # Try ranks from low to high
    for r in range(12, min(24, best_rank)):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            if verify_exact(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
    
    # If we found something better, return it; otherwise return schoolbook
    if best_rank < n * m * p:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        U, V, W = schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
