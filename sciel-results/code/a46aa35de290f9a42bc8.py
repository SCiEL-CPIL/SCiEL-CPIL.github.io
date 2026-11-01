"""Rewritten program for matrix multiplication 2,4,5 rank-minimal decomposition."""
import numpy as np
from itertools import combinations
import random

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-9):
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Build tensor T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Expected: 1 if i in [0,n), j in [0,m), k in [0,p) for the standard matmul tensor?
    # Actually the tensor for matrix multiplication C = A B has entries C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The decomposition identity is:
    # sum_r U[i,j,r] V[j,k,r] W[i,k,r] = delta_{i,k?} no.
    # Standard: T_{i,j,k} = 1 if we are multiplying? Wait, the tensor is T_{i,j,k} such that
    # C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}. So T_{i,j,k} = 1 for all i,j,k? No.
    # Actually T_{i,j,k} is 1 if the indices match the multiplication structure.
    # For C = A B, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor T has T_{i,j,k} = 1 for all valid i,j,k? No, that's not a tensor decomposition of a fixed tensor.
    # The "matrix multiplication tensor" T_{n,m,p} is defined by T_{i,j,k} = 1 if we consider the trilinear form
    # sum_{i,j,k} T_{i,j,k} x_i y_j z_k = sum_i,k C_{i,k} ...
    # Actually, the standard definition is: T_{i,j,k} = 1 if the indices correspond to the multiplication.
    # Specifically, T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p) is NOT correct.
    # The tensor is T_{i,j,k} such that the matmul is: C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}.
    # This implies T_{i,j,k} must be 1 for all i,j,k? No, that would mean C_{i,k} = sum_j A_{i,j} B_{j,k} which is correct.
    # Wait, if T_{i,j,k} = 1 for all i,j,k, then the tensor is all ones.
    # Is the matrix multiplication tensor all ones?
    # Let's check the rank. Rank of all-ones tensor of size 2x4x5 is 1 (if we allow arbitrary reals).
    # But matrix multiplication is not rank 1.
    # The confusion: The tensor T_{n,m,p} is not "all ones".
    # The identity is: sum_{j} A_{i,j} B_{j,k} = C_{i,k}.
    # In tensor notation: <T, A \otimes B \otimes C^T> = 0?
    # The standard tensor T_{n,m,p} has entries T_{i,j,k} = 1 if i,j,k are indices?
    # No, the tensor is defined such that the bilinear map is multiplication.
    # The tensor T is an element of R^{n*m} \otimes R^{m*p} \otimes R^{p*n}.
    # T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}?
    # Let's look at the seed code:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # This implies T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # In schoolbook, for each (i,j,k), there is a term with U=1, V=1, W=1.
    # So T_{i,j,k} = 1 for all i,j,k.
    # So the target tensor IS the all-ones tensor of dimensions (n*m) x (m*p) x (p*n)?
    # No, the dimensions are (n*m) for U, (m*p) for V, (p*n) for W.
    # The indices are:
    # U index: a = i*m + j
    # V index: b = j*p + k
    # W index: c = k*n + i
    # The tensor T_{a,b,c} = 1 if there exists i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i.
    # Since i,j,k are uniquely determined by a,b,c?
    # a -> i = a//m, j = a%m
    # b -> j' = b//p, k = b%p. We need j == j'.
    # c -> k' = c//n, i' = c%n. We need k == k' and i == i'.
    # So T_{a,b,c} = 1 if j(a) == j(b) and k(b) == k(c) and i(c) == i(a).
    # Otherwise T_{a,b,c} = 0.
    # So it is NOT the all-ones tensor. It is a sparse tensor with 1s at specific positions.
    
    # Let's construct the target tensor T as a 3D array of size (n*m, m*p, p*n).
    # T[a, b, c] = 1 if (a//m == c%n) and (a%m == b//p) and (b%p == c//n).
    # Else 0.
    
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Compute decomposed tensor D_{a,b,c} = sum_r U[a,r] V[b,r] W[c,r]
    # This is a tensor product contraction.
    # D = np.einsum('ar,br,cr->abc', U, V, W)
    D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    err = np.max(np.abs(D - target))
    return err < tol

def construct():
    n, m, p = DIMENSIONS
    # Try to find a rank-16 decomposition
    # Strategy: Fix U and V to have simple structures (e.g., 0/1) and solve for W?
    # Or fix U, solve for V, W?
    # Given the asymmetry n=2, we can try to exploit that.
    # The tensor T has support size n*m*p = 40.
    # Rank 16 is a guess based on "2 * 4 * (5-1) + 2" = 34? No.
    # Let's try rank 16 first as per directive "rank-16 decomposition".
    # If we can't find it, we fall back to schoolbook (rank 40).
    # We need integer/half-integer entries.
    
    # Heuristic: Use a random subset of the support to define the rank-1 terms?
    # Or use a structured approach.
    # Let's try to solve a linear system for W given random U, V?
    # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
    # This is linear in W if U, V are fixed.
    # For each r, we have a term.
    # Actually, it's linear in the vectorized W.
    # Let U, V be fixed. Then for each c, we have a vector equation?
    # No, T_{a,b,c} is the sum over r.
    # This is a system of linear equations for the entries of W.
    # Variables: W_{c,r} for c in [0, p*n), r in [0, rank).
    # Equations: For each (a,b,c) in support, sum_r U_{a,r} V_{b,r} W_{c,r} = 1.
    # For (a,b,c) not in support, sum_r U_{a,r} V_{b,r} W_{c,r} = 0.
    # Total equations: n*m*p*n? No, size of T is (n*m)*(m*p)*(p*n).
    # That's 8 * 20 * 10 = 1600 equations.
    # Variables: rank * p * n = 16 * 10 = 160 variables.
    # This is an overdetermined system.
    # We can use least squares, but we need exact integer/half-integer solutions.
    
    # Let's try a specific structured U, V.
    # Since n=2, maybe U can be simple.
    # Let's try to find a solution with U, V, W having entries in {0, 1, 1/2, -1/2, -1}.
    
    # We will perform a random search for U, V with small entries, then solve for W exactly.
    # To make it exact, we can use rational arithmetic or check if the least squares solution is close to half-integers.
    
    rng = np.random.default_rng(42)
    
    # Try rank 16
    target_rank = 16
    num_trials = 100
    
    for trial in range(num_trials):
        # Generate random U, V with entries in {-1, 0, 1}
        # To increase chance of success, maybe use sparse matrices?
        U_rand = rng.integers(-1, 2, size=(n*m, target_rank))
        V_rand = rng.integers(-1, 2, size=(m*p, target_rank))
        
        # Construct the linear system A W = b
        # A is (num_equations x num_vars)
        # num_vars = p*n*target_rank
        # num_equations = n*m*m*p*p*n? No, the size of the tensor.
        # We only need to satisfy the support equations (40) and maybe some zero equations.
        # Actually, satisfying all 1600 equations is required.
        
        # Let's build the matrix A.
        # A_{(a,b,c), (c',r)} = U[a,r] * V[b,r] if c == c' else 0.
        # This is block diagonal in c?
        # For each c, the equation for all (a,b) is:
        # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        # This is a system for W[c, :] given U, V.
        # So we can solve for each c independently?
        # Yes! The equations for different c are decoupled.
        # For a fixed c, we have variables W[c, 0], ..., W[c, rank-1].
        # The equations are for all a in [0, n*m), b in [0, m*p).
        # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        # This is a linear system M_c * W_c = t_c
        # where M_c[a, b, r] = U[a,r] V[b,r].
        # This can be reshaped.
        # Let U_c = U (same for all c), V_c = V.
        # The equation is vec( U^T * diag(W_c) * V )? No.
        # sum_r U[a,r] W[c,r] V[b,r] = (U * diag(W_c) * V^T)_{a,b} = T_c[a,b]
        # So we need to solve U * diag(W_c) * V^T = T_c
        # This is a linear system for W_c.
        # Let X_c = diag(W_c). Then U X_c V^T = T_c.
        # This is a Sylvester-like equation but with diagonal X.
        # We can vectorize: vec(U X_c V^T) = (V \otimes U) vec(X_c)
        # Since X_c is diagonal, vec(X_c) is just W_c.
        # So (V \otimes U) W_c = vec(T_c)
        
        # Let's check if this system is solvable for W_c.
        # A_c = np.kron(V_rand, U_rand)
        # b_c = T_c.flatten()
        # W_c = np.linalg.lstsq(A_c, b_c, rcond=None)[0]
        
        # We need W_c to be half-integers.
        
        T_full = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_full[a, b, c] = 1.0
        
        W_solution = []
        valid = True
        for c in range(p * n):
            T_c = T_full[:, :, c]
            A_c = np.kron(V_rand, U_rand)
            b_c = T_c.flatten()
            
            # Check if system is consistent
            # We'll use least squares and check residual
            W_c, residuals, rank_c, s = np.linalg.lstsq(A_c, b_c, rcond=None)
            
            # Check if W_c is close to half-integers
            # Multiply by 2, round, check
            W_c_2 = W_c * 2
            W_c_rounded = np.round(W_c_2)
            if np.max(np.abs(W_c_2 - W_c_rounded)) > 1e-6:
                valid = False
                break
            W_solution.append(W_c_rounded / 2.0)
            
        if not valid:
            continue
            
        W_sol = np.array(W_solution).T # Shape (p*n, rank)
        
        # Verify
        U_sol = U_rand
        V_sol = V_rand
        
        if _verify(U_sol, V_sol, W_sol):
            # Convert to lists of numbers
            # Ensure they are int or float (half-integers)
            U_list = [[float(x) for x in row] for row in U_sol]
            V_list = [[float(x) for x in row] for row in V_sol]
            W_list = [[float(x) for x in row] for row in W_sol]
            
            # Clean up floats like 1.0 to 1? The problem says integer/half-integer.
            # Returning floats is usually fine if they are exact.
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
