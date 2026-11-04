"""Matrix multiplication 3x3x3 - attempting rank-23 decomposition."""
import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    N = n * m  # 9
    M = m * p  # 9
    P = p * n  # 9
    
    # Schoolbook baseline (rank 27)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the target tensor (matrix multiplication tensor)
    # T[a,b,c,d] = 1 if this corresponds to C[i,k] = sum_j A[i,j]B[j,k]
    # In the decomposition: sum_r U[a,b,r]*V[c,d,r]*W[k,i,r] = delta(a,i)*delta(b,c)*delta(d,k)
    # where indices are: a=i (row of A), b=j (col of A), c=j' (row of B), d=k' (col of B)
    # The output is C[i,k], and W index is (k,i) in the contract.
    
    # Let's build the system of equations.
    # Variables: U[0..8][0..R-1], V[0..8][0..R-1], W[0..8][0..R-1]
    # For each (a,b,c,d,i,k) tuple, we need:
    #   sum_r U[a,b,r] * V[c,d,r] * W[3*k+i, r] = delta(a,i) * delta(b,c) * delta(d,k)
    # But actually, the equation is only constrained when (a,i) match and (b,c) match and (d,k) match.
    # More precisely: for all a,b,c,d: the output C[i,k] for all i,k must be correct.
    # If A = E_{a,b} and B = E_{c,d}, then C should be E_{a,d} if b==c, else 0.
    # So: sum_r U[a,b,r]*V[c,d,r]*W[3*k+i,r] = 1 if (a==i, b==c, d==k) else 0
    
    # Number of equations: 3*3*3*3 * 3*3 = 243, but many are trivially 0.
    # Non-zero equations: when a==i, b==c, d==k: 3*3*3 = 27 equations equal 1.
    # Zero equations: 243 - 27 = 216 equations equal 0.
    
    # This is a system of 243 cubic equations in 3*9*R variables.
    # For R=23: 3*9*23 = 621 variables.
    
    # Strategy: Try to find a rank-23 solution using iterative methods.
    # The known minimal rank for 3x3 matrix multiplication is between 23 and 27.
    # Actually, it's known that rank(3x3) <= 23 (Bini et al. 1980 showed <= 23).
    # But finding an explicit decomposition is non-trivial.
    
    # Let me try a simpler approach: use the known fact that we can potentially
    # reduce from 27 by combining some terms.
    
    # Actually, let me try a local search / gradient-based approach on the error.
    
    def compute_error(U, V, W, R):
        """Compute the error of the decomposition."""
        err = 0.0
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                lhs = 0.0
                                for r in range(R):
                                    lhs += U[a*m+b, r] * V[c*p+d, r] * W[k*n+i, r]
                                rhs = 1.0 if (a == i and b == c and d == k) else 0.0
                                err += (lhs - rhs) ** 2
        return err
    
    # Try to find a lower-rank decomposition.
    # Start from schoolbook and try to perturb and optimize.
    
    best_result = schoolbook()
    best_error = 0.0
    
    # Try random restarts with local optimization
    max_iterations = 50000
    lr = 0.01
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        R = 27  # Start with rank 27, try to reduce
        # Actually, let's just try to find ANY valid decomposition with rank < 27
        # by using a different parametrization.
        
        # Alternative: try rank 24, 25, 26 first as intermediate steps.
        for target_rank in [26, 25, 24, 23]:
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            U = np.random.randn(N, target_rank) * 0.5
            V = np.random.randn(M, target_rank) * 0.5
            W = np.random.randn(P, target_rank) * 0.5
            
            # For a few iterations, do gradient descent on the error
            # The error is sum of cubed terms, so gradients are complex.
            # Let's use a simpler approach: coordinate descent.
            
            improved = False
            for it in range(max_iterations // 5):
                if time.time() > deadline - 10:
                    break
                
                # Pick a random variable to update
                mat = random.choice(['U', 'V', 'W'])
                row = random.randint(0, N-1) if mat == 'U' else (random.randint(0, M-1) if mat == 'V' else random.randint(0, P-1))
                col = random.randint(0, target_rank - 1)
                
                # Try small perturbations
                current_val = U[row, col] if mat == 'U' else (V[row, col] if mat == 'V' else W[row, col])
                
                best_delta = 0
                best_err = compute_error(U.tolist(), V.tolist(), W.tolist(), target_rank)
                
                for delta in [0.01, -0.01, 0.001, -0.001, 0.1, -0.1]:
                    if mat == 'U':
                        U[row, col] = current_val + delta
                        err = compute_error(U.tolist(), V.tolist(), W.tolist(), target_rank)
                        U[row, col] = current_val
                    elif mat == 'V':
                        V[row, col] = current_val + delta
                        err = compute_error(U.tolist(), V.tolist(), W.tolist(), target_rank)
                        V[row, col] = current_val
                    else:
                        W[row, col] = current_val + delta
                        err = compute_error(U.tolist(), V.tolist(), W.tolist(), target_rank)
                        W[row, col] = current_val
                    
                    if err < best_err - 1e-12:
                        best_err = err
                        best_delta = delta
                
                if best_delta != 0:
                    if mat == 'U':
                        U[row, col] += best_delta
                    elif mat == 'V':
                        V[row, col] += best_delta
                    else:
                        W[row, col] += best_delta
                    improved = True
                elif it % 1000 == 0 and not improved:
                    break
            
            if best_err < 1e-6:
                # Found a valid decomposition!
                # Convert to list format
                result = {
                    "U": [[float(x) for x in U[r]] for r in range(N)],
                    "V": [[float(x) for x in V[r]] for r in range(M)],
                    "W": [[float(x) for x in W[r]] for r in range(P)]
                }
                return result
        
        if time.time() > deadline - 10:
            break
    
    # If we didn't find a better decomposition, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
