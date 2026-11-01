"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Tensor dimensions: A is n x m, B is m x p, C is n x p
    # T[i, j, k] = 1 if this triple contributes to C[i,k] via A[i,j]*B[j,k]
    # Flattened: U index = i*m + j (n*m), V index = j*p + k (m*p), W index = k*n + i (p*n)
    
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12
    
    # Build the target tensor as a 3D array
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k  
                # W index: k*n + i
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try to find a low-rank decomposition
    # For rank r: T[i,j,k] = sum_t U[i,t] * V[j,t] * W[k,t]
    
    best_result = None
    start_time = time.time()
    deadline = 140  # seconds, with 10s safety margin
    
    # Strategy: Try ranks from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) isn't right; use known bounds
    # For matrix mult, rank >= max(n*m, m*p, n*p) is not standard
    # Actually, the rank must be at least max(nm, mp, pn) in some cases, but not always
    # Let's just try from reasonable values
    
    # Known: schoolbook is nm*mp/pn... no, it's n*m*p = 48
    # Let's try to find decompositions with rank < 48
    
    # Use a numerical approach: for a given rank r, we can try to solve
    # by fixing two matrices and solving for the third, or using iterative methods
    
    # Alternative: use the fact that for small dimensions, we can do an exhaustive
    # search over possible "patterns" of the decomposition
    
    # Let me try a different approach: use bilinear algorithm structure
    # For 2x4x6, try to use a block-based approach
    
    # Actually, let me try to use a randomized approach with verification
    # For each candidate rank r, generate random U, V, W and check if they work
    
    # Better: use the structure. The tensor T has a specific structure.
    # Let me try to find a decomposition by solving linear systems.
    
    # For rank r, if we fix U and V, then W is determined by:
    # For each (i,j,k): sum_t U[i,t]*V[j,t]*W[k,t] = T[i,j,k]
    # This is linear in W if U,V are fixed.
    
    # Let me try a more systematic approach:
    # 1. Guess a structure for the decomposition
    # 2. Solve for the entries
    
    # For now, let's try to use a numerical optimization approach
    
    def try_rank(r, max_iters=1000):
        """Try to find a rank-r decomposition using iterative refinement."""
        if time.time() - start_time > deadline:
            return None
        
        # Initialize randomly
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        # Normalize to avoid scale issues
        for t in range(r):
            norm = np.linalg.norm(U[:, t]) * np.linalg.norm(V[:, t]) * np.linalg.norm(W[:, t])
            if norm > 1e-10:
                scale = norm ** (1/3)
                U[:, t] /= scale
                V[:, t] /= scale
                W[:, t] /= scale
        
        for it in range(max_iters):
            if time.time() - start_time > deadline:
                return None
            
            # Update U: for each i, solve for U[i,:] given V, W
            # T[i,j,k] = sum_t U[i,t] * V[j,t] * W[k,t]
            # For fixed i, this is: sum_k T[i,j,k] * W[k,t] = U[i,t] * sum_k V[j,t] * W[k,t]... no
            # Actually, it's more complex. Let's use gradient descent.
            
            # Compute current approximation
            # Approx[i,j,k] = sum_t U[i,t] * V[j,t] * W[k,t]
            # This is expensive for large r, but our dimensions are small
            
            # Use a simpler approach: alternating least squares
            # Fix V and W, solve for U:
            # For each i, j: T[i,j,:] = U[i,:] * (V[j,:] ⊙ W...) 
            # Actually, T[i,j,k] = sum_t U[i,t] * V[j,t] * W[k,t]
            # Let M[j,k] = sum_t V[j,t] * W[k,t] (outer product sum)
            # Then T[i,j,k] = sum_t U[i,t] * V[j,t] * W[k,t]
            # This isn't simply U[i,:] dot something unless we restructure.
            
            # Let me use a different formulation:
            # Reshape: For each t, we have outer product U[:,t] ⊗ V[:,t] ⊗ W[:,t]
            # The sum should equal T.
            
            # Gradient descent approach:
            # dL/dU[i,t] = sum_{j,k} (T[i,j,k] - Approx[i,j,k]) * V[j,t] * W[k,t]
            # dL/dV[j,t] = sum_{i,k} (T[i,j,k] - Approx[i,j,k]) * U[i,t] * W[k,t]
            # dL/dW[k,t] = sum_{i,j} (T[i,j,k] - Approx[i,j,k]) * U[i,t] * V[j,t]
            
            lr = 0.01
            for _ in range(10):  # inner iterations
                # Compute Approx
                Approx = np.zeros((nm, mp, pn), dtype=float)
                for t in range(r):
                    Approx += np.einsum('i,j,k->ijk', U[:, t], V[:, t], W[:, t])
                
                # Compute gradients
                diff = T - Approx
                
                # dU
                dU = np.zeros((nm, r), dtype=float)
                for t in range(r):
                    dU[:, t] = np.einsum('ijk,jt,kt->it', diff, V, W)
                
                # dV
                dV = np.zeros((mp, r), dtype=float)
                for t in range(r):
                    dV[:, t] = np.einsum('ijk,it,kt->jt', diff, U, W)
                
                # dW
                dW = np.zeros((pn, r), dtype=float)
                for t in range(r):
                    dW[:, t] = np.einsum('ijk,it,jt->kt', diff, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Check convergence
            err = np.max(np.abs(T - Approx))
            if err < 1e-8:
                # Found a solution! Now try to find exact rational entries
                return (U, V, W)
        
        return None
    
    # Try ranks from a reasonable lower bound
    # For 2x4x6, let's try from 20 upward (schoolbook is 48)
    # Actually, let me first try to see if there's a known good bound
    
    # Let me try a different, more efficient approach:
    # Use the fact that the tensor has a specific structure and try to
    # find a decomposition by solving linear systems directly.
    
    # For a given rank r, the condition is that there exist U, V, W such that
    # sum_t U[i,t] V[j,t] W[k,t] = T[i,j,k]
    
    # This is equivalent to: the matrix T (reshaped) has a certain property.
    
    # Let me try a constructive approach based on the structure.
    # The key observation: T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i=i', j=j', k=k'
    
    # So the nonzero entries are at positions where the indices are "consistent."
    
    # Let me try to use a block decomposition approach.
    # Split A into 2x2 blocks (each 1x2), B into 4x2 blocks (each 2x3)...
    
    # Actually, let me just try the numerical approach for various ranks.
    
    random.seed(42)
    np.random.seed(42)
    
    for r in range(24, 49):  # Try from 24 to 47
        if time.time() - start_time > deadline:
            break
        
        result = try_rank(r, max_iters=200)
        if result is not None:
            U, V, W = result
            
            # Try to find exact rational representations
            # Round to nearest rational with small denominator
            def to_rational_array(arr, max_denom=100):
                """Convert array to fractions with small denominators."""
                result = []
                for val in arr.flatten():
                    frac = Fraction(val).limit_denominator(max_denom)
                    result.append(frac)
                return np.array(result).reshape(arr.shape)
            
            U_frac = to_rational_array(U, max_denom=100)
            V_frac = to_rational_array(V, max_denom=100)
            W_frac = to_rational_array(W, max_denom=100)
            
            # Verify the rational decomposition
            max_err = 0
            for i in range(nm):
                for j in range(mp):
                    for k in range(pn):
                        val = sum(float(U_frac[i, t]) * float(V_frac[j, t]) * float(W_frac[k, t]) for t in range(r))
                        target = T[i, j, k]
                        max_err = max(max_err, abs(val - target))
            
            if max_err < 1e-6:
                # Convert to list format
                U_list = [[float(U_frac[i, t]) for t in range(r)] for i in range(nm)]
                V_list = [[float(V_frac[j, t]) for t in range(r)] for j in range(mp)]
                W_list = [[float(W_frac[k, t]) for t in range(r)] for k in range(pn)]
                
                # Check if entries are integer or half-integer
                def is_half_integer(x):
                    return abs(x * 2 - round(x * 2)) < 1e-9
                
                all_half_int = all(is_half_integer(U_list[i][t]) for i in range(nm) for t in range(r))
                all_half_int = all_half_int and all(is_half_integer(V_list[j][t]) for j in range(mp) for t in range(r))
                all_half_int = all_half_int and all(is_half_integer(W_list[k][t]) for k in range(pn) for t in range(r))
                
                if all_half_int:
                    # Convert to exact half-integers
                    def to_half_int_list(lst):
                        return [[int(round(x * 2)) / 2 for x in row] for row in lst]
                    
                    U_list = to_half_int_list(U_list)
                    V_list = to_half_int_list(V_list)
                    W_list = to_half_int_list(W_list)
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Even if not half-integer, return if close enough (the evaluator may accept)
            # But the contract says integer/half-integer, so let's be strict
            # Try with smaller denominators
            for md in [1, 2, 4, 10, 20, 50]:
                U_frac = to_rational_array(U, max_denom=md)
                V_frac = to_rational_array(V, max_denom=md)
                W_frac = to_rational_array(W, max_denom=md)
                
                max_err = 0
                for i in range(nm):
                    for j in range(mp):
                        for k in range(pn):
                            val = sum(float(U_frac[i, t]) * float(V_frac[j, t]) * float(W_frac[k, t]) for t in range(r))
                            target = T[i, j, k]
                            max_err = max(max_err, abs(val - target))
                
                if max_err < 1e-6:
                    U_list = [[float(U_frac[i, t]) for t in range(r)] for i in range(nm)]
                    V_list = [[float(V_frac[j, t]) for t in range(r)] for j in range(mp)]
                    W_list = [[float(W_frac[k, t]) for t in range(r)] for k in range(pn)]
                    
                    all_half_int = all(is_half_integer(U_list[i][t]) for i in range(nm) for t in range(r))
                    all_half_int = all_half_int and all(is_half_integer(V_list[j][t]) for j in range(mp) for t in range(r))
                    all_half_int = all_half_int and all(is_half_integer(W_list[k][t]) for k in range(pn) for t in range(r))
                    
                    if all_half_int:
                        U_list = to_half_int_list(U_list)
                        V_list = to_half_int_list(V_list)
                        W_list = to_half_int_list(W_list)
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better decomposition found, return schoolbook
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
# EVOLVE-BLOCK-END
