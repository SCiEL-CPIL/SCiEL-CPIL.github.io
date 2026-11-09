"""Rewritten program for rank-minimal exact tensor decomposition of 2x3 * 3x4 matrix mult."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    
    The multiplication tensor T[i,j,k] for (i in 0..n-1, j in 0..m-1, k in 0..p-1)
    satisfies: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    We use a systematic search over small ranks with rational/integer coefficients.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # The multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    def verify_decomposition(U, V, W):
        """Check if the decomposition exactly reproduces the multiplication tensor."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook_decomposition():
        """Return the trivial rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def try_rank_with_fixed_UV(rank, U, V, max_attempts=1000):
        """Given fixed U and V, try to find W such that the tensor equation holds.
        
        For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        This is a linear system in the entries of W.
        """
        n, m, p = DIMENSIONS
        
        # Build the linear system: A * w = b
        # where w is the vectorized W (p*n*rank unknowns)
        # and we have n*m*p equations
        
        num_eqs = n * m * p
        num_unk = p * n * rank
        
        # A is num_eqs x num_unk
        # For equation (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # The coefficient of W[k*n+i, c] in equation (i,j,k) is U[i*m+j,c] * V[j*p+k,c]
        # Note: W[k*n+i, c] appears in equation (i,j,k) with coefficient U[i*m+j,c]*V[j*p+k,c]
        # But W[k*n+i, c] also appears in other equations where the first two indices are different
        
        # Actually, let's think more carefully.
        # Equation (i',j',k'): sum_c U[i'*m+j',c] * V[j'*p+k',c] * W[k'*n+i',c] = 1
        # The unknown W[k'*n+i', c] appears in equation (i',j',k') with coefficient U[i'*m+j',c]*V[j'*p+k',c]
        # And W[k'*n+i', c] also appears in equation (i'',j'',k'') if k'*n+i' = k''*n+i''
        # i.e., if i'' = i' (since k'*n+i' = k''*n+i'' implies i'' = i' mod n, and since 0<=i<n, i''=i')
        # Wait no: k'*n + i' = k''*n + i'' => (k'-k'')*n = i'' - i'
        # Since |i'' - i'| < n, we need k'=k'' and i''=i'.
        
        # So W[k*n+i, c] appears ONLY in equations where the first index is i and third index is k.
        # That is, W[k*n+i, c] appears in equations (i, j, k) for all j in 0..m-1.
        
        # So for fixed i, k: the unknowns are W[k*n+i, c] for c in 0..rank-1
        # They appear in equations (i, j, k) for j in 0..m-1
        # Equation (i, j, k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # So for each (i, k), we have a system of m equations in rank unknowns:
        # For j in 0..m-1: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # This is m equations in rank unknowns. If rank >= m, we might be able to solve.
        
        W = np.zeros((p*n, rank))
        
        for i in range(n):
            for k in range(p):
                # System: for j in 0..m-1: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # Let A_jk[c] = U[i*m+j,c] * V[j*p+k,c]
                # Then sum_c A_jk[c] * W[k*n+i,c] = 1 for each j
                
                # Build the m x rank system
                A = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                
                b = np.ones(m)
                
                # Solve A * w = b where w = W[k*n+i, :]
                # Use least squares if underdetermined, or check consistency
                
                try:
                    # Use np.linalg.lstsq for a solution
                    sol, residuals, rank_actual, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = sol
                except:
                    return None
        
        # Verify
        if verify_decomposition(U, V, W):
            return W
        return None
    
    def find_decomposition_for_rank(rank, max_attempts=2000):
        """Try to find a decomposition of the given rank."""
        n, m, p = DIMENSIONS
        np.random.seed(42 + rank)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Generate random small integer/half-integer matrices for U and V
            # Use values from {-1, -0.5, 0, 0.5, 1} to allow half-integers
            choices = [-1.0, -0.5, 0.0, 0.5, 1.0]
            
            U = np.array([[np.random.choice(choices) for _ in range(rank)] for _ in range(n*m)])
            V = np.array([[np.random.choice(choices) for _ in range(rank)] for _ in range(m*p)])
            
            # Skip if U or V has all zeros in a row (degenerate)
            if np.all(U == 0) or np.all(V == 0):
                continue
            
            # Try to solve for W
            W = try_rank_with_fixed_UV(rank, U, V)
            if W is not None:
                return U, V, W
        
        return None
    
    # Try ranks from low to high
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    min_rank = max(n*m, m*p, p*n)  # 12
    max_rank = n * m * p  # 24 (schoolbook)
    
    best_result = None
    best_rank = max_rank
    
    for rank in range(min_rank, max_rank):
        if time.time() > deadline - 10:
            break
        
        result = find_decomposition_for_rank(rank, max_attempts=5000)
        if result is not None:
            U, V, W = result
            # Verify once more
            if verify_decomposition(U, V, W):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
    
    if best_result is not None:
        return best_result
    else:
        return schoolbook_decomposition()
# EVOLVE-BLOCK-END
