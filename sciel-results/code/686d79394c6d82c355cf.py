"""Find rank-minimal tensor decomposition for 3x3x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin before 160s limit
    
    n, m, p = DIMENSIONS
    # The tensor T has entries T[i, j, k] where:
    # i in [0, n), j in [0, m), k in [0, p)
    # For matrix mult C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 (the coefficient)
    
    # The decomposition: for each rank term r, we have:
    # U[r, i*m+j], V[r, j*p+k], W[r, k*n+i]
    # And the condition is: sum_r U[r, i*m+j] * V[r, j*p+k] = delta_{i,j,k} (i.e., 1 if this is a valid mult term)
    # Actually, the tensor entry T[i,j,k] should equal sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
    # where W is used to extract the output. Let me reconsider.
    
    # The standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We want: C[i,k] = sum_r alpha_r * (sum_j u_r[i,j] * A[i,j]) * (sum_k' v_r[j,k'] * B[j,k'])
    # But the output must match. The W matrix handles the output combination.
    
    # Actually, the convention is:
    # The tensor is T[i, j, k] = 1 for all valid (i,j,k)
    # A decomposition of rank R satisfies:
    # T[i,j,k] = sum_r U[r, i*m+j] * V[r, j*p+k] * W[r, k*n+i]
    # Wait, that's not right either. Let me think again.
    
    # For matrix multiplication, the bilinear map is:
    # (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a trilinear form. The tensor T in indices (i, j, k) has T[i,j,k] = 1.
    # 
    # A rank-1 decomposition term is: u ⊗ v ⊗ w
    # where u is in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    # and (u ⊗ v ⊗ w)[i*m+j, j*p+k, k*n+i] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    # 
    # The sum over r of these should equal T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # But T is only defined on the "diagonal" where the indices are consistent.
    # Actually, the tensor has dimension (n*m) x (m*p) x (p*n) but most entries are 0.
    # T[a, b, c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some valid i,j,k
    # Otherwise T[a,b,c] = 0.
    
    # So we need: sum_r U[r,a] * V[r,b] * W[r,c] = T[a,b,c] for all a in [0,n*m), b in [0,m*p), c in [0,p*n)
    
    # Let's build the tensor T as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Now we want to find rank-minimal U, V, W such that:
    # sum_r U[:, r] @ V[:, r] @ W[:, r] = T (as a 3D tensor)
    # This is: np.einsum('ar,br,cr->abc', U, V, W) = T
    
    # Strategy: Try different ranks R and use optimization to find U, V, W
    # We'll start from a known valid decomposition and try to reduce rank
    
    # Schoolbook decomposition (rank = n*m*p = 63)
    def schoolbook():
        R = n * m * p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank decompositions
    # Use the fact that we can set up equations and try to solve them
    
    # For a given rank R, we have R*(n*m + m*p + p*n) variables
    # and n*m*m*p*p*n equations (but most are trivially satisfied if T=0)
    # The non-trivial equations are those where T=1, which are n*m*p = 63 equations
    # Plus the equations where T=0, which are the rest
    
    # Actually, we need: for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    # This is a system of polynomial equations in the entries of U, V, W
    
    # Let's try a different approach: use the structure of matrix multiplication
    # to find known low-rank decompositions
    
    # For 3x3x3, Strassen's algorithm gives rank 7
    # For 3x3x7, we can think of it as 7 independent 3x3x1 problems
    # or as a combination of smaller problems
    
    # Actually, let's try to use a greedy approach or optimization
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    
    # Let's try using numpy's optimization or a custom approach
    
    # For each candidate rank R, try to find U, V, W
    # We can fix W and solve for U and V, or use alternating optimization
    
    def try_rank(R, max_iters=50, tol=1e-6):
        """Try to find a rank-R decomposition using alternating least squares."""
        # Initialize randomly
        np.random.seed(42 + R)
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                return None, None, None, None
            
            # Update U: for each a, U[a,:] should satisfy
            # sum_r U[a,r] * (V[b,r] * W[c,r] for all b,c where T[a,b,c]=1) = 1
            # This is a linear system for each a
            for a in range(n*m):
                # Find all (b,c) where T[a,b,c] = 1
                bc_pairs = []
                for b in range(m*p):
                    for c in range(p*n):
                        if T[a, b, c] == 1:
                            bc_pairs.append((b, c))
                
                if not bc_pairs:
                    continue
                
                # For this a, we need: sum_r U[a,r] * V[b,r] * W[c,r] = 1 for each (b,c) in bc_pairs
                # Let M be the matrix with rows indexed by bc_pairs and columns by r
                # M[idx, r] = V[b,r] * W[c,r] where (b,c) = bc_pairs[idx]
                # Then we need M @ U[a,:] = ones(len(bc_pairs))
                
                M = np.zeros((len(bc_pairs), R))
                for idx, (b, c) in enumerate(bc_pairs):
                    M[idx, :] = V[b, :] * W[c, :]
                
                # Solve M @ U[a,:] = ones
                try:
                    U[a, :], residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(len(bc_pairs)), rcond=None)
                except:
                    U[a, :] = np.random.randn(R) * 0.1
            
            # Update V similarly
            for b in range(m*p):
                ac_pairs = []
                for a in range(n*m):
                    for c in range(p*n):
                        if T[a, b, c] == 1:
                            ac_pairs.append((a, c))
                
                if not ac_pairs:
                    continue
                
                M = np.zeros((len(ac_pairs), R))
                for idx, (a, c) in enumerate(ac_pairs):
                    M[idx, :] = U[a, :] * W[c, :]
                
                try:
                    V[b, :], residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(len(ac_pairs)), rcond=None)
                except:
                    V[b, :] = np.random.randn(R) * 0.1
            
            # Update W similarly
            for c in range(p*n):
                ab_pairs = []
                for a in range(n*m):
                    for b in range(m*p):
                        if T[a, b, c] == 1:
                            ab_pairs.append((a, b))
                
                if not ab_pairs:
                    continue
                
                M = np.zeros((len(ab_pairs), R))
                for idx, (a, b) in enumerate(ab_pairs):
                    M[idx, :] = U[a, :] * V[b, :]
                
                try:
                    W[c, :], residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(len(ab_pairs)), rcond=None)
                except:
                    W[c, :] = np.random.randn(R) * 0.1
            
            # Check convergence
            # Compute the error
            if iteration % 10 == 0 or iteration == max_iters - 1:
                error = 0.0
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            expected = T[a, b, c]
                            actual = np.dot(U[a, :], np.dot(V[b, :], W[c, :]))
                            if abs(expected - actual) > tol:
                                error += abs(expected - actual)
                
                if error < tol:
                    return U, V, W, error
        
        return None, None, None, None
    
    # Try various ranks
    for R in range(21, best_rank + 1):
        if time.time() > deadline:
            break
        
        U_candidate, V_candidate, W_candidate, error = try_rank(R, max_iters=30)
        
        if U_candidate is not None and error < 1e-6:
            best_rank = R
            best_U, best_V, best_W = U_candidate.copy(), V_candidate.copy(), W_candidate.copy()
            break  # Found the lowest rank, stop
    
    # Convert to integers/half-integers if possible
    def snap_to_half_integer(arr):
        """Try to snap values to integers or half-integers."""
        result = arr.copy()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                # Try nearest integer
                int_val = round(val)
                half_val = round(val * 2) / 2
                if abs(val - int_val) < 1e-9:
                    result[i, j] = int_val
                elif abs(val - half_val) < 1e-9:
                    result[i, j] = half_val
        return result
    
    best_U = snap_to_half_integer(best_U)
    best_V = snap_to_half_integer(best_V)
    best_W = snap_to_half_integer(best_W)
    
    # Verify the decomposition
    def verify(U, V, W):
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    expected = T[a, b, c]
                    actual = sum(U[a, r] * V[b, r] * W[c, r] for r in range(best_rank))
                    if abs(expected - actual) > 1e-9:
                        return False
        return True
    
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p
    
    # Convert to lists for return
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
