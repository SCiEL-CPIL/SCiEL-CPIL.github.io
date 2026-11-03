"""Matrix multiplication 4x4x7 rank-minimal decomposition search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]

# Known upper bounds from literature (Landsberg, Bläser, etc.)
# For 4x4x4: rank 49 (Landsberg), possibly 48
# For 4x4x7: we need to search. Schoolbook is 112.
# Literature suggests M(4,4,7) <= 100 or so. Let's try to find a good decomposition.

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # The tensor T has entries T[i,j,k] where:
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (28, rank)
    # W has shape (p*n, rank) = (28, rank)
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is equivalent to: for all i,j,k:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{i,j,k} (the multiplication tensor)
    
    # Actually, the matrix multiplication tensor for A(4x4) * B(4x7) = C(4x7):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 if this is a valid term, else 0
    # But it's a 3-tensor where the "support" is all (i,j,k) triples
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Wait, let me re-read the convention:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W[k*n+i, r] - this is the output factor with trace convention
    # 
    # The standard matrix multiplication tensor:
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # 
    # In tensor form, the multiplication tensor T has components T_{i,j,k} = 1
    # for all i,j,k (since every product term exists)
    # 
    # The decomposition requires:
    # T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # 
    # where U[i,j,r] indexes U[i*m+j, r], V[j,k,r] indexes V[j*p+k, r],
    # and W[k,i,r] indexes W[k*n+i, r]
    
    # So we need to find rank-r matrices U, V, W such that:
    # for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 4*4*7 = 112 equations in 3*rank variables.
    
    # Strategy: Use a structured decomposition. Start with the schoolbook (rank=112)
    # and try to reduce rank using linear algebra / optimization.
    
    # First, let's try known constructions. For square matrices, Strassen's algorithm
    # gives M(2,2,2)=7. For 4x4x4, we can combine two Strassen algorithms.
    # But 4x4x7 is rectangular.
    
    # Let me try a direct optimization approach with integer/half-integer constraints.
    
    # Approach: Start with a random or structured initial guess and use gradient-free
    # search, or use the fact that we can set up this as a tensor rank problem.
    
    # For now, let's try to find a decomposition with rank < 112 using a simple
    # iterative approach.
    
    # Actually, let's think about this more carefully. The key insight is that we
    # can use bilinear tricks. For 4x4x4, the best known is 49 (Landsberg 2010).
    # For 4x4x7, we can potentially do better than schoolbook.
    
    # Let me try a constructive approach:
    # Split the 4x7 matrix into a 4x4 and a 4x3 part.
    # M(4,4,7) <= M(4,4,4) + M(4,4,3)
    # M(4,4,4) <= 49 (Landsberg)
    # M(4,4,3): this is less standard. Schoolbook is 48.
    
    # Alternatively, use the fact that M(n,m,p) is sub-multiplicative in some sense.
    
    # Let me try a different approach: use a numerical optimization to find a
    # low-rank decomposition, then round to nearest integers/half-integers.
    
    # Given time constraints, let me try a few known ranks and see if we can verify
    # a decomposition exists.
    
    # For a quick win, let's try rank = 100 and see if we can find a solution.
    
    # Actually, the most reliable approach within time limits is to use a structured
    # decomposition based on Strassen-like algorithms.
    
    # Let me implement a simple version: use the fact that we can pack multiple
    # matrix multiplications together.
    
    # For now, return the schoolbook decomposition as a baseline, but let me try
    # to do better.
    
    # Let's try an optimization-based approach:
    
    def try_rank(rank):
        """Try to find a decomposition of given rank using random initialization + optimization."""
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(42)
        
        # Initialize with random values
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # We want: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # 
        # This is a nonlinear system. Let's use a simple iterative method.
        # 
        # Alternative: fix two matrices and solve for the third linearly.
        
        # Let's try: fix U and V, solve for W linearly.
        # For each (k,i) pair, we have:
        # sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) = 1  ... wait, no
        # 
        # Actually: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # 
        # This is not linear in W because U and V are also variables.
        # 
        # Let's use an alternating minimization approach:
        # 1. Fix U, V. For each (k,i), we need:
        #    sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) = 1
        #    But this sum over j makes it coupled.
        
        # Actually, let me reconsider. The equation is:
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        #
        # This is a system of 112 equations. If we fix U and V, then for each
        # (k,i) pair, the equations over j give us:
        # sum_r W[k*n+i, r] * (U[i*m+0, r]*V[0*p+k, r]) = 1  (for j=0)
        # sum_r W[k*n+i, r] * (U[i*m+1, r]*V[1*p+k, r]) = 1  (for j=1)
        # etc.
        #
        # So for fixed U,V, each (k,i) gives m=4 linear equations in W[k*n+i, :]
        # (which has 'rank' unknowns). This is an overdetermined system (4 eqs, 
        # rank unknowns). If rank >= 4, we can solve in least squares.
        
        # Similarly for other alternations.
        
        # Let's implement alternating least squares:
        
        for iteration in range(200):
            if time.time() > deadline - 5:
                return None
            
            # Update W: fix U, V
            # For each (k,i), solve: sum_r W[k*n+i, r] * A[r] = 1 for all j
            # where A[r] = U[i*m+j, r] * V[j*p+k, r] for each j
            # This is: B @ W[k*n+i, :]^T = 1s, where B[j,r] = U[i*m+j,r]*V[j*p+k,r]
            
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    B = np.zeros((m, rank))
                    for j in range(m):
                        B[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    # Solve B @ w = ones(m) in least squares
                    try:
                        w, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(m), rcond=None)
                        W_new[k*n+i, :] = w
                    except:
                        pass
            W = W_new
            
            # Update U: fix V, W
            # For each (i,j), solve: sum_r U[i*m+j, r] * (V[j*p+0,r]*W[0*n+i,r]) = 1 (for k=0)
            # sum_r U[i*m+j, r] * (V[j*p+1,r]*W[1*n+i,r]) = 1 (for k=1)
            # etc.
            
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    B = np.zeros((p, rank))
                    for k in range(p):
                        B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        u, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        U_new[i*m+j, :] = u
                    except:
                        pass
            U = U_new
            
            # Update V: fix U, W
            # For each (j,k), solve: sum_r V[j*p+k, r] * (U[0*m+j,r]*W[k*n+0,r]) = 1 (for i=0)
            # sum_r V[j*p+k, r] * (U[1*m+j,r]*W[k*n+1,r]) = 1 (for i=1)
            # etc.
            
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, rank))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V_new[j*p+k, :] = v
                    except:
                        pass
            V = V_new
            
            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        error += (val - 1.0)**2
            error = np.sqrt(error / (n*m*p))
            
            if error < 1e-8:
                break
        
        if error < 1e-6:
            # Try to round to nearest integers or half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Check if rounded version works
            error_rounded = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                        error_rounded += (val - 1.0)**2
            error_rounded = np.sqrt(error_rounded / (n*m*p))
            
            if error_rounded < 1e-6:
                return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try various ranks from low to high
    best_result = None
    
    # Start with a reasonable range. Schoolbook is 112.
    # Let's try from 80 up to 112.
    for rank in range(80, 113):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            # Convert to lists of lists with integer/half-integer entries
            U_list = [[float(x) for x in U[i]] for i in range(U.shape[0])]
            V_list = [[float(x) for x in V[i]] for i in range(V.shape[0])]
            W_list = [[float(x) for x in W[i]] for i in range(W.shape[0])]
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r] 
                                  for r in range(rank))
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better found, return schoolbook
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
