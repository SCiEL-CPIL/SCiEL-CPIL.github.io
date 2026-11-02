"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5: A is n×m, B is m×p
    # Tensor T[i,j,k] where i in [0,n*m), j in [0,m*p), k in [0,p*n)
    # T[i,j,k] = 1 if i = a*m+b, j = b'*p+c, k = c'*n+a' and b==b' and c==c' and a==a'
    # Actually: C[a,c] = sum_b A[a,b]*B[b,c]
    # U index: a*m+b (row of A)
    # V index: b*p+c (row of B)  
    # W index: c*n+a (entry of C, trace convention W[k*n+i,r] where k=c, i=a)
    
    # Schoolbook rank = n*m*p = 60
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    U[a * m + b, col] = 1
                    V[b * p + c, col] = 1
                    W[c * n + a, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Try to find lower rank decompositions
    # The tensor T is defined by: T[a*m+b, b'*p+c, c'*n+a] = 1 if b==b' and c==c'
    # i.e., T[i,j,k] = 1 if (i mod m) == (j // p) and (j mod p) == (k // n) and (i // m) == (k % n)
    
    # Let's build the tensor explicitly
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m+b, b*p+c, c*n+a] = 1
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But realistically, for these dimensions, the rank is likely 40-60
    
    # Use a simple approach: try to find a rank-r decomposition by solving linear systems
    # For a given rank r, we need U (nm × r), V (mp × r), W (pn × r) such that
    # sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k] for all i,j,k
    
    # This is equivalent to: for each r, the matrix U[:,r] ⊗ V[:,r] ⊗ W[:,r] contributes
    # We can fix U and V and solve for W, or use iterative methods
    
    # Given time constraints, let's try a few candidate ranks with random initialization
    # and use gradient descent or direct solving
    
    deadline = time.time() + 100  # Leave 20s margin
    
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try to verify if a known lower rank works
    # For [3,4,5], let's try ranks 43, 44, ..., 59
    
    for target_rank in range(43, rank_sb):
        if time.time() > deadline:
            break
        # Try random initialization and see if we can converge
        # Use a simple approach: initialize randomly, then use ALS (alternating least squares)
        
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # ALS: fix two, solve for the third
        converged = False
        for it in range(50):
            if time.time() > deadline:
                break
            # Solve for U: for each i, U[i,:] should satisfy sum_r U[i,r] * (V[:,r] ⊗ W[:,r])[i-indexed] = T[i,:,:]
            # This is a linear system: for fixed V,W, U[i,:] * (V⊗W reshaped) = T[i,:,:]
            # V⊗W reshaped: shape (target_rank, m*p*p*n)
            # Actually: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
            # For fixed i: sum_r U[i,r] * (V[:,r] ⊙ W[:,r])[j,k] = T[i,j,k]
            # where (V[:,r] ⊙ W[:,r]) is a vector of length m*p*p*n indexed by (j,k)
            
            # Build the system: A_U[i] is a matrix of shape (m*p*p*n, target_rank)
            # A_U[i][idx, r] = V[j,r] * W[k,r] where idx encodes (j,k)
            
            # This is expensive. Let's use a simpler check.
            # Just compute the residual after a few iterations of a simpler method.
            
            # Actually, let's just check if the random init happens to work (unlikely)
            # or use a smarter approach.
            
            # Let's compute the current decomposition and check error
            # Decomposition: D[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            # This is expensive to compute fully. Let's just do a few ALS steps.
            
            # Fix V and W, solve for U
            # For each i: T[i,:,:] = sum_r U[i,r] * outer(V[:,r], W[:,r]).T reshaped
            # outer(V[:,r], W[:,r]) has shape (m*p, p*n), flatten to (m*p*p*n,)
            # So we need: T[i, :, :] reshaped to (m*p*p*n,) = sum_r U[i,r] * (V[:,r] ⊗ W[:,r])
            # This is a linear system: M_U * U[i,:] = T[i,:,:].flatten()
            # where M_U[:,r] = (V[:,r] ⊗ W[:,r])
            
            # Build M_U (shape: m*p*p*n × target_rank)
            # This is m*p*p*n = 20*15 = 300 rows, target_rank columns
            M_U = np.zeros((m*p*p*n, target_rank))
            for r in range(target_rank):
                M_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
            # Solve for each i
            for i in range(n*m):
                rhs = T[i, :, :].flatten()
                # Solve M_U * U[i,:] = rhs (least squares)
                try:
                    U[i, :], residuals, rank_u, sv = np.linalg.lstsq(M_U, rhs, rcond=None)
                except:
                    break
            if time.time() > deadline:
                break
            
            # Fix U and W, solve for V
            M_V = np.zeros((n*m*p*n, target_rank))
            for r in range(target_rank):
                M_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
            for j in range(m*p):
                rhs = T[:, j, :].reshape(n*m, p*n).flatten()
                try:
                    V[j, :], residuals, rank_v, sv = np.linalg.lstsq(M_V, rhs, rcond=None)
                except:
                    break
            if time.time() > deadline:
                break
            
            # Fix U and V, solve for W
            M_W = np.zeros((n*m*m*p, target_rank))
            for r in range(target_rank):
                M_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
            for k in range(p*n):
                rhs = T[:, :, k].reshape(n*m, m*p).flatten()
                try:
                    W[k, :], residuals, rank_w, sv = np.linalg.lstsq(M_W, rhs, rcond=None)
                except:
                    break
            if time.time() > deadline:
                break
            
            # Check convergence: compute residual
            # D[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            # Compute error
            err = 0
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        d = 0
                        for r in range(target_rank):
                            d += U[i,r]*V[j,r]*W[k,r]
                        err += (d - T[i,j,k])**2
            if err < 1e-10:
                converged = True
                break
        
        if converged:
            best_U, best_V, best_W, best_rank = U, V, W, target_rank
            break
    
    # Convert to lists of integer/half-integer if possible
    def to_halfint(x):
        """Convert to integer or half-integer if close"""
        h = round(x * 2) / 2
        if abs(x - h) < 1e-6:
            return h
        return x
    
    best_U_list = [[to_halfint(best_U[i,j]) for j in range(best_rank)] for i in range(n*m)]
    best_V_list = [[to_halfint(best_V[i,j]) for j in range(best_rank)] for i in range(m*p)]
    best_W_list = [[to_halfint(best_W[i,j]) for j in range(best_rank)] for i in range(p*n)]
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
