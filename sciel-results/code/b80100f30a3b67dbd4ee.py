"""Search for a rank-minimal exact tensor decomposition of 2x3 * 3x5 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

# The multiplication tensor T has entries T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
# We want to find rank r and matrices U (n*m x r), V (m*p x r), W (p*n x r) such that
# T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
#
# The score is the rank r (lower is better). The seed has rank 30 (2*3*5).
#
# Known lower bound: For C_{n,m,p} tensor, rank >= n*m*p / max(n,m,p) is not tight.
# A simple lower bound is max(n*m, m*p, p*n) = max(6, 15, 10) = 15.
# Actually, a better lower bound comes from the "slice rank" or "border rank" arguments.
# For 2x3x5, the known minimal rank is 15 (since 2*3*5=30, and we can potentially halve it).
# Let's try to find a rank 15 decomposition.

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 6, 15, 10
    
    # Try ranks from lower bound upwards
    # Lower bound: max(n*m, m*p, p*n) = 15
    # Let's try to find rank 15 first, then 16, etc.
    
    best_rank = n * m * p  # fallback: schoolbook rank 30
    best_solution = None
    
    # Generate schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    # The tensor is T[i,j,k] = 1 for all i,j,k
    # This is a "flat" tensor - all entries are 1.
    # For such a tensor, rank can be much lower.
    
    # Key insight: T[i,j,k] = 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # If we can find vectors u_i (length r), v_j (length r), w_k (length r) such that
    # sum_c u_i[c] * v_j[c] * w_k[c] = 1 for all i,j,k
    # Then U[i*m+j, c] = u_i[c], V[j*p+k, c] = v_j[c], W[k*n+i, c] = w_k[c]
    
    # This is equivalent to finding a decomposition of the all-ones tensor.
    # The all-ones tensor of size n x m x p has rank at most n*m*p, but can be much less.
    
    # For the all-ones tensor, one approach: use the fact that 1 = 1*1*1
    # So if we set all U, V, W entries to 1/sqrt(r) for r=1, we get:
    # sum_c (1/sqrt(r)) * (1/sqrt(r)) * (1/sqrt(r)) = r * (1/r^{3/2}) = 1/sqrt(r)
    # That's not 1 unless r=1. For r=1: 1*1*1 = 1. So rank 1 works? No!
    # Wait, for r=1: U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = a_{ij} * b_{jk} * c_{ki}
    # We need a_{ij} * b_{jk} * c_{ki} = 1 for all i,j,k.
    # This means a_{ij} * b_{jk} = 1/c_{ki} for all i,j,k.
    # For fixed i,k, a_{ij} * b_{jk} must be constant in j. This is very restrictive.
    # In general, rank 1 does NOT work for the all-ones tensor unless n=m=p=1.
    
    # Let's think differently. The tensor T[i,j,k]=1 can be written as:
    # T[i,j,k] = sum_c alpha_c * u_c[i] * v_c[j] * w_c[k]
    # where u_c is length n, v_c is length m, w_c is length p.
    # Then U[i*m+j, c] = u_c[i], V[j*p+k, c] = v_c[j], W[k*n+i, c] = w_c[k]
    # Wait, that's not right. Let me re-read the convention.
    #
    # The task says: W[k*n+i, r] follows the official trace convention.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # So U is indexed by (i,j), V by (j,k), W by (k,i).
    # This is a "cyclic" indexing.
    
    # For the all-ones tensor, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's try rank r where we use a structured approach.
    # One idea: use Fourier-like bases.
    
    # Actually, let's try a direct numerical search for small ranks.
    # We'll use random initialization and gradient-free or least-squares approaches.
    
    def try_rank(r, max_iters=5000, seed=42):
        """Try to find a rank-r decomposition using iterative fitting."""
        rng = random.Random(seed)
        
        # Random initialization
        U = np.array([[rng.uniform(-1, 1) for _ in range(r)] for _ in range(nm)], dtype=float)
        V = np.array([[rng.uniform(-1, 1) for _ in range(r)] for _ in range(mp)], dtype=float)
        W = np.array([[rng.uniform(-1, 1) for _ in range(r)] for _ in range(pn)], dtype=float)
        
        # Build the target tensor T[i,j,k] = 1
        # We need to solve: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # This is a non-convex problem. Let's use an alternating least squares approach.
        # For fixed V and W, solve for U:
        # For each (i,j): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
        # This is a linear system in U[i*m+j,:]
        
        # For fixed U and W, solve for V:
        # For each (j,k): sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
        
        # For fixed U and V, solve for W:
        # For each (k,i): sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
        
        # Let's implement ALS.
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # For this (i,j), we have r unknowns U[i*m+j, 0..r-1]
                    # Constraints: for each k in range(p): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # This is a system of p equations in r unknowns.
                    # If p <= r, we can solve in least squares sense.
                    A = np.zeros((p, r))
                    b = np.ones(p)
                    for k in range(p):
                        for c in range(r):
                            A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Solve A * x = b in least squares sense
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = x
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # For this (j,k), we have r unknowns V[j*p+k, 0..r-1]
                    # Constraints: for each i in range(n): sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1
                    A = np.zeros((n, r))
                    b = np.ones(n)
                    for i in range(n):
                        for c in range(r):
                            A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k, :] = x
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # For this (k,i), we have r unknowns W[k*n+i, 0..r-1]
                    # Constraints: for each j in range(m): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = x
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (val - 1.0) ** 2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
        
        return best_err, best_U, best_V, best_W
    
    # Try ranks from 15 upward
    for r in range(15, 31):
        if time.time() > deadline - 5:
            break
        
        err, U, V, W = try_rank(r, max_iters=2000, seed=r)
        
        if err < 1e-8:
            # Found a valid decomposition
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better solution found, return the schoolbook solution
    return best_solution

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
