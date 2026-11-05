"""Systematic rank-1 peeling for 2x3x5 matrix multiplication tensor."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Build the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try rank-1 peeling with improved ALS and rational verification
    residual = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    rank = 0
    max_rank = 28  # Target rank (known upper bound for 2x3x5 is around 28)
    
    while rank < max_rank:
        # Find the best rank-1 approximation to residual using ALS
        best_score = -np.inf
        best_u = None
        best_v = None
        best_w = None
        
        # Try multiple random initializations with different seeds
        for trial in range(100):
            np.random.seed(rank * 1000 + trial)
            u = np.random.randn(n*m)
            v = np.random.randn(m*p)
            w = np.random.randn(p*n)
            
            # ALS iterations with more iterations for convergence
            for _ in range(50):
                # Update u: residual contracted with v and w
                u = np.einsum('ijk,j,k->i', residual, v, w)
                # Update v
                v = np.einsum('ijk,i,k->j', residual, u, w)
                # Update w
                w = np.einsum('ijk,i,j->k', residual, u, v)
                
                # Check for convergence
                if np.linalg.norm(u) < 1e-15 or np.linalg.norm(v) < 1e-15 or np.linalg.norm(w) < 1e-15:
                    break
            
            # Compute score
            score = np.einsum('ijk,i,j,k', residual, u, v, w)
            if score > best_score:
                best_score = score
                best_u = u.copy()
                best_v = v.copy()
                best_w = w.copy()
        
        if best_score <= 1e-10:
            break
        
        # Subtract the rank-1 tensor
        residual -= np.einsum('i,j,k->ijk', best_u, best_v, best_w)
        
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        rank += 1
        
        # Check if residual is close to zero
        if np.linalg.norm(residual) < 1e-10:
            break
    
    # Build the factor matrices
    U_mat = np.array(U_list).T  # shape (n*m, rank)
    V_mat = np.array(V_list).T  # shape (m*p, rank)
    W_mat = np.array(W_list).T  # shape (p*n, rank)
    
    # Try to find rational/half-integer approximations
    U_frac = []
    for i in range(n*m):
        row = []
        for c in range(rank):
            val = U_mat[i, c]
            # Try to find a simple fraction with small denominator
            frac = Fraction(val).limit_denominator(1000)
            row.append(float(frac))
        U_frac.append(row)
    
    V_frac = []
    for i in range(m*p):
        row = []
        for c in range(rank):
            val = V_mat[i, c]
            frac = Fraction(val).limit_denominator(1000)
            row.append(float(frac))
        V_frac.append(row)
    
    W_frac = []
    for i in range(p*n):
        row = []
        for c in range(rank):
            val = W_mat[i, c]
            frac = Fraction(val).limit_denominator(1000)
            row.append(float(frac))
        W_frac.append(row)
    
    # Verify the decomposition
    U_arr = np.array(U_frac)
    V_arr = np.array(V_frac)
    W_arr = np.array(W_frac)
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                expected = 1.0
                err = abs(val - expected)
                max_err = max(max_err, err)
    
    # Also check that non-diagonal entries are zero
    for i1 in range(n):
        for j1 in range(m):
            for k1 in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i1 != i2 or j1 != j2 or k1 != k2:
                                val = np.sum(U_arr[i1*m+j1] * V_arr[j2*p+k2] * W_arr[k2*n+i2])
                                err = abs(val)
                                max_err = max(max_err, err)
    
    if max_err > 1e-6:
        # Fall back to schoolbook
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_frac, "V": V_frac, "W": W_frac}
