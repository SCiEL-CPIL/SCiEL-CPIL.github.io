import numpy as np
from scipy.optimize import minimize
from fractions import Fraction
import time

def _tensor_225(U, V, W):
    # U: (2*2, r), V: (2*5, r), W: (5*2, r)
    # Indices: U[i*2+j], V[j*5+k], W[k*2+i]
    # T[i, j, k] = sum_c U[i*2+j, c] V[j*5+k, c] W[k*2+i, c]
    # We want T[i,j,k] = 1 if i,j,k form a valid MM term?
    # No, the tensor for M_{2,2,5} is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # So T[i, j, k] = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # In the tensor product A \otimes B \otimes C, the coefficient of e_i \otimes e_j \otimes e_k is 1 if i,j,k are valid indices.
    # Here i in 0..1, j in 0..1, k in 0..4.
    # T[i,j,k] = 1 for all i,j,k? No.
    # The multiplication tensor is sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}.
    # So T[i, j, k] = 1 if the indices match the structure.
    # In our flattened indices:
    # A index: i*2 + j
    # B index: j*5 + k
    # C index: k*2 + i
    # The tensor value is 1 if these indices are linked.
    # Actually, the tensor T is defined such that T[A_idx, B_idx, C_idx] = 1 if B_idx corresponds to j,k and A_idx to i,j and C_idx to k,i.
    # Specifically, T[i*2+j, j*5+k, k*2+i] = 1.
    # All other entries are 0.
    
    n, m, p = 2, 2, 5
    r = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _error_225(U, V, W, T):
    # Compute tensor from U, V, W
    # U: (4, r), V: (10, r), W: (10, r)
    n, m, p = 2, 2, 5
    r = U.shape[1]
    computed = np.zeros((n*m, m*p, p*n))
    for c in range(r):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        computed += np.einsum('i,j,k->ijk', u, v, w)
    return np.linalg.norm(computed - T)

def _find_decomp_225(rank, seed=42):
    np.random.seed(seed)
    n, m, p = 2, 2, 5
    T = _tensor_225(None, None, None) # Just to get T
    
    # Initialize
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Flatten
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def loss(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        return _error_225(U, V, W, T)
    
    # Optimize
    res = minimize(loss, params, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-8})
    
    if res.success or res.fun < 1e-6:
        U = res.x[:n*m*rank].reshape(n*m, rank)
        V = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        return U, V, W
    return None

