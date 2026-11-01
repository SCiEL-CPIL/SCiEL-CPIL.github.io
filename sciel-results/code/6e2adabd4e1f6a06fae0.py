<<<<<<< SEARCH
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def compute_residual(params, n, m, p, rank):
    """Compute the residual of the tensor decomposition."""
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U = params[:size_u].reshape((n * m, rank))
    V = params[size_u:size_u + size_v].reshape((m * p, rank))
    W = params[size_u + size_v:].reshape((p * n, rank))
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Target: T[i,j,k] = 1 for all i,j,k
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                residual += (val - 1.0) ** 2
    return residual
=======
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def compute_residual_and_grad(params, n, m, p, rank):
    """Compute the residual and gradient of the tensor decomposition."""
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U = params[:size_u].reshape((n, m, rank))
    V = params[size_u:size_u + size_v].reshape((m, p, rank))
    W = params[size_u + size_v:].reshape((p, n, rank))
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    UV = np.einsum('ijr,jkr->ijkr', U, V)
    T = np.einsum('ijkr,kir->ijk', UV, W)
    
    diff = T - 1.0
    residual = np.sum(diff**2)
    
    # Gradient
    grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', diff, V, W)
    grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', diff, U, W)
    grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', diff, U, V)
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    return residual, grad
>>>>>>> REPLACE

