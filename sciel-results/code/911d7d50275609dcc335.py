"""Matrix multiplication 3x4x6 - model-and-optimize with ALS and structured init."""
import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = U.shape[1]
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    expected = np.ones((n, m, p))
    return np.max(np.abs(T - expected)) < tol

def _als_step(U, V, W, n, m, p, r, max_iter=200, tol=1e-10):
    """Alternating least squares: fix two factors, solve for third."""
    for it in range(max_iter):
        if time.time() > _deadline - 3:
            break
        
        # Fix V, W -> solve for U
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # For fixed V,W: U[i,j,c] should satisfy:
        # sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c]) = 1 for all k
        # This is: U[i,j,:] @ diag(V[j,k,:]*W[k,i,:]) = 1_k
        # More precisely: for each (i,j), we need U[i,j,:] such that
        # U[i,j,:] @ (V[j,:,:] * W[:,i,:])^T ... let me think differently.
        
        # Actually: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # For fixed V,W, define M[i,j,k,c] = V[j,k,c] * W[k,i,c]
        # Then T[i,j,k] = sum_c U[i,j,c] * M[i,j,k,c]
        # We want T[i,j,k] = 1, so: U[i,j,:] @ M[i,j,:,k] ... 
        # For each (i,j,k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
        # This is a linear system in U[i,j,c].
        # For fixed (i,j): for all k, sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c]) = 1
        # Let A[i,j] be a k x r matrix where A[i,j][k,c] = V[j,k,c] * W[k,i,c]
        # Then A[i,j] @ U[i,j,:]^T = ones(p)
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # A[k,c] = V[j,k,c] * W[k,i,c]
                A = V[j, :, :] * W[:, i, :]  # shape (p, r)
                # Solve A @ u = ones(p) where u = U[i,j,:]
                try:
                    u, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = u
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        # Check convergence
        if np.max(np.abs(U_new - U)) < tol:
            U = U_new
            break
        U = U_new
        
        # Fix U, W -> solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # For fixed (j,k): for all i, sum_c V[j,k,c] * U[i,j,c] * W[k,i,c] = 1
                # A[i,c] = U[i,j,c] * W[k,i,c]
                A = U.reshape(n, m, r)[:, j, :] * W[k, :, :]  # shape (n, r)
                try:
                    v, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
        
        # Fix U, V -> solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # For fixed (k,i): for all j, sum_c W[k,i,c] * U[i,j,c] * V[j,k,c] = 1
                # A[j,c] = U[i,j,c] * V[j,k,c]
                A = U.reshape(n, m, r)[i, :, :] * V.reshape(m, p, r)[:, k, :]  # shape (m, r)
                try:
                    w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        if np.max(np.abs(W_new - W)) < tol:
            W = W_new
            break
        W = W_new
    
    return U, V, W

def _structured_init(n, m, p, r, seed=0):
    """Create structured initialization from truncated trivial decomposition."""
    np.random.seed(seed)
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Start from trivial and randomly select r columns, or create random structured init
    # Better: random init with some structure
    scale = 1.0 / np.sqrt(r)
    U = np.random.randn(n * m, r) * scale
    V = np.random.randn(m * p, r) * scale
    W = np.random.randn(p * n, r) * scale
    
    return U, V, W

def _polish_scipy(U, V, W, n, m, p, r, max_iter=500):
    """Polish with scipy L-BFGS-B."""
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def loss_and_grad(x):
        Ux = x[:n*m*r].reshape(n, m, r)
        Vx = x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
        Wx = x[n*m*r + m*p*r :].reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', Ux, Vx, Wx)
        err = T - 1.0
        loss = np.sum(err ** 2)
        
        dU = np.einsum('ijk,jkc,kic->ijc', err, Vx, Wx) * 2
        dV = np.einsum('ijk,ijc,kic->jkc', err, Ux, Wx) * 2
        dW = np.einsum('ijk,ijc,jkc->kic', err, Ux, Vx) * 2
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return loss, grad
    
    try:
        result = opt.minimize(loss_and_grad, x0, method='L-BFGS-B',
                              jac=True, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < 1e-8:
            U_opt = result.x[:n*m*r].reshape(n*m, r)
            V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
            return U_opt, V_opt, W_opt, result.fun
    except:
        pass
    
    return U, V, W, np.inf

def construct():
    global _deadline
    start_time = time.time()
    _deadline = 145.0  # Safety margin before 150s
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition as fallback
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = n * m * p  # 72
    
    def try_rank(r, max_restarts=10):
        """Try to find a rank-r decomposition using ALS + scipy polish."""
        for restart in range(max_restarts):
            if time.time() > _deadline - 5:
                return None
            
            seed = 42 + restart * 137
            U0, V0, W0 = _structured_init(n, m, p, r, seed=seed)
            
            # ALS phase
            U_als, V_als, W_als = _als_step(U0, V0, W0, n, m, p, r, max_iter=100)
            
            # Check if ALS converged
            if _verify(U_als, V_als, W_als, n, m, p, tol=1e-6):
                return U_als, V_als, W_als
            
            # Polish with scipy
            U_pol, V_pol, W_pol, loss = _polish_scipy(U_als, V_als, W_als, n, m, p, r, max_iter=300)
            
            if _verify(U_pol, V_pol, W_pol, n, m, p, tol=1e-6):
                return U_pol, V_pol, W_pol
            
            # Try direct scipy from random init as well
            U_rand = np.random.randn(n * m, r) * (1.0 / np.sqrt(r))
            V_rand = np.random.randn(m * p, r) * (1.0 / np.sqrt(r))
            W_rand = np.random.randn(p * n, r) * (1.0 / np.sqrt(r))
            U_pol2, V_pol2, W_pol2, loss2 = _polish_scipy(U_rand, V_rand, W_rand, n, m, p, r, max_iter=200)
            if _verify(U_pol2, V_pol2, W_pol2, n, m, p, tol=1e-6):
                return U_pol2, V_pol2, W_pol2
            
            if time.time() > _deadline - 5:
                break
        
        return None
    
    # Try ranks from just below 72 downward
    for r in [71, 70, 69, 68, 67]:
        if time.time() > _deadline - 15:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best = {'U': U, 'V': V, 'W': W}
                best_rank = r
                break
    
    if best_rank < n * m * p:
        return best
    else:
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
# EVOLVE-BLOCK-END
