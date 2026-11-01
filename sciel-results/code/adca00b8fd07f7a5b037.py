"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the multiplication tensor T of shape (nm, mp, pn)
    # T[a, b, d] = 1 if a = i*m+j, b = j'*p+k, d = k'*n+i' 
    # where j == j' and i == i' (the middle index must match)
    # Actually: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    deadline = time.time() + 100  # 100s budget, leave margin
    
    # Try ranks from schoolbook-1 down
    for rank in range(schoolbook_rank - 1, 6, -1):
        if time.time() > deadline:
            break
        result = _try_rank(T, nm, mp, pn, rank, deadline)
        if result is not None:
            best_result = result
            break  # Found lower rank, stop
    
    return best_result

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank(T, nm, mp, pn, rank, deadline):
    """Try to find a rank-`rank` decomposition of T."""
    np.random.seed(42)
    num_tries = 50
    for trial in range(num_tries):
        if time.time() > deadline:
            return None
        
        # Random initialization
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Optimize: minimize ||sum_c U[:,c] outer V[:,c] outer W[:,c] - T||^2
        # Flatten T for easier handling
        T_flat = T.flatten()
        
        def compute_residual(U, V, W):
            # Compute the tensor sum
            # tensor[a,b,d] = sum_c U[a,c]*V[b,c]*W[d,c]
            tensor = np.einsum('ac,bc,dc->abd', U, V, W)
            return (tensor - T).flatten()
        
        # Use scipy optimize if available, else simple gradient descent
        try:
            from scipy.optimize import minimize
            from scipy.optimize import least_squares
        except ImportError:
            least_squares = None
        
        if least_squares is not None:
            # Flatten variables
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def fun(x):
                Ux = x[:nm*rank].reshape(nm, rank)
                Vx = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                Wx = x[nm*rank+mp*rank:].reshape(pn, rank)
                tensor = np.einsum('ac,bc,dc->abd', Ux, Vx, Wx)
                return (tensor - T).flatten()
            
            try:
                res = least_squares(fun, x0, max_nfev=500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            except Exception:
                continue
            
            if res.cost < 1e-8:
                U_sol = res.x[:nm*rank].reshape(nm, rank)
                V_sol = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W_sol = res.x[nm*rank+mp*rank:].reshape(pn, rank)
                
                # Snap to half-integers
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify
                if _verify(T, U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        else:
            # Simple gradient descent fallback
            lr = 0.01
            for it in range(2000):
                if time.time() > deadline:
                    break
                tensor = np.einsum('ac,bc,dc->abd', U, V, W)
                residual = (tensor - T)
                # Gradient
                dU = np.einsum('abd,bc,dc->ac', residual, V, W)
                dV = np.einsum('abd,ac,dc->bc', residual, U, W)
                dW = np.einsum('abd,ac,bc->dc', residual, U, V)
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if np.sum(residual**2) < 1e-8:
                    break
            
            if np.sum((np.einsum('ac,bc,dc->abd', U, V, W) - T)**2) < 1e-8:
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if _verify(T, U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return None

def _verify(T, U, V, W):
    """Verify that sum_c U[:,c]*V[:,c]*W[:,c] == T exactly (within tolerance)."""
    tensor = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(tensor, T, atol=1e-10)
# EVOLVE-BLOCK-END
