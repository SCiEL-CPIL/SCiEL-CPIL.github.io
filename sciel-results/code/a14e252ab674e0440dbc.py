import time
import numpy as np
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin before 150s kill (55s buffer)

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_strassen_56():
    """Build the rank-56 Strassen decomposition."""
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i*m + j, col_idx] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j*p + k, col_idx] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k*n + i, col_idx] += val

    col = 0
    for k0 in [0, 2, 4, 6]:
        # Part 1: A0 (cols 0,1) * B0 (rows 0,1, cols k0,k0+1)
        add_term({(0,0):1,(1,1):1}, {(0,k0):1,(1,k0+1):1}, {(k0,0):1,(k0+1,1):1}, col); col+=1
        add_term({(1,0):1,(1,1):1}, {(0,k0):1}, {(k0,1):1,(k0+1,1):-1}, col); col+=1
        add_term({(0,0):1}, {(0,k0+1):1,(1,k0+1):-1}, {(k0+1,0):1,(k0+1,1):1}, col); col+=1
        add_term({(1,1):1}, {(1,k0):1,(0,k0):-1}, {(k0,0):1,(k0,1):1}, col); col+=1
        add_term({(0,0):1,(0,1):1}, {(1,k0+1):1}, {(k0,0):-1,(k0+1,0):1}, col); col+=1
        add_term({(1,0):1,(0,0):-1}, {(0,k0):1,(0,k0+1):1}, {(k0+1,1):1}, col); col+=1
        add_term({(0,1):1,(1,1):-1}, {(1,k0):1,(1,k0+1):1}, {(k0,0):1}, col); col+=1
        
        # Part 2: A1 (cols 2,3) * B1 (rows 2,3, cols k0,k0+1)
        add_term({(0,2):1,(1,3):1}, {(2,k0):1,(3,k0+1):1}, {(k0,0):1,(k0+1,1):1}, col); col+=1
        add_term({(1,2):1,(1,3):1}, {(2,k0):1}, {(k0,1):1,(k0+1,1):-1}, col); col+=1
        add_term({(0,2):1}, {(2,k0+1):1,(3,k0+1):-1}, {(k0+1,0):1,(k0+1,1):1}, col); col+=1
        add_term({(1,3):1}, {(3,k0):1,(2,k0):-1}, {(k0,0):1,(k0,1):1}, col); col+=1
        add_term({(0,2):1,(0,3):1}, {(3,k0+1):1}, {(k0,0):-1,(k0+1,0):1}, col); col+=1
        add_term({(1,2):1,(0,2):-1}, {(2,k0):1,(2,k0+1):1}, {(k0+1,1):1}, col); col+=1
        add_term({(0,3):1,(1,3):-1}, {(3,k0):1,(3,k0+1):1}, {(k0,0):1}, col); col+=1
    
    return U, V, W

def _build_schoolbook():
    """Build the rank-64 schoolbook decomposition."""
    n, m, p = 2, 4, 8
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matmul tensor."""
    n, m, p = 2, 4, 8
    T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T_recon, T_target, atol=tol)

def _try_reduce_rank(U, V, W, max_reduction=4):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 factors.
    If a set of columns is linearly dependent in a way that preserves the tensor,
    we can drop one.
    
    Strategy: for each pair of columns, check if one can be expressed as a linear
    combination of others in a way that preserves the tensor equation.
    """
    n, m, p = 2, 4, 8
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Try removing one column at a time
    for trial in range(max_reduction):
        if _time_left() < 30:
            break
        
        reduced = False
        for c in range(current_rank):
            if _time_left() < 30:
                break
            
            # Try to express column c as a linear combination of the others
            # We need: sum_{r != c} alpha_r * (U[:,r], V[:,r], W[:,r]) = (U[:,c], V[:,c], W[:,c])
            # in the tensor sense: sum alpha_r * outer(U[:,r], V[:,r], W[:,r]) = outer(U[:,c], V[:,c], W[:,c])
            
            # This is equivalent to: for all (u,v,w),
            # sum_{r!=c} alpha_r * U[u,r]*V[v,r]*W[w,r] = U[u,c]*V[v,c]*W[w,c]
            
            # Let's try a simpler approach: check if column c is a scalar multiple
            # of another column in all three matrices simultaneously.
            
            for c2 in range(current_rank):
                if c2 == c:
                    continue
                
                # Check if U[:,c] = alpha * U[:,c2] for some alpha
                if np.allclose(U[:,c2], 0, atol=1e-12):
                    continue
                
                # Find alpha from first non-zero entry
                nz = np.nonzero(U[:,c2])[0]
                if len(nz) == 0:
                    continue
                idx = nz[0]
                if U[idx, c2] == 0:
                    continue
                alpha = U[idx, c] / U[idx, c2]
                
                # Check if this alpha works for all entries
                if np.allclose(U[:,c], alpha * U[:,c2], atol=1e-10) and \
                   np.allclose(V[:,c], alpha * V[:,c2], atol=1e-10) and \
                   np.allclose(W[:,c], alpha * W[:,c2], atol=1e-10):
                    # We can drop column c
                    mask = np.ones(current_rank, dtype=bool)
                    mask[c] = False
                    best_U = U[:, mask].copy()
                    best_V = V[:, mask].copy()
                    best_W = W[:, mask].copy()
                    current_rank -= 1
                    reduced = True
                    break
            
            if reduced:
                break
        
        if not reduced:
            break
    
    if _verify(best_U, best_V, best_W):
        return best_U, best_V, best_W, best_rank
    return U, V, W, current_rank

def construct():
    n, m, p = 2, 4, 8
    
    # Start with the rank-56 Strassen decomposition
    U, V, W = _build_strassen_56()
    
    # Verify it's correct
    if not _verify(U, V, W):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Try to reduce the rank
    U_red, V_red, W_red, rank_red = _try_reduce_rank(U, V, W, max_reduction=6)
    
    if rank_red < U.shape[1]:
        return {
            "U": U_red.tolist(),
            "V": V_red.tolist(),
            "W": W_red.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
