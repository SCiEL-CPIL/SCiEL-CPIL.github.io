import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 100

def _verify(U, V, W, tol=1e-10):
    rank = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False, T[a, b, c]
    valid_mask = np.zeros((9, 9, 9), dtype=bool)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                valid_mask[a, b, c] = True
    invalid = T[~valid_mask]
    if np.max(np.abs(invalid)) > tol:
        return False, np.max(np.abs(invalid))
    return True, None

def _schoolbook_decomposition():
    U = np.zeros((9, 27), dtype=float)
    V = np.zeros((9, 27), dtype=float)
    W = np.zeros((9, 27), dtype=float)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1.0
                V[j*3+k, col] = 1.0
                W[k*3+i, col] = 1.0
                col += 1
    return U, V, W

def _als_decomposition(rank, max_iters=5000, tol=1e-12):
    T = np.zeros((9, 9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T[a, b, c] = 1.0
    
    U = np.random.randn(9, rank) * 0.1
    V = np.random.randn(9, rank) * 0.1
    W = np.random.randn(9, rank) * 0.1
    
    for r in range(rank):
        norm = np.linalg.norm(W[:, r])
        if norm > 1e-10:
            W[:, r] /= norm
            
    R = T.copy()
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
            
        # Update U
        for r in range(rank):
            R -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            # Compute m_r = vec(V[:, r] W[:, r]^T)
            # m_r[b*9+c] = V[b, r] * W[c, r]
            m_r = np.outer(V[:, r], W[:, r]).flatten()
            norm2 = np.dot(m_r, m_r)
            
            if norm2 > 1e-12:
                # U[a, r] = sum_{b,c} R[a,b,c] * V[b,r] * W[c,r] / norm2
                # This is R[a, :, :] dot m_r
                # Reshape R to (9, 81)
                R_reshaped = R.reshape(9, 81)
                U[:, r] = np.dot(R_reshaped, m_r) / norm2
            else:
                U[:, r] = 0
            
            R += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
        # Update V
        for r in range(rank):
            R -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            # m_r = vec(U[:, r] W[:, r]^T) ? No.
            # We want to update V[:, r].
            # R[a,b,c] = U[a,r] V[b,r] W[c,r]
            # For fixed r, variable is V[:, r].
            # Coefficient of V[b,r] is U[a,r] W[c,r].
            # m_r[b] = sum_{a,c} R[a,b,c] U[a,r] W[c,r] / norm2
            # Let's form a vector of length 9 for V[:, r].
            # m_r is not a single vector in the same way.
            # We need to compute for each b:
            # V[b, r] = sum_{a,c} R[a,b,c] U[a,r] W[c,r] / (sum_{a,c} (U[a,r] W[c,r])^2)
            
            # Let's compute the numerator and denominator for each b.
            # Numerator[b] = sum_{a,c} R[a,b,c] U[a,r] W[c,r]
            # Denominator = sum_{a,c} (U[a,r] W[c,r])^2 = (sum_a U[a,r]^2) * (sum_c W[c,r]^2)
            
            u_norm2 = np.dot(U[:, r], U[:, r])
            w_norm2 = np.dot(W[:, r], W[:, r])
            denom = u_norm2 * w_norm2
            
            if denom > 1e-12:
                # Numerator[b] = sum_a sum_c R[a,b,c] U[a,r] W[c,r]
                # = sum_a U[a,r] (sum_c R[a,b,c] W[c,r])
                # Let's use einsum
                # tmp[b] = sum_{a,c} R[a,b,c] U[a,r] W[c,r]
                tmp = np.einsum('abc,ar,cr->b', R, U[:, r], W[:, r])
                V[:, r] = tmp / denom
            else:
                V[:, r] = 0
                
            R += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
        # Update W
        for r in range(rank):
            R -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            # W[c, r] = sum_{a,b} R[a,b,c] U[a,r] V[b,r] / denom
            u_norm2 = np.dot(U[:, r], U[:, r])
            v_norm2 = np.dot(V[:, r], V[:, r])
            denom = u_norm2 * v_norm2
            
            if denom > 1e-12:
                tmp = np.einsum('abc,ar,br->c', R, U[:, r], V[:, r])
                W[:, r] = tmp / denom
            else:
                W[:, r] = 0
                
            # Normalize W
            norm = np.linalg.norm(W[:, r])
            if norm > 1e-10:
                W[:, r] /= norm
                
            R += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
        err = np.linalg.norm(R)
        if err < tol:
            return U, V, W
            
    return None

def _try_rank23_decomposition():
    for _ in range(50):
        if time.time() > _DEADLINE - 10:
            break
        U, V, W = _als_decomposition(23, max_iters=1000)
        if U is not None:
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            is_valid, err = _verify(U_rounded, V_rounded, W_rounded)
            if is_valid:
                return U_rounded, V_rounded, W_rounded
    return None

def construct():
    result = _try_rank23_decomposition()
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    U, V, W = _schoolbook_decomposition()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
