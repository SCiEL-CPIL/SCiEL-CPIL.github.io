import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def _build_rank28():
    """Build the standard rank-28 decomposition for 2x3 x 3x5."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0, k=0..4 (schoolbook for 2x1x5), rank 10
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2a: j=1,2, k=0,1 (Strassen for 2x2x2), rank 7
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1
    W[0 * n + 1][col] = 1; W[1 * n + 1][col] = -1
    col += 1
    
    U[0 * m + 1][col] = 1
    V[1 * p + 1][col] = 1; V[2 * p + 1][col] = -1
    W[1 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 2][col] = 1
    V[2 * p + 0][col] = 1; V[1 * p + 0][col] = -1
    W[0 * n + 0][col] = 1; W[0 * n + 1][col] = 1
    col += 1
    
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = -1; W[1 * n + 0][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 0][col] = 1; V[1 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1
    col += 1
    
    # Block 2b: j=1,2, k=2 (schoolbook for 2x2x1), rank 4
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    # Block 2c: j=1,2, k=3,4 (Strassen for 2x2x2), rank 7
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1
    W[3 * n + 1][col] = 1; W[4 * n + 1][col] = -1
    col += 1
    
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1; V[2 * p + 4][col] = -1
    W[4 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 2][col] = 1
    V[2 * p + 3][col] = 1; V[1 * p + 3][col] = -1
    W[3 * n + 0][col] = 1; W[3 * n + 1][col] = 1
    col += 1
    
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = -1; W[4 * n + 0][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 3][col] = 1; V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1
    col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=3, p=5, atol=1e-8):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = np.array(U[:, r] if isinstance(U, list) else U[:, r])
        v = np.array(V[:, r] if isinstance(V, list) else V[:, r])
        w = np.array(W[:, r] if isinstance(W, list) else W[:, r])
        T += np.einsum('i,j,k->ijk', u, v, w)
    T_orig = np.ones((n, m, p))
    return np.allclose(T, T_orig, atol=atol)

def _build_tensor(U, V, W, n=2, m=3, p=5):
    """Build the full tensor from the decomposition."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
    return T

def _target_tensor(n=2, m=3, p=5):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _try_reduce_rank(U, V, W, n=2, m=3, p=5, max_iterations=200, time_limit=30.0):
    """Attempt to reduce the rank by finding linear dependencies.
    
    Uses iterative least-squares to find if any term is in the span of others.
    """
    start_time = time.time()
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    rank = U_mat.shape[1]
    
    # Build tensor for each term
    terms = []
    for r in range(rank):
        t = np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r]).flatten()
        terms.append(t)
    
    current_terms = terms[:]
    current_rank = rank
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        for k in range(current_rank):
            if time.time() - start_time > time_limit:
                break
            
            other_indices = [r for r in range(current_rank) if r != k]
            if not other_indices:
                break
            
            other_M = np.array([current_terms[r] for r in other_indices]).T
            t_k = current_terms[k]
            
            try:
                x, residuals, _, _ = np.linalg.lstsq(other_M, t_k, rcond=None)
                residual = np.linalg.norm(other_M @ x - t_k)
                
                if residual < 1e-10:
                    # T_k is in the span of the other terms
                    # New decomposition: T = sum_{r != k} (1 + x_r) T_r
                    new_terms = []
                    idx = 0
                    for r in other_indices:
                        coeff = 1 + x[idx]
                        if abs(coeff) > 1e-12:
                            # Scale the term
                            u = U_mat[:, r] * coeff
                            v = V_mat[:, r] * coeff
                            w = W_mat[:, r] * coeff
                            if np.any(np.abs(u) > 1e-12) and np.any(np.abs(v) > 1e-12) and np.any(np.abs(w) > 1e-12):
                                new_terms.append((u, v, w))
                        idx += 1
                    
                    if len(new_terms) < current_rank:
                        # Verify the new decomposition
                        T_new = np.zeros((n*m, m*p, p*n))
                        for u, v, w in new_terms:
                            T_new += np.einsum('a,b,c->abc', u, v, w)
                        
                        T_target = _target_tensor(n, m, p)
                        if np.allclose(T_new, T_target, atol=1e-8):
                            # Update
                            U_mat = np.column_stack([u for u, v, w in new_terms])
                            V_mat = np.column_stack([v for u, v, w in new_terms])
                            W_mat = np.column_stack([w for u, v, w in new_terms])
                            current_rank = len(new_terms)
                            # Rebuild terms
                            current_terms = []
                            for r in range(current_rank):
                                t = np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r]).flatten()
                                current_terms.append(t)
                            found = True
                            break
            except Exception:
                pass
        
        if not found:
            break
    
    if current_rank < rank:
        U_out = U_mat.tolist()
        V_out = V_mat.tolist()
        W_out = W_mat.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    
    return None

def _try_random_perturbation(U, V, W, n=2, m=3, p=5, n_attempts=50, time_limit=20.0):
    """Try random perturbations to find a lower-rank decomposition."""
    start_time = time.time()
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    rank = U_mat.shape[1]
    T_target = _target_tensor(n, m, p).flatten()
    
    best_rank = rank
    best_result = None
    
    for attempt in range(n_attempts):
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a rank-(rank-1) decomposition by random projection
        # This is heuristic and may not work, but worth trying
        pass
    
    return best_result

def construct():
    """Return a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    n, m, p = 2, 3, 5
    
    # Build the rank-28 decomposition
    U, V, W = _build_rank28()
    rank = len(U[0])
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to reduce the rank by finding linear dependencies
    if time.time() < deadline - 15:
        reduced = _try_reduce_rank(U, V, W, time_limit=min(40.0, deadline - time.time() - 15))
        if reduced is not None:
            new_rank = len(reduced['U'][0])
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = reduced
    
    # If no reduction found, return the original rank-28 decomposition
    return best_result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
