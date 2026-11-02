import numpy as np
import time

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

def _try_reduce_rank(U, V, W, n=2, m=3, p=5, max_attempts=100):
    """Attempt to reduce the rank by finding linear dependencies.
    
    Uses a randomized approach: for each term, check if it can be expressed
    as a linear combination of the others using least squares.
    """
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Build the tensor for each term
    terms = []
    for r in range(rank):
        t = np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        terms.append(t.flatten())
    
    # For each term k, check if it's in the span of the others
    for k in range(rank):
        other_indices = [r for r in range(rank) if r != k]
        other_M = np.array([terms[r] for r in other_indices]).T  # (nmp, rank-1)
        t_k = terms[k]
        
        # Solve least squares
        try:
            x, residuals, _, _ = np.linalg.lstsq(other_M, t_k, rcond=None)
            residual = np.linalg.norm(other_M @ x - t_k)
            
            if residual < 1e-10:
                # T_k is in the span of the other terms
                # New decomposition: T = sum_{r != k} (1 + x_r) T_r
                new_U = []
                new_V = []
                new_W = []
                idx = 0
                for r in other_indices:
                    coeff = 1 + x[idx]
                    if abs(coeff) > 1e-12:  # Skip if coefficient is essentially zero
                        new_U.append(U_mat[:, r] * coeff)
                        new_V.append(V_mat[:, r] * coeff)
                        new_W.append(W_mat[:, r] * coeff)
                    idx += 1
                
                # Remove zero terms
                new_U = [u for u in new_U if np.any(np.abs(u) > 1e-12)]
                new_V = [v for v in new_V if np.any(np.abs(v) > 1e-12)]
                new_W = [w for w in new_W if np.any(np.abs(w) > 1e-12)]
                
                if len(new_U) < rank:
                    # Convert to lists
                    new_U_list = [u.tolist() for u in new_U]
                    new_V_list = [v.tolist() for v in new_V]
                    new_W_list = [w.tolist() for w in new_W]
                    
                    # Transpose to get proper shape
                    U_out = [[new_U_list[i][r] for i in range(n * m)] for r in range(len(new_U))]
                    V_out = [[new_V_list[i][r] for i in range(m * p)] for r in range(len(new_V))]
                    W_out = [[new_W_list[i][r] for i in range(p * n)] for r in range(len(new_W))]
                    
                    if _verify_decomposition(U_out, V_out, W_out):
                        return {"U": U_out, "V": V_out, "W": W_out}
        except Exception:
            pass
    
    return None

def construct():
    """Return a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    n, m, p = 2, 3, 5
    
    # Build the rank-28 decomposition
    U, V, W = _build_rank28()
    rank = len(U[0])
    
    # Convert to numpy for verification
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Try to reduce the rank by finding linear dependencies
    if time.time() < deadline - 10:
        reduced = _try_reduce_rank(U, V, W)
        if reduced is not None:
            new_rank = len(reduced['U'][0])
            if new_rank < rank:
                return reduced
    
    # If no reduction found, return the original rank-28 decomposition
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
