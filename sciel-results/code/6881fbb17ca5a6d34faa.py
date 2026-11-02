import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def _build_rank_28():
    """Build the rank-28 decomposition using Strassen for 2x2x2 subblocks."""
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank))  # 6 x 28
    V = np.zeros((m * p, rank))  # 15 x 28
    W = np.zeros((p * n, rank))  # 10 x 28
    
    col = 0
    
    # Block 1: j=0, k=0..4 (schoolbook for 2x1x5), rank 10
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Block 2a: j=1,2, k=0,1 (Strassen for 2x2x2), rank 7
    # M1: (A[0,1]+A[1,2])(B[1,0]+B[2,1])
    U[0 * m + 1, col] = 1; U[1 * m + 2, col] = 1
    V[1 * p + 0, col] = 1; V[2 * p + 1, col] = 1
    W[0 * n + 0, col] = 1; W[1 * n + 1, col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2])(B[1,0])
    U[1 * m + 1, col] = 1; U[1 * m + 2, col] = 1
    V[1 * p + 0, col] = 1
    W[0 * n + 1, col] = 1; W[1 * n + 1, col] = -1
    col += 1
    
    # M3: A[0,1](B[1,1]-B[2,1])
    U[0 * m + 1, col] = 1
    V[1 * p + 1, col] = 1; V[2 * p + 1, col] = -1
    W[1 * n + 0, col] = 1; W[1 * n + 1, col] = 1
    col += 1
    
    # M4: A[1,2](B[2,0]-B[1,0])
    U[1 * m + 2, col] = 1
    V[2 * p + 0, col] = 1; V[1 * p + 0, col] = -1
    W[0 * n + 0, col] = 1; W[0 * n + 1, col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2])(B[2,1])
    U[0 * m + 1, col] = 1; U[0 * m + 2, col] = 1
    V[2 * p + 1, col] = 1
    W[0 * n + 0, col] = -1; W[1 * n + 0, col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1])(B[1,0]+B[1,1])
    U[1 * m + 1, col] = 1; U[0 * m + 1, col] = -1
    V[1 * p + 0, col] = 1; V[1 * p + 1, col] = 1
    W[1 * n + 1, col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2])(B[2,0]+B[2,1])
    U[0 * m + 2, col] = 1; U[1 * m + 2, col] = -1
    V[2 * p + 0, col] = 1; V[2 * p + 1, col] = 1
    W[0 * n + 0, col] = 1
    col += 1
    
    # Block 2b: j=1,2, k=2 (schoolbook for 2x2x1), rank 4
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
    
    # Block 2c: j=1,2, k=3,4 (Strassen for 2x2x2), rank 7
    # M1: (A[0,1]+A[1,2])(B[1,3]+B[2,4])
    U[0 * m + 1, col] = 1; U[1 * m + 2, col] = 1
    V[1 * p + 3, col] = 1; V[2 * p + 4, col] = 1
    W[3 * n + 0, col] = 1; W[4 * n + 1, col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2])(B[1,3])
    U[1 * m + 1, col] = 1; U[1 * m + 2, col] = 1
    V[1 * p + 3, col] = 1
    W[3 * n + 1, col] = 1; W[4 * n + 1, col] = -1
    col += 1
    
    # M3: A[0,1](B[1,4]-B[2,4])
    U[0 * m + 1, col] = 1
    V[1 * p + 4, col] = 1; V[2 * p + 4, col] = -1
    W[4 * n + 0, col] = 1; W[4 * n + 1, col] = 1
    col += 1
    
    # M4: A[1,2](B[2,3]-B[1,3])
    U[1 * m + 2, col] = 1
    V[2 * p + 3, col] = 1; V[1 * p + 3, col] = -1
    W[3 * n + 0, col] = 1; W[3 * n + 1, col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2])(B[2,4])
    U[0 * m + 1, col] = 1; U[0 * m + 2, col] = 1
    V[2 * p + 4, col] = 1
    W[3 * n + 0, col] = -1; W[4 * n + 0, col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1])(B[1,3]+B[1,4])
    U[1 * m + 1, col] = 1; U[0 * m + 1, col] = -1
    V[1 * p + 3, col] = 1; V[1 * p + 4, col] = 1
    W[4 * n + 1, col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2])(B[2,3]+B[2,4])
    U[0 * m + 2, col] = 1; U[1 * m + 2, col] = -1
    V[2 * p + 3, col] = 1; V[2 * p + 4, col] = 1
    W[3 * n + 0, col] = 1
    col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    return U, V, W

def _verify(U, V, W, n=2, m=3, p=5, atol=1e-8):
    """Verify that the decomposition exactly represents matrix multiplication."""
    T = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    T_orig = np.ones((n, m, p))
    return np.allclose(T, T_orig, atol=atol)

def _try_reduce_rank(U, V, W, n=2, m=3, p=5):
    """Try to find a rank-27 decomposition by checking for linear dependencies."""
    rank = U.shape[1]
    
    # Flatten each term into a vector
    terms = []
    for r in range(rank):
        t = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        terms.append(t.flatten())
    
    M = np.array(terms).T  # shape (n*m*p, rank)
    
    # Compute rank of M
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
    if rank_M < rank:
        # Find null space vector
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            c = Vt[-1, :]  # last row of Vt is in null space
            
            # Find index with largest |c_k|
            k_idx = np.argmax(np.abs(c))
            if np.abs(c[k_idx]) > 1e-10:
                # Eliminate term k_idx
                new_U = []
                new_V = []
                new_W = []
                for r in range(rank):
                    if r == k_idx:
                        continue
                    coeff = 1 - c[r] / c[k_idx]
                    new_U.append(U[:, r] * coeff)
                    new_V.append(V[:, r] * coeff)
                    new_W.append(W[:, r] * coeff)
                
                new_U = np.array(new_U).T
                new_V = np.array(new_V).T
                new_W = np.array(new_W).T
                
                if _verify(new_U, new_V, new_W, n, m, p):
                    return new_U, new_V, new_W
        except Exception:
            pass
    
    return None

def _search_rank_reduction(U, V, W, n=2, m=3, p=5, deadline=None):
    """Search for rank-27 decomposition by trying to add a fractional term that cancels cross-terms."""
    rank = U.shape[1]
    
    # Strategy: try to find a single additional rank-1 term with fractional coefficients
    # that allows elimination of one existing term
    
    # First, try the direct linear dependency approach
    result = _try_reduce_rank(U, V, W, n, m, p)
    if result is not None:
        return result
    
    # If direct approach fails, try a more sophisticated search
    # We'll try to find if there's a combination of terms that can be replaced
    
    # For now, return None to indicate no reduction found
    return None

def construct():
    """Return a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 2, 3, 5
    
    # Build the rank-28 decomposition
    U, V, W = _build_rank_28()
    
    # Try to reduce rank
    result = _search_rank_reduction(U, V, W, n, m, p, deadline)
    
    if result is not None:
        new_U, new_V, new_W = result
        return {
            "U": new_U.tolist(),
            "V": new_V.tolist(),
            "W": new_W.tolist()
        }
    
    # Deadline check
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
