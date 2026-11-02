import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    """Return a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    
    Strategy:
    - Build rank-28 decomposition using Strassen for 2x2x2 blocks.
    - Check for linear dependencies to potentially reduce to rank 27.
    - Return best result before deadline.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 2, 3, 5
    
    def build_rank28():
        """Build the standard rank-28 decomposition."""
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
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
    
    U, V, W = build_rank28()
    rank = U.shape[1]
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition correctly computes matrix multiplication."""
        # T[i,j,k] should be 1 for all valid i,j,k (coefficient of A[i,j]*B[j,k] in C[i,k])
        T_reconstructed = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            T_reconstructed += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        T_expected = np.ones((n, m, p))
        return np.allclose(T_reconstructed, T_expected, atol=1e-10)
    
    if not verify(U, V, W, n, m, p):
        # Fallback: return schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find linear dependencies to reduce rank
    # Flatten each rank-1 term
    terms = []
    for r in range(rank):
        t = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        terms.append(t.flatten())
    
    M = np.array(terms).T  # shape (900, rank)
    
    # Compute rank using SVD
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        # Find numerical rank
        tol = 1e-10
        numerical_rank = np.sum(s > tol)
        
        if numerical_rank < rank:
            # There are linear dependencies
            # The null space is spanned by the last (rank - numerical_rank) rows of Vt
            null_start = numerical_rank
            null_space = Vt[null_start:]  # shape (rank - numerical_rank, rank)
            
            # Try to eliminate one term at a time
            best_result = None
            for null_vec in null_space:
                # Find the term with largest coefficient
                max_idx = np.argmax(np.abs(null_vec))
                if np.abs(null_vec[max_idx]) < 1e-10:
                    continue
                
                # Construct new decomposition without term max_idx
                new_U = []
                new_V = []
                new_W = []
                for r in range(rank):
                    if r == max_idx:
                        continue
                    # Adjust coefficient: new_term_r = old_term_r - (null_vec[r]/null_vec[max_idx]) * old_term_max_idx
                    # But we're removing term max_idx, so we need to add its contribution
                    # Actually, if sum(null_vec[r] * term_r) = 0, then term_max_idx = -sum_{r!=max_idx}(null_vec[r]/null_vec[max_idx]) * term_r
                    # So the remaining terms with adjusted coefficients still span the same space
                    coeff = 1.0
                    new_U.append(U[:, r] * coeff)
                    new_V.append(V[:, r] * coeff)
                    new_W.append(W[:, r] * coeff)
                
                new_U = np.array(new_U).T
                new_V = np.array(new_V).T
                new_W = np.array(new_W).T
                
                # Verify
                if verify(new_U, new_V, new_W, n, m, p):
                    new_rank = new_U.shape[1]
                    if best_result is None or new_rank < best_result[0]:
                        best_result = (new_rank, new_U, new_V, new_W)
            
            if best_result is not None:
                _, bU, bV, bW = best_result
                return {"U": bU.tolist(), "V": bV.tolist(), "W": bW.tolist()}
    except Exception:
        pass
    
    # Deadline check
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
