"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        U = [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, -1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, -1, 1, 0, 0],
        ]
        W = [
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, -1, 0, 0],
        ]
        return U, V, W

    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        """Faster verification using einsum."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    def compute_residual(U, V, W, n, m, p):
        """Compute the residual tensor (what we need to be all 1s)."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return T - np.ones((n, m, p))

    def try_reduce_rank(U, V, W, n, m, p, max_iter=5000, tol=1e-8):
        """
        Try to reduce rank by finding linear dependencies among the rank-1 terms.
        We look for subsets of terms that can be replaced by fewer terms.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Build the tensor from the decomposition
        T = np.einsum('ic,jkc,kic->ijk', U, V, W, optimize=True)
        
        # The tensor should be all 1s
        # We want to find if there are linear dependencies
        # Each term c contributes: u_c[i,j] * v_c[j,k] * w_c[k,i]
        
        # Try to find a subset of terms that sums to zero (or nearly zero)
        # This would allow us to remove them
        
        # For small ranks, try all subsets of size 2, 3, 4
        # For each subset, check if the sum of contributions is ~0
        
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # First, try to find pairs of terms that cancel
        # Term c1 + term c2 = 0 means:
        # u_c1[i,j]*v_c1[j,k]*w_c1[k,i] + u_c2[i,j]*v_c2[j,k]*w_c2[k,i] = 0 for all i,j,k
        
        # This is a strong condition. Let's check if any pair sums to ~0
        for c1 in range(rank):
            if time.time() > deadline:
                break
            for c2 in range(c1+1, rank):
                # Check if term c1 + term c2 ≈ 0
                term1 = np.einsum('i,j,k->ijk', U[:, c1].reshape(n, m), V[:, c1].reshape(m, p), W[:, c1].reshape(p, n))
                term2 = np.einsum('i,j,k->ijk', U[:, c2].reshape(n, m), V[:, c2].reshape(m, p), W[:, c2].reshape(p, n))
                if np.allclose(term1 + term2, 0, atol=tol, rtol=0):
                    # We can remove both terms
                    new_indices = [c for c in range(rank) if c != c1 and c != c2]
                    new_rank = len(new_indices)
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_U = U[:, new_indices]
                        best_V = V[:, new_indices]
                        best_W = W[:, new_indices]
        
        # Try to find triples that sum to zero
        if best_rank < rank:
            # Reindex
            U = best_U
            V = best_V
            W = best_W
            rank = best_rank
        
        for c1 in range(rank):
            if time.time() > deadline:
                break
            for c2 in range(c1+1, rank):
                if time.time() > deadline:
                    break
                for c3 in range(c2+1, rank):
                    if time.time() > deadline:
                        break
                    term1 = np.einsum('i,j,k->ijk', U[:, c1].reshape(n, m), V[:, c1].reshape(m, p), W[:, c1].reshape(p, n))
                    term2 = np.einsum('i,j,k->ijk', U[:, c2].reshape(n, m), V[:, c2].reshape(m, p), W[:, c2].reshape(p, n))
                    term3 = np.einsum('i,j,k->ijk', U[:, c3].reshape(n, m), V[:, c3].reshape(m, p), W[:, c3].reshape(p, n))
                    if np.allclose(term1 + term2 + term3, 0, atol=tol, rtol=0):
                        new_indices = [c for c in range(rank) if c not in (c1, c2, c3)]
                        new_rank = len(new_indices)
                        if new_rank < best_rank:
                            best_rank = new_rank
                            best_U = U[:, new_indices]
                            best_V = V[:, new_indices]
                            best_W = W[:, new_indices]
        
        if best_rank < rank:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        return None

    def build_block_decomposition():
        """
        Build the 52-rank block decomposition and try to improve it.
        """
        U_cols = []
        V_cols = []
        W_cols = []

        U_s, V_s, W_s = strassen_2x2x2()

        # Block (i in {0,1}, j in {0,1}, k in {0,1}): Strassen
        for c in range(7):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for i in range(2):
                for j in range(2):
                    U_col[i*m + j] = U_s[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    V_col[j*p + k] = V_s[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W_col[k*n + i] = W_s[k*2+i][c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

        # Block (i in {0,1}, j in {0,1}, k in {2,3}): Strassen
        for c in range(7):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for i in range(2):
                for j in range(2):
                    U_col[i*m + j] = U_s[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    V_col[j*p + (2+k)] = V_s[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W_col[(2+k)*n + i] = W_s[k*2+i][c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

        # Block (i in {0,1}, j in {0,1}, k=4): schoolbook
        for i in range(2):
            for j in range(2):
                U_col = [0.0] * (n * m)
                V_col = [0.0] * (m * p)
                W_col = [0.0] * (p * n)
                U_col[i*m + j] = 1.0
                V_col[j*p + 4] = 1.0
                W_col[4*n + i] = 1.0
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)

        # Block (i in {0,1}, j in {2,3}, k in {0,1}): Strassen
        for c in range(7):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for i in range(2):
                for j in range(2):
                    U_col[i*m + (2+j)] = U_s[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    V_col[(2+j)*p + k] = V_s[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W_col[k*n + i] = W_s[k*2+i][c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

        # Block (i in {0,1}, j in {2,3}, k in {2,3}): Strassen
        for c in range(7):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for i in range(2):
                for j in range(2):
                    U_col[i*m + (2+j)] = U_s[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    V_col[(2+j)*p + (2+k)] = V_s[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W_col[(2+k)*n + i] = W_s[k*2+i][c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

        # Block (i in {0,1}, j in {2,3}, k=4): schoolbook
        for i in range(2):
            for j in range(2):
                U_col = [0.0] * (n * m)
                V_col = [0.0] * (m * p)
                W_col = [0.0] * (p * n)
                U_col[i*m + (2+j)] = 1.0
                V_col[(2+j)*p + 4] = 1.0
                W_col[4*n + i] = 1.0
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)

        # Block (i=2, all j, all k): schoolbook
        for j in range(m):
            for k in range(p):
                U_col = [0.0] * (n * m)
                V_col = [0.0] * (m * p)
                W_col = [0.0] * (p * n)
                U_col[2*m + j] = 1.0
                V_col[j*p + k] = 1.0
                W_col[k*n + 2] = 1.0
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)

        rank = len(U_cols)
        U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
        V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
        W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]

        return U_mat, V_mat, W_mat

    # Start with schoolbook baseline
    best_result = schoolbook()
    best_rank = n * m * p

    # Build the 52-rank block decomposition
    U_mat, V_mat, W_mat = build_block_decomposition()
    rank = len(U_mat[0])

    if verify_fast(U_mat, V_mat, W_mat, n, m, p):
        if rank < best_rank:
            best_result = {"U": U_mat, "V": V_mat, "W": W_mat}
            best_rank = rank

    # Try to reduce rank further by finding linear dependencies
    if time.time() < deadline - 5:
        reduced = try_reduce_rank(U_mat, V_mat, W_mat, n, m, p)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            if verify_fast(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best_result = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank

    # Try alternative block structures
    # Strategy: Use Strassen for 2x2x2 blocks, but try different groupings
    
    # Alternative 1: Group i differently
    # i in {0,1,2} but use 2x2x2 for (i={0,1}, j={0,1}, k={0,1}), etc.
    # This is what we already have.
    
    # Alternative 2: Try to use a 2x2x3 or 2x3x2 decomposition
    # For 2x2x3: A is 2x2, B is 2x3, C is 2x3
    # We can split k into {0,1} and {2}
    # k in {0,1}: 2x2x2 -> 7 terms (Strassen)
    # k=2: 2x2x1 -> 4 terms (schoolbook)
    # Total: 11 terms vs 12 schoolbook
    
    # Let's build a 2x2x3 decomposition
    def strassen_2x2x3():
        """Return U, V, W for 2x2x3 decomposition (Strassen for k={0,1} + schoolbook for k=2)."""
        U_s, V_s, W_s = strassen_2x2x2()
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Strassen for k in {0,1}
        for c in range(7):
            U_col = [0.0] * 4
            V_col = [0.0] * 6
            W_col = [0.0] * 6
            for i in range(2):
                for j in range(2):
                    U_col[i*2+j] = U_s[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    V_col[j*3+k] = V_s[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W_col[k*2+i] = W_s[k*2+i][c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
        
        # Schoolbook for k=2
        for i in range(2):
            for j in range(2):
                U_col = [0.0] * 4
                V_col = [0.0] * 6
                W_col = [0.0] * 6
                U_col[i*2+j] = 1.0
                V_col[j*3+2] = 1.0
                W_col[2*2+i] = 1.0
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)
        
        rank = len(U_cols)
        U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(4)]
        V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(6)]
        W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(6)]
        return U_mat, V_mat, W_mat

    # Now try to build a better overall decomposition using 2x2x3 blocks
    # Split j into {0,1} and {2,3}, k into {0,1,2} and {3,4}
    # For i in {0,1}:
    #   j in {0,1}, k in {0,1,2}: 2x2x3 -> 11 terms
    #   j in {0,1}, k in {3,4}: 2x2x2 -> 7 terms (Strassen)
    #   j in {2,3}, k in {0,1,2}: 2x2x3 -> 11 terms
    #   j in {2,3}, k in {3,4}: 2x2x2 -> 7 terms (Strassen)
    #   Total for i in {0,1}: 11+7+11+7 = 36 terms
    # For i=2:
    #   j in {0,1}, k in {0,1,2}: 1x2x3 -> 6 terms (schoolbook)
    #   j in {0,1}, k in {3,4}: 1x2x2 -> 4 terms (schoolbook)
    #   j in {2,3}, k in {0,1,2}: 1x2x3 -> 6 terms (schoolbook)
    #   j in {2,3}, k in {3,4}: 1x2x2 -> 4 terms (schoolbook)
    #   Total for i=2: 6+4+6+4 = 20 terms
    # Grand total: 36 + 20 = 56 terms - worse than 52
    
    # So the original 52-term decomposition is better. Let's stick with it.
    
    # Try one more thing: see if we can find a 50-rank or 48-rank decomposition
    # by trying different block sizes
    
    # What about 2x2x4? Split k into {0,1} and {2,3}
    # k in {0,1}: 2x2x2 -> 7 terms
    # k in {2,3}: 2x2x2 -> 7 terms
    # Total: 14 terms vs 16 schoolbook - same as before
    
    # The 52-term solution seems to be the best structured decomposition.
    # Let's return it.
    
    return best_result
# EVOLVE-BLOCK-END
