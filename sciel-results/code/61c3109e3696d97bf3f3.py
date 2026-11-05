import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Strassen's algorithm for 2x2x2 blocks to reduce the rank from the schoolbook 60
    to 56 by splitting the 3x4x5 multiplication into structured blocks.
    Then attempts systematic rank reduction via linear dependency detection.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(U_blk, V_blk, W_blk, u_idx, v_idx, w_idx):
        for r in range(U_blk.shape[1]):
            U_part = np.zeros(n * m)
            for i, idx in enumerate(u_idx):
                U_part[idx] = U_blk[i, r]
            U_parts.append(U_part)
            
            V_part = np.zeros(m * p)
            for i, idx in enumerate(v_idx):
                V_part[idx] = V_blk[i, r]
            V_parts.append(V_part)
            
            W_part = np.zeros(p * n)
            for i, idx in enumerate(w_idx):
                W_part[idx] = W_blk[i, r]
            W_parts.append(W_part)
    
    # Block 1: A1_top * B_top_left (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_blk, V_blk, W_blk, [0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4])
    
    # Block 2: A1_top * B_top_mid (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_blk, V_blk, W_blk, [0, 1, 4, 5], [2, 3, 7, 8], [6, 9, 7, 10])
    
    # Block 3: A1_top * B_top_right (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block(U_blk, V_blk, W_blk, [0, 1, 4, 5], [4, 9], [12, 13])
    
    # Block 4: A1_bottom * B_bottom_left (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_blk, V_blk, W_blk, [2, 3, 6, 7], [10, 11, 15, 16], [0, 3, 1, 4])
    
    # Block 5: A1_bottom * B_bottom_mid (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block(U_blk, V_blk, W_blk, [2, 3, 6, 7], [12, 13, 17, 18], [6, 9, 7, 10])
    
    # Block 6: A1_bottom * B_bottom_right (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block(U_blk, V_blk, W_blk, [2, 3, 6, 7], [14, 19], [12, 13])
    
    # Block 7: A2_top * B_top_left (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(U_blk, V_blk, W_blk, [8, 9], [0, 1, 5, 6], [2, 5])
    
    # Block 8: A2_top * B_top_mid (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(U_blk, V_blk, W_blk, [8, 9], [2, 3, 7, 8], [8, 11])
    
    # Block 9: A2_top * B_top_right (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block(U_blk, V_blk, W_blk, [8, 9], [4, 9], [14])
    
    # Block 10: A2_bottom * B_bottom_left (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(U_blk, V_blk, W_blk, [10, 11], [10, 11, 15, 16], [2, 5])
    
    # Block 11: A2_bottom * B_bottom_mid (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block(U_blk, V_blk, W_blk, [10, 11], [12, 13, 17, 18], [8, 11])
    
    # Block 12: A2_bottom * B_bottom_right (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block(U_blk, V_blk, W_blk, [10, 11], [14, 19], [14])
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    # Track which block each column belongs to for structured reduction
    # Blocks: 0-6 (2x2x2 Strassen), 7-10 (2x2x1 schoolbook), 11-14 (1x2x2 schoolbook), 15-18 (1x2x1 schoolbook)
    # Actually let me recount:
    # Block 1: 7 cols (indices 0-6)
    # Block 2: 7 cols (indices 7-13)
    # Block 3: 4 cols (indices 14-17)
    # Block 4: 7 cols (indices 18-24)
    # Block 5: 7 cols (indices 25-31)
    # Block 6: 4 cols (indices 32-35)
    # Block 7: 4 cols (indices 36-39)
    # Block 8: 4 cols (indices 40-43)
    # Block 9: 2 cols (indices 44-45)
    # Block 10: 4 cols (indices 46-49)
    # Block 11: 4 cols (indices 50-53)
    # Block 12: 2 cols (indices 54-55)
    # Total: 56
    
    def verify_fast(U, V, W, tol=1e-9):
        """Fast verification using einsum."""
        r = U.shape[1]
        if r == 0:
            return False
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    if not verify_fast(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank using linear algebra on the tensor
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    # Strategy: For each column, check if it can be expressed as a linear combination
    # of other columns in the tensor sense.
    # A column c is redundant if there exist coefficients alpha_d for d != c such that
    # for all (i,j,k): sum_d alpha_d * U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d] = 0
    # This is equivalent to checking if the tensor contribution of column c is in the
    # span of contributions of other columns.
    
    # Build the "tensor space" representation: each column c defines a rank-1 tensor
    # T_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need to check linear dependencies among these rank-1 tensors.
    
    # Vectorize: reshape each column's tensor to a vector of length n*m*p
    def col_to_vector(U_col, V_col, W_col):
        """Convert a single column (vectors of length nm, mp, pn) to a flattened tensor vector."""
        u = U_col.reshape(n, m)
        v = V_col.reshape(m, p)
        w = W_col.reshape(p, n)
        # T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        T = np.einsum('ij,jk,ki->ijk', u, v, w)
        return T.flatten()
    
    # Build the matrix of all column tensors
    # This is expensive for large rank, so we do it incrementally
    
    # First, try simple heuristics:
    # 1. Check for duplicate or proportional columns
    # 2. Check for linear dependencies using QR/SVD on the tensor space
    
    # Efficient approach: build the tensor matrix M where M[:, c] = vectorized T_c
    # Then find the null space / rank deficiency
    
    # For rank 56, the tensor space has dimension 3*4*5 = 60
    # So at most 60 independent tensors. We have 56, so there could be up to 4 dependencies.
    
    # Build the 60 x rank matrix
    tensor_matrix = np.zeros((n * m * p, current_rank))
    for c in range(current_rank):
        tensor_matrix[:, c] = col_to_vector(current_U[:, c], current_V[:, c], current_W[:, c])
    
    # Find the rank of this matrix and identify dependent columns
    # Use SVD to find the null space
    try:
        U_svd, S, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
        # The rank is the number of singular values > tol
        rank_tol = max(tensor_matrix.shape) * np.finfo(float).eps * S[0]
        numeric_rank = np.sum(S > rank_tol)
        
        # If numeric_rank < current_rank, there are dependencies
        # Find which columns can be removed
        # The null space of tensor_matrix gives us the dependencies
        if numeric_rank < current_rank:
            # The right singular vectors corresponding to zero singular values span the null space
            # Vt[numeric_rank:] gives the null space vectors
            null_space = Vt[numeric_rank:]
            
            # For each null space vector, find a column with significant weight
            # and try to remove it
            removed = set()
            for ns_vec in null_space:
                if time.time() > deadline - 10:
                    break
                # Find the column with the largest absolute coefficient in this null vector
                weights = np.abs(ns_vec)
                # Only consider columns not yet removed
                candidates = [c for c in range(current_rank) if c not in removed]
                if not candidates:
                    break
                max_idx = max(candidates, key=lambda c: weights[c])
                if weights[max_idx] > 1e-8:
                    # Try removing this column
                    keep = [c for c in range(current_rank) if c != max_idx]
                    U_test = current_U[:, keep]
                    V_test = current_V[:, keep]
                    W_test = current_W[:, keep]
                    if verify_fast(U_test, V_test, W_test):
                        current_U = U_test
                        current_V = V_test
                        current_W = W_test
                        current_rank -= 1
                        removed.add(max_idx)
                        # Rebuild tensor matrix (expensive but necessary)
                        tensor_matrix = np.zeros((n * m * p, current_rank))
                        for c in range(current_rank):
                            tensor_matrix[:, c] = col_to_vector(current_U[:, c], current_V[:, c], current_W[:, c])
                        U_svd, S, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
                        rank_tol = max(tensor_matrix.shape) * np.finfo(float).eps * (S[0] if len(S) > 0 else 1.0)
                        numeric_rank = np.sum(S > rank_tol)
                        if numeric_rank >= current_rank:
                            break
                        null_space = Vt[numeric_rank:]
            # After SVD-based reduction, do a final greedy pass
            for col in range(current_rank):
                if time.time() > deadline - 10:
                    break
                keep = [c for c in range(current_rank) if c != col]
                U_test = current_U[:, keep]
                V_test = current_V[:, keep]
                W_test = current_W[:, keep]
                if verify_fast(U_test, V_test, W_test):
                    current_U = U_test
                    current_V = V_test
                    current_W = W_test
                    current_rank -= 1
                    break  # restart the loop since indices changed
    except np.linalg.LinAlgError:
        pass
    
    # Final greedy pass (fast version)
    for attempt in range(current_rank):
        if time.time() > deadline - 10:
            break
        reduced = False
        for col in range(current_rank):
            if time.time() > deadline - 5:
                break
            keep = [c for c in range(current_rank) if c != col]
            U_test = current_U[:, keep]
            V_test = current_V[:, keep]
            W_test = current_W[:, keep]
            if verify_fast(U_test, V_test, W_test):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
                reduced = True
                break
        if not reduced:
            break
    
    if not verify_fast(current_U, current_V, current_W):
        current_U = U
        current_V = V
        current_W = W
    
    return {
        "U": current_U.tolist(),
        "V": current_V.tolist(),
        "W": current_W.tolist()
    }
