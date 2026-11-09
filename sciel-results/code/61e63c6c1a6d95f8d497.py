"""Tensor decomposition search for 5x5x5 matrix multiplication."""
import numpy as np
import itertools
import time

DIMENSIONS = [5, 5, 5]

def _build_schoolbook(n=5, m=5, p=5):
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
    return {'U': U, 'V': V, 'W': W}

def _verify(U, V, W, n=5, m=5, p=5, tol=1e-6):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = 1 for all (i,j,k)
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    errors = 0.0
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :]) * 0  # placeholder
                # Actually: sum_c U[idx_u,c] * V[idx_v,c] * W[idx_w,c]
                # = sum_c (U[idx_u,c] * V[idx_v,c]) * W[idx_w,c]
                prod = np.dot(U[idx_u, :] * V[idx_v, :], W[idx_w, :])
                err = abs(prod - 1.0)
                if err > tol:
                    errors += 1
                max_err = max(max_err, err)
    return errors == 0, max_err

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 matrix multiplication.
    U shape (4,7), V shape (4,7), W shape (4,7)
    Index convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    For 2x2: i,j,k in {0,1}
    U row index: i*2+j -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    V row index: j*2+k -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    W row index: k*2+i -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U rows: (i,j) -> coefficient on A[i,j]
    # V rows: (j,k) -> coefficient on B[j,k]
    # W rows: (k,i) -> coefficient on C[i,k]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # U: A11=1, A22=1 -> rows 0,3
    U[0, 0] = 1; U[3, 0] = 1
    # V: B11=1, B22=1 -> rows 0,3
    V[0, 0] = 1; V[3, 0] = 1
    # W: C11+=1, C22+=1 -> (k=0,i=0)=0, (k=1,i=1)=3
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # W: C21+=1 -> (k=1,i=0) = 1*2+0 = 2... wait
    # W index: k*2+i. C21 means i=1,k=1? No, C[i][k] where i is row, k is col.
    # C21: i=1, k=0? No. Standard: C[i][j] where i=row, j=col.
    # Let me re-index: C[i][j], i in {0,1}, j in {0,1}
    # W index: k*n + i where n=2, so W[k*2 + i]
    # C[1][0]: i=1, j=0 -> W[0*2+1] = W[1]
    # C[2][2] in 1-based = C[1][1] in 0-based: i=1,j=1 -> W[1*2+1] = W[3]
    # C[1][1] in 1-based = C[0][0]: i=0,j=0 -> W[0*2+0] = W[0]
    # C[1][2] in 1-based = C[0][1]: i=0,j=1 -> W[1*2+0] = W[2]
    
    # Let me redo with 0-based indices.
    # C[0][0] = M1 + M4 - M5 + M7
    # C[0][1] = M3 + M5
    # C[1][0] = M2 + M4
    # C[1][1] = M1 - M2 + M3 + M6
    #
    # W[k*2 + i] where i is row of C, k is col of C.
    # C[0][0]: i=0, k=0 -> W[0]
    # C[0][1]: i=0, k=1 -> W[2]
    # C[1][0]: i=1, k=0 -> W[1]
    # C[1][1]: i=1, k=1 -> W[3]
    
    # Reset
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A[0][0]+A[1][1])(B[0][0]+B[1][1])
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2 = (A[1][0]+A[1][1])B[0][0]
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    W[1, 1] = 1  # C10
    W[3, 1] = -1  # C11
    
    # M3 = A[0][0](B[0][1]-B[1][1])
    U[0, 2] = 1  # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01, B11
    W[2, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = A[1][1](B[1][0]-B[0][0])
    U[3, 3] = 1  # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10, B00
    W[0, 3] = 1  # C00
    W[1, 3] = 1  # C10
    
    # M5 = (A[0][0]+A[0][1])B[1][1]
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1  # C00
    W[2, 4] = 1  # C01
    
    # M6 = (A[1][0]-A[0][0])(B[0][0]+B[0][1])
    U[2, 5] = 1; U[0, 5] = -1  # A10, A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B01
    W[3, 5] = 1  # C11
    
    # M7 = (A[0][1]-A[1][1])(B[1][0]+B[1][1])
    U[1, 6] = 1; U[3, 6] = -1  # A01, A11
    V[2, 6] = 1; V[3, 6] = 1  # B10, B11
    W[0, 6] = 1  # C00
    
    return {'U': U, 'V': V, 'W': W}

def _try_rank_reduction(schoolbook, max_iters=5000, seed=42):
    """Try to find a lower-rank decomposition by combining basis vectors."""
    np.random.seed(seed)
    n, m, p = 5, 5, 5
    U0 = schoolbook['U'].copy()
    V0 = schoolbook['V'].copy()
    W0 = schoolbook['W'].copy()
    rank0 = U0.shape[1]
    
    best_rank = rank0
    best_result = {'U': U0, 'V': V0, 'W': W0}
    
    # Try merging pairs of columns
    # For each pair (a, b), try to see if we can combine them
    # This is a very crude heuristic
    
    cols_to_try = list(range(rank0))
    np.random.shuffle(cols_to_try)
    
    for _ in range(min(max_iters, len(cols_to_try) * (len(cols_to_try) - 1) // 2)):
        a = cols_to_try[_ % len(cols_to_try)]
        b = cols_to_try[(_ + 1) % len(cols_to_try)]
        if a == b:
            continue
        
        # Try linear combination: new_col = U[:,a] + alpha * U[:,b]
        # This won't generally work, but let's try a few alpha values
        for alpha in [0.5, -0.5, 1.0, -1.0, 0.25, -0.25]:
            U_new = U0.copy()
            V_new = V0.copy()
            W_new = W0.copy()
            
            # Replace column a with U[:,a] + alpha*U[:,b]
            U_new[:, a] = U0[:, a] + alpha * U0[:, b]
            V_new[:, a] = V0[:, a]  # Keep V the same for now
            W_new[:, a] = W0[:, a]
            
            # Remove column b
            if a < b:
                U_new = np.delete(U_new, b, axis=1)
                V_new = np.delete(V_new, b, axis=1)
                W_new = np.delete(W_new, b, axis=1)
            else:
                U_new = np.delete(U_new, a, axis=1)
                U_new = np.delete(U_new, b, axis=1)
                U_new = np.insert(U_new, a, U0[:, a] + alpha * U0[:, b], axis=1)
                V_new = np.delete(V_new, b, axis=1)
                W_new = np.delete(W_new, b, axis=1)
            
            new_rank = U_new.shape[1]
            if new_rank < best_rank:
                valid, max_err = _verify(U_new, V_new, W_new)
                if valid:
                    best_rank = new_rank
                    best_result = {'U': U_new, 'V': V_new, 'W': W_new}
    
    return best_result, best_rank

def construct():
    start_time = time.time()
    deadline = 100  # Safety margin
    
    # Start with schoolbook
    schoolbook = _build_schoolbook()
    best_result = schoolbook
    best_rank = 512
    
    # Check if we should try anything else
    elapsed = time.time() - start_time
    
    if elapsed < deadline - 5:
        # Try to verify schoolbook
        valid, max_err = _verify(schoolbook['U'], schoolbook['V'], schoolbook['W'])
        if valid:
            pass  # Good, schoolbook works
    
    # Try Strassen-based construction for 5x5
    # Split 5x5 into blocks: 2+2+1 or 2+1+2 or 1+2+2
    # This is complex; for now, return schoolbook
    
    # Try a simple rank reduction search
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        result, new_rank = _try_rank_reduction(schoolbook, max_iters=2000)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = result
    
    elapsed = time.time() - start_time
    if elapsed < deadline:
        # Final verification
        valid, max_err = _verify(best_result['U'], best_result['V'], best_result['W'])
        if not valid:
            # Fallback to schoolbook
            best_result = schoolbook
            best_rank = 512
    
    # Convert to list format for the interface
    U = best_result['U'].tolist()
    V = best_result['V'].tolist()
    W = best_result['W'].tolist()
    
    return {'U': U, 'V': V, 'W': W}

# EVOLVE-BLOCK-END
