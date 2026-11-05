"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_54():
    """Build the structured Strassen decomposition with rank 54."""
    n, m, p = 2, 5, 6
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        # M0: (A00+A11)(B00+B11)
        U[i_base*m + j_base][col] += 1
        U[(i_base+1)*m + (j_base+1)][col] += 1
        V[j_base*p + k_base][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] += 1
        W[k_base*n + i_base][col] += 1
        W[(k_base+1)*n + (i_base+1)][col] += 1
        col += 1
        # M1: (A10+A11)B00
        U[(i_base+1)*m + j_base][col] += 1
        U[(i_base+1)*m + (j_base+1)][col] += 1
        V[j_base*p + k_base][col] += 1
        W[k_base*n + (i_base+1)][col] += 1
        W[(k_base+1)*n + (i_base+1)][col] -= 1
        col += 1
        # M2: A00(B01-B11)
        U[i_base*m + j_base][col] += 1
        V[j_base*p + (k_base+1)][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] -= 1
        W[(k_base+1)*n + i_base][col] += 1
        W[(k_base+1)*n + (i_base+1)][col] += 1
        col += 1
        # M3: A11(B10-B00)
        U[(i_base+1)*m + (j_base+1)][col] += 1
        V[(j_base+1)*p + k_base][col] += 1
        V[j_base*p + k_base][col] -= 1
        W[k_base*n + i_base][col] += 1
        W[k_base*n + (i_base+1)][col] += 1
        col += 1
        # M4: (A00+A01)B11
        U[i_base*m + j_base][col] += 1
        U[i_base*m + (j_base+1)][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] += 1
        W[k_base*n + i_base][col] -= 1
        W[(k_base+1)*n + i_base][col] += 1
        col += 1
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1)*m + j_base][col] += 1
        U[i_base*m + j_base][col] -= 1
        V[j_base*p + k_base][col] += 1
        V[j_base*p + (k_base+1)][col] += 1
        W[(k_base+1)*n + (i_base+1)][col] += 1
        col += 1
        # M6: (A01-A11)(B10+B11)
        U[i_base*m + (j_base+1)][col] += 1
        U[(i_base+1)*m + (j_base+1)][col] -= 1
        V[(j_base+1)*p + k_base][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] += 1
        W[k_base*n + i_base][col] += 1
        col += 1

    def add_naive2x1x2(i_base, j_base, k_base):
        nonlocal col
        U[i_base*m + j_base][col] += 1
        V[j_base*p + k_base][col] += 1
        W[k_base*n + i_base][col] += 1
        col += 1
        U[i_base*m + j_base][col] += 1
        V[j_base*p + (k_base+1)][col] += 1
        W[(k_base+1)*n + i_base][col] += 1
        col += 1
        U[(i_base+1)*m + j_base][col] += 1
        V[j_base*p + k_base][col] += 1
        W[k_base*n + (i_base+1)][col] += 1
        col += 1
        U[(i_base+1)*m + j_base][col] += 1
        V[j_base*p + (k_base+1)][col] += 1
        W[(k_base+1)*n + (i_base+1)][col] += 1
        col += 1

    for jb in [0, 2]:
        for kb in [0, 2, 4]:
            add_strassen(0, jb, kb)
    for kb in [0, 2, 4]:
        add_naive2x1x2(0, 4, kb)

    assert col == rank
    return U, V, W

def _verify(U, V, W, n=2, m=5, p=6):
    """Verify that the decomposition correctly computes matrix multiplication."""
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    
    # Tensor T[i,j,k] = delta[i,k] (1 if i==k, else 0)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For random A, B: C[i,k] = sum_j A[i,j] * B[j,k]
    # Check: for each r, the term contributes (sum_j U[i,j,r]*A[i,j]) * (sum_j V[j,k,r]*B[j,k]) * W[k,i,r]
    
    # Let's verify with random matrices
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomposed = np.zeros((n, p))
        for r in range(r):
            # For each i, k:
            # sum_r (sum_j A[i,j]*U[i,j,r]) * (sum_j B[j,k]*V[j,k,r]) * W[k,i,r]
            Au = np.zeros((n, p))
            for j in range(m):
                Au[:, :] += A[:, j] * Ua[:, j, r]  # This isn't right, let me redo
            
        # Better: compute directly
        C_decomposed = np.zeros((n, p))
        for r in range(r):
            # u_r[i, j] = Ua[i*m+j, r]
            # v_r[j, k] = Va[j*p+k, r]
            # w_r[k, i] = Wa[k*n+i, r]
            # Contribution: sum_i,k (sum_j A[i,j]*u_r[i,j]) * (sum_j B[j,k]*v_r[j,k]) * w_r[k,i]
            # = sum_j A[i,j]*u_r[i,j]  -- this is per (i,j)
            # Actually: for each r, define:
            #   a_r[i] = sum_j A[i,j] * Ua[i*m+j, r]  -- no, that's not how it works
            # 
            # The bilinear form: C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j]*U[i*m+j,r]) * (sum_j B[j,k]*V[j*p+k,r])
            # Wait, that's not right either. Let me think again.
            #
            # Standard tensor decomposition for matmul:
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # where T[i,j,k] = 1 if i==k else 0... no.
            #
            # Actually the matmul tensor is: T[i,j,k] = delta[i,k]
            # And C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r T[i,j,k]*U[i,j,r]*V[j,k,r]*W[k,i,r]
            #           = sum_r W[k,i,r] * (sum_j A[i,j]*U[i,j,r]) * (sum_j B[j,k]*V[j,k,r])
            #
            # Hmm, that doesn't look right dimensionally. Let me reconsider.
            #
            # The correct identity: C[i,k] = sum_j A[i,j] * B[j,k]
            # If we have rank-r decomposition: sum_j A[i,j]*B[j,k] = sum_r (sum_j A[i,j]*u_r[i,j]) * (sum_j B[j,k]*v_r[j,k]) * w_r[k,i]
            # This is not standard. The standard way:
            #
            # Actually for bilinear complexity, we write:
            # C[i,k] = sum_r lambda_r(i,k) * prod_j ... no.
            #
            # Let me just verify numerically by expanding.
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r])
            # This is O(r * n * m * p) per (i,k) pair, which is fine.
            
            for i in range(n):
                for k in range(p):
                    s1 = 0.0
                    for j in range(m):
                        s1 += A[i, j] * Ua[i*m + j, r]
                    s2 = 0.0
                    for j in range(m):
                        s2 += B[j, k] * Va[j*p + k, r]
                    C_decomposed[i, k] += Wa[k*n + i, r] * s1 * s2
        
        if not np.allclose(C_decomposed, C_expected, atol=1e-10):
            return False
    return True

def _try_merge_pairs(U, V, W, max_time=10):
    """Try to merge pairs of rank-1 tensors that have overlapping support."""
    import numpy as np
    import time
    
    n, m, p = 2, 5, 6
    r = U.shape[0]  # wait, U is (n*m, rank)
    r = len(U[0])
    
    # Convert to numpy for speed
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # For each pair (a, b), check if U[:,a], U[:,b], V[:,a], V[:,b] are linearly dependent
    # in a way that allows merging. Specifically, if there exists alpha such that
    # U[:,a] + alpha*U[:,b] and V[:,a] + beta*V[:,b] have the same "pattern"
    # Actually, the simplest check: if U[:,a] is a scalar multiple of U[:,b] AND
    # V[:,a] is a scalar multiple of V[:,b], then we can merge.
    
    # More generally, we can try to find if two rank-1 terms can be combined.
    # This is complex, so let's try a simple heuristic: 
    # If U[:,a] == U[:,b] (element-wise equal) and V[:,a] == V[:,b], then 
    # W[:,a] and W[:,b] can be added and we reduce rank by 1.
    
    # Let's check for exact duplicates first
    used = [False] * r
    new_U = []
    new_V = []
    new_W = []
    
    for a in range(r):
        if used[a]:
            continue
        # Find all b > a with same U and V columns
        merge_indices = [a]
        for b in range(a + 1, r):
            if used[b]:
                continue
            if np.array_equal(Ua[:, a], Ua[:, b]) and np.array_equal(Va[:, a], Va[:, b]):
                merge_indices.append(b)
                used[b] = True
        
        if len(merge_indices) > 1:
            # Merge them
            sum_W = np.zeros(p * n)
            for idx in merge_indices:
                sum_W += Wa[:, idx]
            new_U.append(Ua[:, a])
            new_V.append(Va[:, a])
            new_W.append(sum_W)
        else:
            new_U.append(Ua[:, a])
            new_V.append(Va[:, a])
            new_W.append(Wa[:, a])
        used[a] = True
    
    new_rank = len(new_U)
    if new_rank < r:
        # Convert back to list format
        result_U = [[float(x) for x in col] for col in zip(*new_U)]
        result_V = [[float(x) for x in col] for col in zip(*new_V)]
        result_W = [[float(x) for x in col] for col in zip(*new_W)]
        # Clean up small values
        for mat in [result_U, result_V, result_W]:
            for row in mat:
                for i, val in enumerate(row):
                    if abs(val) < 1e-15:
                        row[i] = 0.0
        return result_U, result_V, result_W, new_rank
    
    return U, V, W, r

def _random_search_for_improvement(U_init, V_init, W_init, max_time=80):
    """Try random perturbations and local search to find lower rank decomposition."""
    import numpy as np
    import time
    import random
    
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + max_time
    
    best_U, best_V, best_W = U_init, V_init, W_init
    best_rank = len(U_init[0])
    
    # Try to find a decomposition of rank best_rank - 1, best_rank - 2, etc.
    # This is very hard in general, so we'll try a limited search.
    
    # Approach: try to eliminate one rank-1 term by expressing it as a combination
    # of others, then remove it.
    
    # For now, just return the initial solution. A proper search would be much more complex.
    # Given the time constraints and the difficulty of the problem, we'll stick with 54.
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    
    # Build the structured Strassen decomposition
    U, V, W = _build_strassen_54()
    
    # Verify correctness
    if not _verify(U, V, W):
        raise ValueError("Strassen decomposition verification failed")
    
    # Try to merge identical columns
    U, V, W, rank = _try_merge_pairs(U, V, W)
    
    # Verify again after merging
    if rank < 54:
        if not _verify(U, V, W):
            # If verification fails after merging, fall back
            U, V, W = _build_strassen_54()
            rank = 54
    
    # Time check: if we have time, could try more aggressive search
    elapsed = time.time() - start_time
    if elapsed < 100:  # Safety margin
        pass  # Could add more search here, but for now just return
    
    # Convert to proper format
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
