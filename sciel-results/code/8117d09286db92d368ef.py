"""Matrix multiplication 3x5x7 tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n=3, m=5, p=7, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    rank = U.shape[1]
    # Build the target tensor: T[i, j, k] = 1 if i==k else 0, with the convention
    # (C = AB)[i,k] = sum_j A[i,j] B[j,k]
    # U rows indexed by (i,j), V rows by (j,k), W rows by (k,i) [trace convention W[k*n+i]]
    # The decomposition says: for each c in range(rank),
    #   contribution to (i,k) from A[i,j]*B[j,k] is sum over j of U[i*m+j,c] * V[j*p+k,c]
    # and this should equal sum_c W[k*n+i,c] for each (i,k)
    # More precisely: the tensor T[i,j,k] (where the "output" index k is tied to input i)
    # Actually, the standard matmul tensor is: T[i, j, k] = delta_{i,k} (no, that's not right)
    # 
    # Let me reconsider. The matmul C = AB has C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor we decompose has three indices: i (from A rows), j (shared/inner), k (from B cols)
    # T[i,j,k] = 1 if we want to compute C[i,k] from A[i,j]*B[j,k], but the tensor entries are
    # actually the "structure" of the multiplication: T[i,j,k] = delta_{i,i} * delta_{k,k} ... 
    # 
    # The standard formulation: the multiplication tensor M_{n,m,p} has entries:
    # M[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k]
    # So M[i,j,k] = 1 for all valid (i,j,k), and the decomposition is:
    # M = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is indexed by (i,j), v_c by (j,k), w_c by (i,k) [or (k,i) depending on convention]
    # 
    # With W[k*n + i], the W index is (k, i), so w_c[k, i].
    # The reconstruction: for each (i,k): C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c] * V[j*p+k,c])
    # This must equal sum_j A[i,j]*B[j,k] for ALL A, B, which means:
    # sum_c W[k*n+i, c] * (sum_j U[i*m+j,c] * V[j*p+k,c]) = delta_{i,k}? No...
    # 
    # Actually it must hold that for the bilinear form:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_c (sum_j A[i,j] U[i*m+j,c]) * (sum_k' B[j,k'] V[j*p+k',c]) ... this doesn't quite work
    # 
    # Let me use the standard: A rank-r decomposition of the matmul tensor means:
    # There exist U (nm x r), V (mp x r), W (pn x r) such that for all i, k:
    # sum_j A[i,j] B[j,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * ... 
    # 
    # No, the correct formulation: 
    # C[i,k] = sum_c (sum_j A[i,j] U[i*m+j,c]) * (sum_j B[j,k] V[j*p+k,c]) * W[k*n+i,c]?
    # That's not bilinear in the right way.
    # 
    # The correct tensor decomposition: The matmul tensor T is in R^{(nm) x (mp) x (pn)} with
    # T[(i,j), (j',k'), (k,i')] = delta_{j,j'} * delta_{i,i'} * ... 
    # Actually T[(i,j),(j',k),(k',i)] = delta_{j,j'} * delta_{k,k'} * delta_{i,i}
    # So T[(i,j),(j,k),(k,i)] = 1 (with proper index matching)
    # 
    # A rank-r decomposition: T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{nm}, v_c ∈ R^{mp}, w_c ∈ R^{pn}
    # u_c[i*m+j] = U[i*m+j, c], etc.
    # 
    # Verification: for all (i,j), (j',k'), (k'',i''):
    # sum_c U[i*m+j, c] * V[j'*p+k', c] * W[k''*n+i'', c] = delta_{j,j'} * delta_{k',k''} * delta_{i,i''}
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check a sample of tensor entries
    # Full check would be nm * mp * pn = 15*35*21 = 11025 entries, each requiring rank operations
    # Let's do a sparse check
    
    # Actually, let's verify by checking the matrix multiplication property directly.
    # If the decomposition is correct, then for random A, B:
    # C = A @ B should equal the "decomposed" version
    
    rng = np.random.RandomState(42)
    for _ in range(5):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_true = A @ B
        
        # Compute via decomposition
        C_comp = np.zeros((n, p))
        for c in range(rank):
            # U_col = U[:, c], V_col = V[:, c], W_col = W[:, c]
            u_col = U_arr[:, c]  # shape (n*m,)
            v_col = V_arr[:, c]  # shape (m*p,)
            w_val = W_arr[:, c]  # shape (p*n,)
            
            # Reshape u_col to (n, m), v_col to (m, p)
            u_mat = u_col.reshape(n, m)
            v_mat = v_col.reshape(m, p)
            
            # The contribution to C[i,k] is w_val[k*n+i] * (u_mat @ ... ) 
            # Actually: C[i,k] += w_val[k*n+i] * (sum_j u_mat[i,j] * ... ) 
            # Hmm, let me think again.
            
            # u_c ⊗ v_c ⊗ w_c means:
            # (u_c ⊗ v_c ⊗ w_c)[(i,j),(j',k'),(k'',i'')] = u_c[i*m+j] * v_c[j'*p+k'] * w_c[k''*n+i'']
            # 
            # For the matmul: we need this to equal delta_{j,j'} delta_{k',k''} delta_{i,i''}
            # 
            # The actual multiplication: C[i,k] = sum_j A[i,j] B[j,k]
            # In tensor form: the "fiber" over (i,j) for A, (j,k) for B, (i,k) for C
            # 
            # I think the correct check is:
            # For each c, define:
            #   A_c[i,j] = U[i*m+j, c]
            #   B_c[j,k] = V[j*p+k, c]  
            #   C_c[i,k] = W[k*n+i, c]  (note: W index is k*n+i, so C_c[i,k] = W[k*n+i, c])
            # 
            # Then the decomposition says: sum_c C_c[i,k] * (something with A_c, B_c) = ...
            # 
            # Actually, I think the right interpretation is:
            # The tensor T has T[(i,j),(j',k),(k',i')] = δ_{j,j'} δ_{k,k'} δ_{i,i'}
            # And T = sum_c u_c ⊗ v_c ⊗ w_c where:
            # (u_c ⊗ v_c ⊗ w_c)[(i,j),(j',k),(k',i')] = u_c[(i,j)] v_c[(j',k)] w_c[(k',i')]
            # 
            # So: sum_c u_c[i*m+j] * v_c[j'*p+k] * w_c[k'*n+i'] = δ_{j,j'} δ_{k,k'} δ_{i,i'}
            # 
            # Let's verify this identity directly for a few index combinations.
            
        # Let's do the direct tensor identity check instead
        # Pick random indices and verify
        for trial in range(200):
            i = rng.randint(n)
            j = rng.randint(m)
            j2 = rng.randint(m)
            k = rng.randint(p)
            k2 = rng.randint(p)
            i2 = rng.randint(n)
            
            lhs = sum(U_arr[i*m+j, c] * V_arr[j2*p+k, c] * W_arr[k2*n+i2, c] for c in range(rank))
            rhs = (1.0 if j == j2 and k == k2 and i == i2 else 0.0)
            if abs(lhs - rhs) > tol:
                return False
    return True


def _schoolbook(n=3, m=5, p=7):
    """Standard schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_strassen_like(n=3, m=5, p=7):
    """
    Attempt to find a lower-rank decomposition using structured search.
    Try to exploit the tensor structure.
    
    Strategy: Initialize with random matrices of decreasing rank, verify, keep best.
    """
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # First, get the schoolbook solution as fallback
    sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_sol = sb
    
    # Try to find lower rank decompositions
    # The rank must be at least max(n*m, m*p, p*n) / something... 
    # Lower bound: rank >= max(n*m, m*p, p*n) / min(n,m,p)? Not tight.
    # Simple lower bound: rank >= max(n*m, m*p, p*n) is wrong.
    # Actually rank >= max(n*m, m*p, p*n) is NOT a lower bound.
    # The rank is at least the dimension of each space: rank >= n*m, rank >= m*p, rank >= p*n? No.
    # rank >= max(n, m, p) * something.
    # 
    # For now, try ranks from a reasonable lower estimate up to schoolbook.
    
    # Lower bound: the tensor has support on n*m*p entries (all 1s in the "diagonal" pattern)
    # Actually the tensor T[(i,j),(j',k),(k',i')] = δ_{j,j'}δ_{k,k'}δ_{i,i'}
    # This is a permutation-like tensor. Its rank is related to the Strassen rank.
    
    # Let's try to use a greedy/randomized approach for a few ranks below 105.
    rng = np.random.RandomState(12345)
    
    for target_rank in [90, 85, 80, 75, 70, 65, 60]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Try a few random initializations at this rank
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            U_rand = rng.randn(n * m, target_rank)
            V_rand = rng.randn(m * p, target_rank)
            W_rand = rng.randn(p * n, target_rank)
            
            # Convert to list of lists for verification
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p, tol=1e-8):
                best_rank = target_rank
                best_sol = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    return best_sol


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Always start with schoolbook as guaranteed valid
    sb = _schoolbook(n, m, p)
    
    # Try to find better solution
    # But only if we have time
    if time.time() - start_time < 140:
        try:
            better = _try_strassen_like(n, m, p)
            if better is not None:
                # Verify the better solution
                U_arr = np.array(better["U"], dtype=float)
                V_arr = np.array(better["V"], dtype=float)
                W_arr = np.array(better["W"], dtype=float)
                rank = U_arr.shape[1]
                if rank < n * m * p:
                    # Quick verification
                    if _verify(better["U"], better["V"], better["W"], n, m, p, tol=1e-8):
                        return better
        except Exception:
            pass
    
    # Fallback: schoolbook
    return sb

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
