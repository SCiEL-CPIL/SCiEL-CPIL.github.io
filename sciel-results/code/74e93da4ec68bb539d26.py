"""Tensor decomposition for 4x4 * 4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that U,V,W form a valid tensor decomposition of matmul(n,m,p)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True

def _construct_schoolbook(n=4, m=4, p=5):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_3x3x3_terms():
    """
    Return the 7 rank-1 tensors for 3x3x3 matrix multiplication.
    """
    terms = []
    terms.append(([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]))  # simplified placeholder
    # Correct Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    u1 = [0,0,0, 0,1,0, 0,0,0]; v1 = [1,0,0, 0,0,0, 0,0,1]; w1 = [1,0,0, 0,0,0, 0,0,1]
    terms.append((u1, v1, w1))
    
    # Let me use the standard Strassen decomposition properly
    # For 2x2x2:
    # M1=(A00+A11)(B00+B11)
    # M2=(A10+A11)B00
    # M3=A00(B10-B11)
    # M4=A11(B00-B01)
    # M5=(A00+A01)B11
    # M6=(A10-A00)(B00+B01)
    # M7=(A01-A11)(B10+B11)
    
    # For 3x3x3, we can embed the 2x2 Strassen in the top-left 2x2 block
    # and use schoolbook for the remaining parts
    
    terms2 = []
    # M1: (A00+A11)(B00+B11) -> C00 +=, C11 +=
    u1 = [0]*9; u1[0]=1; u1[4]=1
    v1 = [0]*9; v1[0]=1; v1[4]=1
    w1 = [0]*9; w1[0]=1; w1[4]=1
    terms2.append((u1, v1, w1))
    
    # M2: (A10+A11)B00 -> C10 +=
    u2 = [0]*9; u2[3]=1; u2[4]=1
    v2 = [0]*9; v2[0]=1
    w2 = [0]*9; w2[3]=1
    terms2.append((u2, v2, w2))
    
    # M3: A00(B10-B11) -> C01 +=
    u3 = [0]*9; u3[0]=1
    v3 = [0]*9; v3[3]=1; v3[4]=-1
    w3 = [0]*9; w3[3]=1
    terms2.append((u3, v3, w3))
    
    # M4: A11(B00-B01) -> C10 +=, C11 +=
    u4 = [0]*9; u4[4]=1
    v4 = [0]*9; v4[0]=1; v4[1]=-1
    w4 = [0]*9; w4[0]=1; w4[4]=1
    terms2.append((u4, v4, w4))
    
    # M5: (A00+A01)B11 -> C01 +=, C11 +=
    u5 = [0]*9; u5[0]=1; u5[1]=1
    v5 = [0]*9; v5[4]=1
    w5 = [0]*9; w5[3]=1; w5[4]=1
    terms2.append((u5, v5, w5))
    
    # M6: (A10-A00)(B00+B01) -> C10 +=, C11 +=
    u6 = [0]*9; u6[3]=1; u6[0]=-1
    v6 = [0]*9; v6[0]=1; v6[1]=1
    w6 = [0]*9; w6[0]=1; w6[4]=1
    terms2.append((u6, v6, w6))
    
    # M7: (A01-A11)(B10+B11) -> C01 +=, C11 +=
    u7 = [0]*9; u7[1]=1; u7[4]=-1
    v7 = [0]*9; v7[3]=1; v7[4]=1
    w7 = [0]*9; w7[0]=1; w7[4]=1
    terms2.append((u7, v7, w7))
    
    return terms2

def _construct_strassen_embedded(n=4, m=4, p=5):
    """
    Construct a decomposition using Strassen for the 2x2x2 sub-tensor
    and schoolbook for the boundary.
    """
    strassen_terms = _strassen_3x3x3_terms()
    
    # Start with schoolbook
    result = _construct_schoolbook(n, m, p)
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    rank = n * m * p
    
    # The 7 Strassen terms replace 8 schoolbook terms for the 2x2x2 block
    # So we can potentially reduce rank by 1
    
    # For now, return the schoolbook result as baseline
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_remove_terms(U, V, W, n, m, p, max_iterations=100, time_limit=30):
    """
    Try to remove redundant terms from a decomposition.
    """
    start_time = time.time()
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def _is_valid(U_mat, V_mat, W_mat, r):
        if r <= 0:
            return False
        U3 = U_mat[:, :r].reshape(n, m, -1)
        V3 = V_mat[:, :r].reshape(m, p, -1)
        W3 = W_mat[:, :r].reshape(p, n, -1)
        M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
        return np.allclose(M_decomp, 1.0, atol=1e-9)
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        # Try removing each term
        for t in range(rank):
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, t] = 0
            V_test[:, t] = 0
            W_test[:, t] = 0
            
            if _is_valid(U_test, V_test, W_test, rank):
                U = np.delete(U, t, axis=1)
                V = np.delete(V, t, axis=1)
                W = np.delete(W, t, axis=1)
                rank -= 1
                found = True
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if not found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}, best_rank

def _randomized_search(n, m, p, time_limit=60):
    """
    Randomized search for low-rank decompositions.
    """
    start_time = time.time()
    best_rank = n * m * p
    best_result = _construct_schoolbook(n, m, p)
    
    def _is_valid(U_mat, V_mat, W_mat, r):
        if r <= 0:
            return False
        U3 = U_mat[:, :r].reshape(n, m, -1)
        V3 = V_mat[:, :r].reshape(m, p, -1)
        W3 = W_mat[:, :r].reshape(p, n, -1)
        M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
        return np.allclose(M_decomp, 1.0, atol=1e-9)
    
    # Try various target ranks
    for target_rank in range(60, 80):
        if time.time() - start_time > time_limit:
            break
        if target_rank >= best_rank:
            continue
        
        for restart in range(20):
            if time.time() - start_time > time_limit:
                break
            
            np.random.seed(restart * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Try to adjust W to make the decomposition valid
            for col in range(target_rank):
                if time.time() - start_time > time_limit:
                    break
                
                U_i = U[:, col].reshape(n, m)
                V_i = V[:, col].reshape(m, p)
                S = U_i @ V_i  # shape (n, p)
                
                # Compute current decomposition with cols 0..col-1
                if col > 0:
                    U3 = U[:, :col].reshape(n, m, -1)
                    V3 = V[:, :col].reshape(m, p, -1)
                    W3 = W[:, :col].reshape(p, n, -1)
                    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
                else:
                    M_decomp = np.zeros((n, p, m))
                
                residual = 1.0 - M_decomp  # shape (n, p, m)
                target = np.sum(residual, axis=2)  # shape (n, p)
                
                W_new = np.zeros((p, n))
                valid = np.abs(S) > 1e-10
                W_new.T[valid] = target[valid] / S[valid]
                
                U_test = U.copy()
                V_test = V.copy()
                W_test = W.copy()
                W_test[:, col] = W_new.flatten()
                
                if _is_valid(U_test, V_test, W_test, col + 1):
                    if col + 1 < best_rank:
                        best_rank = col + 1
                        best_result = {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}
                    break
    
    return best_result, best_rank

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_result = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen-embedded approach
    try:
        strassen_result = _construct_strassen_embedded(n, m, p)
        rank_s = len(strassen_result["U"][0]) if strassen_result["U"] else 0
        if rank_s < best_rank:
            if _validate(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
                best_result = strassen_result
                best_rank = rank_s
    except Exception:
        pass
    
    # Try to remove redundant terms
    try:
        U = np.array(best_result["U"], dtype=float)
        V = np.array(best_result["V"], dtype=float)
        W = np.array(best_result["W"], dtype=float)
        greedy_result, greedy_rank = _try_remove_terms(U, V, W, n, m, p, max_iterations=50, time_limit=20)
        if greedy_rank < best_rank:
            if _validate(greedy_result["U"], greedy_result["V"], greedy_result["W"], n, m, p):
                best_result = greedy_result
                best_rank = greedy_rank
    except Exception:
        pass
    
    # Try randomized search with remaining time
    remaining = deadline - time.time()
    if remaining > 15:
        try:
            rand_result, rand_rank = _randomized_search(n, m, p, time_limit=max(10, remaining - 5))
            if rand_rank < best_rank:
                if _validate(rand_result["U"], rand_result["V"], rand_result["W"], n, m, p):
                    best_result = rand_result
                    best_rank = rand_rank
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
