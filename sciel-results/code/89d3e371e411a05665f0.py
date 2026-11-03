"""Tensor decomposition for 4x4 * 4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

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
    
    # Term 1: (A00+A11)(B00+B11)
    u1 = [0]*9; u1[0]=1; u1[4]=1
    v1 = [0]*9; v1[0]=1; v1[4]=1
    w1 = [0]*9; w1[0]=1; w1[4]=1
    terms.append((u1, v1, w1))
    
    # Term 2: (A10+A11)(B00)
    u2 = [0]*9; u2[3]=1; u2[4]=1
    v2 = [0]*9; v2[0]=1
    w2 = [0]*9; w2[0]=1
    terms.append((u2, v2, w2))
    
    # Term 3: A00*(B10-B11)
    u3 = [0]*9; u3[0]=1
    v3 = [0]*9; v3[3]=1; v3[4]=-1
    w3 = [0]*9; w3[3]=1
    terms.append((u3, v3, w3))
    
    # Term 4: A11*(B00-B01)
    u4 = [0]*9; u4[4]=1
    v4 = [0]*9; v4[0]=1; v4[1]=-1
    w4 = [0]*9; w4[0]=1; w4[4]=1
    terms.append((u4, v4, w4))
    
    # Term 5: (A00+A01)*B11
    u5 = [0]*9; u5[0]=1; u5[1]=1
    v5 = [0]*9; v5[4]=1
    w5 = [0]*9; w5[3]=1; w5[4]=1
    terms.append((u5, v5, w5))
    
    # Term 6: (A10-A00)*(B00+B01)
    u6 = [0]*9; u6[3]=1; u6[0]=-1
    v6 = [0]*9; v6[0]=1; v6[1]=1
    w6 = [0]*9; w6[0]=1; w6[4]=1
    terms.append((u6, v6, w6))
    
    # Term 7: (A01-A11)*(B10+B11)
    u7 = [0]*9; u7[1]=1; u7[4]=-1
    v7 = [0]*9; v7[3]=1; v7[4]=1
    w7 = [0]*9; w7[0]=1; w7[4]=1
    terms.append((u7, v7, w7))
    
    return terms

def _construct_strassen_embedded(n=4, m=4, p=5):
    """
    Construct a decomposition using Strassen for the 3x3x3 sub-tensor
    and schoolbook for the boundary.
    """
    strassen_terms = _strassen_3x3x3_terms()
    
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (u, v, w) in enumerate(strassen_terms):
        for i in range(3):
            for j in range(3):
                U[i * m + j][t] = u[i * 3 + j]
        for j in range(3):
            for k in range(3):
                V[j * p + k][t] = v[j * 3 + k]
        for k in range(3):
            for i in range(3):
                W[k * n + i][t] = w[k * 3 + i]
    
    # Add schoolbook terms for the boundary
    boundary_triples = []
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if i >= 3 or k >= 3 or j >= 3:
                    boundary_triples.append((i, k, j))
    
    for (i, k, j) in boundary_triples:
        U[i * m + j][rank] = 1
        V[j * p + k][rank] = 1
        W[k * n + i][rank] = 1
        rank += 1
    
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _greedy_build(n, m, p, max_rank, time_limit=30):
    """
    Local search: start with Strassen-embedded, then try to reduce rank
    by removing redundant terms or merging pairs.
    """
    start_time = time.time()
    
    result = _construct_strassen_embedded(n, m, p)
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    rank = U.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def _is_valid(U_mat, V_mat, W_mat, r):
        U3 = U_mat[:, :r].reshape(n, m, -1)
        V3 = V_mat[:, :r].reshape(m, p, -1)
        W3 = W_mat[:, :r].reshape(p, n, -1)
        M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
        return np.allclose(M_decomp, 1.0, atol=1e-9)
    
    # Try removing terms
    for attempt in range(500):
        if time.time() - start_time > time_limit:
            break
        if rank <= max_rank:
            break
        
        found_merge = False
        order = list(range(rank))
        np.random.shuffle(order)
        
        for t in order:
            if time.time() - start_time > time_limit:
                break
            
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
                found_merge = True
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if not found_merge:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}, best_rank

def _pairwise_merge_search(n, m, p, start_result, time_limit=60):
    """
    More aggressive search: try to replace pairs of terms with single terms,
    and try to find low-rank completions by solving linear systems.
    """
    start_time = time.time()
    
    U = np.array(start_result["U"], dtype=float)
    V = np.array(start_result["V"], dtype=float)
    W = np.array(start_result["W"], dtype=float)
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
    
    # Phase 1: Try removing single terms (already done in _greedy_build, but redo)
    for t in range(rank):
        if time.time() - start_time > time_limit:
            break
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
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            t -= 1  # re-check same index
    
    # Phase 2: Try replacing pairs with a single term
    # For each pair (a, b), check if there exists a single term that can replace them
    # This means: the sum of terms a and b should be expressible as a rank-1 term
    # In other words, U[:,a]+U[:,b], V[:,a]+V[:,b], W[:,a]+W[:,b] should satisfy
    # the tensor equation when the other terms are present
    for _round in range(200):
        if time.time() - start_time > time_limit:
            break
        if rank <= 2:
            break
        
        found = False
        # Try all pairs
        for a in range(rank):
            if time.time() - start_time > time_limit:
                break
            for b in range(a+1, rank):
                # Check if removing both a and b and adding a new term works
                # New term: we need to find u, v, w such that
                # sum_{t not in {a,b}} u_t v_t w_t + u_new v_new w_new = target
                # This is complex; instead, try if the pair is redundant
                U_test = U.copy()
                V_test = V.copy()
                W_test = W.copy()
                U_test[:, a] = 0
                U_test[:, b] = 0
                V_test[:, a] = 0
                V_test[:, b] = 0
                W_test[:, a] = 0
                W_test[:, b] = 0
                
                # Check if the remaining terms already give the correct result
                # (i.e., the pair is completely redundant)
                r_remaining = rank - 2
                if r_remaining > 0 and _is_valid(U_test, V_test, W_test, r_remaining):
                    # Remove both
                    keep = [i for i in range(rank) if i not in (a, b)]
                    U = U[:, keep]
                    V = V[:, keep]
                    W = W[:, keep]
                    rank = r_remaining
                    found = True
                    best_rank = rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            
            if found:
                break
        
        if not found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}, best_rank

def _randomized_greedy(n, m, p, time_limit=90):
    """
    Randomized approach: start from random initializations and greedily build
    a low-rank decomposition by solving linear systems.
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
    
    # Try different target ranks
    for target_rank in range(45, 80):
        if time.time() - start_time > time_limit:
            break
        if target_rank >= best_rank:
            continue
        
        # Multiple random restarts
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with random values
            np.random.seed(restart * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Greedily fix columns to satisfy the tensor equation
            success = False
            for col in range(target_rank):
                if time.time() - start_time > time_limit:
                    break
                
                # Compute the residual
                if col == 0:
                    U3 = U[:, :1].reshape(n, m, -1)
                    V3 = V[:, :1].reshape(m, p, -1)
                    W3 = W[:, :1].reshape(p, n, -1)
                    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
                else:
                    U3 = U[:, :col].reshape(n, m, -1)
                    V3 = V[:, :col].reshape(m, p, -1)
                    W3 = W[:, :col].reshape(p, n, -1)
                    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
                
                residual = 1.0 - M_decomp  # shape (n, p, m)
                
                # For the next term, we need u, v, w such that
                # u[i,j] * v[j,k] * w[k,i] = residual[i,k,j] for all i,k,j
                # This is a system of n*p*m equations in (n*m + m*p + p*n) unknowns
                # But it's nonlinear (trilinear). 
                # Instead, fix two and solve for the third.
                
                # Fix u and v, solve for w:
                # w[k,i] = residual[i,k,j] / (u[i,j] * v[j,k]) for all j
                # This is overdetermined; use least squares
                # Actually, for each (k,i), we need:
                # sum_j u[i,j] * v[j,k] * w[k,i] = residual[i,k,j]... 
                # Wait, the equation is: for each (i,k), sum_j u[i,j]*v[j,k]*w[k,i]... 
                # No, w depends on (k,i), not j. So the equation is:
                # For each (i,k): u[i,:] dot v[:,k] * w[k,i] = sum_j residual[i,k,j]
                # That's not right either. Let me re-derive.
                
                # The tensor equation is: for each (i,k,j):
                #   sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1
                # So for term col, we need:
                #   U[i,j,col] * V[j,k,col] * W[k,i,col] = residual[i,k,j]
                # for all i,k,j.
                
                # This is a system of 80 equations in 32 unknowns (for n=4,m=4,p=5:
                # U has 16 entries, V has 20, W has 20, total 56 unknowns)
                # Wait, U is (4*4)=16, V is (4*5)=20, W is (5*4)=20, total 56.
                # And we have 4*5*4=80 equations. Overdetermined.
                
                # Strategy: fix U and V, solve for W via least squares
                # For each (k,i): W[k,i] * sum_j U[i,j] * V[j,k] = sum_j residual[i,k,j]
                # Let S[i,k] = sum_j U[i,j] * V[j,k]
                # Then W[k,i] = sum_j residual[i,k,j] / S[i,k] if S[i,k] != 0
                
                U_i = U[:, col].reshape(n, m)
                V_i = V[:, col].reshape(m, p)
                
                # S[i,k] = sum_j U[i,j] * V[j,k]
                S = U_i @ V_i  # shape (n, p)
                
                # target[i,k] = sum_j residual[i,k,j]
                target = np.sum(residual, axis=2)  # shape (n, p)
                
                # W[k,i] = target[i,k] / S[i,k]
                W_new = np.zeros((p, n))
                valid = np.abs(S) > 1e-10
                W_new.T[valid] = target[valid] / S[valid]
                
                # Check if this works
                U_test = U.copy()
                V_test = V.copy()
                W_test = W.copy()
                U_test[:, col] = U_i.flatten()
                V_test[:, col] = V_i.flatten()
                W_test[:, col] = W_new.flatten()
                
                if _is_valid(U_test, V_test, W_test, col + 1):
                    if col + 1 < best_rank:
                        best_rank = col + 1
                        best_result = {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}
                    success = True
                    break
                else:
                    # Perturb and try again
                    U[:, col] += np.random.randn(n * m) * 0.1
                    V[:, col] += np.random.randn(m * p) * 0.1
            
            if success:
                break
    
    return best_result, best_rank

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_result = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try the Strassen-embedded approach
    try:
        strassen_result = _construct_strassen_embedded(n, m, p)
        rank_s = len(strassen_result["U"][0]) if strassen_result["U"] else 0
        if rank_s < best_rank:
            if _validate(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
                best_result = strassen_result
                best_rank = rank_s
    except Exception:
        pass
    
    # Try greedy improvement
    try:
        greedy_result, greedy_rank = _greedy_build(n, m, p, best_rank, time_limit=30)
        if greedy_rank < best_rank:
            if _validate(greedy_result["U"], greedy_result["V"], greedy_result["W"], n, m, p):
                best_result = greedy_result
                best_rank = greedy_rank
    except Exception:
        pass
    
    # Try pairwise merge search
    try:
        merge_result, merge_rank = _pairwise_merge_search(n, m, p, best_result, time_limit=40)
        if merge_rank < best_rank:
            if _validate(merge_result["U"], merge_result["V"], merge_result["W"], n, m, p):
                best_result = merge_result
                best_rank = merge_rank
    except Exception:
        pass
    
    # Try randomized approach with remaining time
    remaining = deadline - time.time()
    if remaining > 10:
        try:
            rand_result, rand_rank = _randomized_greedy(n, m, p, time_limit=max(10, remaining - 5))
            if rand_rank < best_rank:
                if _validate(rand_result["U"], rand_result["V"], rand_result["W"], n, m, p):
                    best_result = rand_result
                    best_rank = rand_rank
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
