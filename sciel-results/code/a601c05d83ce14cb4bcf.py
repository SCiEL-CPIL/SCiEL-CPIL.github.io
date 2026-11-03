import time
import itertools

N, M, P = 3, 4, 4
NM, MP, PN = N*M, M*P, P*N
TOTAL_TENSORS = NM*MP*PN  # 3*4*4*4*4*4 = 768? No: 12*16*12 = 2304

def _verify(U, V, W, tol=1e-10):
    """Verify using einsum."""
    import numpy as np
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _try_reduce_rank(U, V, W, deadline):
    """
    Try to reduce rank by checking if any rank-1 term is a linear combination
    of the others. We test this by checking if removing term 'c' and adjusting
    coefficients can still produce the same tensor.
    
    For each term c, we check if the tensor T_c = u_c ⊗ v_c ⊗ w_c can be
    expressed as sum of other terms. This is equivalent to checking if the
    Kronecker product vector of T_c is in the span of the others.
    """
    import numpy as np
    import time
    
    rank = len(U[0])
    if rank <= 1:
        return None
    
    # Build the tensor vectors for each term
    # Each term c corresponds to a vector in R^(N*M*P)
    # T_c[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    term_vectors = []
    for c in range(rank):
        vec = np.zeros(N*M*P)
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    vec[i*M*P + j*P + k] = U[i*M+j][c] * V[j*P+k][c] * W[k*N+i][c]
        term_vectors.append(vec)
    
    # For each term, check if it's in the span of the others
    # This is expensive, so we limit the search
    for c in range(rank):
        if time.time() > deadline:
            break
        
        # Build matrix of all other term vectors
        other_indices = [i for i in range(rank) if i != c]
        if len(other_indices) == 0:
            continue
            
        # Create matrix with other vectors as columns
        A = np.column_stack([term_vectors[i] for i in other_indices])
        b = term_vectors[c]
        
        # Check if b is in the column space of A
        # Use least squares and check residual
        try:
            coeffs, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
            if len(residuals) > 0:
                resid = residuals[0]
                if resid < 1e-8:
                    # Term c is in the span of others, we can remove it
                    new_U = [U[i][:c] + U[i][c+1:] for i in range(NM)]
                    new_V = [V[i][:c] + V[i][c+1:] for i in range(MP)]
                    new_W = [W[i][:c] + W[i][c+1:] for i in range(PN)]
                    return {'U': new_U, 'V': new_V, 'W': new_W}
        except:
            pass
    
    return None

def _schoolbook():
    rank = NM*MP*PN
    U = [[0]*rank for _ in range(NM)]
    V = [[0]*rank for _ in range(MP)]
    W = [[0]*rank for _ in range(PN)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M+j][c] = 1
                V[j*P+k][c] = 1
                W[k*N+i][c] = 1
                c += 1
    return {'U': U, 'V': V, 'W': W}

def _strassen_44():
    rank = 44
    U = [[0]*rank for _ in range(NM)]
    V = [[0]*rank for _ in range(MP)]
    W = [[0]*rank for _ in range(PN)]
    
    # 7 Strassen steps for 2x2 block
    s_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # 4 blocks for the 2x2 output blocks
    blocks = [(0,0,0,0,0,0), (0,2,2,0,0,0), (0,0,0,2,0,2), (0,2,2,2,0,2)]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for (u_c, v_c, w_c) in s_steps:
            for (a, b), val in u_c.items():
                U[(a+xi)*M + (b+xj)][col] += val
            for (b, c), val in v_c.items():
                V[(b+yj)*P + (c+yk)][col] += val
            for (a, c), val in w_c.items():
                W[(c+zk)*N + (a+zi)][col] += val
            col += 1
    
    # Bottom row (row 2) done naively: 4x4 = 16 terms
    for j in range(M):
        for k in range(P):
            U[2*M + j][col] = 1
            V[j*P + k][col] = 1
            W[k*N + 2][col] = 1
            col += 1
    
    assert col == rank
    return {'U': U, 'V': V, 'W': W}

def _try_reduce_by_merge(U, V, W):
    """
    Try to find pairs of rank-1 tensors that can be merged into one.
    Two rank-1 terms with U vectors u1,u2; V v1,v2; W w1,w2 can be merged
    if u1 ⊗ u2 ⊗ v1 ⊗ v2 ⊗ w1 ⊗ w2 allows cancellation.
    
    Simplest merge: if u1=-u2 and v1=v2 and w1=w2 -> zero out. 
    Or more generally, try to find if two terms are identical -> combine.
    """
    # This is a heuristic - try small modifications
    pass

def _gaussian_reduce(U, V, W):
    """
    Try to reduce rank by finding linear dependencies among rank-1 terms.
    This is computationally expensive, so we limit the search.
    """
    pass

def construct():
    start = time.time()
    deadline = start + 120  # 30s safety margin
    
    best_result = _strassen_44()
    ok, err = _verify(best_result['U'], best_result['V'], best_result['W'])
    if not ok:
        best_result = _schoolbook()
    
    # Try to reduce rank by checking for linear dependencies
    if time.time() < deadline - 5:
        reduced = _try_reduce_rank(best_result['U'], best_result['V'], best_result['W'], deadline)
        if reduced is not None:
            ok2, err2 = _verify(reduced['U'], reduced['V'], reduced['W'])
            if ok2:
                best_result = reduced
    
    return best_result
