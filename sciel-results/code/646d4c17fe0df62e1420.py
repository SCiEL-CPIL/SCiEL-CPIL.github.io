import time
import numpy as np
from itertools import product

DEADLINE = 120.0
START_TIME = time.time()

def _check_deadline():
    return time.time() - START_TIME > DEADLINE - 10

def _construct_trivial(n, m, p):
    """
    Construct the trivial (schoolbook) tensor decomposition.
    Returns U, V, W as lists of lists.
    U: (n*m, n*m*p)
    V: (m*p, n*m*p)
    W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                r += 1
                
    return U, V, W

def _tensor_check(U, V, W, n, m, p):
    """
    Check if the decomposition is valid for all i, k.
    C[i, k] = sum_{j, r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Should equal sum_{j} A[i, j] B[j, k].
    We check the tensor identity:
    T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    Must equal delta_{i, j, k} where the tensor for matmul has 1s at (i, j, k)
    corresponding to C[i, k] += A[i, j] B[j, k].
    Wait, the tensor for matmul is T_{i,j,k} = 1 if it contributes to C[i,k] via A[i,j]B[j,k].
    Actually, the tensor is defined such that C[i,k] = sum_{j,r} A[i,j] B[j,k] W... no.
    
    The decomposition is:
    C[i, k] = sum_{j=0}^{m-1} sum_{r=0}^{R-1} A[i, j] B[j, k] * (something?)
    
    Let's look at the definition in the prompt:
    "W[k*n+i,r] corresponds to C[i, k] += U[.,r] * V[.,r] * W[k*n+i, r]"
    This phrasing is slightly ambiguous. Usually:
    C[i, k] = sum_{r} (sum_j U[i*m+j, r] A[i, j]) * (sum_j V[j*p+k, r] B[j, k]) * W[k*n+i, r]
    
    Let's verify with trivial case:
    U[i*m+j, r] is 1 only if r corresponds to (i, j, k').
    V[j*p+k, r] is 1 only if r corresponds to (i', j, k).
    W[k'*n+i, r] is 1 only if r corresponds to (i, j, k).
    
    For the product to be non-zero, r must match (i, j, k) for all three.
    So U[i*m+j, r] = 1 iff r=(i,j,k).
    V[j*p+k, r] = 1 iff r=(i,j,k).
    W[k*n+i, r] = 1 iff r=(i,j,k).
    
    Then C[i, k] = sum_{j, r} A[i, j] B[j, k] delta_{r, (i,j,k)} * 1
    = sum_{j} A[i, j] B[j, k].
    This is correct.
    
    So the check is:
    For all i, j, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    For all other combinations?
    The tensor T is trilinear.
    T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    We require T_{i,j,k} = 1 for all i, j, k?
    No.
    In matrix multiplication C = A B,
    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    The tensor of the bilinear map is T such that
    C_{i,k} = sum_{j, r} A_{i,j} B_{j,k} T_{i,j,k,r}? No.
    
    The standard definition:
    A bilinear map f: U x V -> W can be written as sum_r u_r(a) v_r(b) w_r.
    Here, the inputs are A (flattened) and B (flattened).
    The output is C (flattened).
    But the "multiplication" structure is specific.
    
    Let's stick to the explicit formula given in the prompt's Tensor class docstring or similar:
    C[i, k] = sum_{j, r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i, j] * B[j, k] ?
    No, U and V *are* the linear combinations.
    
    Correct interpretation:
    Let x = flatten(A), y = flatten(B).
    z_r = sum_i U[i, r] x_i = sum_{i,j} U[i*m+j, r] A[i, j]
    y_r = sum_j V[j, r] y_j = sum_{j,k} V[j*p+k, r] B[j, k]
    C[i, k] = sum_r z_r y_r W[k*n+i, r]
    
    So C[i, k] = sum_r (sum_{j} U[i*m+j, r] A[i, j] ... wait, U index is i*m+j.
    z_r = sum_{i', j'} U[i'*m+j', r] A[i', j'].
    This sums over ALL i', j'.
    
    So C[i, k] = sum_r W[k*n+i, r] * (sum_{i', j'} U[i'*m+j', r] A[i', j']) * (sum_{j', k'} V[j'*p+k', r] B[j', k'])
    
    We want this to equal sum_{j} A[i, j] B[j, k].
    
    This implies that for a fixed r, the term in the sum must be "aligned" with i, j, k.
    Specifically, the coefficient of A[i, j] B[j, k] in the expansion must be 1.
    The coefficient of A[i, j] B[j, k] in the r-th term is:
    W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r].
    
    So the condition is:
    For all i, j, k:
    sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
    
    And for all other combinations of indices that don't form a valid (i, j, k) path?
    Actually, the indices i, j, k are linked.
    U depends on i, j.
    V depends on j, k.
    W depends on i, k.
    
    So the "tensor" is defined on the indices (i, j, k).
    T_{i, j, k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    We require T_{i, j, k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    Is that sufficient?
    Yes, because the expansion of C[i, k] is:
    C[i, k] = sum_r W[k*n+i, r] (sum_{i', j'} U[i'*m+j', r] A[i', j']) (sum_{j', k'} V[j'*p+k', r] B[j', k'])
    = sum_{i', j', j', k'} W[k*n+i, r] U[i'*m+j', r] V[j'*p+k', r] A[i', j'] B[j', k']
    
    We want the coefficient of A[i, j] B[j, k] to be 1.
    In the sum above, the term A[i, j] B[j, k] appears when i'=i, j'=j, and j'=j, k'=k.
    So the coefficient is sum_r W[k*n+i, r] U[i*m+j, r] V[j*p+k, r].
    This must be 1.
    
    What about other terms? e.g., A[i, j] B[j', k] with j != j'?
    The coefficient is sum_r W[k*n+i, r] U[i*m+j, r] V[j'*p+k, r].
    This must be 0.
    
    So the full condition is:
    For all i, k, j, j':
    sum_r W[k*n+i, r] U[i*m+j, r] V[j'*p+k, r] = delta_{j, j'}
    
    This is equivalent to:
    For all i, k:
    (U_i * V_k^T)_{j, j'} = delta_{j, j'}?
    Where U_i is the vector of U[i*m+0..m-1, :] and V_k is vector of V[0*p+k .. (m-1)p+k, :].
    Actually, let M_{i,k} be the m x m matrix with entries M_{j, j'} = sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r].
    We require M_{i,k} = I_m for all i, k.
    
    So the check is:
    For all i in [0, n):
    For all k in [0, p):
        Compute M_{i,k} (m x m matrix).
        Check if M_{i,k} == Identity(m).
    """
    # This check is O(n * p * m^2 * rank).
    # n=2, p=7, m=4. 2*7*16*R. R~50. 11200 ops. Fast.
    I_m = np.eye(m)
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m), dtype=float)
            for j in range(m):
                for jp in range(m):
                    # M[j, jp] = sum_r U[i*m+j, r] V[jp*p+k, r] W[k*n+i, r]
                    # Vectorized:
                    # u_vec = U[i*m+j, :]
                    # v_vec = V[jp*p+k, :]
                    # w_val = W[k*n+i, :]
                    # M[j, jp] = dot(u_vec * v_vec, w_val)
                    pass
            
            # Vectorized check for speed
            # U_i: shape (m, R) -> rows j
            # V_k: shape (m, R) -> rows jp (indexed by jp*p+k)
            # W_ik: shape (R,)
            
            U_sub = U[i*m : i*m+m, :] # (m, R)
            V_sub = V[0*m*p + k : m*p + k : p, :] # (m, R) -- careful with stride
            # V index is j*p+k. So for fixed k, indices are 0*p+k, 1*p+k, ..., (m-1)*p+k.
            # These are not contiguous if p > 1.
            V_indices = np.arange(m) * p + k
            V_sub = V[V_indices, :] # (m, R)
            
            W_vec = W[k*n + i, :] # (R,)
            
            # M[j, jp] = sum_r U_sub[j, r] * V_sub[jp, r] * W_vec[r]
            # M = (U_sub * W_vec) @ V_sub.T
            # U_sub * W_vec broadcasts (m, R) * (R,) -> (m, R)
            M = (U_sub * W_vec) @ V_sub.T
            
            if not np.allclose(M, I_m, atol=1e-9):
                return False
    return True

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 7
    
    try:
        # 1. Initialize with Trivial Decomposition
        U, V, W = _construct_trivial(n, m, p)
        current_rank = U.shape[1]
        
        # 2. Iterative Reduction
        # We try to remove columns.
        # A column r can be removed if its removal doesn't break the identity.
        # However, simply removing a column breaks the identity.
        # We need to "absorb" the contribution of column r into other columns.
        # Or, we can look for linear dependencies.
        
        # A more robust approach for small ranks:
        # Randomized Search for Rank Minimization.
        # Start with Trivial (56).
        # Try to find a valid decomposition of rank R < 56.
        # We can use the fact that Strassen gives 7 for 2x2x2.
        # 2x4x7 is close to 2x4x4 (Rank 28 via 4x Strassen) + 2x4x3 (Rank 22 via 2x Strassen + 2x Schoolbook).
        # Total 50.
        
        # Let's try to construct the 50-rank solution explicitly but programmatically
        # to avoid the errors in the previous hardcoded version.
        
        def build_strassen_2x2x2():
            n, m, p = 2, 2, 2
            R = 7
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            # P1 = (A11+A22)(B11+B22)
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[3, 0] = 1
            
            # P2 = (A21+A22)A11
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[2, 1] = 1; W[3, 1] = -1
            
            # P3 = A11(B12-B22)
            U[0, 2] = 1
            V[1, 2] = 1; V[3, 2] = -1
            W[1, 2] = 1; W[3, 2] = 1
            
            # P4 = A22(B21-B11)
            U[3, 3] = 1
            V[2, 3] = 1; V[0, 3] = -1
            W[0, 3] = 1; W[2, 3] = 1
            
            # P5 = (A11+A12)B22
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[0, 4] = -1; W[3, 4] = 1
            
            # P6 = (A21-A11)(B11+B12)
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            W[3, 5] = 1
            
            # P7 = (A12-A22)(B21+B22)
            U[1, 6] = 1; U[3, 6] = -1
            V[2, 6] = 1; V[3, 6] = 1
            W[0, 6] = 1; W[2, 6] = -1
            
            return U, V, W

        def build_schoolbook(n, m, p):
            R = n * m * p
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, r] = 1
                        V[j*p+k, r] = 1
                        W[k*n+i, r] = 1
                        r += 1
            return U, V, W

        def embed(U_src, V_src, W_src, n_src, m_src, p_src, n_tgt, m_tgt, p_tgt, n_map, m_map, p_map):
            U = np.zeros((n_tgt*m_tgt, U_src.shape[1]))
            V = np.zeros((m_tgt*p_tgt, V_src.shape[1]))
            W = np.zeros((p_tgt*n_tgt, W_src.shape[1]))
            
            for i in range(n_src):
                for j in range(m_src):
                    si = i*m_src + j
                    ti = n_map[i]*m_tgt + m_map[j]
                    U[ti, :] = U_src[si, :]
                    
            for j in range(m_src):
                for k in range(p_src):
                    sj = j*p_src + k
                    tj = m_map[j]*p_tgt + p_map[k]
                    V[tj, :] = V_src[sj, :]
                    
            for k in range(p_src):
                for i in range(n_src):
                    sk = k*n_src + i
                    tk = p_map[k]*n_tgt + n_map[i]
                    W[tk, :] = W_src[sk, :]
                    
            return U, V, W

        # Build 2x2x2 Strassen
        U222, V222, W222 = build_strassen_2x2x2()
        
        # Build 2x2x1 Schoolbook
        U221, V221, W221 = build_schoolbook(2, 2, 1)
        
        # Build 2x2x3
        # Sum of 2x2x2 (on p=0,1) and 2x2x1 (on p=2)
        U223_1, V223_1, W223_1 = embed(U222, V222, W222, 2, 2, 2, 2, 2, 3, [0,1], [0,1], [0,1])
        U223_2, V223_2, W223_2 = embed(U221, V221, W221, 2, 2, 1, 2, 2, 3, [0,1], [0,1], [2])
        
        U223 = np.hstack([U223_1, U223_2])
        V223 = np.hstack([V223_1, V223_2])
        W223 = np.hstack([W223_1, W223_2])
        
        # Build 2x4x3
        # Sum of two 2x2x3 blocks
        # Block 1: m=0,1
        U243_1, V243_1, W243_1 = embed(U223, V223, W223, 2, 2, 3, 2, 4, 3, [0,1], [0,1], [0,1,2])
        # Block 2: m=2,3
        U243_2, V243_2, W243_2 = embed(U223, V223, W223, 2, 2, 3, 2, 4, 3, [0,1], [2,3], [0,1,2])
        
        U243 = np.hstack([U243_1, U243_2])
        V243 = np.hstack([V243_1, V243_2])
        W243 = np.hstack([W243_1, W243_2])
        
        # Build 2x4x4
        # Sum of four 2x2x2 blocks
        # Block 1: m=0,1, p=0,1
        U244_1, V244_1, W244_1 = embed(U222, V222, W222, 2, 2, 2, 2, 4, 4, [0,1], [0,1], [0,1])
        # Block 2: m=2,3, p=0,1
        U244_2, V244_2, W244_2 = embed(U222, V222, W222, 2, 2, 2, 2, 4, 4, [0,1], [2,3], [0,1])
        # Block 3: m=0,1, p=2,3
        U244_3, V244_3, W244_3 = embed(U222, V222, W222, 2, 2, 2, 2, 4, 4, [0,1], [0,1], [2,3])
        # Block 4: m=2,3, p=2,3
        U244_4, V244_4, W244_4 = embed(U222, V222, W222, 2, 2, 2, 2, 4, 4, [0,1], [2,3], [2,3])
        
        U244 = np.hstack([U244_1, U244_2, U244_3, U244_4])
        V244 = np.hstack([V244_1, V244_2, V244_3, V244_4])
        W244 = np.hstack([W244_1, W244_2, W244_3, W244_4])
        
        # Build 2x4x7
        # Sum of 2x4x4 (p=0..3) and 2x4x3 (p=4..6)
        U247_1, V247_1, W247_1 = embed(U244, V244, W244, 2, 4, 4, 2, 4, 7, [0,1], [0,1,2,3], [0,1,2,3])
        U247_2, V247_2, W247_2 = embed(U243, V243, W243, 2, 4, 3, 2, 4, 7, [0,1], [0,1,2,3], [4,5,6])
        
        U_final = np.hstack([U247_1, U247_2])
        V_final = np.hstack([V247_1, V247_2])
        W_final = np.hstack([W247_1, W247_2])
        
        # Verify
        if _tensor_check(U_final, V_final, W_final, n, m, p):
            return {
                "U": U_final.tolist(),
                "V": V_final.tolist(),
                "W": W_final.tolist()
            }
        else:
            # Fallback to trivial if construction failed
            U_t, V_t, W_t = _construct_trivial(n, m, p)
            return {
                "U": U_t.tolist(),
                "V": V_t.tolist(),
                "W": W_t.tolist()
            }

    except Exception as e:
        # Fallback to trivial
        U_t, V_t, W_t = _construct_trivial(n, m, p)
        return {
            "U": U_t.tolist(),
            "V": V_t.tolist(),
            "W": W_t.tolist()
        }

if __name__ == "__main__":
    res = construct()
    print("Constructed.")
