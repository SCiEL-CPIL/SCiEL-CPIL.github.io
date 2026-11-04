"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _tensor_from_matrices(U, V, W):
    """Compute the tensor from factor matrices."""
    n_a, rank = U.shape
    n_b = V.shape[0]
    n_c = W.shape[0]
    T = np.zeros((n_a, n_b, n_c))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return T

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # Rank-17 decomposition:
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2] (7 terms)
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2] (4 terms)
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3] (6 terms)
    
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    terms = []
    
    # Part 1: Strassen for 2x2 x 2x2
    # A1 = [[a00, a01], [a10, a11]] corresponds to U indices 0,1,3,4
    # B1_left = [[b00, b01], [b10, b11]] corresponds to V indices 0,1,3,4
    # C[0:2, 0:2] corresponds to W indices 0,1,2,3
    
    # M1 = (a00+a11)(b00+b11)
    terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]))
    # M2 = (a10+a11)b00
    terms.append(([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]))
    # M3 = a00(b10-b11)
    terms.append(([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]))
    # M4 = a11(b00-b10)
    terms.append(([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]))
    # M5 = (a00+a01)b11
    terms.append(([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]))
    # M6 = (a10-a00)(b00+b01)
    terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    # M7 = (a01-a11)(b10+b11)
    terms.append(([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]))
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    # B1_right = [b02, b12] corresponds to V indices 2, 5
    # C[0,2] = a00*b02 + a01*b12 -> W index 4
    # C[1,2] = a10*b02 + a11*b12 -> W index 5
    terms.append(([0], [1], [2], [1], [4], [1]))
    terms.append(([1], [1], [5], [1], [4], [1]))
    terms.append(([3], [1], [2], [1], [5], [1]))
    terms.append(([4], [1], [5], [1], [5], [1]))
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    # A2 = [a02, a12] corresponds to U indices 2, 5
    # B2 = [b20, b21, b22] corresponds to V indices 6, 7, 8
    # C[0,0] = a02*b20 -> W index 0
    # C[0,1] = a02*b21 -> W index 1
    # C[0,2] = a02*b22 -> W index 2
    # C[1,0] = a12*b20 -> W index 3
    # C[1,1] = a12*b21 -> W index 4
    # C[1,2] = a12*b22 -> W index 5
    terms.append(([2], [1], [6], [1], [0], [1]))
    terms.append(([2], [1], [7], [1], [1], [1]))
    terms.append(([2], [1], [8], [1], [2], [1]))
    terms.append(([5], [1], [6], [1], [3], [1]))
    terms.append(([5], [1], [7], [1], [4], [1]))
    terms.append(([5], [1], [8], [1], [5], [1]))
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
            
    # Verify
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U, V, W, T):
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
