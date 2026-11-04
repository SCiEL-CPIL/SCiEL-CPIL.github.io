"""Rewritten program: deterministic rank-17 tensor decomposition for 2x3x3 matrix multiplication."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(result, T, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline with safety margin

    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6

    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))

    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val

    # Rank-17 decomposition for 2x3x3 matrix multiplication.
    # Index mapping:
    # a: A[i,j] -> i*3+j
    #   a0=A[0,0], a1=A[0,1], a2=A[0,2], a3=A[1,0], a4=A[1,1], a5=A[1,2]
    # b: B[j,k] -> j*3+k
    #   b0=B[0,0], b1=B[0,1], b2=B[0,2], b3=B[1,0], b4=B[1,1], b5=B[1,2],
    #   b6=B[2,0], b7=B[2,1], b8=B[2,2]
    # c: C[i,k] -> k*2+i
    #   c0=C[0,0], c1=C[1,0], c2=C[0,1], c3=C[1,1], c4=C[0,2], c5=C[1,2]

    # Part 1: Strassen for A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2]
    # M1 = (a0+a4)*(b0+b4)
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    # M2 = (a3+a4)*b0
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    # M3 = a0*(b1-b4)
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    # M4 = a4*(b3-b0)
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    # M5 = (a0+a1)*b4
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    # M6 = (a3-a0)*(b0+b1)
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    # M7 = (a1-a4)*(b3+b4)
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})

    # Part 2: Direct for A[0:2, 0:2] x B[0:2, 2] -> C[:, 2]
    # C[0,2] = a0*b2 + a1*b5
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    # C[1,2] = a3*b2 + a4*b5
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})

    # Part 3: Direct for A[0:2, 2] x B[2, 0:3] -> C[0:2, 0:3]
    # C[0,0] += a2*b6
    set_term(11, {2:1}, {6:1}, {0:1})
    # C[0,1] += a2*b7
    set_term(12, {2:1}, {7:1}, {2:1})
    # C[0,2] += a2*b8
    set_term(13, {2:1}, {8:1}, {4:1})
    # C[1,0] += a5*b6
    set_term(14, {5:1}, {6:1}, {1:1})
    # C[1,1] += a5*b7
    set_term(15, {5:1}, {7:1}, {3:1})
    # C[1,2] += a5*b8
    set_term(16, {5:1}, {8:1}, {5:1})

    # Verify
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook rank 18
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
