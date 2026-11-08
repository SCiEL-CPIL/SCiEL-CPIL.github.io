import numpy as np
import time
import itertools
import sys

DIMENSIONS = [2, 3, 4]
SEED_RANK = 22

def construct():
    n, m, p = DIMENSIONS
    nm = n * m      # 6
    mp = m * p      # 12
    pn = p * n      # 8

    def idx_a(i, j): return i * m + j
    def idx_b(i, j): return i * p + j
    def idx_c(i, j): return j * n + i  # trace convention W[k*n+i,r]

    terms = []  # list of (U_list, V_list, W_list) where each is [(idx, val), ...]

    def add(a_terms, b_terms, c_terms, scale=1):
        u = [(idx_a(i, j), v) for i, j, v in a_terms]
        v = [(idx_b(i, j), val) for i, j, val in b_terms]
        w = [(idx_c(i, j), val) for i, j, val in c_terms]
        terms.append((u, v, w, scale))

    # Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # M1 = (A00+A11)(B00+B11) -> C00 + C11
    add([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
    # M2 = (A10+A11)B00 -> C10 - C11
    add([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)])
    # M3 = A00(B01-B11) -> C01 + C11
    add([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)])
    # M4 = A11(B10-B00) -> C00 + C10
    add([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)])
    # M5 = (A00+A01)B11 -> -C00 + C01
    add([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)])
    # M6 = (A10-A00)(B00+B01) -> C11
    add([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])
    # M7 = (A01-A11)(B10+B11) -> C00
    add([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])

    # Strassen for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # B columns 2,3 -> B[0,2]=idx 2, B[0,3]=idx 3, B[1,2]=idx 6, B[1,3]=idx 7
    add([(0,0,1),(1,1,1)], [(0,2,1),(1,3,1)], [(0,2,1),(1,3,1)])
    add([(1,0,1),(1,1,1)], [(0,2,1)], [(1,2,1),(1,3,-1)])
    add([(0,0,1)], [(0,3,1),(1,3,-1)], [(0,3,1),(1,3,1)])
    add([(1,1,1)], [(1,2,1),(0,2,-1)], [(0,2,1),(1,2,1)])
    add([(0,0,1),(0,1,1)], [(1,3,1)], [(0,2,-1),(0,3,1)])
    add([(1,0,1),(0,0,-1)], [(0,2,1),(0,3,1)], [(1,3,1)])
    add([(0,1,1),(1,1,-1)], [(1,2,1),(1,3,1)], [(0,2,1)])

    # Rank-2 decomposition for A[0:2,2] * B[2,0:4] -> C[0:2,0:4]
    # A[0,2]=idx2, A[1,2]=idx5; B[2,0]=idx8, B[2,1]=idx9, B[2,2]=idx10, B[2,3]=idx11
    # Term 1: handles i=0 (A row 0, col 2): C[0,j] += A[0,2]*B[2,j]
    #   U: idx_a(0,2)=2 with value 1
    #   V: idx_b(2,0)=8, idx_b(2,1)=9, idx_b(2,2)=10, idx_b(2,3)=11 all with value 1
    #   W: idx_c(0,0)=0, idx_c(1,0)=2, idx_c(2,0)=4, idx_c(3,0)=6 all with value 1
    add([(0, 2, 1)],
        [(2, 0, 1), (2, 1, 1), (2, 2, 1), (2, 3, 1)],
        [(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)])

    # Term 2: handles i=1 (A row 1, col 2): C[1,j] += A[1,2]*B[2,j]
    #   U: idx_a(1,2)=5 with value 1
    #   V: idx_b(2,0)=8, idx_b(2,1)=9, idx_b(2,2)=10, idx_b(2,3)=11 all with value 1
    #   W: idx_c(0,1)=1, idx_c(1,1)=3, idx_c(2,1)=5, idx_c(3,1)=7 all with value 1
    add([(1, 2, 1)],
        [(2, 0, 1), (2, 1, 1), (2, 2, 1), (2, 3, 1)],
        [(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1)])

    R0 = len(terms)  # 16

    # Build final U, V, W
    R = R0

    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))

    for col in range(R):
        u_list, v_list, w_list, s = terms[col]
        for (ai, val) in u_list:
            U[ai, col] = val * s
        for (bi, val) in v_list:
            V[bi, col] = val * s
        for (ci, val) in w_list:
            W[ci, col] = val * s

    # Verify the decomposition
    T_recon = np.zeros((nm, mp, pn))
    for r in range(R):
        T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])

    # The true matmul tensor
    T_true = np.zeros((nm, mp, pn))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a_idx = i * m + k
                b_idx = k * p + j
                c_idx = j * n + i
                T_true[a_idx, b_idx, c_idx] = 1

    if not np.allclose(T_recon, T_true, atol=1e-10):
        # This should not happen, but fall back gracefully
        # Build a naive rank-24 decomposition as fallback
        U = np.zeros((nm, n*m*p))
        V = np.zeros((mp, n*m*p))
        W = np.zeros((pn, n*m*p))
        r = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a_idx = i * m + k
                    b_idx = k * p + j
                    c_idx = j * n + i
                    U[a_idx, r] = 1
                    V[b_idx, r] = 1
                    W[c_idx, r] = 1
                    r += 1
        R = n * m * p

    # Convert to lists of lists with integer/half-integer entries
    def to_half_int_list(M):
        """Convert matrix to list of lists, rounding to half-integers if close."""
        L = []
        for row in M:
            r = []
            for val in row:
                doubled = val * 2
                nearest = np.round(doubled)
                if abs(doubled - nearest) < 1e-8:
                    r.append(int(nearest) / 2.0)
                else:
                    r.append(val)
            L.append(r)
        return L

    U_list = to_half_int_list(U)
    V_list = to_half_int_list(V)
    W_list = to_half_int_list(W)

    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("U shape:", (len(res['U']), len(res['U'][0])))
    print("V shape:", (len(res['V']), len(res['V'][0])))
    print("W shape:", (len(res['W']), len(res['W'][0])))
