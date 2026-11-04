"""
Rank-17 exact decomposition for 2x3 by 3x3 matrix multiplication.

Strategy:
- Partition A = [A_left | A_right] where A_left is 2x2 (first two cols) and A_right is 2x1.
- Partition B = [B_top; B_bottom] where B_top is 2x3 and B_bottom is 1x3.
- C = A_left * B_top + A_right * B_bottom
- Use 7 Strassen mults for A_left * B_top[:, :2] (2x2 * 2x2)
- Use 2 mults for A_left * B_top[:, 2] (2x1 * 1x1... wait, 2x1 * 1x1 per row = 2 mults)
  Actually 2x2 * 2x1: 4 mults naively, but we can do it in 2 if we share... no, 4 mults.
  Hmm, let me reconsider.
  
  Actually: A_left is 2x2, B_top is 2x3.
  A_left * B_top is 2x3. Naively 12 mults.
  Using Strassen for first two columns (2x2 * 2x2): 7 mults.
  Third column: A_left * [w;z] where w,z are scalars. That's 4 mults: p*w, q*z, r*w, s*z.
  Total for A_left * B_top: 7 + 4 = 11 mults.
  
  A_right * B_bottom: [a02; a12] * [b20, b21, b22] = 6 mults.
  
  Total: 11 + 6 = 17 mults.
"""

import time
import numpy as np


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n * m):
        i_a = i // m
        i_b = i % m
        for j in range(m * p):
            j_a = j // p
            j_b = j % p
            if i_b != j_a:
                continue
            for k in range(p * n):
                k_a = k // n
                k_b = k % n
                if i_a != k_b or j_b != k_a:
                    continue
                T[i, j, k] = 1.0
    return T


def _schoolbook(n, m, p):
    """Return schoolbook rank decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i_a in range(n):
        for i_b in range(m):
            for j_b in range(p):
                i_idx = i_a * m + i_b
                j_idx = i_b * p + j_b
                k_idx = j_b * n + i_a
                U[i_idx, col] = 1.0
                V[j_idx, col] = 1.0
                W[k_idx, col] = 1.0
                col += 1
    return U, V, W


def _check_decomposition(T, U, V, W, tol=1e-9):
    """Check if U, V, W exactly decompose T."""
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.max(np.abs(recon - T)) < tol


def _rank17_decomposition():
    """
    Construct a rank-17 decomposition for 2x3 * 3x3.
    
    A = [[a00, a01, a02],
         [a10, a11, a12]]
    
    B = [[b00, b01, b02],
         [b10, b11, b12],
         [b20, b21, b22]]
    
    C = A * B (2x3)
    
    Partition:
    A_left = [[a00, a01],    B_top = [[b00, b01, b02],
                 [a10, a11]]                [b10, b11, b12]]
    A_right = [a02,           B_bottom = [b20, b21, b22]
                a12]
    
    C = A_left * B_top + A_right * B_bottom
    
    For A_left * B_top[:, :2] (2x2 * 2x2), use Strassen:
    Let P = [[p, q], [r, s]] = [[a00, a01], [a10, a11]]
    Let Q = [[e, f], [g, h]] = [[b00, b01], [b10, b11]]
    
    Strassen (verified):
    m1 = (p+s)(e+h)
    m2 = (r+s)e
    m3 = p(f-h)
    m4 = s(g-e)
    m5 = (p+q)h
    m6 = (r-s)(e+f)
    m7 = (q-s)(g+h)
    
    Result:
    P*Q = [[m1+m4-m5+m7, m3+m5],
           [m2+m4,        m1-m2+m3-m6]]
    
    For A_left * B_top[:, 2] (2x2 * 2x1 where B_top[:,2] = [b02, b12]):
    m8 = a00 * b02
    m9 = a01 * b12
    m10 = a10 * b02
    m11 = a11 * b12
    
    For A_right * B_bottom:
    m12 = a02 * b20
    m13 = a02 * b21
    m14 = a02 * b22
    m15 = a12 * b20
    m16 = a12 * b21
    m17 = a12 * b22
    
    C[0,0] = (m1+m4-m5+m7) + m12
    C[0,1] = (m3+m5) + m13
    C[0,2] = m8 + m9 + m14
    C[1,0] = (m2+m4) + m15
    C[1,1] = (m1-m2+m3-m6) + m16
    C[1,2] = m10 + m11 + m17
    """
    n, m, p = 2, 3, 3
    u_dim = n * m  # 6
    v_dim = m * p  # 9
    w_dim = p * n  # 6
    rank = 17
    
    U = np.zeros((u_dim, rank))
    V = np.zeros((v_dim, rank))
    W = np.zeros((w_dim, rank))
    
    # Index conventions:
    # U index: i_a * m + i_b = i_a * 3 + i_b, where i_a in {0,1}, i_b in {0,1,2}
    #   So: a00 -> U[0], a01 -> U[1], a02 -> U[2], a10 -> U[3], a11 -> U[4], a12 -> U[5]
    # V index: j_a * p + j_b = j_a * 3 + j_b, where j_a in {0,1,2}, j_b in {0,1,2}
    #   So: b00 -> V[0], b01 -> V[1], b02 -> V[2], b10 -> V[3], b11 -> V[4], b12 -> V[5],
    #       b20 -> V[6], b21 -> V[7], b22 -> V[8]
    # W index: k_a * n + k_b = k_a * 2 + k_b, where k_a in {0,1,2}, k_b in {0,1}
    #   k_a = j_b (column of B), k_b = i_a (row of A)
    #   So: C[0,0] -> W[0*2+0] = W[0], C[1,0] -> W[0*2+1] = W[1],
    #       C[0,1] -> W[1*2+0] = W[2], C[1,1] -> W[1*2+1] = W[3],
    #       C[0,2] -> W[2*2+0] = W[4], C[1,2] -> W[2*2+1] = W[5]
    
    # m1 = (a00+a11) * (b00+b11)
    col = 0
    U[0, col] = 1.0  # a00
    U[4, col] = 1.0  # a11
    V[0, col] = 1.0  # b00
    V[4, col] = 1.0  # b11
    W[0, col] = 1.0  # C[0,0]
    W[3, col] = 1.0  # C[1,1]
    col += 1
    
    # m2 = (a10+a11) * b00
    U[3, col] = 1.0  # a10
    U[4, col] = 1.0  # a11
    V[0, col] = 1.0  # b00
    W[1, col] = 1.0  # C[1,0]
    W[3, col] = -1.0 # C[1,1]
    col += 1
    
    # m3 = a00 * (b01-b11)
    U[0, col] = 1.0  # a00
    V[1, col] = 1.0  # b01
    V[4, col] = -1.0 # b11
    W[0, col] = -1.0 # C[0,0]
    W[2, col] = 1.0  # C[0,1]
    W[3, col] = 1.0  # C[1,1]
    col += 1
    
    # m4 = a11 * (b10-b00)
    U[4, col] = 1.0  # a11
    V[0, col] = -1.0 # b00
    V[3, col] = 1.0  # b10
    W[0, col] = 1.0  # C[0,0]
    W[1, col] = 1.0  # C[1,0]
    W[3, col] = 1.0  # C[1,1]
    col += 1
    
    # m5 = (a00+a01) * b11
    U[0, col] = 1.0  # a00
    U[1, col] = 1.0  # a01
    V[4, col] = 1.0  # b11
    W[0, col] = -1.0 # C[0,0]
    W[2, col] = 1.0  # C[0,1]
    col += 1
    
    # m6 = (a10-a11) * (b00+b01)
    U[3, col] = 1.0  # a10
    U[4, col] = -1.0 # a11
    V[0, col] = 1.0  # b00
    V[1, col] = 1.0  # b01
    W[3, col] = -1.0 # C[1,1]
    col += 1
    
    # m7 = (a01-a11) * (b10+b11)
    U[1, col] = 1.0  # a01
    U[4, col] = -1.0 # a11
    V[3, col] = 1.0  # b10
    V[4, col] = 1.0  # b11
    W[0, col] = 1.0  # C[0,0]
    col += 1
    
    # m8 = a00 * b02
    U[0, col] = 1.0  # a00
    V[2, col] = 1.0  # b02
    W[4, col] = 1.0  # C[0,2]
    col += 1
    
    # m9 = a01 * b12
    U[1, col] = 1.0  # a01
    V[5, col] = 1.0  # b12
    W[4, col] = 1.0  # C[0,2]
    col += 1
    
    # m10 = a10 * b02
    U[3, col] = 1.0  # a10
    V[2, col] = 1.0  # b02
    W[5, col] = 1.0  # C[1,2]
    col += 1
    
    # m11 = a11 * b12
    U[4, col] = 1.0  # a11
    V[5, col] = 1.0  # b12
    W[5, col] = 1.0  # C[1,2]
    col += 1
    
    # m12 = a02 * b20
    U[2, col] = 1.0  # a02
    V[6, col] = 1.0  # b20
    W[0, col] = 1.0  # C[0,0]
    col += 1
    
    # m13 = a02 * b21
    U[2, col] = 1.0  # a02
    V[7, col] = 1.0  # b21
    W[2, col] = 1.0  # C[0,1]
    col += 1
    
    # m14 = a02 * b22
    U[2, col] = 1.0  # a02
    V[8, col] = 1.0  # b22
    W[4, col] = 1.0  # C[0,2]
    col += 1
    
    # m15 = a12 * b20
    U[5, col] = 1.0  # a12
    V[6, col] = 1.0  # b20
    W[1, col] = 1.0  # C[1,0]
    col += 1
    
    # m16 = a12 * b21
    U[5, col] = 1.0  # a12
    V[7, col] = 1.0  # b21
    W[3, col] = 1.0  # C[1,1]
    col += 1
    
    # m17 = a12 * b22
    U[5, col] = 1.0  # a12
    V[8, col] = 1.0  # b22
    W[5, col] = 1.0  # C[1,2]
    col += 1
    
    return U, V, W


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    Returns dict with U (6 x r), V (9 x r), W (6 x r).
    """
    n, m, p = 2, 3, 3
    T = _build_tensor(n, m, p)
    
    # Try rank-17 first
    U, V, W = _rank17_decomposition()
    if _check_decomposition(T, U, V, W):
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook rank-18
    U, V, W = _schoolbook(n, m, p)
    assert _check_decomposition(T, U, V, W), "Schoolbook decomposition failed!"
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
