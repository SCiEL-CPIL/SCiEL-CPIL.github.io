"""Rewritten program using tensor product of Strassen decompositions for 4x4x4 matrix multiplication."""

# Strassen matrices for 2x2x2 (rank 7)
# U_2: 4x7 (rows are a,b,c,d; cols are M1..M7)
# V_2: 4x7 (rows are e,f,g,h; cols are M1..M7)
# W_2: 4x7 (rows indexed by k*2+i; cols are M1..M7)

_U2 = [
    [1, 0, 1, 0, 1, 0, 1],   # a
    [0, 0, -1, 0, 0, 1, 0],  # b
    [0, 1, 0, -1, 0, 0, 1],  # c
    [1, 1, 0, 1, -1, 0, -1]  # d
]

_V2 = [
    [1, 1, 0, 0, -1, 0, 0],  # e
    [0, 0, 1, 0, 0, 1, 0],   # f
    [0, 0, 0, 1, 0, 0, 1],   # g
    [1, 0, 1, 1, 1, -1, 0]   # h
]

# W_2: rows indexed by k*2+i
# Index 0: k=0, i=0 -> z_00 = M1 + M4 - M5 - M7
# Index 1: k=0, i=1 -> z_10 = M2 + M4
# Index 2: k=1, i=0 -> z_01 = M3 + M5
# Index 3: k=1, i=1 -> z_11 = M1 - M2 + M3 + M6
_W2 = [
    [1, 0, 0, 1, -1, 0, -1],  # z_00
    [0, 1, 0, 1, 0, 0, 0],    # z_10
    [0, 0, 1, 0, 1, 0, 0],    # z_01
    [1, -1, 1, 0, 0, 1, 0]    # z_11
]


def _tensor_product(A, B):
    """Compute the Kronecker/tensor product of two matrices.
    
    If A is (n1, k1) and B is (n2, k2), the result is (n1*n2, k1*k2).
    Row index: r1 * n2 + r2
    Col index: c1 * k2 + c2
    Entry: A[r1, c1] * B[r2, c2]
    """
    n1, k1 = len(A), len(A[0])
    n2, k2 = len(B), len(B[0])
    res = [[0] * (k1 * k2) for _ in range(n1 * n2)]
    for r1 in range(n1):
        for r2 in range(n2):
            row_idx = r1 * n2 + r2
            for c1 in range(k1):
                a_val = A[r1][c1]
                if a_val == 0:
                    continue
                for c2 in range(k2):
                    b_val = B[r2][c2]
                    res[row_idx][c1 * k2 + c2] += a_val * b_val
    return res


def construct():
    """Construct rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    
    Uses the tensor product of two Strassen (rank-7) decompositions.
    Returns {'U': U, 'V': V, 'W': W} with shapes (16, 49), (16, 49), (16, 49).
    """
    U = _tensor_product(_U2, _U2)
    V = _tensor_product(_V2, _V2)
    W = _tensor_product(_W2, _W2)
    return {"U": U, "V": V, "W": W}
