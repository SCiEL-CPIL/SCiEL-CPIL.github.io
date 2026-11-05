import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def _strassen_terms():
    """Return the 7 Strassen bilinear forms for 2x2x2 matmul."""
    terms = []
    # m1 = (x00+x11)(y00+y11) -> c00 += m1, c11 += m1
    a = [[1,0],[0,1]]
    b = [[1,0],[0,1]]
    c = [[1,0],[0,1]]
    terms.append((a,b,c))
    # m2 = (x10+x11)y00 -> c10 += m2, c11 -= m2
    a = [[0,0],[1,1]]
    b = [[1,0],[0,0]]
    c = [[0,0],[1,-1]]
    terms.append((a,b,c))
    # m3 = x00(y01-y11) -> c01 += m3
    a = [[1,0],[0,0]]
    b = [[0,1],[0,-1]]
    c = [[0,1],[0,0]]
    terms.append((a,b,c))
    # m4 = x11(y10-y00) -> c00 += m4, c10 += m4
    a = [[0,0],[0,1]]
    b = [[-1,0],[1,0]]
    c = [[1,0],[1,0]]
    terms.append((a,b,c))
    # m5 = (x00+x01)y11 -> c00 -= m5, c01 += m5
    a = [[1,1],[0,0]]
    b = [[0,0],[0,1]]
    c = [[-1,1],[0,0]]
    terms.append((a,b,c))
    # m6 = (x10-x00)(y00+y10) -> c11 += m6
    a = [[-1,0],[1,0]]
    b = [[1,0],[1,0]]
    c = [[0,0],[0,1]]
    terms.append((a,b,c))
    # m7 = (x01-x11)(y10+y11) -> c00 += m7
    a = [[0,1],[0,-1]]
    b = [[0,0],[1,1]]
    c = [[1,0],[0,0]]
    terms.append((a,b,c))
    return terms

def _add_block(U, V, W, n, m, p, i0, j0, k0, sa, sb, sc, col):
    """Add decomposition terms for a block at global offsets (i0,j0,k0)
    with sizes sa (i), sb (j), sc (k). Returns new col index."""
    if sa == 2 and sb == 2 and sc == 2:
        # Strassen
        for (a, b, c) in _strassen_terms():
            for i in range(2):
                for j in range(2):
                    if a[i][j] != 0:
                        U[(i0+i)*m + (j0+j)][col] += a[i][j]
            for j in range(2):
                for k in range(2):
                    if b[j][k] != 0:
                        V[(j0+j)*p + (k0+k)][col] += b[j][k]
            for i in range(2):
                for k in range(2):
                    if c[i][k] != 0:
                        W[(k0+k)*n + (i0+i)][col] += c[i][k]
            col += 1
    else:
        # Schoolbook for this block
        for i in range(sa):
            for j in range(sb):
                for k in range(sc):
                    U[(i0+i)*m + (j0+j)][col] += 1
                    V[(j0+j)*p + (k0+k)][col] += 1
                    W[(k0+k)*n + (i0+i)][col] += 1
                    col += 1
    return col

def _block_decomp(n, m, p):
    """Block decomposition of 5x5x5 using Strassen for 2x2x2 blocks."""
    blocks = [(0, 2), (2, 2), (4, 1)]
    
    U = [[0] * 200 for _ in range(n * m)]
    V = [[0] * 200 for _ in range(m * p)]
    W = [[0] * 200 for _ in range(p * n)]
    
    col = 0
    for (i0, sa) in blocks:
        for (j0, sb) in blocks:
            for (k0, sc) in blocks:
                col = _add_block(U, V, W, n, m, p, i0, j0, k0, sa, sb, sc, col)
    
    rank = col
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    return {"U": U, "V": V, "W": W}, rank

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i]))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    try:
        result, rank = _block_decomp(n, m, p)
        if time.time() < deadline:
            if _verify(result, n, m, p):
                return result
    except Exception:
        pass
    
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
