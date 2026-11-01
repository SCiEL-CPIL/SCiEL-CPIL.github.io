"""Matrix multiplication 2x5x5 — rank-minimal exact tensor decomposition.

Uses Strassen's algorithm on 2x2x2 sub-blocks to reduce rank from 50 to 46.
Falls back to schoolbook rank-50 if Strassen verification fails.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return the 7 rank-1 terms for 2x2x2 matrix multiplication tensor.
    
    Each term is (u, v, w) where:
    - u has shape 4, indexed by i*2+j for i,j in {0,1}
    - v has shape 4, indexed by j*2+k for j,k in {0,1}  
    - w has shape 4, indexed by k*2+i for k,i in {0,1}
    
    The sum over terms of u[i*2+j]*v[j*2+k]*w[k*2+i] should equal 1 for all i,j,k.
    Derived from Strassen's algorithm:
      p1=(a+d)(e+h), p2=(c+d)e, p3=a(f-h), p4=d(g-e),
      p5=(a+b)h, p6=(c-a)(e+f), p7=(b-d)(g+h)
      C11=p1+p4-p5+p7, C12=p3+p5, C21=p2+p4, C22=p1-p2+p3+p6
    """
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    # Verify: sum_r u_r[i*2+j]*v_r[j*2+k]*w_r[k*2+i] == 1 for all i,j,k
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for u, v, w in terms:
                    total += u[i*2+j] * v[j*2+k] * w[k*2+i]
                if total != 1:
                    return None  # Invalid decomposition
    return terms


def _verify_exact(U, V, W, n, m, p, rng):
    """Check the tensor identity exactly on random integer matrices."""
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    rank = Ua.shape[1]
    for _ in range(8):
        A = rng.integers(-5, 6, size=(n, m))
        B = rng.integers(-5, 6, size=(m, p))
        C_true = A @ B
        C_rec = np.zeros((n, p), dtype=np.int64)
        for r in range(rank):
            ur = Ua[:, r].reshape(n, m)
            vr = Va[:, r].reshape(m, p)
            wr = Wa[:, r].reshape(p, n)
            a_part = ur * A
            b_part = vr * B
            M = a_part @ b_part
            C_rec += M * wr.T
        if not np.array_equal(C_true, C_rec):
            return False
    return True


def _schoolbook(n, m, p):
    """Guaranteed-exact schoolbook decomposition of rank n*m*p."""
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


def _strassen_255():
    """Construct rank-46 decomposition using Strassen on 2x2x2 sub-blocks.
    
    Partition j-index (0..4) into pairs (0,1), (2,3), and single (4).
    Partition k-index (0..4) into pairs (0,1), (2,3), and single (4).
    
    For each (j-pair, k-pair) combination, use Strassen rank-7.
    For mixed/single cases, use schoolbook.
    """
    n, m, p = 2, 5, 5
    strassen_terms = _strassen_222()
    if strassen_terms is None:
        return None  # Fall back to schoolbook
    
    # j-groups: [(0,1), (2,3), (4)]
    # k-groups: [(0,1), (2,3), (4)]
    j_groups = [(0, 1), (2, 3), (4,)]
    k_groups = [(0, 1), (2, 3), (4,)]
    
    U = {}  # (i, j) -> list of contributions
    V = {}  # (j, k) -> list of contributions
    W = {}  # (k, i) -> list of contributions
    
    rank = 0
    # We'll build column by column
    U_cols = []
    V_cols = []
    W_cols = []
    
    for jg in j_groups:
        for kg in k_groups:
            if len(jg) == 2 and len(kg) == 2:
                # 2x2x2 Strassen block: rank 7
                j0, j1 = jg
                k0, k1 = kg
                for u, v, w in strassen_terms:
                    U_col = [0] * (n * m)
                    V_col = [0] * (m * p)
                    W_col = [0] * (p * n)
                    # u indexed by i*2+j_local, j_local in {0,1} -> j = j0+j_local
                    for i in range(2):
                        for jl in range(2):
                            U_col[i * m + (j0 + jl)] = u[i * 2 + jl]
                    # v indexed by j_local*2+k_local
                    for jl in range(2):
                        for kl in range(2):
                            V_col[(j0 + jl) * p + (k0 + kl)] = v[jl * 2 + kl]
                    # w indexed by k_local*2+i
                    for kl in range(2):
                        for i in range(2):
                            W_col[(k0 + kl) * n + i] = w[kl * 2 + i]
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
                    rank += 1
            elif len(jg) == 2 and len(kg) == 1:
                # 2x2x1: schoolbook rank 4
                j0, j1 = jg
                k0 = kg[0]
                for i in range(2):
                    for j in [j0, j1]:
                        U_col = [0] * (n * m)
                        V_col = [0] * (m * p)
                        W_col = [0] * (p * n)
                        U_col[i * m + j] = 1
                        V_col[j * p + k0] = 1
                        W_col[k0 * n + i] = 1
                        U_cols.append(U_col)
                        V_cols.append(V_col)
                        W_cols.append(W_col)
                        rank += 1
            elif len(jg) == 1 and len(kg) == 2:
                # 2x1x2: schoolbook rank 4
                j0 = jg[0]
                k0, k1 = kg
                for i in range(2):
                    for k in [k0, k1]:
                        U_col = [0] * (n * m)
                        V_col = [0] * (m * p)
                        W_col = [0] * (p * n)
                        U_col[i * m + j0] = 1
                        V_col[j0 * p + k] = 1
                        W_col[k * n + i] = 1
                        U_cols.append(U_col)
                        V_cols.append(V_col)
                        W_cols.append(W_col)
                        rank += 1
            else:
                # 2x1x1: schoolbook rank 2
                j0 = jg[0]
                k0 = kg[0]
                for i in range(2):
                    U_col = [0] * (n * m)
                    V_col = [0] * (m * p)
                    W_col = [0] * (p * n)
                    U_col[i * m + j0] = 1
                    V_col[j0 * p + k0] = 1
                    W_col[k0 * n + i] = 1
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
                    rank += 1
    
    # Convert to matrices
    U = np.array(U_cols).T.tolist()
    V = np.array(V_cols).T.tolist()
    W = np.array(W_cols).T.tolist()
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120.0
    
    rng = np.random.default_rng(12345)
    
    # Try Strassen-based rank-46 decomposition first
    result = _strassen_255()
    
    # Verify if Strassen worked
    if result is not None and _verify_exact(result["U"], result["V"], result["W"], n, m, p, rng):
        if time.time() < deadline:
            return result
    
    # Fall back to schoolbook rank-50
    result = _schoolbook(n, m, p)
    return result
# EVOLVE-BLOCK-END
