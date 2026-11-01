"""Matrix multiplication 2x3 by 3x3, rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    """Check U,V,W of given shapes reconstruct the matmul tensor exactly."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # Build tensor T[k*n+i, j*m+? ... ] ; use convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # T[i*m+j? ] ... The matmul tensor: C[i,k] = sum_j A[i,j] B[j,k]
    # So tensor T[i, j, k] = 1 (delta). Index U row = i*m+j, V row = j*p+k, W row = k*n+i.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Reconstruct via rank-r: T[a,b,c] = sum_c U[a,c] V[b,c] W[c_,c]? No: T[a,b,c] = sum_rank U[a,r]*V[b,r]*W[c,r]
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    if recon.shape != T.shape:
        return False
    return np.allclose(recon, T, atol=1e-9)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_rank15():
        # Stothers-style rank-15 for M_{2,3,3}.
        # A is 2x3 (rows i=0,1; cols j=0,1,2), B is 3x3 (rows j; cols k).
        # C[i,k] = sum_j A[i,j] B[j,k].
        #
        # We introduce the 3 column-sums of B across k via grouping, and exploit the
        # 2-row output with a rank-2 identity on (i) using the relation that for two
        # outputs we can use 3 multiplications for a 2x? via Winograd.
        #
        # Concrete scheme (rank 15):
        # Let B's three rows be b0,b1,b2 (each length 3).
        # Define B-col-combos:
        #   We will form, for each of 5 "column patterns" of B, the vector
        #   s_j = sum over pattern of b_j.
        #
        # Use the identity for a 2x3 times 3x3:
        #   C = [ A0 ; A1 ] B.
        #   C0 = A0 B, C1 = A1 B.
        #   Write B = [b0;b1;b2].
        #   C0_k = A0_0 b0_k + A0_1 b1_k + A0_2 b2_k
        #   C1_k = A1_0 b0_k + A1_1 b1_k + A1_2 b2_k
        #
        # Combine: for each k we need two linear forms in (b0_k,b1_k,b2_k) with
        # coefficients (A0_0,A0_1,A0_2) and (A1_0,A1_1,A1_2).
        #
        # Use the rank-3 decomposition of the 2x3 "dual" multiplication:
        #   Given vectors x = (x0,x1,x2), y=(y0,y1,y2) (the two rows restricted to
        #   a fixed column k), produce two outputs u = x·e, v = y·e where e=(b0_k,b1_k,b2_k)?
        #   No: u = sum_j A0_j b_jk, v = sum_j A1_j b_jk.
        #   This is: [u; v] = diag? It's M_{2,3,3} transposed per column.
        #
        # Per column k, the sub-problem is: given e=(b0k,b1k,b2k) (length 3) and the
        # 2x3 matrix A, compute 2 outputs. That is exactly M_{2,3,1} with "matrix" A and
        # vector e. R(M_{2,3,1}) = 4? Let's use rank-4 per column? 3 columns * 4 = 12, plus?
        #
        # M_{2,3,1}: C_i = sum_j A[i,j] e_j, i=0,1. 2 outputs, each dot of length 3.
        # Schoolbook 6. Can we do 4? Use the identity:
        #   Let p1 = (A0_0+A0_1) e0 + A0_2 e2
        #   Let p2 = A0_0 e0 + (A0_1+A1_1) e1
        #   Let p3 = A1_0 e0 + A1_1 e1 + (A1_2+A0_2) e2
        #   Let p4 = A1_0 e0 + A1_1 e1 + A1_2 e2  (this is v itself)
        #   Hmm need to recover both u and v.
        #
        # Known: R(M_{2,2,1}) = 3 (via 3 multiplications for 2x2 times vector? no,
        # 2x2 . vector = 2 outputs, R=3? Actually M_{2,2,1} has 4 terms, R=3 is false;
        # R(M_{2,2,1}) = 4? The product of 2x2 matrix and 2-vector: each output is a dot of 2,
        # total 4 mults, no sharing -> R=4.)
        #
        # So per-column sharing is not available. Back to rank-15 via cross-column grouping.
        #
        # Cross-column grouping: group the 3 output columns k into a structure.
        # Use the fact that B is 3x3: define B' = B + B^T? Not generally.
        #
        # Given the difficulty and risk, return None to trigger fallback.
        return None

    # Try the rank-15 candidate if it exists and verifies.
    cand = try_rank15()
    if cand is not None and _verify(cand["U"], cand["V"], cand["W"], n, m, p):
        return cand

    # Guaranteed-correct schoolbook fallback.
    return schoolbook()
# EVOLVE-BLOCK-END
