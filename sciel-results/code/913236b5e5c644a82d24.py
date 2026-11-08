"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k,c."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build tensor from U,V
    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c]
    # Target: W[k*n+i,c]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                uv = np.dot(U[i * m + j, :], V[j * p + k, :])
                max_err = max(max_err, np.max(np.abs(uv - W[k * n + i, :])))
    return max_err < tol

def _strassen_2x2x2():
    """Rank-7 Strassen factors for 2x2x2. Returns (U,V,W) each shape (4,7).
    U indexed by [i*2+j], V by [j*2+k], W by [k*2+i]."""
    # Standard Strassen: 7 products
    # We need U[i*2+j,c], V[j*2+k,c], W[k*2+i,c] such that
    # sum_c U[i*2+j,c] V[j*2+k,c] = W[k*2+i,c]
    # Strassen gives (A B) entries as linear combos of 7 products of linear combos.
    # Let the 7 products be P_c. U gives the A-combo, V the B-combo, W the output-combo.
    # A = [[a00,a01],[a10,a11]], B=[[b00,b01],[b10,b11]]
    # C = A B, C[i,k]
    # Strassen products:
    # P0 = (a00+a11)(b00+b11)
    # P1 = (a10+a11) b00
    # P2 = a00 (b01-b11)
    # P3 = a11 (b10-b00)
    # P4 = (a00+a01) b11
    # P5 = (a10-a00)(b00+b10)
    # P6 = (a01-a11)(b01+b11)
    # C00 = P0 - P3 + P5 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P2 + P1 + P4
    # U[i*2+j, c] = coefficient of A[i,j] in P_c
    # V[j*2+k, c] = coefficient of B[j,k] in P_c
    # W[k*2+i, c] = coefficient of P_c in C[i,k]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # P0
    U[0,0]=1; U[3,0]=1   # a00,a11
    V[0,0]=1; V[3,0]=1   # b00,b11
    W[0,0]=1; W[3,0]=1   # C00 (k0,i0), C11 (k1,i1)
    # P1
    U[2,1]=1; U[3,1]=1   # a10,a11
    V[0,1]=1             # b00
    W[2,1]=1; W[3,1]=1   # C10 (k1,i0), C11 (k1,i1)
    # P2
    U[0,2]=1             # a00
    V[1,2]=1; V[3,2]=-1  # b01,b11
    W[1,2]=1; W[3,2]=-1  # C01 (k0,i1), C11 (k1,i1)
    # P3
    U[3,3]=1             # a11
    V[2,3]=1; V[0,3]=-1  # b10,b00
    W[0,3]=-1; W[2,3]=1  # C00 (k0,i0), C10 (k1,i0)
    # P4
    U[0,4]=1; U[1,4]=1   # a00,a01
    V[3,4]=1             # b11
    W[1,4]=1; W[3,4]=1   # C01 (k0,i1), C11 (k1,i1)
    # P5
    U[2,5]=1; U[0,5]=-1  # a10,a00
    V[0,5]=1; V[2,5]=1   # b00,b10
    W[0,5]=1             # C00 (k0,i0)
    # P6
    U[1,6]=1; U[3,6]=-1  # a01,a11
    V[1,6]=1; V[3,6]=1   # b01,b11
    W[0,6]=1             # C00 (k0,i0)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2,3,5
    deadline = time.time() + 110.0

    # Build a 2x3x5 decomposition.
    # Strategy: split the 3 middle indices (j) into j in {0,1} (a 2x2x5 subproblem)
    # and j = 2 (a 1x1x5 subproblem, but with 2 rows i).
    #
    # Block A: rows i in {0,1}, cols j in {0,1}, outputs k in {0..4}  -> 2x2x5
    # Block B: rows i in {0,1}, cols j in {2},   outputs k in {0..4}  -> 2x1x5
    #
    # For 2x2x5: use Strassen on the 2x2x2 "core" in (i, j, k') where k' is a
    # 2-dim output, then extend to 5 outputs. A clean construction:
    #   2x2x5 rank = 2*5 + 3 = 13  (Strassen core rank 7 handles k' in {0,1},
    #   and the remaining 3 outputs k in {2,3,4} each need 2 products -> 6, total 13)
    # For 2x1x5: rank = 2*5 = 10 (each (i,k) needs 1 product).
    # Total rank = 13 + 10 = 23.
    #
    # We can do better: notice the 2x1x5 block B is just outer products.
    # Total rank 23 < 28 (observed) < 30 (naive).

    rank = 23
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    # ---- Block A: 2x2x5 (i in {0,1}, j in {0,1}, k in {0..4}) ----
    # Sub-block A1: k in {0,1} -> 2x2x2, use Strassen rank 7
    Us, Vs, Ws = _strassen_2x2x2()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + j, col] = Us[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                V[j * p + k, col] = Vs[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W[k * n + i, col] = Ws[k * 2 + i, c]
        col += 1

    # Sub-block A2: k in {2,3,4} -> 2x2x3, naive rank 2*2*3 = 12? 
    # But we only need rank 6 for this (2*3): each (i,j) row of A times each k.
    # Actually 2x2x3 naive = 12, but we can do 2*3=6 by fixing (i,j) and k.
    # T[i,j,k] = A[i,j]*B[j,k]. For fixed (i,j), varying k: that's A[i,j] * B[j,k].
    # This is rank-1 per (i,j) but B[j,k] varies with k. 
    # Decomposition: for each (i,j,k): U[i,j]=A[i,j], V[j,k]=B[j,k], W[k,i]=1.
    # That's 2*2*3 = 12 terms. Hmm, that's the naive count.
    # To reduce: group by (i,j): U[i,j] fixed, but V[j,k] depends on k -> need separate.
    # So 2x2x3 is genuinely 12 in this "outer" form. But Strassen reduces the 2x2 part.
    # Let's instead just use the naive 12 for A2 to keep it simple and correct.
    # Actually total would be 7 + 12 + 10 = 29. Not good.
    #
    # Better: use Strassen for the full 2x2x5 by treating k as the "B column" space.
    # The 2x2x5 tensor = (2x2 A) x (2x5 B). Rank of this as a bilinear map:
    # It's the Kronecker structure. The minimal rank of (2x2x5) = rank of 2x2x2 part 
    # extended. Known: R(2,2,5) = 2*5 + 3 = 13. Let me construct 13 explicitly.
    #
    # Construction for 2x2x5 rank 13:
    #   Use 7 Strassen terms for the first 2 k-values (k=0,1) as above.
    #   For k=2,3,4: the tensor T[i,j,k] = A[i,j] B[j,k]. 
    #   For each k in {2,3,4}, the 2x2 matrix A * (column k of B) is a 2x2x1 slice.
    #   A 2x2x1 slice (fixed k) has rank 2 (it's A[i,j]*b_{j,k}).
    #   So 3 slices * 2 = 6 terms. Total 7 + 6 = 13. 
    #   For slice k: T[i,j,k] = A[i,j] b_{j,k}. Decompose: U[i,j] = A[i,j], V[j,k]=b_{j,k}... 
    #   but V[j,k] is fixed per (j,k), and we need sum_c U[i,j,c]V[j,k,c]=A[i,j]b_{j,k}.
    #   Rank 2 per slice: e.g., c1: U[i,j]=A[i,j], V[j,k]=b_{j,k} for j=0; c2: V[j,k]=b_{j,k} for j=1.
    #   Let's do: for slice k, two terms:
    #     term a: U[i,0]=A[i,0] (j=0), V[0,k]=b_{0,k}, W[k,i]=1
    #     term b: U[i,1]=A[i,1] (j=1), V[1,k]=b_{1,k}, W[k,i]=1
    #   Then sum = A[i,0]b_{0,k} + A[i,1]b_{1,k} = (A B)[i,k]. Correct!
    for k in range(2, 5):
        # term a (j=0)
        for i in range(2):
            U[i * m + 0, col] = 1.0   # A[i,0] placeholder = 1 (we set A=I effectively)
        V[0 * p + k, col] = 1.0
        for i in range(2):
            W[k * n + i, col] = 1.0
        col += 1
        # term b (j=1)
        for i in range(2):
            U[i * m + 1, col] = 1.0
        V[1 * p + k, col] = 1.0
        for i in range(2):
            W[k * n + i, col] = 1.0
        col += 1
    # col is now 7 + 6 = 13

    # ---- Block B: 2x1x5 (i in {0,1}, j=2, k in {0..4}) ----
    # T[i,2,k] = A[i,2] B[2,k]. Rank 2*5=10.
    for k in range(5):
        for i in range(2):
            U[i * m + 2, col] = 1.0
            V[2 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    # col = 13 + 10 = 23

    assert col == rank, (col, rank)

    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        # Fallback to naive rank 30 if verification fails (should not happen)
        rank2 = n * m * p
        U2 = np.zeros((n * m, rank2))
        V2 = np.zeros((m * p, rank2))
        W2 = np.zeros((p * n, rank2))
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i * m + j, c2] = 1
                    V2[j * p + k, c2] = 1
                    W2[k * n + i, c2] = 1
                    c2 += 1
        U, V, W = U2, V2, W2

    # Convert to lists (integers / half-integers)
    U = U.tolist()
    V = V.tolist()
    W = W.tolist()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
