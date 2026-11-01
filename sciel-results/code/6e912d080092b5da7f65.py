import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    rank = 11
    
    # Initialize factor matrices
    # U: (n*m, rank) = (4, 11)
    # V: (m*p, rank) = (6, 11)
    # W: (p*n, rank) = (6, 11)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper functions to populate matrices based on indices
    # A is n x m, B is m x p, C is p x n (Wait, task says W[k*n+i] which implies C is p x n? 
    # Let's check standard matrix mult: A(n x m) * B(m x p) = C(n x p).
    # Task says: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually W corresponds to the output tensor C.
    # If C is n x p, index is [i, k] or [k, i]?
    # "W[k*n+i]" suggests the first index is k (0..p-1) and second is i (0..n-1).
    # This implies the output matrix is stored as p x n? Or is it just an indexing convention?
    # Let's look at the seed program:
    # set_w(0, 0, 0, 1) -> W[0*n+0] = 1.
    # In Strassen for 2x2, C_00 is top-left.
    # If C is 2x2, indices are (0,0), (0,1), (1,0), (1,1).
    # If W[k*n+i] is used, and k is col, i is row?
    # Let's trace seed:
    # M1 contributes to C_00 and C_11.
    # Seed: set_w(0,0,0,1) and set_w(1,1,0,1).
    # If W is 2x2 (p=2, n=2), W[0*2+0]=W[0], W[1*2+1]=W[3].
    # This corresponds to C[0,0] and C[1,1] if row-major [row*n+col]?
    # No, [k*n+i]. If k=0, i=0 -> 0. If k=1, i=1 -> 3.
    # In a 2x2 matrix C, C[0,0] is index 0, C[1,1] is index 3 in row-major [row*2+col].
    # So k=row, i=col?
    # But the dimension of W is (p*n, rank).
    # If C is n x p, size is n*p.
    # If C is p x n, size is p*n.
    # Standard mult: A(n x m) B(m x p) -> C(n x p).
    # However, the task specifies W shape (p*n, rank). This strongly implies the output tensor is viewed as p x n or the indices are swapped.
    # Let's look at the "trace convention" hint.
    # In tensor decomposition $C_{ik} = \sum_r U_{ir} V_{jk} W_{kr}$? No.
    # Standard: $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # Tensor form: $T_{i,j,k} = \delta_{ij} \delta_{jk}$? No.
    # $T_{i,j,k} = 1$ if $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    # Decomposition: $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}$?
    # Here U is (n*m, rank), so index (i*m+j).
    # V is (m*p, rank), so index (j*p+k).
    # W is (p*n, rank), so index (k*n+i).
    # This matches the contraction:
    # $\sum_r U_{i,m,j,r} V_{j,p,k,r} W_{k,n,i,r} = C_{i,k}$?
    # Let's check the indices in the summation.
    # U depends on i, j. V depends on j, k. W depends on k, i.
    # Summing over r gives a value for (i, k).
    # So the output is C[i][k].
    # The index in W is k*n + i.
    # So W is stored in a vector of length p*n.
    # This is consistent with C being n x p, but the storage index in W is swapped (k first, then i).
    
    def add_u(i, j, r, val):
        idx = i * m + j
        U[idx][r] += val
        
    def add_v(j, k, r, val):
        idx = j * p + k
        V[idx][r] += val
        
    def add_w(k, i, r, val):
        idx = k * n + i
        W[idx][r] += val

    # --- Strassen Decomposition for C[:, 0:2] ---
    # We are computing C = A * B.
    # A is 2x2, B is 2x3.
    # Let B = [B_left, B_right] where B_left is 2x2 (cols 0,1) and B_right is 2x1 (col 2).
    # C = [C_left, C_right].
    # C_left (2x2) is computed via Strassen.
    # C_right (2x1) is computed via standard mult.

    # Strassen Factors (Rank 7)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Recombination for C_left:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # r=0: M1
    add_u(0, 0, 0, 1); add_u(1, 1, 0, 1)
    add_v(0, 0, 0, 1); add_v(1, 1, 0, 1)
    # W contributions:
    # C00 gets +1 (i=0, k=0) -> w(0,0)
    # C11 gets +1 (i=1, k=1) -> w(1,1)
    add_w(0, 0, 0, 1)
    add_w(1, 1, 0, 1)

    # r=1: M2
    add_u(1, 0, 1, 1); add_u(1, 1, 1, 1)
    add_v(0, 0, 1, 1)
    # C10 gets +1 (i=1, k=0) -> w(0,1)
    # C11 gets -1 (i=1, k=1) -> w(1,1)
    add_w(0, 1, 1, 1)
    add_w(1, 1, 1, -1)

    # r=2: M3
    add_u(0, 0, 2, 1)
    add_v(0, 1, 2, 1); add_v(1, 1, 2, -1)
    # C01 gets +1 (i=0, k=1) -> w(1,0)
    # C11 gets +1 (i=1, k=1) -> w(1,1)
    add_w(1, 0, 2, 1)
    add_w(1, 1, 2, 1)

    # r=3: M4
    add_u(1, 1, 3, 1)
    add_v(1, 0, 3, 1); add_v(0, 0, 3, -1)
    # C00 gets +1 (i=0, k=0) -> w(0,0)
    # C10 gets +1 (i=1, k=0) -> w(0,1)
    add_w(0, 0, 3, 1)
    add_w(0, 1, 3, 1)

    # r=4: M5
    add_u(0, 0, 4, 1); add_u(0, 1, 4, 1)
    add_v(1, 1, 4, 1)
    # C00 gets -1 (i=0, k=0) -> w(0,0)
    # C01 gets +1 (i=0, k=1) -> w(1,0)
    add_w(0, 0, 4, -1)
    add_w(1, 0, 4, 1)

    # r=5: M6
    add_u(1, 0, 5, 1); add_u(0, 0, 5, -1)
    add_v(0, 0, 5, 1); add_v(0, 1, 5, 1)
    # C11 gets +1 (i=1, k=1) -> w(1,1)
    add_w(1, 1, 5, 1)

    # r=6: M7
    add_u(0, 1, 6, 1); add_u(1, 1, 6, -1)
    add_v(1, 0, 6, 1); add_v(1, 1, 6, 1)
    # C00 gets +1 (i=0, k=0) -> w(0,0)
    add_w(0, 0, 6, 1)

    # --- Standard Multiplication for C[:, 2] ---
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # r=7: A00 * B02
    add_u(0, 0, 7, 1)
    add_v(0, 2, 7, 1)
    # C02 gets +1 (i=0, k=2) -> w(2,0)
    add_w(2, 0, 7, 1)

    # r=8: A01 * B12
    add_u(0, 1, 8, 1)
    add_v(1, 2, 8, 1)
    # C02 gets +1 (i=0, k=2) -> w(2,0)
    add_w(2, 0, 8, 1)

    # r=9: A10 * B02
    add_u(1, 0, 9, 1)
    add_v(0, 2, 9, 1)
    # C12 gets +1 (i=1, k=2) -> w(2,1)
    add_w(2, 1, 9, 1)

    # r=10: A11 * B12
    add_u(1, 1, 10, 1)
    add_v(1, 2, 10, 1)
    # C12 gets +1 (i=1, k=2) -> w(2,1)
    add_w(2, 1, 10, 1)

    if time.time() > deadline:
        # Return current best (which is complete)
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
