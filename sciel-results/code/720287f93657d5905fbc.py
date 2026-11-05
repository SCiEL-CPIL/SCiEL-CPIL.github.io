import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=6
    # U: 15 x R
    # V: 30 x R
    # W: 18 x R
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1
        
    # Helper: Add 2x2x2 Strassen
    # u_rows: [r0, r1] indices in U
    # v_rows: [r0, r1] indices in V
    # w_cols: [c0, c1] indices in W
    # The 2x2 block of A is at U[u_rows[0]], U[u_rows[1]] with columns j0, j1
    # The 2x2 block of B is at V[v_rows[0]], V[v_rows[1]] with columns k0, k1
    # The 2x2 block of C is at W[:, w_cols[0]], W[:, w_cols[1]]
    # Note: U index is i*m + j. V index is j*p + k. W index is k*n + i.
    
    def add_strassen_222(i_base, j_base, k_base):
        # i_base, j_base, k_base are the starting indices for the 2x2 blocks
        # A rows: i_base, i_base+1
        # A cols: j_base, j_base+1
        # B rows: j_base, j_base+1
        # B cols: k_base, k_base+1
        # C rows: k_base, k_base+1
        # C cols: i_base, i_base+1
        
        # U indices for A
        u00 = i_base * m + j_base
        u01 = i_base * m + j_base + 1
        u10 = (i_base + 1) * m + j_base
        u11 = (i_base + 1) * m + j_base + 1
        
        # V indices for B
        v00 = j_base * p + k_base
        v01 = j_base * p + k_base + 1
        v10 = (j_base + 1) * p + k_base
        v11 = (j_base + 1) * p + k_base + 1
        
        # W indices for C
        w00 = k_base * n + i_base
        w01 = k_base * n + i_base + 1
        w10 = (k_base + 1) * n + i_base
        w11 = (k_base + 1) * n + i_base + 1
        
        # M1 = (A00+A11)(B00+B11)
        add_term(u00, 1, v00, 1, w00, 1)
        add_term(u00, 1, v00, 1, w01, 0) # Wait, W coefficients for C11, C12, C21, C22
        # This is getting complicated to manage with add_term directly because
        # one term contributes to multiple W entries?
        # No, in tensor decomposition, each term r has a specific U_r, V_r, W_r.
        # U_r is a vector of size n*m. V_r is m*p. W_r is p*n.
        # My add_term adds to specific positions.
        # But Strassen terms like M1 contribute to C11, C22?
        # C11 = M1 + M4 - M5 + M7
        # C22 = M1 - M2 + M3 + M6
        # So M1 has coefficient 1 in C11 and 1 in C22.
        # So for term r (M1):
        # U_r has 1 at u00 and u11.
        # V_r has 1 at v00 and v11.
        # W_r has 1 at w00 and w11.
        
        # Let's rewrite add_strassen to handle this correctly.
        pass

    # Let's restart the helper with clear logic.
    
    # Reset matrices
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # A: 2x2 at (i_base, j_base)
        # B: 2x2 at (j_base, k_base)
        # C: 2x2 at (k_base, i_base)
        
        u00 = i_base * m + j_base
        u01 = i_base * m + j_base + 1
        u10 = (i_base + 1) * m + j_base
        u11 = (i_base + 1) * m + j_base + 1
        
        v00 = j_base * p + k_base
        v01 = j_base * p + k_base + 1
        v10 = (j_base + 1) * p + k_base
        v11 = (j_base + 1) * p + k_base + 1
        
        w00 = k_base * n + i_base
        w01 = k_base * n + i_base + 1
        w10 = (k_base + 1) * n + i_base
        w11 = (k_base + 1) * n + i_base + 1
        
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        U[u00][col] = 1; U[u11][col] = 1
        V[v00][col] = 1; V[v11][col] = 1
        W[w00][col] = 1; W[w11][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00 -> C10
        U[u10][col] = 1; U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M3 = A00(B10-B11) -> C01, C11
        U[u00][col] = 1
        V[v10][col] = 1; V[v11][col] = -1
        W[w01][col] = 1; W[w11][col] = 1
        col += 1
        
        # M4 = A11(B01-B00) -> C00, C10
        U[u11][col] = 1
        V[v01][col] = 1; V[v00][col] = -1
        W[w00][col] = 1; W[w10][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11 -> C00, C01
        U[u00][col] = 1; U[u01][col] = 1
        V[v11][col] = 1
        W[w00][col] = -1; W[w01][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01) -> C11
        U[u10][col] = 1; U[u00][col] = -1
        V[v00][col] = 1; V[v01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11) -> C00, C11
        U[u01][col] = 1; U[u11][col] = -1
        V[v10][col] = 1; V[v11][col] = 1
        W[w00][col] = 1; W[w11][col] = 1
        col += 1

    def add_schoolbook_221(i_base, j_base, k_idx):
        # A: 2x2 at (i_base, j_base)
        # B: 2x1 at (j_base, k_idx)
        # C: 2x1 at (k_idx, i_base)
        # R(2,2,1) = 2
        
        u00 = i_base * m + j_base
        u01 = i_base * m + j_base + 1
        u10 = (i_base + 1) * m + j_base
        u11 = (i_base + 1) * m + j_base + 1
        
        v0 = j_base * p + k_idx
        v1 = (j_base + 1) * p + k_idx
        
        w0 = k_idx * n + i_base
        w1 = k_idx * n + i_base + 1
        
        # Term 1: A_col0 * B_0
        U[u00][col] = 1
        U[u10][col] = 1
        V[v0][col] = 1
        W[w0][col] = 1
        W[w1][col] = 1
        col += 1
        
        # Term 2: A_col1 * B_1
        U[u01][col] = 1
        U[u11][col] = 1
        V[v1][col] = 1
        W[w0][col] = 1
        W[w1][col] = 1
        col += 1

    def add_block_223(i_base, j_base, k_base):
        # 2x2x3
        # Split k into k_base (1 col) and k_base+1, k_base+2 (2 cols)
        # Strassen for 2 cols (k_base+1, k_base+2)?
        # Strassen requires 2x2 B.
        # So use Strassen for cols k_base+1, k_base+2.
        # And Schoolbook 221 for col k_base.
        
        # Strassen on (i_base, j_base, k_base+1)
        # Wait, Strassen uses 2x2 B.
        # So k indices are k_base+1, k_base+2.
        add_strassen_block(i_base, j_base, k_base + 1)
        
        # Schoolbook 221 on (i_base, j_base, k_base)
        add_schoolbook_221(i_base, j_base, k_base)

    def add_block_213(i_base, j_base, k_base):
        # 2x1x3
        # A: 2x1 at (i_base, j_base)
        # B: 1x3 at (j_base, k_base)
        # C: 2x3 at (k_base, i_base)
        # Rank 3.
        
        u0 = i_base * m + j_base
        u1 = (i_base + 1) * m + j_base
        
        v0 = j_base * p + k_base
        v1 = j_base * p + k_base + 1
        v2 = j_base * p + k_base + 2
        
        w00 = k_base * n + i_base
        w01 = k_base * n + i_base + 1
        w10 = (k_base + 1) * n + i_base
        w11 = (k_base + 1) * n + i_base + 1
        w20 = (k_base + 2) * n + i_base
        w21 = (k_base + 2) * n + i_base + 1
        
        # Term 1 for k_base
        U[u0][col] = 1
        U[u1][col] = 1
        V[v0][col] = 1
        W[w00][col] = 1
        W[w01][col] = 1
        col += 1
        
        # Term 2 for k_base+1
        U[u0][col] = 1
        U[u1][col] = 1
        V[v1][col] = 1
        W[w10][col] = 1
        W[w11][col] = 1
        col += 1
        
        # Term 3 for k_base+2
        U[u0][col] = 1
        U[u1][col] = 1
        V[v2][col] = 1
        W[w20][col] = 1
        W[w21][col] = 1
        col += 1

    def add_block_233(i_base, j_base, k_base):
        # 2x3x3
        # Split j into j_base (2 cols) and j_base+2 (1 col)
        # add_block_223 for j_base
        add_block_223(i_base, j_base, k_base)
        # add_block_213 for j_base+2
        add_block_213(i_base, j_base + 2, k_base)

    def add_block_236(i_base, j_base, k_base):
        # 2x3x6
        # Split k into two 2x3x3 blocks
        add_block_233(i_base, j_base, k_base)
        add_block_233(i_base, j_base, k_base + 3)

    def add_block_226(i_base, j_base, k_base):
        # 2x2x6
        # Split k into two 2x2x3 blocks
        add_block_223(i_base, j_base, k_base)
        add_block_223(i_base, j_base, k_base + 3)

    # Main Construction
    
    # Block 1: i=0,1; j=0,1,2; k=0..5 -> 2x3x6
    add_block_236(0, 0, 0)
    
    # Block 2: i=0,1; j=3,4; k=0..5 -> 2x2x6
    add_block_226(0, 3, 0)
    
    # Block 3: i=2; j=0..4; k=0..5 -> 1x5x6
    # Rank 6.
    # i_base = 2
    for k in range(6):
        U[2 * m][col] = 1 # Only row 2 of A (which is index 2*5 + j? No, U is 15xR)
        # U index for A[2, j] is 2*m + j.
        # We need to sum over j.
        # The term is U[2*m+j] * V[j*p+k] * W[k*n+2].
        # We established rank 6 for 1x5x6.
        # Term r=k:
        # U[2*m + j] = 1 for all j?
        # V[j*p + k] = 1 for all j?
        # W[k*n + 2] = 1?
        # Sum over j: (1) * (1) * (1) = 1?
        # No, the sum is over r.
        # For fixed k, we have one term r=k.
        # U_{2j, r=k} = 1 for all j.
        # V_{jk, r=k} = 1 for all j.
        # W_{k2, r=k} = 1.
        # Then $\sum_j U_{2j, k} V_{jk, k} W_{k2, k} = \sum_j 1 \cdot 1 \cdot 1 = 5$.
        # We need 1.
        # So we should divide by 5? No, integer/half-integer.
        # Wait, the decomposition is $C_{ik} = \sum_{j,r} A_{ij} B_{jk} U_{ir} V_{jr} W_{kr}$.
        # My previous check:
        # $\sum_r U_{2,r} V_{j,r} W_{k,r}$.
        # Here $U$ is indexed by $i$ (which is 2) and $r$.
        # $V$ is indexed by $j$ and $r$.
        # $W$ is indexed by $k$ and $r$.
        # So for term $r=k$:
        # $U_{2, k} = 1$. (Note: U row index is $i$, not $ij$. The matrix U is $n \times r$. The index $j$ is in V.)
        # Ah, I confused the matrix shapes in my head.
        # U is $n \times r$. Index is $i, r$.
        # V is $m \times r$. Index is $j, r$.
        # W is $p \times r$. Index is $k, r$.
        # In the code, U is flattened to $n*m$ rows?
        # The interface says U is `(n*m, rank)`.
        # So U[row] where row = `i*m + j`.
        # So U is not $n \times r$, it is $(nm) \times r$.
        # The tensor is $T_{ij, jk, ki}$.
        # Term r: $u_{ij, r} v_{jk, r} w_{ki, r}$.
        
        # So for 1x5x6:
        # i=2.
        # Term r=k.
        # $u_{2j, r} = 1$ for all j.
        # $v_{jk, r} = 1$ for all j.
        # $w_{k2, r} = 1$.
        # Sum over j: $\sum_j 1 \cdot 1 \cdot 1 = 5$.
        # We need 1.
        # So we must use fractions or change the decomposition.
        # $u_{2j, r} = 1/5$?
        # "integer/half-integer entries".
        # 1/5 is not allowed.
        
        # Alternative:
        # $u_{2j, r} = 1$.
        # $v_{jk, r} = 1$.
        # $w_{k2, r} = 1/5$? No.
        
        # Can we do 1x5x6 with integer coefficients?
        # Rank 6.
        # If we use rank 5?
        # $R(1, m, p) = p$.
        # Is it possible to do it with integers?
        # Yes, if we don't need to divide by 5.
        # Wait, the sum is $\sum_j u_{2j} v_{jk} w_{k2}$.
        # If $u_{2j}=1, v_{jk}=1, w_{k2}=1$, sum is 5.
        # We need 1.
        # So we can set $w_{k2} = 1/5$? No.
        
        # How about this:
        # $u_{2j, r} = \delta_{j, ?}$
        # If we use 5 terms, one for each j?
        # Term r=j.
        # $u_{2j, r} = \delta_{j, r} = 1$.
        # $v_{jk, r} = \delta_{j, r} = 1$.
        # $w_{k2, r} = 1$.
        # Sum over r: $\sum_r u_{2j, r} v_{jk, r} w_{k2, r}$.
        # For fixed j, k:
        # Only r=j contributes.
        # $u_{2j, j} = 1$.
        # $v_{jj, j} = 1$.
        # $w_{k2, j} = 1$.
        # Sum = 1.
        # This works!
        # Rank 5?
        # We used r=0..4.
        # But W has index k.
        # $w_{k2, r}$ is non-zero for all k?
        # In this construction, $w_{k2, r} = 1$ for all k.
        # So for each r, it contributes to all k.
        # This gives $C_{2, k} = \sum_j A_{2, j} B_{j, k}$.
        # This is correct.
        # Rank is 5 (one for each j).
        
        # Let's verify.
        # Term r (corresponds to j=r):
        # $U_{2r, r} = 1$. (Other U entries 0)
        # $V_{r k, r} = 1$ for all k. (Other V entries 0)
        # $W_{k 2, r} = 1$ for all k.
        
        # Check $C_{2, k}$:
        # $\sum_{j, r} A_{2, j} B_{j, k} U_{2j, r} V_{jk, r} W_{k2, r}$
        # Only $j=r$ and $r=j$?
        # For a specific term r:
        # $U_{2j, r}$ is non-zero only if $j=r$. Value 1.
        # $V_{jk, r}$ is non-zero only if $j=r$. Value 1.
        # $W_{k2, r}$ is 1.
        # So term r contributes $A_{2, r} B_{r, k}$.
        # Sum over r=0..4 gives $\sum_{r=0}^4 A_{2, r} B_{r, k}$.
        # This is exactly $C_{2, k}$.
        # So Rank 5 for 1x5x6.
        
        for r in range(5):
            U[2 * m + r][col] = 1
            for k in range(6):
                V[r * p + k][col] = 1
                W[k * n + 2][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
