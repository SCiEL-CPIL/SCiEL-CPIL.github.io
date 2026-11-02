"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 37 solution:
    # M_{3,4,5} = M_{3,3,5} (cols 0-2 of B) + M_{3,1,5} (col 3 of B)
    # M_{3,1,5} rank 1
    # M_{3,3,5} = M_{3,3,3} (cols 0-2 of B) + M_{3,3,2} (cols 3-4 of B? No, B is 3x5)
    # Wait, my previous split was:
    # A (3x4) -> A1 (3x3) + A2 (3x1)
    # B (4x5) -> B1 (3x5) + B2 (1x5)
    # C = A1 B1 + A2 B2
    # A1 B1 is 3x3 by 3x5 -> M_{3,3,5}
    # A2 B2 is 3x1 by 1x5 -> M_{3,1,5}
    
    # M_{3,3,5} = A1 (3x3) * B1 (3x5)
    # Split B1 (3x5) into B11 (3x3) + B12 (3x2)
    # A1 B11 is M_{3,3,3}
    # A1 B12 is M_{3,3,2}
    
    # So:
    # 1. M_{3,1,5}: Rank 1
    # 2. M_{3,3,3}: Rank 27 (Schoolbook)
    # 3. M_{3,3,2}: Rank 9 (Strassen 2x2x2 + 1x3x2 + 2x1x2)
    
    # Total Rank: 1 + 27 + 9 = 37.
    
    rank = 37
    U = [[0] * rank for _ in range(n * m)] # 12 x 37
    V = [[0] * rank for _ in range(m * p)] # 20 x 37
    W = [[0] * rank for _ in range(p * n)] # 15 x 37
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1
        
    # 1. M_{3,1,5}: A2 (3x1) * B2 (1x5)
    # A2 is col 3 of A (indices 3, 7, 11 in U flat? No, A is 3x4. Index i*4+j.
    # A2 is j=3. i in 0..2. U indices: 0*4+3=3, 1*4+3=7, 2*4+3=11.
    # B2 is row 3 of B (indices 3*5..3*5+4 in V flat? No, B is 4x5. Index j*5+k.
    # B2 is j=3. k in 0..4. V indices: 3*5+k = 15..19.
    # C is 3x5. Index k*3+i. W indices: k*3+i.
    # Term: u_i * v_j * w_k.
    # We need sum u v w = 1 for all i,k.
    # Use u_i=1, v_3=1, w_k=1.
    add_term(3, 15, 0, 1) # i=0, j=3, k=0 -> W[0*3+0]=W[0]
    # Wait, the term is u_i * v_j * w_k.
    # We need to fill U, V, W such that sum U[i,j] V[j,k] W[k,i] = 1 for all i,k.
    # My add_term adds to a specific column.
    # For rank 1:
    # U[3][col] = 1 (i=0, j=3)
    # U[7][col] = 1 (i=1, j=3)
    # U[11][col] = 1 (i=2, j=3)
    # V[15..19][col] = 1 (j=3, k=0..4)
    # W[k*3+i][col] = 1 for all k,i.
    # This is 9 non-zeros in W.
    # But in tensor decomposition, each rank-1 term has 1 non-zero in U, 1 in V, 1 in W.
    # So I must add 9 terms for this part?
    # No, $M_{3,1,5}$ rank is 1?
    # $T_{i,0,k} = 1$.
    # $T = \sum u_i v_0 w_k$.
    # If rank is 1, then $T_{i,0,k} = u_i w_k$.
    # This means $u_i$ and $w_k$ are fixed vectors.
    # So in the decomposition:
    # $U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
    # Here $j=0$ (relative to the sub-tensor).
    # So for the term $r$:
    # $U_{i, 3, r} = u_i$.
    # $V_{3, k, r} = v_k$? No, $v_{0, r}$.
    # $W_{k, i, r} = w_{ki}$? No, $w_{k, i, r}$.
    # Wait, $W$ is $p \times n$. Index $k, i$.
    # So $W_{k, i, r}$ must be 1 for all $k,i$ if $u_i, v_0, w_k$ are 1?
    # No, $T_{i,0,k} = \sum_r U_{i,0,r} V_{0,k,r} W_{k,i,r}$.
    # If we set $U_{i,0,r} = 1$, $V_{0,k,r} = 1$, then $\sum_r W_{k,i,r} = 1$.
    # We can set $W_{k,i,r} = 1$ for one $r$ and 0 for others.
    # So yes, it is 1 term in the sum, but it populates multiple entries in U, V, W.
    # So `add_term` is wrong. I need to add a full rank-1 tensor.
    
    # Let's restructure.
    # A rank-1 tensor is defined by vectors u, v, w.
    # It contributes to U[i, j, r] = u[i], V[j, k, r] = v[j], W[k, i, r] = w[k] ??
    # No, $T_{i,j,k} = u_i v_j w_k$.
    # In matrix mult: $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # $T_{i,j,k} = 1$.
    # So $u_i v_j w_k = 1$.
    # This implies $u_i=1, v_j=1, w_k=1$.
    # So for a term $r$:
    # $U_{i,j,r} = 1$
    # $V_{j,k,r} = 1$
    # $W_{k,i,r} = 1$
    
    # So for $M_{3,1,5}$ (indices i in 0..2, j=3, k in 0..4):
    # Term 0:
    # U[i, 3, 0] = 1 for i in 0..2
    # V[3, k, 0] = 1 for k in 0..4
    # W[k, i, 0] = 1 for k in 0..4, i in 0..2
    # This is 1 rank term.
    
    col = 0
    # 1. M_{3,1,5}
    for i in range(3):
        U[i * 4 + 3][col] = 1
    for k in range(5):
        V[3 * 5 + k][col] = 1
    for k in range(5):
        for i in range(3):
            W[k * 3 + i][col] = 1
    col += 1
    
    # 2. M_{3,3,3} (Schoolbook)
    # i in 0..2, j in 0..2, k in 0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    # col is now 1 + 27 = 28.
    
    # 3. M_{3,3,2}
    # A1 (3x3, j in 0..2) * B12 (3x2, k in 3..4)
    # Split A1 into A1_top (2x3, i in 0..1) and A1_bot (1x3, i=2)
    # B12 is 3x2.
    
    # A1_top * B12 is M_{2,3,2}.
    # Split middle dim 3 into 2+1.
    # A1_top (2x3) -> A1_top_1 (2x2, j in 0..1) and A1_top_2 (2x1, j=2)
    # B12 (3x2) -> B12_1 (2x2, j in 0..1) and B12_2 (1x2, j=2)
    
    # A1_top_1 * B12_1 is M_{2,2,2}. Rank 7 (Strassen).
    # A1_top_2 * B12_2 is M_{2,1,2}. Rank 1.
    
    # A1_bot * B12 is M_{1,3,2}. Rank 1.
    
    # Total 7 + 1 + 1 = 9 terms.
    
    # 3a. M_{2,1,2} (A1_top_2 * B12_2)
    # i in 0..1, j=2, k in 3..4
    # u_i=1, v_2=1, w_k=1.
    for i in range(2):
        U[i * 4 + 2][col] = 1
    V[2 * 5 + 0][col] = 1 # Wait, B12 is cols 3,4 of B.
    # B12_2 is row 2 of B12. B12 rows are 0,1,2. So row 2.
    # B12_2 is 1x2.
    # V index: j=2, k=3,4.
    V[2 * 5 + 3][col] = 1
    V[2 * 5 + 4][col] = 1
    for k in [3, 4]:
        for i in range(2):
            W[k * 3 + i][col] = 1
    col += 1
    
    # 3b. M_{1,3,2} (A1_bot * B12)
    # i=2, j in 0..2, k in 3..4
    # u_2=1, v_j=1, w_k=1.
    U[2 * 4 + 0][col] = 1
    U[2 * 4 + 1][col] = 1
    U[2 * 4 + 2][col] = 1
    V[0 * 5 + 3][col] = 1
    V[0 * 5 + 4][col] = 1
    V[1 * 5 + 3][col] = 1
    V[1 * 5 + 4][col] = 1
    V[2 * 5 + 3][col] = 1
    V[2 * 5 + 4][col] = 1
    for k in [3, 4]:
        W[k * 3 + 2][col] = 1
    col += 1
    
    # 3c. M_{2,2,2} (A1_top_1 * B12_1)
    # i in 0..1, j in 0..1, k in 3..4
    # Strassen for 2x2x2.
    # A1_top_1 is A[i, j] for i in 0..1, j in 0..1.
    # B12_1 is B[j, k] for j in 0..1, k in 3..4.
    # Map to local indices:
    # A_local[i, j] = A[i, j]
    # B_local[j, k'] = B[j, k'+3]
    # C_local[i, k'] = C[i, k'+3]
    
    # Strassen terms for 2x2x2:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-c)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # We need to decompose these into rank-1 terms for the tensor.
    # This is tricky because Strassen combines products.
    # Each Strassen product corresponds to a rank-1 term in the *combined* multiplication, but in the tensor decomposition, we need to express the tensor T as sum of rank-1.
    # Strassen shows that T_{2,2,2} has rank 7.
    # The 7 terms are specific combinations.
    
    # Term 1: (A00 + A11)(B00 + B11)
    # T1 = (A00+A11) (B00+B11) I?
    # No, $C_{00} = P_1 + P_4 - P_5 + P_7$.
    # This means the tensor $T$ is a sum of 7 tensors $T_r$.
    # $T_r[i,j,k]$ are the coefficients.
    
    # Let's use the known 7-term decomposition for $M_{2,2,2}$.
    # The terms are:
    # 1. u=[1,1], v=[1,0], w=[1,0] ? No.
    # I will use the standard Strassen decomposition vectors.
    # Ref: "Strassen's algorithm"
    
    # Term 1:
    # u = [1, 1], v = [1, 1], w = [1, 0] ?
    # Let's check P1 = (A00+A11)(B00+B11).
    # This contributes to C00 and C11?
    # C00 += P1. C11 += P1.
    # So w = [1, 1]?
    # If u=[1,1], v=[1,1], w=[1,1], then T = u v w^T.
    # T[0,0,0] = 1. T[0,0,1] = 1. T[1,0,0] = 1. ...
    # This is not correct for Strassen.
    
    # I will implement the 7 terms explicitly.
    # Term 1:
    U[0*4+0][col] = 1; U[1*4+1][col] = 1
    V[0*5+3][col] = 1; V[1*5+4][col] = 1
    W[3*3+0][col] = 1; W[4*3+1][col] = 1
    col += 1
    
    # This is getting complicated to guess.
    # Given the risk, I will fall back to Schoolbook for M_{2,2,2} as well.
    # Rank 8 for M_{2,2,2} (schoolbook 4x4? No, 2x2x2 schoolbook is 4).
    # So M_{2,2,2} Rank 4.
    # Total Rank = 1 (3,1,5) + 27 (3,3,3) + 1 (2,1,2) + 1 (1,3,2) + 4 (2,2,2) = 34.
    
    # This is safer.
    
    # 3c (revised). M_{2,2,2} (Schoolbook)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i * 4 + j][col] = 1
                V[j * 5 + (3 + k)][col] = 1
                W[(3 + k) * 3 + i][col] = 1
                col += 1
                
    # col should be 1 + 27 + 1 + 1 + 4 = 34.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
