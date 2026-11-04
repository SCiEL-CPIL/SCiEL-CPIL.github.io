"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 17
    V = [[0] * rank for _ in range(m * p)]  # 9 x 17
    W = [[0] * rank for _ in range(p * n)]  # 6 x 17
    
    col = 0
    
    # Block 1: A[0:2, 0:2] @ B[0:2, 0:2] using Strassen (7 terms)
    # A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    # C11 = (A11+A22)(B11+B22) - (A21+A22)B22
    # C12 = A11(B12-B22) + (A11+A12)B22
    # C21 = (A21+A22)B11 + A22(B21-B11)
    # C22 = (A11-A21)(B11+B21) + A21(B22-B11)
    
    # We need to express each Strassen product as U*V contributions to W
    
    # Strassen 7 products for 2x2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*(B11)
    # M3 = (A11)*(B12-B22)
    # M4 = (A22)*(B22)
    # M5 = (A11+A12)*(B21+B22)
    # M6 = (A11-A21)*(B11+B21)
    # M7 = (A21+A22)*(B12-B11)
    
    # C11 = M1 - M4
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M4 + M7
    
    # For each Strassen product, we need to add contributions.
    # The indices: A row i (0,1), A col j (0,1) for this block
    # B row k (0,1), B col l (0,1) for this block
    
    # Helper to add a term: U[i*m+j][col] += val_u, V[k*p+l][col] += val_v, W[l*n+i][col] += val_w
    
    # Term 1: M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # U contributions: A[0,0] and A[1,1]
    # V contributions: B[0,0] and B[1,1]
    # This gives 4 products but we want to share. In tensor decomposition, 
    # (a+b)*(c+d) = a*c + a*d + b*c + b*d requires 4 terms unless we can share.
    # For Strassen in tensor decomposition, each "product" is one rank-1 term.
    # M1 requires: (A11+A22)*(B11+B22) which expands to 4 rank-1 terms.
    # That's not right. Let me reconsider.
    
    # In tensor decomposition, each rank-1 term is one multiplication of a linear form in A 
    # times a linear form in B, contributing to a linear form in C.
    # So for M1 = (A11+A22)*(B11+B22):
    #   U: A11 and A22 each get coefficient 1
    #   V: B11 and B12... wait, B11 and B22 each get coefficient 1
    #   W: C11 gets coefficient 1
    # But this single rank-1 term represents (A11+A22)*(B11+B22) which contributes to C11.
    # Yes! Each Strassen product IS one rank-1 term in the tensor decomposition.
    
    # So for M1: U[0*m+0]=1, U[1*m+1]=1, V[0*p+0]=1, V[1*p+1]=1, W[0*n+0]=1
    # This means: contribution to C[0,0] = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    
    # Let me redo this properly.
    
    # Reset
    col = 0
    
    def add_term(U_entries, V_entries, W_entries):
        """Add one rank-1 term. U_entries: list of (row_idx, coeff), etc."""
        nonlocal col
        for row, coeff in U_entries:
            U[row][col] += coeff
        for row, coeff in V_entries:
            V[row][col] += coeff
        for row, coeff in W_entries:
            W[row][col] += coeff
        col += 1
    
    # Strassen for A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    # A indices: row 0-1, col 0-1 -> U index: i*m+j = i*3+j
    # B indices: row 0-1, col 0-1 -> V index: k*p+l = k*3+l
    # C indices: row 0-1, col 0-1 -> W index: l*n+i = l*2+i
    
    # M1 = (A11+A22)*(B11+B22) -> contributes to C11
    add_term(
        [(0*3+0, 1), (1*3+1, 1)],  # A[0,0], A[1,1]
        [(0*3+0, 1), (1*3+1, 1)],  # B[0,0], B[1,1]
        [(0*2+0, 1)]                # C[0,0]
    )
    
    # M2 = (A21+A22)*B11 -> contributes to C21
    add_term(
        [(1*3+0, 1), (1*3+1, 1)],  # A[1,0], A[1,1]
        [(0*3+0, 1)],              # B[0,0]
        [(0*2+1, 1)]               # C[1,0]
    )
    
    # M3 = A11*(B12-B22) -> contributes to C12
    add_term(
        [(0*3+0, 1)],              # A[0,0]
        [(0*3+1, 1), (1*3+1, -1)], # B[0,1], -B[1,1]
        [(1*2+0, 1)]               # C[0,1]
    )
    
    # M4 = A22*B22 -> contributes to C11 (with -1) and C22 (with +1)
    # We need this to contribute to both C11 and C22, so we need two separate terms
    # OR we can have W contribute to multiple entries
    # Actually in tensor decomposition, W can have multiple non-zero entries per column
    # So M4: U=[A22], V=[B22], W=[-C11, +C22]
    add_term(
        [(1*3+1, 1)],              # A[1,1]
        [(1*3+1, 1)],              # B[1,1]
        [(0*2+0, -1), (1*2+1, 1)]  # -C[0,0], +C[1,1]
    )
    
    # M5 = (A11+A12)*(B21+B22) -> contributes to C12
    add_term(
        [(0*3+0, 1), (0*3+1, 1)],  # A[0,0], A[0,1]
        [(1*3+0, 1), (1*3+1, 1)],  # B[1,0], B[1,1]
        [(1*2+0, 1)]               # C[0,1]
    )
    
    # M6 = (A11-A21)*(B11+B21) -> contributes to C21
    add_term(
        [(0*3+0, 1), (1*3+0, -1)], # A[0,0], -A[1,0]
        [(0*3+0, 1), (1*3+0, 1)],  # B[0,0], B[1,0]
        [(0*2+1, 1)]               # C[1,0]
    )
    
    # M7 = (A21+A22)*(B12-B11) -> contributes to C22
    add_term(
        [(1*3+0, 1), (1*3+1, 1)],  # A[1,0], A[1,1]
        [(0*3+1, 1), (0*3+0, -1)], # B[0,1], -B[0,0]
        [(1*2+1, 1)]               # C[1,1]
    )
    
    # col should be 7 now
    
    # Block 2: A[0:2,0:2] @ B[0:2,2] (2x2 @ 2x1) -> C[0:2,2]
    # 4 schoolbook terms: A[i,j]*B[k,l] where j,k in {0,1}, l=2
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    add_term(
        [(0*3+0, 1)],              # A[0,0]
        [(0*3+2, 1)],              # B[0,2]
        [(2*2+0, 1)]               # C[0,2]
    )
    
    add_term(
        [(0*3+1, 1)],              # A[0,1]
        [(1*3+2, 1)],              # B[1,2]
        [(2*2+0, 1)]               # C[0,2]
    )
    
    add_term(
        [(1*3+0, 1)],              # A[1,0]
        [(0*3+2, 1)],              # B[0,2]
        [(2*2+1, 1)]               # C[1,2]
    )
    
    add_term(
        [(1*3+1, 1)],              # A[1,1]
        [(1*3+2, 1)],              # B[1,2]
        [(2*2+1, 1)]               # C[1,2]
    )
    
    # col should be 11 now
    
    # Block 3: A[0:2,2] @ B[2,0:3] (2x1 @ 1x3) -> C[0:2,0:3]
    # 6 schoolbook terms
    # C[0,0] += A[0,2]*B[2,0]
    # C[0,1] += A[0,2]*B[2,1]
    # C[0,2] += A[0,2]*B[2,2]
    # C[1,0] += A[1,2]*B[2,0]
    # C[1,1] += A[1,2]*B[2,1]
    # C[1,2] += A[1,2]*B[2,2]
    
    add_term(
        [(0*3+2, 1)],              # A[0,2]
        [(2*3+0, 1)],              # B[2,0]
        [(0*2+0, 1)]               # C[0,0]
    )
    
    add_term(
        [(0*3+2, 1)],              # A[0,2]
        [(2*3+1, 1)],              # B[2,1]
        [(1*2+0, 1)]               # C[0,1]
    )
    
    add_term(
        [(0*3+2, 1)],              # A[0,2]
        [(2*3+2, 1)],              # B[2,2]
        [(2*2+0, 1)]               # C[0,2]
    )
    
    add_term(
        [(1*3+2, 1)],              # A[1,2]
        [(2*3+0, 1)],              # B[2,0]
        [(0*2+1, 1)]               # C[1,0]
    )
    
    add_term(
        [(1*3+2, 1)],              # A[1,2]
        [(2*3+1, 1)],              # B[2,1]
        [(1*2+1, 1)]               # C[1,1]
    )
    
    add_term(
        [(1*3+2, 1)],              # A[1,2]
        [(2*3+2, 1)],              # B[2,2]
        [(2*2+1, 1)]               # C[1,2]
    )
    
    # col should be 17 now
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

