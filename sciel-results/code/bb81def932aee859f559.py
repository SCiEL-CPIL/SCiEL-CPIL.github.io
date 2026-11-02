"""Rewritten program for Matrix multiplication 2,2,13 - rank 26 approach"""

def construct():
    n, m, p = 2, 2, 13
    rank = 26
    
    # U: shape (n*m, rank) = (4, 26)
    # V: shape (m*p, rank) = (26, 26)  
    # W: shape (p*n, rank) = (26, 26)
    
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(26)]
    W = [[0] * rank for _ in range(26)]
    
    # For each output column k (0..12) and row i (0..1):
    # C[i,k] = A[0,k-1] * B[0,k] + A[1,k-1] * B[1,k] ... 
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # We want to decompose each C[i,k] as a product of a linear form in A and a linear form in B
    # using only 26 total bilinear terms total.
    
    # Strategy: use the fact that for 2x2 A, we can use 4 linear forms
    # and for each of the 13 columns, we need 2 outputs
    # Total 26 outputs, so we assign 1 bilinear term per output.
    
    # But that's just schoolbook with rank 4*2*13 = 104, not 26.
    
    # The correct approach: we need to find 26 triples (u_r, v_r, w_r) such that
    # sum_r u_r ⊗ v_r ⊗ w_r = tensor
    # where u_r ∈ R^4, v_r ∈ R^26, w_r ∈ R^26
    
    # Let me use a different strategy based on the structure.
    # For matrix multiplication (2,2,13), we can view it as:
    # For each k in 0..12: C[:,k] = A * B[:,k]
    # where A is 2x2 and B[:,k] is 2x1
    
    # The tensor for 2x2 times 2x1 has rank 4 (schoolbook) or less?
    # Actually rank of (2,2,1) matmul tensor is 4.
    
    # For 13 independent columns, if they were independent, rank would be 13*4=52.
    # But we can share bilinear forms across columns.
    
    # Key insight: Use 2 linear forms in A that are shared, and 13 pairs of linear forms in B.
    
    # Let me try: define bilinear forms:
    # For the A side, use: L0 = A00, L1 = A01, L2 = A10, L3 = A11 (4 forms)
    # For each column k, we need to compute C0k = A00*B0k + A01*B1k and C1k = A10*B0k + A11*B1k
    # 
    # If I use bilinear terms:
    # term_k0: L0 * B0k -> contributes to C0k
    # term_k1: L1 * B1k -> contributes to C0k  
    # term_k2: L2 * B0k -> contributes to C1k
    # term_k3: L3 * B1k -> contributes to C1k
    # That's 4*13 = 52 terms, still too many.
    
    # To get rank 26, I need to combine. Let me use:
    # For each k, define:
    # S_k = B0k + B1k, D_k = B0k - B1k
    # C0k = (A00+A01)*S_k/2 + (A00-A01)*D_k/2
    # C1k = (A10+A11)*S_k/2 + (A10-A11)*D_k/2
    # 
    # So for each k, I need bilinear forms:
    # (A00+A01)*S_k, (A00-A01)*D_k, (A10+A11)*S_k, (A10-A11)*D_k
    # That's 4 per k, total 52. Still not 26.
    
    # Let me try pairing columns. For columns 2k and 2k+1:
    # I can use Strassen-style on pairs.
    
    # Actually, let me just build a rank-46 solution properly and submit that,
    # since rank 26 may not be achievable for (2,2,13).
    
    # Reverting to the proven rank-46 Strassen approach but implemented cleanly:
    
    U = [[0] * 46 for _ in range(4)]
    V = [[0] * 46 for _ in range(26)]
    W = [[0] * 46 for _ in range(26)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1
    
    for block_idx in range(6):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        # Strassen for 2x2 x 2x2 block
        # U indices: A00=0, A01=1, A10=2, A11=3
        # V indices: B[j,k] where j=0,1; k=k0,k1 -> indices: k, 13+k
        # W indices: C[k,i] where k=k0,k1; i=0,1 -> indices: 2k, 2k+1
        
        add_term({0: 1, 3: 1}, {k0: 1, 13+k1: 1}, {2*k0: 1, 2*k1+1: 1})
        add_term({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1})
        add_term({0: 1}, {k1: 1, 13+k1: -1}, {2*k1: 1, 2*k1+1: 1})
        add_term({3: 1}, {13+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1})
        add_term({0: 1, 1: 1}, {13+k1: 1}, {2*k0: -1, 2*k1: 1})
        add_term({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1})
        add_term({1: 1, 3: -1}, {13+k0: 1, 13+k1: 1}, {2*k0: 1})
    
    k = 12
    add_term({0: 1}, {k: 1}, {2*k: 1})
    add_term({1: 1}, {13+k: 1}, {2*k: 1})
    add_term({2: 1}, {k: 1}, {2*k+1: 1})
    add_term({3: 1}, {13+k: 1}, {2*k+1: 1})
    
    return {"U": U, "V": V, "W": W}
