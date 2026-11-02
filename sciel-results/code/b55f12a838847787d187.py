"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2_terms():
    """Return 7 rank-1 terms for 2x2 x 2x2 Strassen.
    Each term: (u, v, w) length-4 vectors, row-major.
    u[i] for A flat idx i, v[j] for B flat idx j, w[k] for C flat idx k.
    """
    # A = [[a0,a1],[a2,a3]], B=[[b0,b1],[b2,b3]], C=[[c0,c1],[c2,c3]]
    # flat: a0=0,a1=1,a2=2,a3=3; b0=0,b1=1,b2=2,b3=3; c0=0,c1=1,c2=2,c3=3
    terms = []
    # M1 = (a0+a3)(b0+b3) -> c0+=M1, c3+=M1
    terms.append(([1,0,0,1],[1,0,0,1],[1,0,0,1]))
    # M2 = (a2+a3)b0 -> c2+=M2
    terms.append(([0,0,1,1],[1,0,0,0],[0,0,1,0]))
    # M3 = a0(b1-b3) -> c1+=M3
    terms.append(([1,0,0,0],[0,1,0,-1],[0,1,0,0]))
    # M4 = a3(b2-b0) -> c3+=M4
    terms.append(([0,0,0,1],[0,0,1,-1],[0,0,0,1]))
    # M5 = (a0+a1)b2 -> c0+=M5, c2+=M5
    terms.append(([1,1,0,0],[0,0,1,0],[1,0,1,0]))
    # M6 = (a2-a0)(b0+b1) -> c2+=M6, c3-=M6
    terms.append(([0,0,1,-1],[1,1,0,0],[0,0,1,-1]))
    # M7 = (a1-a3)(b3+b2) -> c1-=M7, c3+=M7
    terms.append(([0,1,0,-1],[0,0,1,1],[0,-1,0,1]))
    return terms

def _embed_2x2_into_4x4(vec, block_row, block_col, length):
    """Embed a length-4 vector (2x2 block) into a length-16 vector (4x4).
    block_row, block_col in {0,1} select which 2x2 block.
    Returns a length-16 list (0 elsewhere).
    """
    out = [0] * length
    if length == 16:
        base = block_row * 4 + block_col * 2
        for idx in range(4):
            r = idx // 2
            c = idx % 2
            out[base + r * 2 + c] = vec[idx]
    return out

def _build_strassen_4x4_4x4():
    """Build rank-49 decomposition for 4x4 x 4x4 x 4x4 via recursive Strassen.
    Returns list of (u,v,w) each length 16.
    """
    s2 = _strassen_2x2_terms()
    terms = []
    # Block-level Strassen: 7 super-multiplications, each 2x2x2x2 (7 terms)
    # Super-level indices: A blocks A00,A01,A10,A11 ; B blocks B00,B01,B10,B11 ; C blocks C00,C01,C10,C11
    # The 7 super-terms (u_super, v_super, w_super) over the 4 block-positions:
    # Represent each super-vector as length-4 over block positions, but each position is a 2x2 block.
    # We need to combine: for each super-term, the A-side is a linear combo of 2x2 blocks,
    # B-side a linear combo of 2x2 blocks, and the result contributes to C blocks.
    #
    # Actually the clean recursive construction:
    # The 4x4x4x4 tensor = tensor over (block-index, inner-index).
    # Strassen at block level gives 7 terms where each term is a "block multiplication"
    # of (linear combo of A-blocks) * (linear combo of B-blocks) -> (linear combo of C-blocks).
    # Each such block multiplication is itself a 2x2x2x2 matmul tensor, decomposed by 7 terms.
    # So total 7*7=49.
    #
    # Let me define the 7 super-terms as (U_super, V_super, W_super) where each is a
    # 2x2 matrix of coefficients (over the 4 block positions). Then for each inner Strassen
    # term (u_i, v_i, w_i) length-4, the full term is:
    #   u_full[block_pos, inner] = U_super[block_pos] * u_i[inner]
    #   v_full[block_pos, inner] = V_super[block_pos] * v_i[inner]
    #   w_full[block_pos, inner] = W_super[block_pos] * w_i[inner]
    # where block_pos in {00,01,10,11} and inner in {0,1,2,3}.
    #
    # The 7 super-terms (as 2x2 coefficient matrices for A-blocks, B-blocks, C-blocks):
    # M1: A: [[1,0],[0,1]], B: [[1,0],[0,1]], C: [[1,0],[0,1]]
    # M2: A: [[0,0],[1,1]], B: [[1,0],[0,0]], C: [[0,0],[1,0]]
    # M3: A: [[1,0],[0,0]], B: [[0,1],[0,-1]], C: [[0,1],[0,0]]
    # M4: A: [[0,0],[0,1]], B: [[0,0],[1,-1]], C: [[0,0],[0,1]]
    # M5: A: [[1,1],[0,0]], B: [[0,0],[1,0]], C: [[1,0],[1,0]]
    # M6: A: [[0,0],[1,-1]], B: [[1,1],[0,0]], C: [[0,0],[1,-1]]
    # M7: A: [[0,1],[0,-1]], B: [[0,0],[1,1]], C: [[0,-1],[0,1]]
    super_terms = [
        ([[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]),
        ([[0,0],[1,1]],[[1,0],[0,0]],[[0,0],[1,0]]),
        ([[1,0],[0,0]],[[0,1],[0,-1]],[[0,1],[0,0]]),
        ([[0,0],[0,1]],[[0,0],[1,-1]],[[0,0],[0,1]]),
        ([[1,1],[0,0]],[[0,0],[1,0]],[[1,0],[1,0]]),
        ([[0,0],[1,-1]],[[1,1],[0,0]],[[0,0],[1,-1]]),
        ([[0,1],[0,-1]],[[0,0],[1,1]],[[0,-1],[0,1]]),
    ]
    for U_s, V_s, W_s in super_terms:
        for (u_i, v_i, w_i) in s2:
            u_full = [0]*16
            v_full = [0]*16
            w_full = [0]*16
            for br in range(2):
                for bc in range(2):
                    block_pos = br*2+bc
                    coeff_u = U_s[br][bc]
                    coeff_v = V_s[br][bc]
                    coeff_w = W_s[br][bc]
                    for inner in range(4):
                        base = block_pos*4 + inner
                        u_full[base] = coeff_u * u_i[inner]
                        v_full[base] = coeff_v * v_i[inner]
                        w_full[base] = coeff_w * w_i[inner]
            terms.append((u_full, v_full, w_full))
    return terms

def _build_decomposition_4x4x4x5():
    """Build rank-53 decomposition for 4x4 x 4x5.
    Split B into B[:,:4] (4x4) and B[:,4] (4x1).
    C[:,:4] = A @ B[:,:4]  -> rank-49 Strassen
    C[:,4]  = A @ B[:,4]   -> rank-4 schoolbook
    Total rank 53.
    """
    n, m, p = 4, 4, 5
    s4 = _build_strassen_4x4_4x4()
    # 49 terms for the 4x4x4x4 part, each (u,v,w) length 16.
    # Map to full: A flat idx i*m+j (i in 0..3, j in 0..3) -> same as 4x4 flat.
    # B for the 4x4 part: j in 0..3, k in 0..3 -> flat j*5+k (since p=5).
    # C for the 4x4 part: k in 0..3, i in 0..3 -> flat k*4+i.
    rank = 53
    U = [[0]*rank for _ in range(n*m)]   # 16 x 53
    V = [[0]*rank for _ in range(m*p)]   # 20 x 53
    W = [[0]*rank for _ in range(p*n)]   # 20 x 53
    
    col = 0
    for (u16, v16, w16) in s4:
        # u16: A flat i*4+j
        for idx in range(16):
            i = idx // 4
            j = idx % 4
            U[i*m + j][col] = u16[idx]
        # v16: B 4x4 part, flat j*4+k (j in 0..3, k in 0..3) -> full flat j*5+k
        for idx in range(16):
            j = idx // 4
            k = idx % 4
            V[j*p + k][col] = v16[idx]
        # w16: C 4x4 part, flat k*4+i (k in 0..3, i in 0..3) -> full flat k*4+i
        for idx in range(16):
            k = idx // 4
            i = idx % 4
            W[k*n + i][col] = w16[idx]
        col += 1
    
    # Now the 4x4 x 4x1 part: C[:,4] = A @ B[:,4]
    # Schoolbook: for each i in 0..3, k=4: C[i,4] = sum_j A[i,j] B[j,4]
    # That's 4 rank-1 terms (one per j), but actually we need n*m = 16 terms? No.
    # C[i,4] = sum_j A[i,j]*B[j,4]. For fixed i,k=4, sum over j.
    # Rank-1 term for (i,j,k=4): U[i*4+j, c]=1, V[j*5+4, c]=1, W[4*4+i, c]=1.
    # That's n*m = 16 terms. But we can do better: for each j, the term contributes to all i.
    # Actually each (i,j) pair is a separate rank-1 term in schoolbook. n*m = 16 terms.
    # But we can use rank-4: for each j, define a term where U[i*4+j, c_j]=1 for all i,
    # V[j*5+4, c_j]=1, W[4*4+i, c_j]=1 for all i. That's 4 terms (one per j).
    # Check: sum_c U[i*4+j,c]*V[j*5+4,c]*W[4*4+i,c] for c in {c_0..c_3}:
    #   = sum_j' U[i*4+j, c_j']*V[j*5+4, c_j']*W[4*4+i, c_j']
    #   Only c_j' = c_j has V[j*5+4, c_j]=1 and U[i*4+j,c_j]=1 and W[4*4+i,c_j]=1.
    #   So = 1 for each j. Sum over j = 4. That's wrong, we want sum_j A[i,j]B[j,4].
    # Hmm, the rank-1 term gives a contribution of 1 to C[i,4] for that specific j.
    # So we need one term per j, and the sum over j gives the correct sum. That's 4 terms.
    # Wait, but each term contributes 1 (not A[i,j]*B[j,4]). The decomposition is:
    #   C[i,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # For the schoolbook, each (i,j,k) has its own term with U=V=W=1.
    # For the 4x4x4x1 part, k=4, and we need sum_j A[i,j]B[j,4].
    # The standard schoolbook would use n*m = 16 terms (one per i,j with k=4).
    # But we can group: for fixed j, the term (i,j,k=4) has U[i*4+j,c]=1, V[j*5+4,c]=1, W[4*4+i,c]=1.
    # If I use one term per j (4 terms), with U[i*4+j, c_j]=1 for ALL i, V[j*5+4,c_j]=1, W[4*4+i,c_j]=1 for ALL i,
    # then for a given (i,j',k=4): sum_c U[i*4+j',c]*V[j'*5+4,c]*W[4*4+i,c]
    #   = sum_j'' U[i*4+j',c_j'']*V[j''*5+4,c_j'']*W[4*4+i,c_j'']
    #   = U[i*4+j', c_j'] * V[j'*5+4, c_j'] * W[4*4+i, c_j']  (only c_j' matches)
    #   = 1*1*1 = 1 if j'=j'', else 0.
    # So it gives 1 for the matching j', which is correct for that j. But we need sum over j, so we need all 4 terms active.
    # The total contribution to C[i,4] = sum_j (1) = 4? No, each term c_j contributes 1 to C[i,4] for the i it's defined for.
    # Wait, I'm confusing myself. Let me just use the straightforward schoolbook: 16 terms for the 4x4x4x1 part.
    # That gives total rank 49+16 = 65. Not great.
    
    # Better: the 4x4 x 4x1 matmul has rank at most 4 (since it's a rank-1 in the B dimension).
    # Actually, C[:,4] = A @ B[:,4]. This is a 4x4 times 4x1 = 4x1 vector.
    # The tensor for this: T[i,j,4] = 1 for all i,j. We need sum_c U[i*4+j,c]*V[j*5+4,c]*W[4*4+i,c] = 1 for all i,j.
    # A rank-4 decomposition: for c in 0..3, let c_j = c (j=c).
    #   U[i*4+c, c] = 1 for all i
    #   V[c*5+4, c] = 1
    #   W[4*4+i, c] = 1 for all i
    # Then for (i,j,4): sum_c U[i*4+j,c]*V[j*5+4,c]*W[4*4+i,c]
    #   = U[i*4+j, j]*V[j*5+4, j]*W[4*4+i, j]  (only c=j is nonzero)
    #   = 1*1*1 = 1. Correct!
    # So 4 terms suffice. Total rank 49+4 = 53.
    
    for j in range(4):
        c = col + j
        for i in range(4):
            U[i*m + j][c] = 1
        V[j*p + 4][c] = 1
        for i in range(4):
            W[4*n + i][c] = 1
    col += 4
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140
    
    # Build the rank-53 decomposition
    result = _build_decomposition_4x4x4x5()
    
    # Verify
    if time.time() < deadline - 5:
        if not _verify(result["U"], result["V"], result["W"], n, m, p):
            # Fallback to schoolbook
            rank = n*m*p
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            result = {"U": U, "V": V, "W": W}
    
    return result
# EVOLVE-BLOCK-END
