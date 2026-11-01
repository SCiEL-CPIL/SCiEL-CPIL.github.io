"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2 * 2x2 multiplication.
    
    Each term is (u_local, v_local, w_local) where each is a list of (local_idx, coeff).
    Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    M1 = (x00+x11)(y00+y11) -> contributes to P00(+1), P11(+1), P01(-1), P10(-1)
    M2 = (x10+x11)y00       -> contributes to P10(+1), P11(-1)
    M3 = x00(y10-y00)       -> contributes to P01(+1), P11(+1)
    M4 = x11(y01-y10)       -> contributes to P00(+1), P10(+1)
    M5 = (x00+x01)y11       -> contributes to P10(+1), P11(-1)  -- wait let me recheck
    M6 = (x10-x00)(y00+y10) -> contributes to P00(-1), P01(+1), P10(+1), P11(-1)
    M7 = (x01-x11)(y01+y11) -> contributes to P00(+1), P01(+1), P10(-1), P11(-1)
    
    Standard Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A10)B11
    M6 = (A01-A10)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    So for each term, the W contributions are:
    M1: C00+=1, C11+=1
    M2: C10+=1, C11-=1
    M3: C01+=1, C11+=1
    M4: C00+=1, C10+=1
    M5: C00-=1, C01+=1
    M6: C00+=1, C11+=1  -- wait, let me recheck
    
    Actually standard:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    M1: W00=+1, W11=+1
    M2: W10=+1, W11=-1
    M3: W01=+1, W11=+1
    M4: W00=+1, W10=+1
    M5: W00=-1, W01=+1
    M6: W00=+1, W11=+1  -- hmm, M6 only appears in C11? Let me check.
    
    Wait: C11 = M1 - M2 + M3 + M6. So M6 contributes +1 to C11.
    But M6 = (A01-A10)(B00+B10). Let me verify:
    (A01-A10)(B00+B10) = A01*B00 + A01*B10 - A10*B00 - A10*B10
    
    Hmm, this doesn't seem to only contribute to C11. Let me use a verified Strassen formulation.
    
    Actually, let me just use the well-known formulation and verify numerically.
    """
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    terms.append(([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]))
    
    # M2 = (A10+A11)B00
    terms.append(([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]))
    
    # M3 = A00(B10-B11)
    terms.append(([(0,1)], [(2,1),(3,-1)], [(1,1),(3,1)]))
    
    # M4 = A11(B01-B10)
    terms.append(([(3,1)], [(1,1),(2,-1)], [(0,1),(2,1)]))
    
    # M5 = (A00+A10)B11
    terms.append(([(0,1),(2,1)], [(3,1)], [(0,-1),(1,1)]))
    
    # M6 = (A01-A10)(B00+B10)
    terms.append(([(1,1),(2,-1)], [(0,1),(2,1)], [(0,1),(1,-1),(2,1),(3,-1)]))
    
    # M7 = (A01-A11)(B01+B11)
    terms.append(([(1,1),(3,-1)], [(1,1),(3,1)], [(0,1),(1,1),(2,-1),(3,-1)]))
    
    return terms


def _build_rank28():
    """Build a rank-28 decomposition for 2x4 * 4x4 using block Strassen.
    
    A (2x4) = [A_L | A_R] where A_L, A_R are 2x2
    B (4x4) = [[B_TL, B_TR], [B_BL, B_BR]]
    C (2x4) = [C_L | C_R]
    
    C_L = A_L * B_TL + A_R * B_BL
    C_R = A_L * B_TR + A_R * B_BR
    
    Each 2x2*2x2 product uses 7 Strassen terms, total 4*7 = 28.
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_terms = _strassen_2x2_terms()
    
    # Index mappings
    # A (2x4): U index = i*m + j = i*4 + j
    #   A_L (cols 0,1): i*4+0, i*4+1, i*4+2, i*4+3 -> local (0,0)=i*4+0, (0,1)=i*4+1, (1,0)=i*4+4, (1,1)=i*4+5
    #   Wait, A is 2x4. A[i,j] where i=0,1 and j=0,1,2,3.
    #   U index for A[i,j] is i*4+j.
    #   A_L: j=0,1. A_L[0,0]=A[0,0]=U[0], A_L[0,1]=A[0,1]=U[1], A_L[1,0]=A[1,0]=U[4], A_L[1,1]=A[1,1]=U[5]
    #   A_R: j=2,3. A_R[0,0]=A[0,2]=U[2], A_R[0,1]=A[0,3]=U[3], A_R[1,0]=A[1,2]=U[6], A_R[1,1]=A[1,3]=U[7]
    
    def map_u(side, local_idx):
        # side: 'L' or 'R'
        # local_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        r = local_idx // 2
        c = local_idx % 2
        if side == 'L':
            j = c
        else:
            j = 2 + c
        return r * 4 + j
    
    # B (4x4): V index = j*p + k = j*4 + k where j=0..3 (row of B), k=0..3 (col of B)
    #   B_TL: rows 0,1 cols 0,1. B_TL[0,0]=V[0*4+0]=V[0], B_TL[0,1]=V[0*4+1]=V[1], B_TL[1,0]=V[1*4+0]=V[4], B_TL[1,1]=V[1*4+1]=V[5]
    #   B_TR: rows 0,1 cols 2,3. B_TR[0,0]=V[0*4+2]=V[2], B_TR[0,1]=V[0*4+3]=V[3], B_TR[1,0]=V[1*4+2]=V[6], B_TR[1,1]=V[1*4+3]=V[7]
    #   B_BL: rows 2,3 cols 0,1. B_BL[0,0]=V[2*4+0]=V[8], B_BL[0,1]=V[2*4+1]=V[9], B_BL[1,0]=V[3*4+0]=V[12], B_BL[1,1]=V[3*4+1]=V[13]
    #   B_BR: rows 2,3 cols 2,3. B_BR[0,0]=V[2*4+2]=V[10], B_BR[0,1]=V[2*4+3]=V[11], B_BR[1,0]=V[3*4+2]=V[14], B_BR[1,1]=V[3*4+3]=V[15]
    
    def map_v(block, local_idx):
        # block: 'TL','TR','BL','BR'
        r = local_idx // 2
        c = local_idx % 2
        if block == 'TL':
            j, k = r, c
        elif block == 'TR':
            j, k = r, 2+c
        elif block == 'BL':
            j, k = 2+r, c
        elif block == 'BR':
            j, k = 2+r, 2+c
        return j * 4 + k
    
    # C (2x4): W index = k*n + i = k*2 + i where k=0..3 (col of C), i=0..1 (row of C)
    #   C_L: cols 0,1. C_L[0,0]=W[0*2+0]=W[0], C_L[0,1]=W[1*2+0]=W[2], C_L[1,0]=W[0*2+1]=W[1], C_L[1,1]=W[1*2+1]=W[3]
    #   C_R: cols 2,3. C_R[0,0]=W[2*2+0]=W[4], C_R[0,1]=W[3*2+0]=W[6], C_R[1,0]=W[2*2+1]=W[5], C_R[1,1]=W[3*2+1]=W[7]
    
    def map_w(side, local_idx):
        # side: 'L' or 'R'
        r = local_idx // 2
        c = local_idx % 2
        if side == 'L':
            k = c
        else:
            k = 2 + c
        return k * 2 + r
    
    def apply_strassen(a_side, b_block, c_side, start_col):
        for t in strassen_terms:
            u_local, v_local, w_local = t
            for idx, coeff in u_local:
                U[map_u(a_side, idx), start_col] += coeff
            for idx, coeff in v_local:
                V[map_v(b_block, idx), start_col] += coeff
            for idx, coeff in w_local:
                W[map_w(c_side, idx), start_col] += coeff
            start_col += 1
        return start_col
    
    col = apply_strassen('L', 'TL', 'L', 0)
    col = apply_strassen('R', 'BL', 'L', col)
    col = apply_strassen('L', 'TR', 'R', col)
    col = apply_strassen('R', 'BR', 'R', col)
    
    return U, V, W


def _verify(U, V, W, n=2, m=4, p=4):
    """Verify the decomposition. Returns max error."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _try_reduce(U, V, W, time_limit=100):
    """Try to reduce rank by finding linear dependencies among the 28 terms.
    
    We look for a non-trivial combination of terms that sums to zero tensorially.
    If term r can be expressed as a linear combination of other terms, we can eliminate it.
    """
    start_time = time.time()
    n, m, p = 2, 4, 4
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # For each pair of terms (r1, r2), check if there's a scalar c such that
    # term_r1 - c * term_r2 is in the span of the remaining terms.
    # This is a hard problem. Let's try a simpler approach:
    # Check if any term is a scalar multiple of another (unlikely for Strassen).
    
    # Alternative: use the fact that the tensor has symmetries.
    # The 2x4*4x4 tensor might have rank < 28. Let's try to find it.
    
    # Build the full tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should be 1 for all i,j,k.
    
    # To reduce rank, we need to find a linear dependency among the rank-1 tensors
    # T_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # If sum_r alpha_r * T_r = 0 for some non-trivial alpha, we can reduce rank.
    
    # This is equivalent to finding a non-trivial solution to:
    # sum_r alpha_r * U[:,r] * V[:,r] * W[:,r] = 0 (as a tensor)
    
    # Let's try: for each column r, check if T_r is in the span of the others.
    # T_r can be represented as a vector of size n*m*p = 32 (flattened tensor).
    # But T_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Build the matrix of all T_r as vectors.
    T_vecs = np.zeros((rank, n*m*p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_vecs[r, i*m*p + j*p + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check if any row is a linear combination of others.
    # Use rank-revealing QR or SVD.
    try:
        U_qr, R_qr = np.linalg.qr(T_vecs.T)
        # R_qr is (n*m*p) x rank. The diagonal entries indicate rank.
        diag = np.diag(R_qr)
        tol = 1e-10
        num_rank = np.sum(np.abs(diag) > tol)
        if num_rank < rank:
            # There's a dependency. Find it.
            # The null space of T_vecs gives the dependency.
            # T_vecs @ alpha = 0
            U_svd, S_svd, Vt_svd = np.linalg.svd(T_vecs, full_matrices=False)
            null_space = Vt_svd[-1:]  # last row of Vt
            alpha = null_space[0]
            
            # Find the largest |alpha_r| to identify which term to eliminate.
            r_elim = np.argmax(np.abs(alpha))
            
            # Normalize: make alpha[r_elim] = 1
            if abs(alpha[r_elim]) < 1e-15:
                return U, V, W
            alpha = alpha / alpha[r_elim]
            
            # Check: sum_{r != r_elim} alpha[r] * T_r should equal T_{r_elim}
            # i.e., T_{r_elim} = sum_{r != r_elim} alpha[r] * T_r
            # This means we can replace term r_elim with the combination.
            # But we need integer/half-integer entries. Let's check if alpha is close to rational.
            
            # For now, just check if this actually reduces the rank.
            # Remove term r_elim and see if the tensor is still correct.
            # Actually, this doesn't directly give us a lower-rank decomposition.
            # We'd need to redistribute the contributions.
            
            pass
    except:
        pass
    
    return U, V, W


def _build_schoolbook():
    """Build the rank-32 schoolbook decomposition."""
    n, m, p = 2, 4, 4
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Build rank-28 Strassen decomposition
    U, V, W = _build_rank28()
    
    # Verify
    err = _verify(U, V, W)
    
    if err < 1e-9:
        # Valid rank-28 decomposition
        # Try to reduce further if time permits
        if time.time() < deadline - 20:
            U, V, W = _try_reduce(U, V, W, time_limit=min(100, deadline - time.time() - 10))
            err = _verify(U, V, W)
            if err >= 1e-9:
                # Reduction failed, go back to rank-28
                U, V, W = _build_rank28()
    else:
        # Rank-28 failed, fall back to schoolbook
        U, V, W = _build_schoolbook()
    
    # Final deadline guard
    if time.time() > deadline:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
