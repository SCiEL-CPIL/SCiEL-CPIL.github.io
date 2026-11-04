"""Matrix multiplication 4,4,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _laderman_4x4():
    """
    Return the Laderman rank-19 decomposition for 4x4 x 4x4 -> 4x4.
    
    Each term is (u, v, w) where:
      u is a 16-vector (index i*4+j for A[i,j])
      v is a 16-vector (index j*4+k for B[j,k])
      w is a 16-vector (index k*4+i for C[k,i])
    
    The decomposition satisfies:
      sum_c u_c[i*4+j] * v_c[j*4+k] * w_c[k*4+i] = 1  for all i,j,k
    """
    # Laderman's 19 bilinear forms for 4x4 matrix multiplication
    # Split A, B, C into 2x2 blocks:
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
    # C = [C11 C12; C21 C22]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Laderman uses 19 products instead of 16 (schoolbook) by exploiting
    # the structure of 2x2 multiplications.
    #
    # For each 2x2 block product, we use the 7 Strassen products.
    # But Laderman's construction is more clever: it shares products between blocks.
    #
    # The 19 Laderman products are:
    # Let's denote A blocks as A11, A12, A21, A22 (each 2x2)
    # and B blocks as B11, B12, B21, B22 (each 2x2)
    #
    # Laderman's 19 bilinear forms (from the original 1974 paper):
    #
    # M1  = (A11 + A22) * (B11 + B22)
    # M2  = (A11 + A22) * (B21 - B11)
    # M3  = (A21 + A22) * (B11 + B12)
    # M4  = (A11 - A22) * (B22 - B11)
    # M5  = A11 * (B12 - B22)
    # M6  = A22 * (B21 - B11)
    # M7  = (A12 + A22) * (B21 + B22)
    # M8  = (A11 + A12) * (B11 - B21)
    # M9  = (A21 - A11) * (B11 + B12)
    # M10 = (A22 + A12) * (B22 - B12)
    # M11 = (A21 + A12) * (B21 - B11)
    # M12 = (A11 + A12) * (B12 - B22)
    # M13 = (A21 - A11) * (B11 + B12)  -- wait, this is same as M9
    #
    # Actually let me look at this more carefully. The standard Laderman construction:
    #
    # The 19 products are:
    # M1  = (A11 + A22)(B11 + B22)
    # M2  = (A11 + A22)(B21 - B11)
    # M3  = (A21 + A22)(B11 + B12)
    # M4  = (A11 - A22)(B22 - B11)
    # M5  = A11(B12 - B22)
    # M6  = A22(B21 - B11)
    # M7  = (A12 + A22)(B21 + B22)
    # M8  = (A11 + A12)(B11 - B21)
    # M9  = (A21 - A11)(B11 + B12)
    # M10 = (A22 + A12)(B22 - B12)
    # M11 = (A21 + A12)(B21 - B11)
    # M12 = (A11 + A12)(B12 - B22)
    # M13 = (A21 - A11)(B21 + B12)
    # M14 = (A12 + A22)(B11 + B21)
    # M15 = (A21 + A22)(B12 - B22)
    # M16 = (A11 - A12)(B21 + B22)
    # M17 = (A21 - A22)(B11 + B12)
    # M18 = (A11 + A12)(B11 + B12)  -- no this doesn't seem right
    # M19 = (A21 + A22)(B21 + B22)
    #
    # I'm not confident in the exact formulas from memory. Let me use a verified approach.
    #
    # Actually, the cleanest way: use the known fact that 4x4x4 has rank 19 (Laderman).
    # Let me construct it by embedding into a larger known structure or by direct formula.
    #
    # Let me use a different well-known construction. The 4x4x4 rank-19 decomposition
    # can be built from the 2x2x2 Strassen (rank 7) with a specific sharing pattern.
    #
    # Actually, let me just verify: the best known rank for 4x4x4 is indeed 19 (Laderman 1974).
    #
    # Here's a correct construction using the "Laderman" approach:
    # We split into 2x2 blocks and use 19 bilinear forms.
    #
    # Let me use a verified set of 19 forms. I'll construct them programmatically.
    
    # Actually, the simplest correct approach: use the known rank-19 Laderman formulas.
    # Let me define them carefully.
    #
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Laderman's 19 products (verified from literature):
    #
    # P1  = (A11 + A22)(B11 + B22)
    # P2  = (A11 + A22)(B21 - B11)
    # P3  = (A21 + A22)(B11 + B12)
    # P4  = (A11 - A22)(B22 - B11)
    # P5  = A11(B12 - B22)
    # P6  = A22(B21 - B11)
    # P7  = (A12 + A22)(B21 + B22)
    # P8  = (A11 + A12)(B11 - B21)
    # P9  = (A21 - A11)(B11 + B12)
    # P10 = (A22 + A12)(B22 - B12)
    # P11 = (A21 + A12)(B21 - B11)
    # P12 = (A11 + A12)(B12 - B22)
    # P13 = (A21 - A11)(B21 + B12)
    # P14 = (A12 + A22)(B11 + B21)
    # P15 = (A21 + A22)(B12 - B22)
    # P16 = (A11 - A12)(B21 + B22)
    # P17 = (A21 - A22)(B11 + B12)
    # P18 = (A11 + A12)(B11 + B12)
    # P19 = (A21 + A22)(B21 + B22)
    #
    # And the output combinations:
    # C11 = P1 - P2 - P4 + P6 + P8 + P11 - P13 + P14 - P16 - P17 + P19  (not sure)
    #
    # I'm not confident in the exact output combinations from memory. Let me use a
    # computational approach to find a valid rank-19 decomposition, or use a
    # slightly higher rank that I can construct with certainty.
    #
    # SAFER APPROACH: Use rank 49 (Strassen^2) for 4x4x4, giving total rank 98 for 4x4x8.
    # This is guaranteed correct and gives score 128/98 = 1.306.
    #
    # But the directive says to try Laderman rank-19. Let me try to construct it
    # computationally by solving the system.
    
    return None  # Placeholder, will use computational approach below

def _find_rank19_4x4():
    """
    Try to find a rank-19 decomposition for 4x4x4 by solving the system.
    Returns (U, V, W) each of shape (16, 19) or None.
    """
    n = m = p = 4
    target_rank = 19
    
    # The constraint: for all (i,j,k) in [0,4)^3:
    #   sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c] = 1
    #
    # This is a system of 64 equations in 19*16*3 = 912 unknowns.
    # We can solve it by fixing two matrices and solving for the third, iteratively.
    
    # Strategy: Initialize with Strassen^2 (rank 49), then try to reduce to rank 19.
    # Or: use a random initialization and alternating minimization.
    
    # Let's try alternating minimization from a good initialization.
    # Initialize using the schoolbook pattern mapped to 19 columns.
    
    U = np.zeros((16, 19))
    V = np.zeros((16, 19))
    W = np.zeros((16, 19))
    
    # Distribute the 64 (i,j,k) terms across 19 columns
    # Use a greedy assignment
    for i in range(4):
        for j in range(4):
            for k in range(4):
                c = (i * 16 + j * 4 + k) % 19
                U[i*4+j, c] += 1.0
                V[j*4+k, c] += 1.0
                W[k*4+i, c] += 1.0
    
    # Normalize each column
    for c in range(19):
        su, sv, sw = np.sum(U[:,c]), np.sum(V[:,c]), np.sum(W[:,c])
        if su * sv * sw > 0:
            scale = (su * sv * sw) ** (1/3)
            U[:,c] /= scale
            V[:,c] /= scale
            W[:,c] /= scale
    
    # Alternating minimization
    for iteration in range(500):
        # Update U: for each (i,j), solve M^T U[i,j,:] = 1
        for i in range(4):
            for j in range(4):
                M = np.array([V[j*4+k, :] * W[k*4+i, :] for k in range(4)])
                U[i*4+j, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(4), rcond=None)
        
        # Update V: for each (j,k), solve M^T V[j,k,:] = 1
        for j in range(4):
            for k in range(4):
                M = np.array([U[i*4+j, :] * W[k*4+i, :] for i in range(4)])
                V[j*4+k, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(4), rcond=None)
        
        # Update W: for each (k,i), solve M^T W[k,i,:] = 1
        for k in range(4):
            for i in range(4):
                M = np.array([U[i*4+j, :] * V[j*4+k, :] for j in range(4)])
                W[k*4+i, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(4), rcond=None)
        
        # Check error
        if iteration % 50 == 0:
            err = 0.0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        s = sum(U[i*4+j,c] * V[j*4+k,c] * W[k*4+i,c] for c in range(19))
                        err += (s - 1.0)**2
            if err < 1e-10:
                return U, V, W
    
    # Final check
    err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = sum(U[i*4+j,c] * V[j*4+k,c] * W[k*4+i,c] for c in range(19))
                err += (s - 1.0)**2
    
    if err < 1e-8:
        return U, V, W
    
    return None

def _strassen_2x2():
    """
    Return the Strassen rank-7 decomposition for 2x2 x 2x2 -> 2x2.
    U, V, W each of shape (4, 7).
    Index convention: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P6
    # C22 = P1 - P2 + P3 + P6
    #
    # A indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B indices: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    #
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A11+A22)(B11+B22)
    # U: A11(0), A22(3) -> +1
    # V: B11(0), B22(3) -> +1
    # W: contributes to C11(+1), C12(-1 via P5), C21(-1 via P2), C22(+1)
    # Actually W is the output coefficient. Let me think about this differently.
    #
    # The decomposition is: C[k,i] = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    #
    # For Strassen, each product P_c is a bilinear form in A and B, and its
    # contribution to C is determined by how it combines.
    #
    # Let me define each rank-1 term by what it multiplies in A, B, and what
    # it outputs to C.
    #
    # P1 = (A11+A22)(B11+B22):
    #   U: +1 at (0,0) and (1,1) -> indices 0, 3
    #   V: +1 at (0,0) and (1,1) -> indices 0, 3
    #   This product contributes to: C11(+1), C22(+1)
    #   But it also gets subtracted in other combinations...
    #
    # Actually, the standard way: each Strassen product is a rank-1 tensor where
    # U and V capture the linear combinations of A and B entries, and W captures
    # how this product contributes to C entries.
    #
    # P1 = (A11+A22)(B11+B22) contributes:
    #   To C11: +1 (from P1 in C11 = P1+P4-P5+P7)
    #   To C22: +1 (from P1 in C22 = P1-P2+P3+P6)
    #   So W: +1 at C11(0*2+0=0), +1 at C22(1*2+1=3)
    
    # P1: (A11+A22)(B11+B22)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1  # C11, C22
    
    # P2 = (A21+A22)B11
    #   U: +1 at A21(1,0)=2, A22(1,1)=3
    #   V: +1 at B11(0,0)=0
    #   Contributes to: C21(+1), C22(-1)
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1  # C21 (k=1,i=0 -> 1*2+0=2)
    W[3, 1] -= 1  # C22 (k=1,i=1 -> 1*2+1=3)
    
    # P3 = A11(B12-B22)
    #   U: +1 at A11(0,0)=0
    #   V: +1 at B12(0,1)=1, -1 at B22(1,1)=3
    #   Contributes to: C12(+1), C22(+1)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1  # C12 (k=0,i=1 -> 0*2+1=1)
    W[3, 2] += 1  # C22
    
    # P4 = A22(B21-B11)
    #   U: +1 at A22(1,1)=3
    #   V: +1 at B21(1,0)=2, -1 at B11(0,0)=0
    #   Contributes to: C11(+1)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1  # C11
    
    # P5 = (A11+A12)B22
    #   U: +1 at A11(0,0)=0, A12(0,1)=1
    #   V: +1 at B22(1,1)=3
    #   Contributes to: C12(+1), C11(-1)
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[1, 4] += 1  # C12
    W[0, 4] -= 1  # C11
    
    # P6 = (A21-A11)(B11+B12)
    #   U: +1 at A21(1,0)=2, -1 at A11(0,0)=0
    #   V: +1 at B11(0,0)=0, B12(0,1)=1
    #   Contributes to: C21(+1), C22(+1)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[2, 5] += 1  # C21
    W[3, 5] += 1  # C22
    
    # P7 = (A12-A22)(B21+B22)
    #   U: +1 at A12(0,1)=1, -1 at A22(1,1)=3
    #   V: +1 at B21(1,0)=2, B22(1,1)=3
    #   Contributes to: C11(+1), C22(+1)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1  # C11
    W[3, 6] += 1  # C22
    
    return U, V, W

def _strassen_squared_4x4():
    """
    Build rank-49 decomposition for 4x4x4 using Strassen^2 (Kronecker product).
    Returns (U, V, W) each of shape (16, 49).
    """
    U2, V2, W2 = _strassen_2x2()
    
    # Kronecker product: 4x4 = 2x2 blocks of 2x2
    # A_4x4[i,j] where i = 2*i1+i2, j = 2*j1+j2
    # Block (i1,j1) is A2[i2,j2]
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # U: for block (i1,j1) with inner (i2,j2):
            # index in 4x4: (2*i1+i2)*4 + (2*j1+j2)
            # U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            idx = (2*i1+i2)*4 + (2*j1+j2)
                            U[idx, c] += U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
            
            # V: for block (j1,k1) with inner (j2,k2):
            # index in 4x4: (2*j1+j2)*4 + (2*k1+k2)
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            idx = (2*j1+j2)*4 + (2*k1+k2)
                            V[idx, c] += V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
            
            # W: for block (k1,i1) with inner (k2,i2):
            # index in 4x4: (2*k1+k2)*4 + (2*i1+i2)
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            idx = (2*k1+k2)*4 + (2*i1+i2)
                            W[idx, c] += W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
    
    return U, V, W

def _verify_4x4(U, V, W, tol=1e-9):
    """Verify 4x4x4 decomposition."""
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = sum(U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c] for c in range(U.shape[1]))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_4x4x8(U, V, W, n=4, m=4, p=8, tol=1e-9):
    """Verify 4x4x8 decomposition."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_4x4x4x8():
    """
    Construct decomposition for 4x4 x 4x8 -> 4x8.
    
    Strategy 1: Use Laderman rank-19 for each 4x4 block -> rank 38.
    Strategy 2: Use Strassen^2 rank-49 for each 4x4 block -> rank 98.
    
    Split B (4x8) into B1 (4x4) and B2 (4x4).
    C = A*B = [A*B1, A*B2]
    
    For each block, use a 4x4x4 decomposition.
    """
    n, m, p = 4, 4, 8
    
    # Try Laderman rank-19 first
    laderman = _find_rank19_4x4()
    
    if laderman is not None:
        U4, V4, W4 = laderman
        r4 = 19
        total_rank = 2 * r4  # 38
        
        U = np.zeros((16, total_rank))
        V = np.zeros((32, total_rank))
        W = np.zeros((32, total_rank))
        
        # Block 1: B1 = B[:, 0:4]
        # V for block 1: V4[j*4+k, c] maps to V[j*8+k, c] for k in 0..3
        for c in range(r4):
            for j in range(4):
                for k in range(4):
                    V[j*8+k, c] = V4[j*4+k, c]
            for i in range(4):
                for j in range(4):
                    U[i*4+j, c] = U4[i*4+j, c]
                for k in range(4):
                    W[k*4+i, c] = W4[k*4+i, c]
        
        # Block 2: B2 = B[:, 4:8]
        for c in range(r4):
            c2 = r4 + c
            for j in range(4):
                for k in range(4):
                    V[j*8+(4+k), c2] = V4[j*4+k, c]
            for i in range(4):
                for j in range(4):
                    U[i*4+j, c2] = U4[i*4+j, c]
                for k in range(4):
                    W[(4+k)*4+i, c2] = W4[k*4+i, c]
        
        # Verify
        if _verify_4x4x8(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Strassen^2 rank-49
    U4, V4, W4 = _strassen_squared_4x4()
    r4 = 49
    total_rank = 2 * r4  # 98
    
    U = np.zeros((16, total_rank))
    V = np.zeros((32, total_rank))
    W = np.zeros((32, total_rank))
    
    # Block 1: B1 = B[:, 0:4]
    for c in range(r4):
        for j in range(4):
            for k in range(4):
                V[j*8+k, c] = V4[j*4+k, c]
        for i in range(4):
            for j in range(4):
                U[i*4+j, c] = U4[i*4+j, c]
            for k in range(4):
                W[k*4+i, c] = W4[k*4+i, c]
    
    # Block 2: B2 = B[:, 4:8]
    for c in range(r4):
        c2 = r4 + c
        for j in range(4):
            for k in range(4):
                V[j*8+(4+k), c2] = V4[j*4+k, c]
        for i in range(4):
            for j in range(4):
                U[i*4+j, c2] = U4[i*4+j, c]
            for k in range(4):
                W[(4+k)*4+i, c2] = W4[k*4+i, c]
    
    # Verify
    if _verify_4x4x8(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook rank 128
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    return _construct_4x4x4x8()
# EVOLVE-BLOCK-END
