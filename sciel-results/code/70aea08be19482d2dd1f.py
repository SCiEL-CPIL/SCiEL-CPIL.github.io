"""Matrix multiplication 2,4,7 tensor decomposition using structured construction."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 130

def _strassen_222():
    """Return rank-7 Strassen decomposition for 2x2x2.
    U: (4,7), V: (4,7), W: (4,7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen formulas for C = A*B where A,B are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[i,j],[k,l]]
    
    # M1 = (a+d)*(e+h)
    U[0,0]=1; U[3,0]=1  # a=A[0,0]=idx0, d=A[1,1]=idx3
    V[0,0]=1; V[3,0]=1  # e=B[0,0]=idx0, h=B[1,1]=idx3
    W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=1  # i,j,k,l all +1
    
    # M2 = (c+d)*e
    U[2,1]=1; U[3,1]=1  # c=A[1,0]=idx2, d=A[1,1]=idx3
    V[0,1]=1            # e=B[0,0]=idx0
    W[1,0]=-1; W[3,0]=1  # k=-1, l=+1
    
    # M3 = a*(f-h)
    U[0,2]=1            # a=A[0,0]=idx0
    V[1,2]=1; V[3,2]=-1 # f=B[0,1]=idx1, h=B[1,1]=idx3
    W[0,0]=1; W[1,0]=1  # i=+1, j=+1
    
    # M4 = d*(g-e)
    U[3,3]=1            # d=A[1,1]=idx3
    V[2,3]=1; V[0,3]=-1 # g=B[1,0]=idx2, e=B[0,0]=idx0
    W[1,0]=1; W[3,0]=1  # j=+1, l=+1
    
    # M5 = (a+b)*h
    U[0,4]=1; U[1,4]=1  # a=A[0,0]=idx0, b=A[0,1]=idx1
    V[3,4]=1            # h=B[1,1]=idx3
    W[0,0]=-1; W[2,0]=-1 # i=-1, k=-1
    
    # M6 = (c-a)*(e+f)
    U[2,5]=1; U[0,5]=-1 # c=A[1,0]=idx2, a=A[0,0]=idx0
    V[0,5]=1; V[1,5]=1  # e=B[0,0]=idx0, f=B[0,1]=idx1
    W[2,0]=1; W[3,0]=1  # k=+1, l=+1
    
    # M7 = (b-d)*(g+h)
    U[1,6]=1; U[3,6]=-1 # b=A[0,1]=idx1, d=A[1,1]=idx3
    V[2,6]=1; V[3,6]=1  # g=B[1,0]=idx2, h=B[1,1]=idx3
    W[0,0]=-1; W[2,0]=-1 # i=-1, k=-1
    
    return U, V, W

def _schoolbook_221():
    """Return rank-4 schoolbook decomposition for 2x2x1.
    A (2x2) * B (2x1) = C (2x1)
    U: (4,4), V: (2,4), W: (2,4)
    U[i*2+j, r], V[j*1+k, r] = V[j, r] (p=1), W[k*2+i, r]
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # C[0] = a*e + b*g, C[1] = c*e + d*g
    # M1 = a*e
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    # M2 = b*g
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    # M3 = c*e
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    # M4 = d*g
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition."""
    rank = U.shape[1]
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        result += np.einsum('ij,jk,ki->ijk', U_[:,:,r], V_[:,:,r], W_[:,:,r])
    
    if np.max(np.abs(result - 1.0)) > tol:
        return False
    return True

def _add_block(U_global, V_global, W_global, offset, 
               U_block, V_block, W_block,
               i_off, j_off, k_off,
               n, m, p):
    """Add a block decomposition to global matrices.
    U_block: (n_s*m_s, rank_s) for sub-problem at rows i_off:i_off+n_s, cols j_off:j_off+m_s
    V_block: (m_s*p_s, rank_s) 
    W_block: (p_s*n_s, rank_s)
    """
    rank_s = U_block.shape[1]
    U_block_rows = U_block.shape[0]
    V_block_rows = V_block.shape[0]
    W_block_rows = W_block.shape[0]
    
    # Map local U indices to global
    # Local U: [i_s*m_s + j_s, r] -> Global: [(i_off+i_s)*m + (j_off+j_s), offset+r]
    n_s = int(np.sqrt(U_block_rows / m_s)) if False else None
    # We need to know the local dimensions
    # Actually, let's pass them explicitly
    
    return False  # This approach is getting complicated

def _construct_2x4x7_direct():
    """Directly construct a rank-50 decomposition for 2x4x7.
    
    Decomposition strategy:
    A (2x4) * B (4x7) = C (2x7)
    
    Split: A = [A1 A2] where A1, A2 are 2x2
           B = [B1 B2 B3 B4] where each Bi is 2x2, plus B5 is 2x1
           Actually B is 4x7, so split into 4x2, 4x2, 4x2, 4x1
           B = [B1 B2 B3 B4] where B1,B2,B3 are 4x2 and B4 is 4x1
    
    C = A*B = A1*[B1;B2]_top + A2*[B3;B4]_bottom ... no this doesn't work cleanly.
    
    Better: A (2x4) = [A1 | A2], A1, A2 are 2x2
    B (4x7) = [B1; B2], B1, B2 are 2x7
    
    C = A1*B1 + A2*B2
    
    Now A1*B1 is 2x2 * 2x7. Split B1 = [B1a B1b B1c B1d] where first three are 2x2, last is 2x1
    A1*B1 = A1*B1a + A1*B1b + A1*B1c + A1*B1d
    - A1*B1a: 2x2*2x2, Strassen rank 7
    - A1*B1b: 2x2*2x2, Strassen rank 7
    - A1*B1c: 2x2*2x2, Strassen rank 7
    - A1*B1d: 2x2*2x1, schoolbook rank 4
    Total for A1*B1: 7+7+7+4 = 25
    
    Similarly A2*B2: 25
    
    Total: 50
    
    Let me build this.
    """
    # Global: n=2, m=4, p=7
    # U: (8, 50), V: (28, 50), W: (14, 50)
    
    S_U, S_V, S_W = _strassen_222()   # each (4, 7)
    S2_U, S2_V, S2_W = _schoolbook_221()  # U(4,4), V(2,4), W(2,4)
    
    U = np.zeros((8, 50))
    V = np.zeros((28, 50))
    W = np.zeros((14, 50))
    
    offset = 0
    
    # Helper to add a 2x2*2x2 Strassen block
    def add_strassen(i_off, j_off, k_off, off):
        """Add Strassen for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]"""
        # U local: (i_l*2 + j_l) where i_l=0,1 (row), j_l=0,1 (col of A sub)
        # Global U index: (i_off + i_l)*4 + (j_off + j_l)  [since m=4 globally]
        # V local: (j_l*2 + k_l) where j_l=0,1 (row of B sub = col of A sub), k_l=0,1 (col of B sub)
        # Global V index: (j_off + j_l)*7 + (k_off + k_l)  [since p=7 globally]
        # W local: (k_l*2 + i_l) where k_l=0,1 (row of C sub = col of B sub), i_l=0,1 (col of C sub = row of A sub)
        # Global W index: (k_off + k_l)*2 + (i_off + i_l)  [since n=2 globally]
        
        for r in range(7):
            for il in range(2):
                for jl in range(2):
                    u_val = S_U[il*2+jl, r]
                    if u_val != 0:
                        U[(i_off+il)*4 + (j_off+jl), off+r] += u_val
            for jl in range(2):
                for kl in range(2):
                    v_val = S_V[jl*2+kl, r]
                    if v_val != 0:
                        V[(j_off+jl)*7 + (k_off+kl), off+r] += v_val
            for kl in range(2):
                for il in range(2):
                    w_val = S_W[kl*2+il, r]
                    if w_val != 0:
                        W[(k_off+kl)*2 + (i_off+il), off+r] += w_val
        return off + 7
    
    # Helper to add a 2x2*2x1 schoolbook block
    def add_school_221(i_off, j_off, k_off, off):
        """Add schoolbook for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+1]"""
        # U: same as above
        # V local: (j_l*1 + 0) = j_l, so V[j_l, r]
        # Global V index: (j_off + j_l)*7 + k_off  [k_off is the single column]
        # W local: (0*2 + i_l) = i_l, so W[i_l, r]
        # Global W index: k_off*2 + (i_off + i_l)
        
        for r in range(4):
            for il in range(2):
                for jl in range(2):
                    u_val = S2_U[il*2+jl, r]
                    if u_val != 0:
                        U[(i_off+il)*4 + (j_off+jl), off+r] += u_val
            for jl in range(2):
                v_val = S2_V[jl, r]
                if v_val != 0:
                    V[(j_off+jl)*7 + k_off, off+r] += v_val
            for il in range(2):
                w_val = S2_W[il, r]
                if w_val != 0:
                    W[k_off*2 + (i_off+il), off+r] += w_val
        return off + 4
    
    # A1 (rows 0-1 of A) * B1 (rows 0-1 of B)
    # B1 is 2x7, split into 2x2, 2x2, 2x2, 2x1
    # A1 is 2x2 (cols 0-1 of A)
    
    # A1*B1a: A[0:2, 0:2] * B[0:2, 0:2]
    offset = add_strassen(0, 0, 0, offset)  # i_off=0, j_off=0, k_off=0
    
    # A1*B1b: A[0:2, 0:2] * B[0:2, 2:4]
    offset = add_strassen(0, 0, 2, offset)
    
    # A1*B1c: A[0:2, 0:2] * B[0:2, 4:6]
    offset = add_strassen(0, 0, 4, offset)
    
    # A1*B1d: A[0:2, 0:2] * B[0:2, 6:7]
    offset = add_school_221(0, 0, 6, offset)
    
    # A2 (rows 0-1 of A, cols 2-3) * B2 (rows 2-3 of B)
    # A2 is 2x2 (cols 2-3 of A)
    # B2 is 2x7, split into 2x2, 2x2, 2x2, 2x1
    
    # A2*B2a: A[0:2, 2:4] * B[2:4, 0:2]
    offset = add_strassen(0, 2, 0, offset)
    
    # A2*B2b: A[0:2, 2:4] * B[2:4, 2:4]
    offset = add_strassen(0, 2, 2, offset)
    
    # A2*B2c: A[0:2, 2:4] * B[2:4, 4:6]
    offset = add_strassen(0, 2, 4, offset)
    
    # A2*B2d: A[0:2, 2:4] * B[2:4, 6:7]
    offset = add_school_221(0, 2, 6, offset)
    
    assert offset == 50, f"Expected offset 50, got {offset}"
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the structured rank-50 decomposition
    U, V, W = _construct_2x4x7_direct()
    
    # Verify
    if _verify_fast(U, V, W, n, m, p):
        best_U, best_V, best_W = U, V, W
        best_rank = 50
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U = np.zeros((n*m, rank_sb))
        V = np.zeros((m*p, rank_sb))
        W = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank_sb
    
    # Try to improve with random search
    if time.time() - start_time < DEADLINE - 20:
        for rank in range(best_rank - 1, 34, -1):
            if time.time() - start_time > DEADLINE - 15:
                break
            rng = np.random.RandomState(rank * 7919)
            for trial in range(30):
                if time.time() - start_time > DEADLINE - 15:
                    break
                U_r = np.zeros((n*m, rank))
                V_r = np.zeros((m*p, rank))
                W_r = np.zeros((p*n, rank))
                for i in range(n*m):
                    nz = rng.choice(rank, size=max(1,int(rank*0.4)), replace=False)
                    for c in nz:
                        U_r[i,c] = rng.choice([-1,1])
                for j in range(m*p):
                    nz = rng.choice(rank, size=max(1,int(rank*0.4)), replace=False)
                    for c in nz:
                        V_r[j,c] = rng.choice([-1,1])
                for k in range(p*n):
                    nz = rng.choice(rank, size=max(1,int(rank*0.4)), replace=False)
                    for c in nz:
                        W_r[k,c] = rng.choice([-1,1])
                if _verify_fast(U_r, V_r, W_r, n, m, p):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = rank
                    break
            if best_rank < 50:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
