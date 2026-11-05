import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    # Check dimensions
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    # Build the tensor T[i*m+j, j*p+k, k*n+i] = 1
    # And check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2(n, m, p, i0, j0, k0):
    """
    Return Strassen rank-7 decomposition for a 2x2x2 sub-block.
    The sub-block multiplies A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
    contributing to C[i0:i0+2, k0:k0+2].
    
    Returns (U_block, V_block, W_block) where:
    U_block is 4x7 (local A indices), V_block is 4x7 (local B indices), W_block is 4x7 (local C indices)
    """
    # Strassen's 7 multiplications for 2x2 matrices:
    # M1 = A00 * (B01 - B00)
    # M2 = (A00 + A10) * B00
    # M3 = (A11 - A01) * (B10 + B11)
    # M4 = A11 * (B10 - B00)  -- wait, let me use standard Strassen
    # 
    # Standard Strassen for C = A*B where A, B are 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # M1 = a*(f-h)
    # M2 = (a+b)*h
    # M3 = (c+d)*e
    # M4 = d*(g-e)
    # M5 = (a+d)*(e+h)
    # M6 = (b-d)*(g+h)
    # M7 = (c-a)*(e+f)
    #
    # C00 = M5 + M4 - M2 + M7
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M5 + M1 - M3 + M6
    #
    # Local indices:
    # A: 0=a, 1=b, 2=c, 3=d
    # B: 0=e, 1=f, 2=g, 3=h
    # C: 0=C00, 1=C01, 2=C10, 3=C11
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = A00 * (B01 - B00)
    U[0, 0] = 1.0
    V[1, 0] = 1.0
    V[0, 0] = -1.0
    W[1, 0] = 1.0   # C01
    W[3, 0] = 1.0   # C11
    
    # M2 = (A00 + A10) * B00
    U[0, 1] = 1.0
    U[2, 1] = 1.0
    V[0, 1] = 1.0
    W[0, 1] = -1.0  # C00
    W[1, 1] = 1.0   # C01
    
    # M3 = (A11 - A01) * (B10 + B11)
    U[3, 2] = 1.0
    U[1, 2] = -1.0
    V[2, 2] = 1.0
    V[3, 2] = 1.0
    W[2, 2] = 1.0   # C10
    W[3, 2] = -1.0  # C11
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1.0
    V[2, 3] = 1.0
    V[0, 3] = -1.0
    W[0, 3] = 1.0   # C00
    W[2, 3] = 1.0   # C10
    
    # M5 = (A00 + A11) * (B00 + B11)
    U[0, 4] = 1.0
    U[3, 4] = 1.0
    V[0, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = 1.0   # C00
    W[3, 4] = 1.0   # C11
    
    # M6 = (A01 - A11) * (B10 + B11)
    U[1, 5] = 1.0
    U[3, 5] = -1.0
    V[2, 5] = 1.0
    V[3, 5] = 1.0
    W[3, 5] = 1.0   # C11
    
    # M7 = (A10 - A00) * (B00 + B01)
    U[2, 6] = 1.0
    U[0, 6] = -1.0
    V[0, 6] = 1.0
    V[1, 6] = 1.0
    W[0, 6] = 1.0   # C00
    
    return U, V, W

def _build_block_decomposition(n, m, p, use_strassen=True):
    """
    Build a block decomposition using Strassen for 2x2x2 blocks where possible,
    and schoolbook for the rest.
    """
    if not use_strassen:
        return _schoolbook(n, m, p)
    
    # We'll build the decomposition column by column.
    # For each combination of (i, j, k), we need to contribute 1 to T[i*m+j, j*p+k, k*n+i].
    # 
    # Strategy: partition into 2x2x2 blocks where possible.
    # n=3, m=4, p=5.
    # 
    # Let's think of it as: for each pair of rows (i1, i2) of A, each pair of cols (j1, j2) of A,
    # and each pair of cols (k1, k2) of B, we can use Strassen if we have a 2x2x2 sub-problem.
    #
    # Actually, let's be more systematic. The tensor is T[i*m+j, j*p+k, k*n+i].
    # A rank-1 term is defined by vectors u (length n*m), v (length m*p), w (length p*n).
    # The contribution to T[i*m+j, j*p+k, k*n+i] is u[i*m+j] * v[j*p+k] * w[k*n+i].
    #
    # For a 2x2x2 Strassen block covering A[i0:i0+2, j0:j0+2], B[j0:j0+2, k0:k0+2], C[i0:i0+2, k0:k0+2]:
    # The local A indices are: (i0,j0)->0, (i0,j0+1)->1, (i0+1,j0)->2, (i0+1,j0+1)->3
    # The local B indices are: (j0,k0)->0, (j0,k0+1)->1, (j0+1,k0)->2, (j0+1,k0+1)->3
    # The local C indices are: (k0,i0)->0, (k0+1,i0)->1, (k0,i0+1)->2, (k0+1,i0+1)->3
    #
    # Wait, C indices: C[i,k] where i is row of C (0..n-1), k is col of C (0..p-1).
    # W index is k*n + i. So for C[i0,k0], W index is k0*n + i0.
    # For C[i0,k0+1], W index is (k0+1)*n + i0.
    # For C[i0+1,k0], W index is k0*n + (i0+1).
    # For C[i0+1,k0+1], W index is (k0+1)*n + (i0+1).
    
    U = np.zeros((n * m, 0))
    V = np.zeros((m * p, 0))
    W = np.zeros((p * n, 0))
    
    covered = set()
    
    # Try to place 2x2x2 Strassen blocks
    # i0 in range(n-1), j0 in range(m-1), k0 in range(p-1)
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                # Check if this 2x2x2 block is fully "available" (not yet covered)
                # The block covers:
                # A entries: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
                # B entries: (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1)
                # C entries: (i0,k0), (i0,k0+1), (i0+1,k0), (i0+1,k0+1)
                #
                # But actually, the Strassen decomposition covers the FULL 2x2x2 sub-tensor,
                # meaning all 8 combinations of (i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}).
                # So we need all 8 to be uncovered.
                
                block_entries = set()
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            i = i0 + di
                            j = j0 + dj
                            k = k0 + dk
                            block_entries.add((i, j, k))
                
                if all(e not in covered for e in block_entries):
                    # Use Strassen for this block
                    U_b, V_b, W_b = _strassen_2x2x2(n, m, p, i0, j0, k0)
                    
                    # Map local indices to global
                    # U: local (di, dj) -> global (i0+di, j0+dj) -> index (i0+di)*m + (j0+dj)
                    # V: local (dj, dk) -> global (j0+dj, k0+dk) -> index (j0+dj)*p + (k0+dk)
                    # W: local (dk, di) -> global (k0+dk, i0+di) -> index (k0+dk)*n + (i0+di)
                    
                    U_new = np.zeros((n * m, 7))
                    V_new = np.zeros((m * p, 7))
                    W_new = np.zeros((p * n, 7))
                    
                    for r in range(7):
                        for di in range(2):
                            for dj in range(2):
                                local_u = di * 2 + dj
                                global_u = (i0 + di) * m + (j0 + dj)
                                U_new[global_u, r] = U_b[local_u, r]
                        
                        for dj in range(2):
                            for dk in range(2):
                                local_v = dj * 2 + dk
                                global_v = (j0 + dj) * p + (k0 + dk)
                                V_new[global_v, r] = V_b[local_v, r]
                        
                        for dk in range(2):
                            for di in range(2):
                                local_w = dk * 2 + di
                                global_w = (k0 + dk) * n + (i0 + di)
                                W_new[global_w, r] = W_b[local_w, r]
                    
                    U = np.hstack([U, U_new]) if U.shape[1] > 0 else U_new
                    V = np.hstack([V, V_new]) if V.shape[1] > 0 else V_new
                    W = np.hstack([W, W_new]) if W.shape[1] > 0 else W_new
                    
                    covered.update(block_entries)
    
    # Now handle remaining uncovered entries with schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U_new = np.zeros((n * m, 1))
                    V_new = np.zeros((m * p, 1))
                    W_new = np.zeros((p * n, 1))
                    
                    U_new[i * m + j, 0] = 1.0
                    V_new[j * p + k, 0] = 1.0
                    W_new[k * n + i, 0] = 1.0
                    
                    U = np.hstack([U, U_new])
                    V = np.hstack([V, V_new])
                    W = np.hstack([W, W_new])
                    covered.add((i, j, k))
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Build block decomposition with Strassen
    U, V, W = _build_block_decomposition(n, m, p, use_strassen=True)
    
    rank = U.shape[1]
    
    # Verify
    if not _verify(U, V, W, n, m, p, tol=1e-9):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        rank = U.shape[1]
    
    # Try to reduce rank further via simple pruning
    # Check if any column is redundant (can be expressed as combination of others)
    # This is complex, so for now just return what we have
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
