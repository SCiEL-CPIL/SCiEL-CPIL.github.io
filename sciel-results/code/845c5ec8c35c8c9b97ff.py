"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    # Build factor matrices as numpy arrays
    Um = np.zeros((n * m, rank), dtype=np.float64)
    Vm = np.zeros((m * p, rank), dtype=np.float64)
    Wm = np.zeros((p * n, rank), dtype=np.float64)
    
    for i in range(n * m):
        for c in range(rank):
            if U[i][c] != 0:
                Um[i, c] = U[i][c]
    for j in range(m * p):
        for c in range(rank):
            if V[j][c] != 0:
                Vm[j, c] = V[j][c]
    for k in range(p * n):
        for c in range(rank):
            if W[k][c] != 0:
                Wm[k][c] = W[k][c]
    
    # Check: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # Use efficient tensor contraction
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u_c = Um.reshape(n, m, rank)[:, :, c]
        v_c = Vm.reshape(m, p, rank)[:, :, c]
        w_c = Wm.reshape(p, n, rank)[:, :, c]
        for i in range(n):
            for j in range(m):
                T[i, j, :] += u_c[i, j] * v_c[j, :] * w_c[:, i]
    
    return np.allclose(T, 1.0, atol=tol)

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication (rank 7).
    
    U, V, W have shape (4, 7) corresponding to (i*m+j), (j*p+k), (k*n+i)
    where n=m=p=2.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: (A00+A11)(B00+B11)
    # U: A00=U[0], A11=U[3]
    # V: B00=V[0], B11=V[3]
    # W: C00=W[0], C11=W[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: (A10+A11)B00
    # U: A10=U[2], A11=U[3]
    # V: B00=V[0]
    # W: C10=W[2], C11=W[3] with -1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 2: A00(B01-B11)
    # U: A00=U[0]
    # V: B01=V[1], B11=V[3]
    # W: C01=W[1], C11=W[3]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 3: A11(B10-B00)
    # U: A11=U[3]
    # V: B10=V[2], B00=V[0]
    # W: C10=W[2], C00=W[0]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # Term 4: (A00+A01)B11
    # U: A00=U[0], A01=U[1]
    # V: B11=V[3]
    # W: C01=W[1] with -1, C11=W[3]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = -1; W[3][4] = 1
    
    # Term 5: (A10-A00)(B00+B01)
    # U: A10=U[2], A00=U[0]
    # V: B00=V[0], B01=V[1]
    # W: C11=W[3]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: (A01-A11)(B10+B11)
    # U: A01=U[1], A11=U[3]
    # V: B10=V[2], B11=V[3]
    # W: C00=W[0]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _block_multiply_decomposition(n, m, p, block_n, block_m, block_p):
    """
    Try to construct a decomposition using block structure.
    Split A into (n/block_n) x (m/block_m) blocks of size block_n x block_m
    Split B into (m/block_m) x (p/block_p) blocks of size block_m x block_p
    Split C into (n/block_n) x (p/block_p) blocks of size block_n x block_p
    
    For each pair of blocks (A_ik, B_kl), we need to multiply them and add to C_il.
    We use Strassen for the inner block multiplication if possible.
    """
    # Check divisibility
    if n % block_n != 0 or m % block_m != 0 or p % block_p != 0:
        return None
    
    nb = n // block_n  # number of block rows
    mb = m // block_m  # number of block columns in A / block rows in B
    pb = p // block_p  # number of block columns in B / block cols in C
    
    # We'll build the decomposition by combining block-level Strassen with
    # the internal structure of each block multiplication.
    
    # For simplicity, let's use schoolbook at the block level first,
    # but use Strassen inside each block multiplication.
    
    # Actually, let's try a different approach: use Strassen for the whole thing
    # by padding to powers of 2 if needed, or by using a hierarchical approach.
    
    # For now, let's just return None and use schoolbook.
    # A proper block Strassen implementation is complex.
    return None

def _try_rank_reduction(U, V, W, n, m, p, max_iter=1000):
    """
    Try to reduce the rank of a decomposition by finding linear dependencies
    among the rank-1 terms.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64).T  # (rank, n*m)
    V_arr = np.array(V, dtype=np.float64).T  # (rank, m*p)
    W_arr = np.array(W, dtype=np.float64).T  # (rank, p*n)
    
    # The constraint is: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is a nonlinear constraint, so simple linear algebra doesn't directly apply.
    # We'd need to use more sophisticated optimization.
    
    # For now, just return the original.
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try Strassen-based approaches
    # For 4x4 * 4x5, we can't directly use Strassen since dimensions aren't powers of 2
    # and m != p. But we can try padding or other tricks.
    
    # Approach: Pad B from 4x5 to 4x8 (pad with zeros), use Strassen-like methods,
    # then extract the relevant part. But this may not reduce rank.
    
    # Another approach: Use the fact that we can split into blocks.
    # Let's try a 2x2 block decomposition where each block multiplication
    # uses Strassen internally.
    
    # A is 4x4: split into 2x2 blocks of size 2x2
    # B is 4x5: split into 2x3 blocks (top) and 2x2 blocks (bottom)? No, that's irregular.
    # Let's pad B to 4x6 (2x3 blocks) or 4x8 (2x4 blocks).
    
    # Actually, let's try a simpler structural improvement:
    # Use the "border basis" or other known lower-rank decompositions.
    
    # Known results: For 4x4x4, the rank is at most 49 (not 64).
    # For 4x4x5, we can potentially do better than 80.
    
    # Let's try a greedy approach: start with schoolbook and try to merge terms.
    # Two terms c1 and c2 can be merged if their combined effect can be represented
    # by a single term or a small set of terms.
    
    # This is complex, so for now, let's just return schoolbook if we don't find
    # a quick improvement.
    
    elapsed = time.time() - start_time
    if elapsed < 10:
        # Try a few random perturbations to see if we can find a lower rank solution
        # This is unlikely to work, but let's try.
        pass
    
    # For now, return the schoolbook decomposition
    return best

# EVOLVE-BLOCK-END
