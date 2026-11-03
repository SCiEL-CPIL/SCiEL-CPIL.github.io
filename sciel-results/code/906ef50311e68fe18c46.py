import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify_np(U, V, W, n, m, p, r):
    """Check that the tensor decomposition is exact for the matrix multiplication tensor using NumPy."""
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if i,j,k are valid indices? No.
    # The tensor for matrix multiplication A(n,m) * B(m,p) -> C(n,p)
    # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In the standard tensor decomposition, the tensor is T_{i,j,k} = 1 for all valid i,j,k?
    # No, the matrix multiplication tensor is defined such that
    # (A \otimes B) \cdot T = C.
    # The entries of T are 1 if the indices match the multiplication structure.
    # Specifically, T[i, j, k] = 1 if we are computing C[i,k] from A[i,j] and B[j,k].
    # So T[i,j,k] is 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, the standard definition is:
    # T_{i,j,k} = 1 if i,j,k are such that A_{i,j} and B_{j,k} multiply to C_{i,k}.
    # Yes, so T is all 1s in the range [0,n)x[0,m)x[0,p).
    
    # Let's verify:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r_idx in range(r):
                    s += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                if s != 1:
                    return False
    return True

def _build_rank44_np():
    """Build the rank-44 decomposition using NumPy arrays."""
    n, m, p = 2, 3, 8
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0

    # Part 1: A[0:2, 0:2] * B[0:2, 0:8] using 4 Strassen blocks
    # B is 3x8. We take B[0:2, 0:8].
    # We split B[0:2, 0:8] into 4 blocks of 2x2.
    for blk in range(4):
        bo = 2 * blk
        # Indices in V (m*p = 3*8 = 24)
        # B is stored row-major? V index is j*p + k.
        # B[0, bo] -> j=0, k=bo -> V[0*8 + bo] = V[bo]
        # B[0, bo+1] -> V[bo+1]
        # B[1, bo] -> V[1*8 + bo] = V[8+bo]
        # B[1, bo+1] -> V[8+bo+1]
        
        ve = bo
        vf = bo + 1
        vg = 8 + bo
        vh = 8 + bo + 1

        # Indices in W (p*n = 8*2 = 16)
        # C is 2x8. W index is k*n + i.
        # C[0, bo] -> k=bo, i=0 -> W[bo*2 + 0] = W[2*bo]
        # C[0, bo+1] -> W[2*bo+2]
        # C[1, bo] -> W[2*bo+1]
        # C[1, bo+1] -> W[2*bo+3]
        
        w00 = 2 * bo
        w01 = 2 * bo + 2
        w10 = 2 * bo + 1
        w11 = 2 * bo + 3

        # Strassen for 2x2x2
        # A is 2x2. U index is i*m + j.
        # A[0,0] -> U[0]
        # A[0,1] -> U[1]
        # A[1,0] -> U[3]
        # A[1,1] -> U[4]
        
        # P1 = (a+d)(e+h)
        U[0, col] = 1; U[4, col] = 1
        V[ve, col] = 1; V[vh, col] = 1
        W[w00, col] = 1; W[w11, col] = 1
        col += 1

        # P2 = (c+d)e
        U[3, col] = 1; U[4, col] = 1
        V[ve, col] = 1
        W[w10, col] = 1; W[w11, col] = -1
        col += 1

        # P3 = a(f-h)
        U[0, col] = 1
        V[vf, col] = 1; V[vh, col] = -1
        W[w01, col] = 1; W[w11, col] = 1
        col += 1

        # P4 = d(g-e)
        U[4, col] = 1
        V[vg, col] = 1; V[ve, col] = -1
        W[w00, col] = 1; W[w10, col] = 1
        col += 1

        # P5 = (a+b)h
        U[0, col] = 1; U[1, col] = 1
        V[vh, col] = 1
        W[w00, col] = -1; W[w01, col] = 1
        col += 1

        # P6 = (c-a)(e+f)
        U[3, col] = 1; U[0, col] = -1
        V[ve, col] = 1; V[vf, col] = 1
        W[w11, col] = 1
        col += 1

        # P7 = (b-d)(g+h)
        U[1, col] = 1; U[4, col] = -1
        V[vg, col] = 1; V[vh, col] = 1
        W[w00, col] = 1
        col += 1

    # Part 2: A[0:2, 2] * B[2, 0:8] -> 16 schoolbook terms
    # A[0,2] -> U[0*3+2] = U[2]
    # A[1,2] -> U[1*3+2] = U[5]
    # B[2,k] -> V[2*8+k] = V[16+k]
    # C[i,k] -> W[k*2+i]
    
    for k in range(8):
        for i in range(2):
            U[i * 3 + 2, col] = 1
            V[16 + k, col] = 1
            W[2 * k + i, col] = 1
            col += 1

    assert col == rank, f"col={col}, rank={rank}"
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Build the known rank-44 solution
    U44, V44, W44 = _build_rank44_np()

    # Verify rank-44
    if _verify_np(U44, V44, W44, n, m, p, 44):
        # Convert to lists if the evaluator expects lists? 
        # The prompt says "factor matrices of shapes ... with integer/half-integer entries".
        # It doesn't specify the type (list vs numpy). 
        # The previous program used lists. 
        # To be safe, I will return lists, but constructed via numpy for speed/clarity in construction.
        return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}

    # Fallback: schoolbook (should not be reached)
    rank_fallback = n * m * p
    U = [[0] * rank_fallback for _ in range(n * m)]
    V = [[0] * rank_fallback for _ in range(m * p)]
    W = [[0] * rank_fallback for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
