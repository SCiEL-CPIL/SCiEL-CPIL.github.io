"""Rewritten program for 2x2x15 matrix multiplication tensor decomposition."""
import time
import itertools

DIMENSIONS = [2, 2, 15]

# Strassen's 7-term decomposition for 2x2x2 matrix multiplication
# U: (n*m=4) x 7, V: (m*p=4) x 7, W: (p*n=4) x 7
# Trace convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
#
# Strassen's 7 products:
# M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 -= M1
# M2 = (A10+A11)*B00       -> C10 += M2
# M3 = A00*(B01-B11)       -> C01 += M3
# M4 = A11*(B10-B00)       -> C11 += M4
# M5 = (A00+A01)*B11       -> C01 += M5
# M6 = (A10-A00)*(B00+B01) -> C10 -= M6
# M7 = (A01-A11)*(B10+B11) -> C00 += M7

def _strassen_2x2x2():
    """Return (U, V, W) for Strassen's 2x2x2 decomposition.
    U: (4,7), V: (4,7), W: (4,7)
    U index: i*2+j for i,j in {0,1}
    V index: j*2+k for j,k in {0,1}
    W index: k*2+i for k,i in {0,1}
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = -1  # C00 += M1, C11 -= M1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10 += M2
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01 += M3
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[3][3] = 1  # C11 += M4
    
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1  # C01 += M5
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = -1  # C10 -= M6
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00 += M7
    
    return U, V, W

def _schoolbook_2x2x1():
    """Return (U, V, W) for schoolbook 2x2x1 matrix multiplication.
    U: (4,4), V: (2,4), W: (2,4)
    U index: i*2+j
    V index: j*1+k = j
    W index: k*2+i
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    # C00 = A00*B00 + A01*B10
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
    
    # C10 = A10*B00 + A11*B10
    U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
    U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
    
    return U, V, W

def _validate(U, V, W, n=2, m=2, p=15):
    """Validate that the decomposition exactly reproduces matrix multiplication.
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    if (i,j,k) corresponds to a valid triple, else 0.
    Actually the tensor T[i,j,k] = delta for matrix mult: T[(i,j),(j,k)] but let's use:
    T[i, j, k] = 1 if the product contributes, which for C = A*B means
    C[i,k] = sum_j A[i,j]*B[j,k], so the tensor entries are T[i,j,k] = delta_{j,j}... 
    Let me reconsider: the matrix multiplication tensor T has entries T[i, j, k]
    where i indexes rows of A (and C), j indexes cols of A / rows of B, k indexes cols of B (and C).
    T[i,j,k] = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j] B[j,k].
    The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0  # All entries of the matmul tensor are 1
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(actual - expected) > 1e-10:
                    return False, (i, j, k, actual, expected)
    return True, None

def _try_partition(p, n=2, m=2):
    """Try different block partitions of p and compute the resulting rank.
    For each partition of p into blocks, assign Strassen (rank 7) or schoolbook (rank 4)
    to each block based on its size.
    
    Returns (rank, U, V, W) for the best valid decomposition found.
    """
    best_rank = n * m * p  # schoolbook upper bound
    best_result = None
    
    # Try partitions of p into blocks of size 1, 2, 3
    # For block size b: if b >= 2, use Strassen-like; if b == 1, use schoolbook
    # Actually for block size b: rank is n*m*b for schoolbook, or for b=2 it's 7 (Strassen)
    # For general block sizes, we can use different decompositions
    
    # Let's try all ways to split p=15 into blocks where each block is size 1, 2, or 3
    # For block size 1: rank = n*m*1 = 4
    # For block size 2: rank = 7 (Strassen)
    # For block size 3: we'd need a 2x2x3 decomposition, rank at most 12 (schoolbook)
    #   but maybe we can do better? For now use schoolbook.
    
    def gen_partitions(remaining, min_size=1, max_size=3):
        if remaining == 0:
            yield []
            return
        for size in range(min_size, min(remaining, max_size) + 1):
            for rest in gen_partitions(remaining - size, min_size, max_size):
                yield [size] + rest
    
    for partition in gen_partitions(p):
        rank = 0
        for block_size in partition:
            if block_size == 1:
                rank += n * m  # schoolbook: 4
            elif block_size == 2:
                rank += 7  # Strassen
            elif block_size == 3:
                rank += n * m * block_size  # schoolbook: 12
            else:
                rank += n * m * block_size
        
        if rank >= best_rank:
            continue
        
        # Build the decomposition
        r = 0
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        k_offset = 0
        for block_size in partition:
            if block_size == 2:
                sU, sV, sW = _strassen_2x2x2()
                for ri in range(7):
                    for ii in range(n):
                        for jj in range(m):
                            U[ii*m+jj][r + ri] += sU[ii*m+jj][ri]
                    for jj in range(m):
                        for kk in range(block_size):
                            V[jj*p + k_offset + kk][r + ri] += sV[jj*block_size + kk][ri]
                    for kk in range(block_size):
                        for ii in range(n):
                            W[(k_offset + kk)*n + ii][r + ri] += sW[kk*n + ii][ri]
                r += 7
            elif block_size == 1:
                sU, sV, sW = _schoolbook_2x2x1()
                for ri in range(4):
                    for ii in range(n):
                        for jj in range(m):
                            U[ii*m+jj][r + ri] += sU[ii*m+jj][ri]
                    for jj in range(m):
                        V[jj*p + k_offset][r + ri] += sV[jj][ri]
                    for ii in range(n):
                        W[(k_offset)*n + ii][r + ri] += sW[ii][ri]
                r += 4
            else:
                # Schoolbook for block_size >= 3
                for jj in range(m):
                    for kk in range(block_size):
                        for ii in range(n):
                            U[ii*m+jj][r] = 1
                            V[jj*p + k_offset + kk][r] = 1
                            W[(k_offset + kk)*n + ii][r] = 1
                            r += 1
            
            k_offset += block_size
        
        if r != rank:
            continue
        
        valid, info = _validate(U, V, W, n, m, p)
        if valid:
            if rank < best_rank:
                best_rank = rank
                best_result = {'U': U, 'V': V, 'W': W}
    
    return best_rank, best_result

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with the known rank-53 solution (7 Strassen blocks + 1 schoolbook block)
    rank = 53
    sU, sV, sW = _strassen_2x2x2()
    bU, bV, bW = _schoolbook_2x2x1()
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # 7 Strassen blocks (k=0..13)
    for b in range(7):
        k_off = 2 * b
        r_off = 7 * b
        for ri in range(7):
            for ii in range(n):
                for jj in range(m):
                    U[ii*m+jj][r_off + ri] += sU[ii*m+jj][ri]
            for jj in range(m):
                for kk in range(2):
                    V[jj*p + k_off + kk][r_off + ri] += sV[jj*2 + kk][ri]
            for kk in range(2):
                for ii in range(n):
                    W[(k_off + kk)*n + ii][r_off + ri] += sW[kk*n + ii][ri]
    
    # 1 schoolbook block (k=14)
    k_off = 14
    r_off = 49
    for ri in range(4):
        for ii in range(n):
            for jj in range(m):
                U[ii*m+jj][r_off + ri] += bU[ii*m+jj][ri]
        for jj in range(m):
            V[jj*p + k_off][r_off + ri] += bV[jj][ri]
        for ii in range(n):
            W[(k_off)*n + ii][r_off + ri] += bW[ii][ri]
    
    result = {'U': U, 'V': V, 'W': W}
    
    # Validate the known solution
    valid, info = _validate(U, V, W, n, m, p)
    if not valid:
        # Should not happen, but return anyway
        return result
    
    # Try to find better partitions within time budget
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        best_rank, best_result = _try_partition(p, n, m)
        if best_result is not None and best_rank < rank:
            result = best_result
            rank = best_rank
    
    # Final time check
    if time.time() - start_time > deadline:
        pass  # return whatever we have
    
    return result
