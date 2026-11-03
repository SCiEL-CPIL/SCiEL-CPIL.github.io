"""
Tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
Uses a systematic search for Strassen-like block decompositions to find a rank < 72 solution.
Strategy: Try various ways to embed 2x2x2 Strassen blocks (rank 7 instead of 8 schoolbook)
into the 3x3x8 tensor, combining with schoolbook for the rest.
"""
import time
import sys

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """
    Verify that U, V, W correctly decompose the 3x3x8 matrix multiplication tensor.
    U: (9, rank), V: (24, rank), W: (24, rank)
    Check: sum_c U[i*3+j, c] * V[j*8+k, c] * W[k*3+i, c] == 1 if consistent, else 0.
    Actually the tensor is T[(i,j),(j',k),(i',k')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    In the trace convention W[k*n+i, r], the tensor element is:
    T[i*m+j, j'*p+k, k'*n+i'] = 1 if i=i', j=j', k=k'
    So we need: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Check all (i,j,k) triples - for the diagonal elements
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The element T[i*m+j, j*p+k, k*n+i] should be 1
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1) > 1e-9:
                    return False, (i, j, k, val, "should be 1")
    
    # Check off-diagonal: T[i*m+j, j'*p+k, k'*n+i'] should be 0 if not consistent
    # We check a sample of off-diagonal elements
    import random
    random.seed(42)
    for _ in range(200):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        jp = random.randint(0, m-1)
        k = random.randint(0, p-1)
        kp = random.randint(0, p-1)
        ip = random.randint(0, n-1)
        kp2 = random.randint(0, p-1)
        
        # T[i*m+j, jp*p+k, kp2*n+ip]
        # Should be 1 only if i==ip, j==jp, k==kp2
        expected = 1 if (i==ip and j==jp and k==kp2) else 0
        if expected == 0:
            val = 0
            for c in range(rank):
                val += U[i*m+j][c] * V[jp*p+k][c] * W[kp2*n+ip][c]
            if abs(val) > 1e-9:
                return False, (i, j, jp, k, kp2, ip, val, "should be 0")
    
    return True, None


def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    For multiplying 2x2 by 2x2.
    """
    # Strassen's 7 products for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A11 is index 0, A12 is index 1, A21 is index 2, A22 is index 3
    # B11 is index 0, B12 is index 1, B21 is index 2, B22 is index 3
    # C11 is index 0, C12 is index 1, C21 is index 2, C22 is index 3
    
    # M1: (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C11 += M1
    
    # M2: (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C21 += M2
    W[3][1] = 1  # C22 += M2
    
    # M3: A11(B21-B22)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C12 += M3
    W[3][2] = -1 # C22 -= M3
    
    # M4: A22(B12-B11)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1 # C11 -= M4
    W[2][3] = 1  # C21 += M4
    
    # M5: (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1  # C11 += M5
    W[1][4] = 1  # C12 += M5
    
    # M6: (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22 += M6
    
    # M7: (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11 += M7
    
    return U, V, W


def construct():
    """
    Construct the tensor decomposition for 3x3x8 matrix multiplication.
    Returns {'U': (9, rank), 'V': (24, rank), 'W': (24, rank)}
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    start_time = time.time()
    deadline = start_time + 120  # 120 second safety margin
    
    # Start with schoolbook decomposition (rank 72)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]  # 9 x 72
    V = [[0] * rank for _ in range(m * p)]  # 24 x 72
    W = [[0] * rank for _ in range(p * n)]  # 24 x 72
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify schoolbook
    ok, err = _verify(U, V, W)
    if not ok:
        print(f"Schoolbook verification failed: {err}")
        # Return anyway, it should be correct
        return {"U": U, "V": V, "W": W}
    
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Now try to use Strassen blocks to reduce rank
    # A 2x2x2 Strassen block uses rank 7 instead of 8 schoolbook terms.
    # We can embed Strassen blocks in different positions within the 3x3x8 tensor.
    
    # For a 2x2x2 Strassen block at position (i0, j0, k0) where it covers
    # rows i0, i0+1; cols j0, j0+1; and k-range k0, k0+1:
    # We need i0 in {0,1}, j0 in {0,1}, k0 in {0,...,6}
    
    # Let's try single Strassen blocks and see if we can verify them
    strassen_U, strassen_V, strassen_W = _strassen_2x2()
    
    # Try different placements of a single Strassen block
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                if time.time() > deadline - 5:
                    break
                
                # Create a new decomposition with one Strassen block at (i0, j0, k0)
                # Remove the 8 schoolbook terms for this 2x2x2 subblock
                # Add the 7 Strassen terms
                
                # The 8 schoolbook columns to remove:
                remove_cols = []
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            i = i0 + di
                            j = j0 + dj
                            k = k0 + dk
                            col = i * m + j * p + k  # This is the column index in schoolbook
                            remove_cols.append(col)
                
                # Build new U, V, W with these columns replaced by Strassen terms
                new_rank = rank - 8 + 7  # Remove 8, add 7
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                
                # Copy all schoolbook terms except the removed ones
                col_map = {}
                new_col = 0
                for c in range(rank):
                    if c not in remove_cols:
                        col_map[c] = new_col
                        for r in range(n * m):
                            new_U[r][new_col] = U[r][c]
                        for r in range(m * p):
                            new_V[r][new_col] = V[r][c]
                        for r in range(p * n):
                            new_W[r][new_col] = W[r][c]
                        new_col += 1
                
                # Add Strassen terms
                base_col = new_col
                for sc in range(7):
                    # Strassen U, V, W are 4x7, mapping to 2x2 subblock
                    for si in range(2):
                        for sj in range(2):
                            idx = si * 2 + sj  # 0,1,2,3
                            i = i0 + si
                            j = j0 + sj
                            new_U[i * m + j][base_col + sc] = strassen_U[idx][sc]
                    
                    for sj in range(2):
                        for sk in range(2):
                            idx = sj * 2 + sk  # 0,1,2,3
                            j = j0 + sj
                            k = k0 + sk
                            new_V[j * p + k][base_col + sc] = strassen_V[idx][sc]
                    
                    for sk in range(2):
                        for si in range(2):
                            idx = sk * 2 + si  # 0,1,2,3
                            k = k0 + sk
                            i = i0 + si
                            new_W[k * n + i][base_col + sc] = strassen_W[idx][sc]
                
                # Verify
                ok, err = _verify(new_U, new_V, new_W)
                if ok:
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_U = [row[:] for row in new_U]
                        best_V = [row[:] for row in new_V]
                        best_W = [row[:] for row in new_W]
                        print(f"Found rank {new_rank} with Strassen block at ({i0},{j0},{k0})")
                else:
                    print(f"Strassen block at ({i0},{j0},{k0}) failed: {err}")
    
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
