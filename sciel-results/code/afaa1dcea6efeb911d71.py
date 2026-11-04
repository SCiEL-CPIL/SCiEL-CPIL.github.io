<<<<<<< SEARCH
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_block(U, V, W, i0, j0, k0, col_start, n, m, p):
    """
    Add Strassen decomposition for 2x2 A-block at (i0, j0), 2x2 B-block at (j0, k0).
    A indices: i0, i0+1 ; j0, j0+1
    B indices: j0, j0+1 ; k0, k0+1
    C indices: i0, i0+1 ; k0, k0+1
    W row index for C[i,k] is k*n+i
    """
    terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]
    
    col = col_start
    for t in range(7):
        a_coeffs = terms[t][0]
        b_coeffs = terms[t][1]
        c_coeffs = terms[t][2]
        
        for di in range(2):
            for dj in range(2):
                U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]
        
        for dj in range(2):
            for dk in range(2):
                V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]
        
        for di in range(2):
            for dk in range(2):
                W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
        
        col += 1
    
    return col

def _schoolbook_block(U, V, W, i0, j0, k0, col_start, n, m, p):
    """Add schoolbook decomposition for 1x2 A-block, 2x2 B-block."""
    col = col_start
    for dj in range(2):
        for dk in range(2):
            U[i0*m + (j0+dj)][col] = 1
            V[(j0+dj)*p + (k0+dk)][col] = 1
            W[(k0+dk)*n + i0][col] = 1
            col += 1
    return col

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110

    # Strategy: Use Strassen on i-pair {0,1} for all j-pairs and k-pairs,
    # and schoolbook for i=2.
    # i-groups: {0,1} (Strassen), {2} (schoolbook)
    # j-groups: {0,1}, {2,3}
    # k-groups: {0,1}, {2,3}, {4,5}, {6,7}
    # Rank: 2*4*7 + 2*4*4 = 56 + 32 = 88

    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Strassen block for i={0,1}
            col = _strassen_block(U, V, W, 0, j0, k0, col, n, m, p)
            # Schoolbook for i=2
            col = _schoolbook_block(U, V, W, 2, j0, k0, col, n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import random

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_block(U, V, W, i0, j0, k0, col_start, n, m, p):
    """
    Add Strassen decomposition for 2x2 A-block at (i0, j0), 2x2 B-block at (j0, k0).
    A indices: i0, i0+1 ; j0, j0+1
    B indices: j0, j0+1 ; k0, k0+1
    C indices: i0, i0+1 ; k0, k0+1
    W row index for C[i,k] is k*n+i
    """
    terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]
    
    col = col_start
    for t in range(7):
        a_coeffs = terms[t][0]
        b_coeffs = terms[t][1]
        c_coeffs = terms[t][2]
        
        for di in range(2):
            for dj in range(2):
                U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]
        
        for dj in range(2):
            for dk in range(2):
                V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]
        
        for di in range(2):
            for dk in range(2):
                W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
        
        col += 1
    
    return col

def _schoolbook_block(U, V, W, i0, j0, k0, col_start, n, m, p):
    """Add schoolbook decomposition for 1x2 A-block, 2x2 B-block."""
    col = col_start
    for dj in range(2):
        for dk in range(2):
            U[i0*m + (j0+dj)][col] = 1
            V[(j0+dj)*p + (k0+dk)][col] = 1
            W[(k0+dk)*n + i0][col] = 1
            col += 1
    return col

def _build_decomposition(n, m, p, i_groups, j_groups, k_groups, use_strassen):
    """
    Build a decomposition given partitioning of indices into groups.
    i_groups: list of tuples of i indices
    j_groups: list of tuples of j indices  
    k_groups: list of tuples of k indices
    use_strassen: list of bools for each i_group (True if Strassen, False if schoolbook)
    Returns (U, V, W, rank) or None if invalid
    """
    # Calculate rank
    rank = 0
    for i_g, j_g, k_g in zip(i_groups, j_groups, k_groups):
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    if len(ig) == 2 and len(jg) == 2 and len(kg) == 2:
                        rank += 7
                    elif len(ig) == 1:
                        rank += len(jg) * len(kg)
                    elif len(jg) == 1:
                        rank += len(ig) * len(kg)
                    elif len(kg) == 1:
                        rank += len(ig) * len(jg)
                    else:
                        # Mixed case - use schoolbook
                        rank += len(ig) * len(jg) * len(kg)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                if len(ig) == 2 and len(jg) == 2 and len(kg) == 2:
                    col = _strassen_block(U, V, W, ig[0], jg[0], kg[0], col, n, m, p)
                elif len(ig) == 1:
                    # Schoolbook for single i
                    for dj in range(len(jg)):
                        for dk in range(len(kg)):
                            U[ig[0]*m + jg[dj]][col] = 1
                            V[jg[dj]*p + kg[dk]][col] = 1
                            W[kg[dk]*n + ig[0]][col] = 1
                            col += 1
                elif len(jg) == 1:
                    for di in range(len(ig)):
                        for dk in range(len(kg)):
                            U[ig[di]*m + jg[0]][col] = 1
                            V[jg[0]*p + kg[dk]][col] = 1
                            W[kg[dk]*n + ig[di]][col] = 1
                            col += 1
                elif len(kg) == 1:
                    for di in range(len(ig)):
                        for dj in range(len(jg)):
                            U[ig[di]*m + jg[dj]][col] = 1
                            V[jg[dj]*p + kg[0]][col] = 1
                            W[kg[0]*n + ig[di]][col] = 1
                            col += 1
                else:
                    # Full schoolbook
                    for di in range(len(ig)):
                        for dj in range(len(jg)):
                            for dk in range(len(kg)):
                                U[ig[di]*m + jg[dj]][col] = 1
                                V[jg[dj]*p + kg[dk]][col] = 1
                                W[kg[dk]*n + ig[di]][col] = 1
                                col += 1
    
    if _verify(U, V, W, n, m, p):
        return U, V, W, rank
    return None

def _generate_partitions(n, m, p):
    """Generate all possible partitions of indices into groups of size 1 or 2."""
    def gen_part(size):
        """Generate all partitions of [0..size-1] into groups of size 1 or 2."""
        if size == 0:
            yield []
        elif size == 1:
            yield [(0,)]
        elif size == 2:
            yield [(0, 1)]
            yield [(0,), (1,)]
        else:
            # Try putting element 0 in a singleton
            for rest in gen_part(size - 1):
                yield [(0,)] + [(x + 1 for x in g) for g in rest]
            # Try putting element 0 with element 1
            if size >= 2:
                for rest in gen_part(size - 2):
                    yield [(0, 1)] + [(x + 2 for x in g) for g in rest]
    
    i_parts = list(gen_part(n))
    j_parts = list(gen_part(m))
    k_parts = list(gen_part(p))
    
    return i_parts, j_parts, k_parts

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # First try the known good decomposition (rank 88)
    U, V, W, rank = _build_decomposition(
        n, m, p,
        [(0, 1), (2,)],  # i_groups
        [(0, 1), (2, 3)],  # j_groups
        [(0, 1), (2, 3), (4, 5), (6, 7)],  # k_groups
        [True, False]
    )
    if U is not None:
        return {"U": U, "V": W, "W": W}
    
    # Search for better decompositions
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    i_parts, j_parts, k_parts = _generate_partitions(n, m, p)
    
    # Generate all combinations of partitions
    candidates = []
    for ip in i_parts:
        for jp in j_parts:
            for kp in k_parts:
                # Calculate potential rank
                rank = 0
                for ig in ip:
                    for jg in jp:
                        for kg in kp:
                            if len(ig) == 2 and len(jg) == 2 and len(kg) == 2:
                                rank += 7
                            elif len(ig) == 1:
                                rank += len(jg) * len(kg)
                            elif len(jg) == 1:
                                rank += len(ig) * len(kg)
                            elif len(kg) == 1:
                                rank += len(ig) * len(jg)
                            else:
                                rank += len(ig) * len(jg) * len(kg)
                candidates.append((rank, ip, jp, kp))
    
    # Sort by rank (ascending)
    candidates.sort(key=lambda x: x[0])
    
    for rank, ip, jp, kp in candidates:
        if rank >= best_rank:
            break
        if time.time() > deadline - 10:
            break
        
        result = _build_decomposition(n, m, p, ip, jp, kp, [False]*len(ip))
        if result is not None:
            U, V, W, actual_rank = result
            if actual_rank < best_rank:
                best_rank = actual_rank
                best_result = {"U": U, "V": V, "W": W}
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
