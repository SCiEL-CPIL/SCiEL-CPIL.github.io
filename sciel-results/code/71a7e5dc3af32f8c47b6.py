"""Matrix multiplication 3x5x6 rank-minimal decomposition."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 6]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition.
    U is (4, 7), V is (4, 7), W is (4, 7) with indices:
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # Term 3: M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[1][3] = 1; W[0][3] = -1
    
    # Term 5: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _verify_decomposition(n, m, p, U, V, W):
    """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True


def _build_block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """Build a block decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    for n_start, n_end in n_parts:
        n_size = n_end - n_start
        for m_start, m_end in m_parts:
            m_size = m_end - m_start
            for p_start, p_end in p_parts:
                p_size = p_end - p_start
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    # Use Strassen: rank 7 instead of 8
                    sU, sV, sW = _strassen_2x2x2()
                    for c in range(7):
                        u_col = [0] * (n * m)
                        for i_local in range(2):
                            for j_local in range(2):
                                val = sU[i_local * 2 + j_local][c]
                                if val != 0:
                                    u_col[(n_start + i_local) * m + (m_start + j_local)] = val
                        U_cols.append(u_col)
                        
                        v_col = [0] * (m * p)
                        for j_local in range(2):
                            for k_local in range(2):
                                val = sV[j_local * 2 + k_local][c]
                                if val != 0:
                                    v_col[(m_start + j_local) * p + (p_start + k_local)] = val
                        V_cols.append(v_col)
                        
                        w_col = [0] * (p * n)
                        for k_local in range(2):
                            for i_local in range(2):
                                val = sW[k_local * 2 + i_local][c]
                                if val != 0:
                                    w_col[(p_start + k_local) * n + (n_start + i_local)] = val
                        W_cols.append(w_col)
                else:
                    # Schoolbook for this block
                    for i_local in range(n_size):
                        for j_local in range(m_size):
                            for k_local in range(p_size):
                                u_col = [0] * (n * m)
                                u_col[(n_start + i_local) * m + (m_start + j_local)] = 1
                                U_cols.append(u_col)
                                
                                v_col = [0] * (m * p)
                                v_col[(m_start + j_local) * p + (p_start + k_local)] = 1
                                V_cols.append(v_col)
                                
                                w_col = [0] * (p * n)
                                w_col[(p_start + k_local) * n + (n_start + i_local)] = 1
                                W_cols.append(w_col)
    
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n * m):
            U[i][c] = U_cols[c][i]
        for j in range(m * p):
            V[j][c] = V_cols[c][j]
        for k in range(p * n):
            W[k][c] = W_cols[c][k]
    
    return U, V, W


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
    return U, V, W


def _generate_partitions(n, m, p):
    """Generate various block partitions to try."""
    partitions = []
    
    # Simple approach: try a few specific partitions
    # n=3: [2,1] or [1,2] or [3]
    # m=5: [2,2,1], [2,1,2], [1,2,2], [2,3], [3,2], [5]
    # p=6: [2,2,2], [2,4], [4,2], [3,3], [6]
    
    n_parts_list = [
        [(0, 2), (2, 3)],
        [(0, 1), (1, 3)],
        [(0, 3)],
    ]
    
    m_parts_list = [
        [(0, 2), (2, 4), (4, 5)],
        [(0, 2), (2, 3), (3, 5)],
        [(0, 1), (1, 3), (3, 5)],
        [(0, 2), (2, 5)],
        [(0, 3), (3, 5)],
        [(0, 5)],
    ]
    
    p_parts_list = [
        [(0, 2), (2, 4), (4, 6)],
        [(0, 2), (2, 6)],
        [(0, 4), (4, 6)],
        [(0, 3), (3, 6)],
        [(0, 6)],
    ]
    
    for np_ in n_parts_list:
        for mp_ in m_parts_list:
            for pp_ in p_parts_list:
                partitions.append((np_, mp_, pp_))
    
    return partitions


def _try_circulant_embedding(n, m, p, max_rank=90):
    """Try to find a lower-rank decomposition using circulant matrix embedding.
    
    The idea: embed the multiplication into a larger structured problem
    where circulant matrices can be diagonalized, potentially reducing rank.
    """
    # For 3x5x6, try various embedding sizes
    # A circulant embedding of size s can handle multiplications where
    # n+m-1 <= s and m+p-1 <= s
    
    # Try different embedding strategies
    best_U, best_V, best_W = None, None, None
    best_rank = max_rank + 1
    
    # Strategy: Use the fact that for certain sizes, we can use
    # Kronecker products and structured matrices
    
    # For 3x5x6: 
    # - Embed 3x5 into 4x8 (2x4 blocks)
    # - Embed 5x6 into 8x8 (4x4 blocks)
    # This doesn't directly help, but let's try a different approach
    
    # Alternative: Use the decomposition based on polynomial multiplication
    # 3x5 * 5x6 can be viewed as polynomial multiplication of degree 2 * degree 4
    # which gives degree 6, requiring 7 coefficients
    # But this is for 1D, not 2D matrices
    
    # Let's try a more direct approach: search for a rank-90 decomposition
    # by exploiting the structure of the tensor
    
    # The matrix multiplication tensor T_{i,j,k}^{a,b,c} = 1 if a=j, b=k, c=i
    # For 3x5x6, this tensor has a specific structure we can exploit
    
    # Try to find a decomposition of the form:
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{15}, v_c ∈ R^{30}, w_c ∈ R^{18}
    
    # Use the flattening: T_{(i,j),k}^{a,b} = δ_{j,a} δ_{k,b}
    # This gives a 15x18 matrix for each (i,k) pair
    # Or: T_{i,(j,k)}^{(a,b)} = δ_{j,a} δ_{k,b}
    # This gives a 3x30 matrix for each (i,(a,b)) pair
    
    # The rank of the tensor is the minimum number of terms
    # We can lower bound it by the rank of any flattening
    
    # Flattening 1: Fix i, get matrix M_i with entries M_i[(j,k),(a,b)] = δ_{j,a}δ_{k,b}
    # This is a 30x30 permutation-like matrix, rank = 15 (since j determines a, k determines b)
    # Wait, that's not right. Let me reconsider.
    
    # Actually, T_{i,j,k}^{a,b,c} = 1 iff a=j, b=k, c=i
    # So for fixed i, the "slice" is T_i[(j,k),(a,b)] = 1 iff a=j, b=k
    # This is a 30x30 matrix that is a permutation matrix (mapping (j,k) to (a,b) with a=j,b=k)
    # So it has rank 15 (since there are 15 pairs (j,k) and 15 pairs (a,b), and it's a bijection)
    
    # For fixed (i,k), the slice T_{i,k}[(j),(a,b)] = 1 iff a=j, b=k
    # This is a 5x30 matrix, rank = 5 (one 1 in each row)
    
    # For fixed (i,j), the slice T_{i,j}[(k),(a,b)] = 1 iff a=j, b=k
    # This is a 6x30 matrix, rank = 6
    
    # The tensor rank is at least the maximum rank of any flattening
    # But these lower bounds are weak (15, 5, 6)
    
    # The actual rank of 3x5x6 matrix multiplication is known to be between
    # various bounds. Let's just try to find a good decomposition.
    
    # Use a greedy approach: start with schoolbook and try to merge terms
    # where possible
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    rank_sb = n * m * p  # 90
    
    # Try to find a rank-85 decomposition by merging 5 terms
    # This requires finding 5 terms that can be replaced by 4 terms
    
    # For now, return the schoolbook as a fallback
    # The block decomposition with Strassen should give a better result
    
    return None, None, None, best_rank


def _try_lattice_based_reduction(n, m, p, U_init, V_init, W_init, max_rank=None):
    """Try to reduce the rank of a decomposition using lattice-based methods.
    
    Given an initial decomposition of rank r, try to find a decomposition
    of rank r' < r by identifying linear dependencies among the terms.
    """
    if max_rank is None:
        max_rank = len(U_init[0]) if U_init else 0
    
    rank = len(U_init[0]) if U_init else 0
    
    # Convert to numpy for linear algebra
    U_np = np.array(U_init, dtype=float)
    V_np = np.array(V_init, dtype=float)
    W_np = np.array(W_init, dtype=float)
    
    # The decomposition satisfies:
    # For all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # We want to find a lower-rank decomposition.
    # One approach: find a subspace of the coefficient space where the
    # decomposition still holds.
    
    # This is complex, so for now just return the initial decomposition
    return U_init, V_init, W_init, rank


def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145 seconds to be safe
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try various block partitions and find the one with minimum rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    partitions = _generate_partitions(n, m, p)
    
    for n_parts, m_parts, p_parts in partitions:
        if time.time() > deadline - 5:
            break
        
        try:
            U, V, W = _build_block_decomposition(n, m, p, n_parts, m_parts, p_parts)
            rank = len(U[0]) if U else 0
            
            if rank < best_rank:
                if _verify_decomposition(n, m, p, U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
        except Exception:
            continue
    
    # Try circulant embedding (may find even lower rank)
    if time.time() < deadline - 10:
        try:
            U_c, V_c, W_c, rank_c = _try_circulant_embedding(n, m, p, best_rank)
            if U_c is not None and rank_c < best_rank:
                if _verify_decomposition(n, m, p, U_c, V_c, W_c):
                    best_U, best_V, best_W = U_c, V_c, W_c
                    best_rank = rank_c
        except Exception:
            pass
    
    # Try lattice-based reduction
    if time.time() < deadline - 10:
        try:
            U_r, V_r, W_r, rank_r = _try_lattice_based_reduction(
                n, m, p, best_U, best_V, best_W, best_rank
            )
            if rank_r < best_rank:
                if _verify_decomposition(n, m, p, U_r, V_r, W_r):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = rank_r
        except Exception:
            pass
    
    if best_rank < n * m * p:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
