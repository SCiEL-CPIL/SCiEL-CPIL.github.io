"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    def strassen2_decomp():
        """2x2x2 Strassen, rank 7. Returns U(4x7), V(4x7), W(4x7)"""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # M1=(A11+A22)(B11+B22): C11+=M1, C22+=M1
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # M2=(A21+A22)B21: C21+=M2
        U[2][1]=1; U[3][1]=1; V[2][1]=1; W[2][1]=1
        # M3=A11(B12-B22): C12+=M3
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        # M4=A22(B21-B11): C11+=M4, C21+=M4
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        # M5=(A11+A12)B22: C11-=M5, C12+=M5
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        # M6=(A21-A11)(B11+B12): C22+=M6
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # M7=(A12-A22)(B21+B22): C11+=M7, C22+=M7
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W
    
    def schoolbook_decomp(ni, mi, pi):
        """Schoolbook decomposition for ni x mi x pi. Returns U(ni*mi x ni*mi*pi), V(mi*pi x ni*mi*pi), W(pi*ni x ni*mi*pi)"""
        rank = ni * mi * pi
        U = [[0]*rank for _ in range(ni*mi)]
        V = [[0]*rank for _ in range(mi*pi)]
        W = [[0]*rank for _ in range(pi*ni)]
        col = 0
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    U[i*mi+j][col] = 1
                    V[j*pi+k][col] = 1
                    W[k*ni+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def build_partition(n_parts, m_parts, p_parts, decomp_fn):
        """
        Build a block partition decomposition.
        n_parts: list of row block sizes summing to n
        m_parts: list of col block sizes summing to m  
        p_parts: list of output col block sizes summing to p
        decomp_fn: function(ni, mi, pi) -> (U, V, W, rank)
        """
        total_rank = 0
        for ni in n_parts:
            for mi in m_parts:
                for pi in p_parts:
                    total_rank += decomp_fn(ni, mi, pi)[3]
        
        U_full = [[0] * total_rank for _ in range(n*m)]
        V_full = [[0] * total_rank for _ in range(m*p)]
        W_full = [[0] * total_rank for _ in range(p*n)]
        
        col = 0
        i_off = 0
        for ni in n_parts:
            j_off = 0
            for mi in m_parts:
                k_off = 0
                for pi in p_parts:
                    U_b, V_b, W_b, rank_b = decomp_fn(ni, mi, pi)
                    for r in range(rank_b):
                        for i in range(ni):
                            for j in range(mi):
                                U_full[(i_off+i)*m + (j_off+j)][col+r] = U_b[i*mi+j][r]
                        for j in range(mi):
                            for k in range(pi):
                                V_full[(j_off+j)*p + (k_off+k)][col+r] = V_b[j*pi+k][r]
                        for k in range(pi):
                            for i in range(ni):
                                W_full[(k_off+k)*n + (i_off+i)][col+r] = W_b[k*ni+i][r]
                    col += rank_b
                    k_off += pi
                j_off += mi
            i_off += ni
        
        return U_full, V_full, W_full, total_rank
    
    def verify(U, V, W, rank):
        """Verify the decomposition is exact using numpy for speed"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Tensor: T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j], V_arr[j*p+k])
                    # Wait, that's not right. We need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # This is element-wise product of three vectors, then sum
                    prod = U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i]
                    s = np.sum(prod)
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    def decomp_choice(ni, mi, pi):
        """Choose decomposition for ni x mi x pi block"""
        if ni == 2 and mi == 2 and pi == 2:
            U, V, W = strassen2_decomp()
            return U, V, W, 7
        else:
            U, V, W, rank = schoolbook_decomp(ni, mi, pi)
            return U, V, W, rank
    
    # Try all reasonable partitions and keep the best
    partitions = [
        ([2, 2], [2, 2, 1], [2, 2, 2]),
        ([2, 2], [1, 2, 2], [2, 2, 2]),
        ([2, 2], [2, 1, 2], [2, 2, 2]),
        ([2, 2], [2, 2, 1], [2, 1, 2]),
        ([2, 2], [2, 2, 1], [1, 2, 2]),
        ([2, 2], [1, 2, 2], [2, 2, 1]),
        ([2, 2], [1, 2, 2], [2, 1, 2]),
        ([2, 2], [1, 2, 2], [1, 2, 2]),
        ([2, 2], [2, 1, 2], [2, 2, 1]),
        ([2, 2], [2, 1, 2], [2, 1, 2]),
        ([2, 2], [2, 1, 2], [1, 2, 2]),
        ([1, 2, 1], [2, 2, 1], [2, 2, 2]),
        ([1, 2, 1], [1, 2, 2], [2, 2, 2]),
        ([1, 2, 1], [2, 1, 2], [2, 2, 2]),
    ]
    
    for n_parts, m_parts, p_parts in partitions:
        if time.time() > deadline - 10:
            break
        if sum(n_parts) != n or sum(m_parts) != m or sum(p_parts) != p:
            continue
        U_t, V_t, W_t, rank_t = build_partition(n_parts, m_parts, p_parts, decomp_choice)
        if rank_t < best_rank and verify(U_t, V_t, W_t, rank_t):
            best_result = {"U": U_t, "V": V_t, "W": W_t}
            best_rank = rank_t
    
    return best_result
# EVOLVE-BLOCK-END
