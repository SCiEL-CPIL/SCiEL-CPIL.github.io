"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24

    def schoolbook():
        """Standard schoolbook decomposition with rank = n*m*p = 120."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return U, V, W, rank

    def verify(U, V, W, rank):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    def strassen_2x2x2():
        """Standard Strassen 2x2x2 decomposition, rank 7.
        
        Returns (U, V, W) where:
        U has shape (4, 7) - A indices: (0,0), (0,1), (1,0), (1,1)
        V has shape (4, 7) - B indices: (0,0), (0,1), (1,0), (1,1)
        W has shape (4, 7) - C indices: (0,0), (0,1), (1,0), (1,1)
        
        The convention is C[k,i] = sum_j A[i,j] * B[j,k]
        So A index = (i,j), B index = (j,k), C index = (k,i)
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Index mapping: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i
        # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        # M1 = (A00+A11)(B00+B11): C00 += M1, C11 += M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B10: C10 += M2
        U[2][1] = 1; U[3][1] = 1
        V[2][1] = 1
        W[2][1] = 1
        
        # M3 = A00(B01-B11): C01 += M3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4 = A11(B10-B00): C00 += M4, C10 += M4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00+A01)B11: C00 -= M5, C01 += M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B01): C11 += M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11): C00 += M7, C11 += M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W

    def build_hybrid_decomposition():
        """Build a decomposition using Strassen for 2x2x2 sub-blocks.
        
        Strategy: Partition the 4x5x6 multiplication into blocks and use
        Strassen where the block is 2x2x2, schoolbook otherwise.
        
        Partition:
        - n: [0,2) and [2,4)
        - m: [0,2), [2,4), [4,5)
        - p: [0,2), [2,4), [4,6)
        
        For each combination of (n_block, m_block, p_block):
        - If all three blocks have size 2, use Strassen (rank 7 instead of 8)
        - Otherwise, use schoolbook for that sub-block
        """
        U_list = []  # (a_index, value)
        V_list = []  # (b_index, value)
        W_list = []  # (c_index, value)
        
        # Block boundaries
        n_blocks = [(0, 2), (2, 4)]
        m_blocks = [(0, 2), (2, 4), (4, 5)]
        p_blocks = [(0, 2), (2, 4), (4, 6)]
        
        for ni, (n_start, n_end) in enumerate(n_blocks):
            for mi, (m_start, m_end) in enumerate(m_blocks):
                for pi, (p_start, p_end) in enumerate(p_blocks):
                    n_size = n_end - n_start
                    m_size = m_end - m_start
                    p_size = p_end - p_start
                    
                    # Check if this is a 2x2x2 block
                    if n_size == 2 and m_size == 2 and p_size == 2:
                        # Use Strassen
                        U_s, V_s, W_s = strassen_2x2x2()
                        for r in range(7):
                            for u_idx, val in enumerate(U_s):
                                if val != 0:
                                    a = (n_start + u_idx // 2) * m + (m_start + u_idx % 2)
                                    U_list.append((a, val))
                            for v_idx, val in enumerate(V_s):
                                if val != 0:
                                    b = (m_start + v_idx // 2) * p + (p_start + v_idx % 2)
                                    V_list.append((b, val))
                            for w_idx, val in enumerate(W_s):
                                if val != 0:
                                    c = (p_start + w_idx // 2) * n + (n_start + w_idx % 2)
                                    W_list.append((c, val))
                    else:
                        # Use schoolbook for this sub-block
                        for i in range(n_start, n_end):
                            for j in range(m_start, m_end):
                                for k in range(p_start, p_end):
                                    a = i * m + j
                                    b = j * p + k
                                    c = k * n + i
                                    U_list.append((a, 1))
                                    V_list.append((b, 1))
                                    W_list.append((c, 1))
        
        # Build the final matrices
        rank = len(U_list)
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        for r in range(rank):
            a, u_val = U_list[r]
            U[a][r] = u_val
            b, v_val = V_list[r]
            V[b][r] = v_val
            c, w_val = W_list[r]
            W[c][r] = w_val
        
        return U, V, W, rank

    def build_block_decomposition():
        """Build decomposition using more aggressive blocking.
        
        Try different partitions to find the minimum rank.
        """
        best_rank = float('inf')
        best_U = best_V = best_W = None
        
        # Try different partitions
        partitions = [
            # (n_splits, m_splits, p_splits)
            ([0, 2, 4], [0, 2, 4, 5], [0, 2, 4, 6]),
            ([0, 2, 4], [0, 3, 5], [0, 3, 6]),
            ([0, 1, 2, 3, 4], [0, 2, 4, 5], [0, 2, 4, 6]),
            ([0, 2, 4], [0, 1, 2, 3, 4, 5], [0, 2, 4, 6]),
            ([0, 2, 4], [0, 2, 4, 5], [0, 1, 2, 3, 4, 5, 6]),
        ]
        
        for n_splits, m_splits, p_splits in partitions:
            if time.time() > deadline - 30:
                break
            
            U_list = []
            V_list = []
            W_list = []
            
            for ni in range(len(n_splits) - 1):
                n_start, n_end = n_splits[ni], n_splits[ni + 1]
                n_size = n_end - n_start
                
                for mi in range(len(m_splits) - 1):
                    m_start, m_end = m_splits[mi], m_splits[mi + 1]
                    m_size = m_end - m_start
                    
                    for pi in range(len(p_splits) - 1):
                        p_start, p_end = p_splits[pi], p_splits[pi + 1]
                        p_size = p_end - p_start
                        
                        if n_size == 2 and m_size == 2 and p_size == 2:
                            # Use Strassen
                            U_s, V_s, W_s = strassen_2x2x2()
                            for r in range(7):
                                for u_idx, val in enumerate(U_s):
                                    if val != 0:
                                        a = (n_start + u_idx // 2) * m + (m_start + u_idx % 2)
                                        U_list.append((a, val))
                                for v_idx, val in enumerate(V_s):
                                    if val != 0:
                                        b = (m_start + v_idx // 2) * p + (p_start + v_idx % 2)
                                        V_list.append((b, val))
                                for w_idx, val in enumerate(W_s):
                                    if val != 0:
                                        c = (p_start + w_idx // 2) * n + (n_start + w_idx % 2)
                                        W_list.append((c, val))
                        else:
                            # Use schoolbook
                            for i in range(n_start, n_end):
                                for j in range(m_start, m_end):
                                    for k in range(p_start, p_end):
                                        a = i * m + j
                                        b = j * p + k
                                        c = k * n + i
                                        U_list.append((a, 1))
                                        V_list.append((b, 1))
                                        W_list.append((c, 1))
            
            rank = len(U_list)
            if rank < best_rank:
                best_rank = rank
                U = [[0] * rank for _ in range(nm)]
                V = [[0] * rank for _ in range(mp)]
                W = [[0] * rank for _ in range(pn)]
                
                for r in range(rank):
                    a, u_val = U_list[r]
                    U[a][r] = u_val
                    b, v_val = V_list[r]
                    V[b][r] = v_val
                    c, w_val = W_list[r]
                    W[c][r] = w_val
                
                best_U, best_V, best_W = U, V, W
        
        if best_U is not None:
            return best_U, best_V, best_W, best_rank
        else:
            return None, None, None, float('inf')

    # Try the hybrid decomposition first
    try:
        U, V, W, rank = build_hybrid_decomposition()
        if verify(U, V, W, rank):
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
        else:
            best_result = None
            best_rank = float('inf')
    except Exception:
        best_result = None
        best_rank = float('inf')

    # Try the block decomposition with different partitions
    try:
        U, V, W, rank = build_block_decomposition()
        if U is not None and rank < best_rank:
            if verify(U, V, W, rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
    except Exception:
        pass

    # Fallback to schoolbook
    if best_result is None:
        U, V, W, rank = schoolbook()
        if not verify(U, V, W, rank):
            raise ValueError("Schoolbook decomposition failed verification")
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank

    return best_result
# EVOLVE-BLOCK-END
