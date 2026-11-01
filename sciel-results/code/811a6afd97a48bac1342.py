"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    def schoolbook():
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
    
    # Rank-32 decomposition using Strassen for 2x2x2 blocks
    # Split p=9 into blocks: 2+2+2+2+1
    # Each 2x2x2 block uses Strassen (rank 7), 2x2x1 uses schoolbook (rank 4)
    # Total: 7*4 + 4 = 32
    
    def strassen_2x2x2_block(k_offset, rank_offset):
        """Generate U, V, W entries for a 2x2x2 Strassen block starting at k_offset."""
        # Strassen's 7 products for C = A * B where A, B are 2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1 - M4 + M5
        # C01 = M3 + M4
        # C10 = M2 + M6
        # C11 = M1 - M3 + M7
        
        # For 2x2x2, the tensor T[i,j,k] = 1 for all i,j,k in {0,1}
        # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        
        # U entries (4 entries: i*m+j for i,j in {0,1})
        # V entries (4 entries: j*p+k for j,k in {0,1})
        # W entries (4 entries: k*n+i for k,i in {0,1})
        
        u_entries = []
        v_entries = []
        w_entries = []
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1; V: B00=1, B11=1; W: C00 gets +1, C11 gets +1
        u_entries.append([1, 0, 1, 0])  # A00=1, A01=0, A10=0, A11=1
        v_entries.append([1, 0, 0, 1])  # B00=1, B01=0, B10=0, B11=1
        w_entries.append([1, 0, 0, 1])  # C00=1, C01=0, C10=0, C11=1
        
        # M2 = (A10+A11)B00
        u_entries.append([0, 0, 1, 1])
        v_entries.append([1, 0, 0, 0])
        w_entries.append([0, 0, 1, 0])  # C10 gets +1
        
        # M3 = A00(B10-B11)
        u_entries.append([1, 0, 0, 0])
        v_entries.append([0, 0, 1, -1])
        w_entries.append([0, 1, 0, 0])  # C01 gets +1
        
        # M4 = A11(B01-B00)
        u_entries.append([0, 0, 0, 1])
        v_entries.append([-1, 1, 0, 0])
        w_entries.append([-1, 1, 0, 0])  # C00 gets -1, C01 gets +1
        
        # M5 = (A00+A01)B11
        u_entries.append([1, 1, 0, 0])
        v_entries.append([0, 0, 0, 1])
        w_entries.append([1, 0, 0, 0])  # C00 gets +1
        
        # M6 = (A10-A00)(B00+B10)
        u_entries.append([-1, 0, 1, 0])
        v_entries.append([1, 0, 1, 0])
        w_entries.append([0, 0, 1, 0])  # C10 gets +1
        
        # M7 = (A01-A11)(B01+B11)
        u_entries.append([0, 1, 0, -1])
        v_entries.append([0, 1, 0, 1])
        w_entries.append([0, 0, 0, 1])  # C11 gets +1
        
        return u_entries, v_entries, w_entries
    
    def schoolbook_2x2x1_block(k_offset, rank_offset):
        """Generate U, V, W entries for a 2x2x1 schoolbook block (single k)."""
        # T[i,j,k] = 1 for i,j in {0,1}, single k
        # 4 terms
        u_entries = []
        v_entries = []
        w_entries = []
        for i in range(2):
            for j in range(2):
                u_entry = [0, 0, 0, 0]
                u_entry[i * 2 + j] = 1
                v_entry = [0, 0, 0, 0]
                v_entry[j] = 1  # only one k, so index is just j
                w_entry = [0, 0, 0, 0]
                w_entry[i] = 1  # only one k, so index is just i
                u_entries.append(u_entry)
                v_entries.append(v_entry)
                w_entries.append(w_entry)
        return u_entries, v_entries, w_entries
    
    # Build rank-32 decomposition
    # Blocks: k=0,1 (Strassen), k=2,3 (Strassen), k=4,5 (Strassen), k=6,7 (Strassen), k=8 (schoolbook)
    U = [[0] * 32 for _ in range(n * m)]
    V = [[0] * 32 for _ in range(m * p)]
    W = [[0] * 32 for _ in range(p * n)]
    
    rank_offset = 0
    # 4 Strassen blocks for k=0,1; k=2,3; k=4,5; k=6,7
    for block_idx in range(4):
        k_offset = block_idx * 2
        u_entries, v_entries, w_entries = strassen_2x2x2_block(k_offset, rank_offset)
        for c, (ue, ve, we) in enumerate(zip(u_entries, v_entries, w_entries)):
            col = rank_offset + c
            # U: 2x2 matrix entries, indexed by i*m+j = i*2+j
            for idx in range(4):
                U[idx][col] = ue[idx]
            # V: 2x2 block for k_offset, k_offset+1, indexed by j*2+(k-k_offset)
            for idx in range(4):
                j = idx // 2
                k_local = idx % 2
                k_global = k_offset + k_local
                V[j * p + k_global][col] = ve[idx]
            # W: 2x2 block for k_offset, k_offset+1, indexed by k_local*n+i
            for idx in range(4):
                k_local = idx // 2
                i = idx % 2
                k_global = k_offset + k_local
                W[k_global * n + i][col] = we[idx]
        rank_offset += 7
    
    # 1 schoolbook block for k=8
    k_offset = 8
    u_entries, v_entries, w_entries = schoolbook_2x2x1_block(k_offset, rank_offset)
    for c, (ue, ve, we) in enumerate(zip(u_entries, v_entries, w_entries)):
        col = rank_offset + c
        for idx in range(4):
            U[idx][col] = ue[idx]
        # ve has length 2: ve[0] for j=0, ve[1] for j=1, all at k=k_offset
        for j in range(2):
            V[j * p + k_offset][col] = ve[j]
        # we has length 2: we[0] for i=0, we[1] for i=1, all at k=k_offset
        for i in range(2):
            W[k_offset * n + i][col] = we[i]
    rank_offset += 4
    
    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
