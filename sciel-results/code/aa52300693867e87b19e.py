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
    
    # Build target tensor entries
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.append((a, b, c))
    
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
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(r):
                        s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    def strassen2_decomp():
        """2x2x2 Strassen, rank 7"""
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
    
    def schoolbook_block(ns, ms, ps):
        """Schoolbook decomposition for ns x ms x ps block."""
        rank = ns * ms * ps
        U = [[0]*rank for _ in range(ns*ms)]
        V = [[0]*rank for _ in range(ms*ps)]
        W = [[0]*rank for _ in range(ps*ns)]
        col = 0
        for i in range(ns):
            for j in range(ms):
                for k in range(ps):
                    U[i*ms+j][col] = 1
                    V[j*ps+k][col] = 1
                    W[k*ns+i][col] = 1
                    col += 1
        return U, V, W
    
    # Strategy: Use block partitioning with Strassen for 2x2x2 blocks
    # Try multiple partitions and find the best one
    
    def try_partition(n_parts, m_parts, p_parts):
        """Try a specific partition scheme."""
        # n_parts, m_parts, p_parts are lists of sizes
        # e.g., n_parts = [2, 2] means n=4 split into 2+2
        
        n_offsets = [0]
        for sz in n_parts[:-1]:
            n_offsets.append(n_offsets[-1] + sz)
        m_offsets = [0]
        for sz in m_parts[:-1]:
            m_offsets.append(m_offsets[-1] + sz)
        p_offsets = [0]
        for sz in p_parts[:-1]:
            p_offsets.append(p_offsets[-1] + sz)
        
        # Calculate total rank
        total_rank = 0
        block_decomps = {}
        for ni, ns in enumerate(n_parts):
            for mi, ms in enumerate(m_parts):
                for pi, ps in enumerate(p_parts):
                    if ns == 2 and ms == 2 and ps == 2:
                        U_b, V_b, W_b = strassen2_decomp()
                        block_decomps[(ni, mi, pi)] = (U_b, V_b, W_b)
                        total_rank += 7
                    else:
                        U_b, V_b, W_b = schoolbook_block(ns, ms, ps)
                        block_decomps[(ni, mi, pi)] = (U_b, V_b, W_b)
                        total_rank += ns * ms * ps
        
        if total_rank >= schoolbook_rank:
            return None, total_rank
        
        # Build full matrices
        U_full = [[0] * total_rank for _ in range(nm)]
        V_full = [[0] * total_rank for _ in range(mp)]
        W_full = [[0] * total_rank for _ in range(pn)]
        
        col = 0
        for ni, ns in enumerate(n_parts):
            for mi, ms in enumerate(m_parts):
                for pi, ps in enumerate(p_parts):
                    U_b, V_b, W_b = block_decomps[(ni, mi, pi)]
                    block_rank = len(U_b[0])
                    no = n_offsets[ni]
                    mo = m_offsets[mi]
                    po = p_offsets[pi]
                    
                    for i in range(ns):
                        for j in range(ms):
                            for r in range(block_rank):
                                U_full[(no+i)*m + (mo+j)][col+r] = U_b[i*ms+j][r]
                    
                    for j in range(ms):
                        for k in range(ps):
                            for r in range(block_rank):
                                V_full[(mo+j)*p + (po+k)][col+r] = V_b[j*ps+k][r]
                    
                    for k in range(ps):
                        for i in range(ns):
                            for r in range(block_rank):
                                W_full[(po+k)*n + (no+i)][col+r] = W_b[k*ns+i][r]
                    
                    col += block_rank
        
        # Verify
        if verify(U_full, V_full, W_full):
            return {"U": U_full, "V": V_full, "W": W_full}, total_rank
        return None, total_rank
    
    # Try various partitions
    best_rank = schoolbook_rank
    best_decomp = best_result
    
    # Partition 1: n=2+2, m=2+3, p=2+4
    # 2x2x2 blocks: (0,0,0), (1,1,0) -> 2 blocks, save 2
    # Total: 120 - 2 = 118
    result, rank = try_partition([2,2], [2,3], [2,4])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 2: n=2+2, m=3+2, p=2+4
    # 2x2x2 blocks: (0,1,0), (1,0,0) -> 2 blocks, save 2
    result, rank = try_partition([2,2], [3,2], [2,4])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 3: n=2+2, m=2+3, p=4+2
    # 2x2x2 blocks: (0,0,1), (1,1,1) -> 2 blocks, save 2
    result, rank = try_partition([2,2], [2,3], [4,2])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 4: n=2+2, m=3+2, p=4+2
    # 2x2x2 blocks: (0,1,1), (1,0,1) -> 2 blocks, save 2
    result, rank = try_partition([2,2], [3,2], [4,2])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 5: n=1+3, m=2+3, p=2+4
    # 2x2x2 blocks: none (no 1x2x2 or 3x2x2 that are all 2)
    # Actually no 2x2x2 blocks here
    
    # Partition 6: n=2+2, m=1+4, p=2+4
    # 2x2x2 blocks: none (m parts are 1 and 4)
    
    # Partition 7: n=2+2, m=2+2+1, p=2+4
    # 2x2x2 blocks: (0,0,0), (1,0,0), (0,1,0), (1,1,0) -> 4 blocks, save 4
    # Total: 120 - 4 = 116
    result, rank = try_partition([2,2], [2,2,1], [2,4])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 8: n=2+2, m=2+2+1, p=4+2
    # 2x2x2 blocks: (0,0,1), (1,0,1), (0,1,1), (1,1,1) -> 4 blocks, save 4
    result, rank = try_partition([2,2], [2,2,1], [4,2])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 9: n=2+2, m=1+2+2, p=2+4
    # 2x2x2 blocks: (0,1,0), (1,1,0), (0,2,0), (1,2,0) -> 4 blocks, save 4
    result, rank = try_partition([2,2], [1,2,2], [2,4])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 10: n=2+2, m=1+2+2, p=4+2
    # 2x2x2 blocks: (0,1,1), (1,1,1), (0,2,1), (1,2,1) -> 4 blocks, save 4
    result, rank = try_partition([2,2], [1,2,2], [4,2])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 11: n=2+2, m=2+3, p=2+2+2
    # 2x2x2 blocks: (0,0,0), (1,0,0), (0,0,1), (1,0,1), (0,0,2), (1,0,2) -> 6 blocks
    # But m=2+3, so m parts are [2,3]. For mi=0, ms=2. For mi=1, ms=3.
    # 2x2x2 blocks require ms=2, so only mi=0 works
    # (0,0,0), (1,0,0), (0,0,1), (1,0,1), (0,0,2), (1,0,2) -> 6 blocks, save 6
    # Total: 120 - 6 = 114
    result, rank = try_partition([2,2], [2,3], [2,2,2])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 12: n=2+2, m=3+2, p=2+2+2
    # 2x2x2 blocks: mi=1 (ms=2): (0,1,0), (1,1,0), (0,1,1), (1,1,1), (0,1,2), (1,1,2) -> 6 blocks
    result, rank = try_partition([2,2], [3,2], [2,2,2])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 13: n=2+2, m=2+2+1, p=2+2+2
    # 2x2x2 blocks: mi=0 (ms=2), mi=1 (ms=2): 
    # (0,0,0), (1,0,0), (0,0,1), (1,0,1), (0,0,2), (1,0,2),
    # (0,1,0), (1,1,0), (0,1,1), (1,1,1), (0,1,2), (1,1,2) -> 12 blocks
    # Total: 120 - 12 = 108
    result, rank = try_partition([2,2], [2,2,1], [2,2,2])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 14: n=2+2, m=1+2+2, p=2+2+2
    # 2x2x2 blocks: mi=1 (ms=2), mi=2 (ms=2): 12 blocks
    result, rank = try_partition([2,2], [1,2,2], [2,2,2])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 15: n=2+2, m=2+2+1, p=2+2+2
    # Already tried as partition 13
    
    # Partition 16: n=2+2, m=1+1+2+1, p=2+2+2
    # 2x2x2 blocks: mi=2 (ms=2): 6 blocks
    result, rank = try_partition([2,2], [1,1,2,1], [2,2,2])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 17: n=2+2, m=2+2+1, p=2+2+2
    # Already tried as partition 13
    
    # Partition 18: n=1+1+2, m=2+2+1, p=2+2+2
    # 2x2x2 blocks: ni=2 (ns=2): 12 blocks
    result, rank = try_partition([1,1,2], [2,2,1], [2,2,2])
    if result is not None and rank < best_rank:
        best_rank = rank
        best_decomp = result
    
    # Partition 19: n=2+2, m=2+2+1, p=2+2+2
    # Already tried as partition 13
    
    # Partition 20: n=2+2, m=2+2+1, p=2+2+2
    # Already tried as partition 13
    
    # Return the best result found
    return best_decomp
# EVOLVE-BLOCK-END
