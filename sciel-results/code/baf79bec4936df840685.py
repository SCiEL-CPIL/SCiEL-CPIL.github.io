"""Matrix multiplication 2,2,15 tensor rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 120

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    """Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_rank53():
    """Build the rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_2x2x8_strassen():
    """Build rank-28 decomposition for 2x2x8 using four independent Strassen pairs."""
    n, m, p = 2, 2, 8
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    
    for pair in range(4):
        col_offset = pair * 7
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            V[j * p + k0, col_offset:col_offset+7] = V_str[j * 2 + 0, :]
            V[j * p + k1, col_offset:col_offset+7] = V_str[j * 2 + 1, :]
        for i in range(n):
            W[k0 * n + i, col_offset:col_offset+7] = W_str[0 * 2 + i, :]
            W[k1 * n + i, col_offset:col_offset+7] = W_str[1 * 2 + i, :]
    
    return U, V, W

def _build_2x2x7_schoolbook():
    """Build rank-28 schoolbook decomposition for 2x2x7."""
    n, m, p = 2, 2, 7
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _search_2x2x7_low_rank(deadline):
    """
    Search for a decomposition of 2x2x7 with rank < 28.
    Uses structured integer pattern search.
    Returns (U, V, W) if found with rank < 28, else None.
    """
    n, m, p = 2, 2, 7
    best_rank = 28
    best_result = None
    
    # Try rank 20-27
    for target_rank in range(20, 28):
        if time.time() > deadline - 5:
            break
        
        # Try different block decompositions of 7
        # 7 = 2+2+2+1, 7 = 3+2+2, 7 = 3+4, 7 = 4+3, 7 = 5+2, 7 = 6+1
        block_configs = [
            [2, 2, 2, 1],  # 3*7 + 4 = 25
            [3, 2, 2],     # Need rank(2x2x3) + 7 + 7
            [3, 4],        # Need rank(2x2x3) + rank(2x2x4)
            [4, 3],        # Need rank(2x2x4) + rank(2x2x3)
            [5, 2],        # Need rank(2x2x5) + 7
            [6, 1],        # Need rank(2x2x6) + 4
        ]
        
        for config in block_configs:
            if time.time() > deadline - 5:
                break
            
            # Calculate expected rank from known decompositions
            # rank(2x2x2) = 7, rank(2x2x1) = 4
            # rank(2x2x3) <= 11 (7+4), rank(2x2x4) <= 14 (2*7)
            # rank(2x2x5) <= 18 (7+7+4), rank(2x2x6) <= 21 (3*7)
            
            expected_rank = 0
            valid = True
            for bsize in config:
                if bsize == 1:
                    expected_rank += 4
                elif bsize == 2:
                    expected_rank += 7
                elif bsize == 3:
                    expected_rank += 11  # 7 + 4
                elif bsize == 4:
                    expected_rank += 14  # 2 * 7
                elif bsize == 5:
                    expected_rank += 18  # 7 + 7 + 4
                elif bsize == 6:
                    expected_rank += 21  # 3 * 7
                else:
                    valid = False
                    break
            
            if not valid or expected_rank >= best_rank:
                continue
            
            # Build the decomposition
            U_block = np.zeros((4, expected_rank))
            V_block = np.zeros((14, expected_rank))
            W_block = np.zeros((14, expected_rank))
            
            col_offset = 0
            k_offset = 0
            for bsize in config:
                if bsize == 1:
                    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
                    U_block[:, col_offset:col_offset+4] = U_sb
                    for j in range(2):
                        V_block[j * 7 + k_offset, col_offset:col_offset+4] = V_sb[j, :]
                    for i in range(2):
                        W_block[k_offset * 2 + i, col_offset:col_offset+4] = W_sb[i, :]
                    col_offset += 4
                    k_offset += 1
                elif bsize == 2:
                    U_str, V_str, W_str = _build_strassen_2x2x2()
                    U_block[:, col_offset:col_offset+7] = U_str
                    for j in range(2):
                        for kl in range(2):
                            V_block[j * 7 + k_offset + kl, col_offset:col_offset+7] = V_str[j * 2 + kl, :]
                    for kl in range(2):
                        for i in range(2):
                            W_block[(k_offset + kl) * 2 + i, col_offset:col_offset+7] = W_str[kl * 2 + i, :]
                    col_offset += 7
                    k_offset += 2
                elif bsize == 3:
                    # Use Strassen for first 2 + schoolbook for last 1
                    U_str, V_str, W_str = _build_strassen_2x2x2()
                    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
                    U_block[:, col_offset:col_offset+7] = U_str
                    for j in range(2):
                        for kl in range(2):
                            V_block[j * 7 + k_offset + kl, col_offset:col_offset+7] = V_str[j * 2 + kl, :]
                    for kl in range(2):
                        for i in range(2):
                            W_block[(k_offset + kl) * 2 + i, col_offset:col_offset+7] = W_str[kl * 2 + i, :]
                    col_offset += 7
                    k_offset += 2
                    U_block[:, col_offset:col_offset+4] = U_sb
                    for j in range(2):
                        V_block[j * 7 + k_offset, col_offset:col_offset+4] = V_sb[j, :]
                    for i in range(2):
                        W_block[k_offset * 2 + i, col_offset:col_offset+4] = W_sb[i, :]
                    col_offset += 4
                    k_offset += 1
                elif bsize == 4:
                    # Two Strassen blocks
                    U_str, V_str, W_str = _build_strassen_2x2x2()
                    for pair in range(2):
                        po = col_offset + pair * 7
                        ko = k_offset + pair * 2
                        U_block[:, po:po+7] = U_str
                        for j in range(2):
                            for kl in range(2):
                                V_block[j * 7 + ko + kl, po:po+7] = V_str[j * 2 + kl, :]
                        for kl in range(2):
                            for i in range(2):
                                W_block[(ko + kl) * 2 + i, po:po+7] = W_str[kl * 2 + i, :]
                    col_offset += 14
                    k_offset += 4
                elif bsize == 5:
                    # Two Strassen + one schoolbook
                    U_str, V_str, W_str = _build_strassen_2x2x2()
                    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
                    for pair in range(2):
                        po = col_offset + pair * 7
                        ko = k_offset + pair * 2
                        U_block[:, po:po+7] = U_str
                        for j in range(2):
                            for kl in range(2):
                                V_block[j * 7 + ko + kl, po:po+7] = V_str[j * 2 + kl, :]
                        for kl in range(2):
                            for i in range(2):
                                W_block[(ko + kl) * 2 + i, po:po+7] = W_str[kl * 2 + i, :]
                    col_offset += 14
                    k_offset += 4
                    U_block[:, col_offset:col_offset+4] = U_sb
                    for j in range(2):
                        V_block[j * 7 + k_offset, col_offset:col_offset+4] = V_sb[j, :]
                    for i in range(2):
                        W_block[k_offset * 2 + i, col_offset:col_offset+4] = W_sb[i, :]
                    col_offset += 4
                    k_offset += 1
                elif bsize == 6:
                    # Three Strassen blocks
                    U_str, V_str, W_str = _build_strassen_2x2x2()
                    for pair in range(3):
                        po = col_offset + pair * 7
                        ko = k_offset + pair * 2
                        U_block[:, po:po+7] = U_str
                        for j in range(2):
                            for kl in range(2):
                                V_block[j * 7 + ko + kl, po:po+7] = V_str[j * 2 + kl, :]
                        for kl in range(2):
                            for i in range(2):
                                W_block[(ko + kl) * 2 + i, po:po+7] = W_str[kl * 2 + i, :]
                    col_offset += 21
                    k_offset += 6
            
            # Verify
            if col_offset != expected_rank:
                continue
            
            # Check V and W dimensions
            if V_block.shape[0] != 14 or W_block.shape[0] != 14:
                continue
            
            if _verify_decomposition(U_block, V_block, W_block, 2, 2, 7):
                best_rank = expected_rank
                best_result = (U_block, V_block, W_block)
                break
        
        if best_result is not None and best_rank <= target_rank:
            break
    
    return best_result

def _build_combined_7_8(n, m, p, U7, V7, W7, rank7):
    """Build combined decomposition: 2x2x7 block + 2x2x8 block."""
    U8, V8, W8 = _build_2x2x8_strassen()
    rank8 = 28
    total_rank = rank7 + rank8
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # First 7 columns of k: use the 2x2x7 decomposition
    U[:, :rank7] = U7
    for j in range(m):
        for k in range(7):
            V[j * p + k, :rank7] = V7[j * 7 + k, :]
    for k in range(7):
        for i in range(n):
            W[k * n + i, :rank7] = W7[k * 2 + i, :]
    
    # Last 8 columns of k: use the 2x2x8 Strassen decomposition
    U[:, rank7:] = U8
    for j in range(m):
        for k in range(8):
            V[j * p + (7 + k), rank7:] = V8[j * 8 + k, :]
    for k in range(8):
        for i in range(n):
            W[(7 + k) * n + i, rank7:] = W8[k * 2 + i, :]
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 10
    n, m, p = DIMENSIONS
    
    # Baseline: rank 53
    U, V, W = _build_rank53()
    best_rank = 53
    best_U, best_V, best_W = U, V, W
    
    # Verify baseline
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
        best_rank = 60
        best_U, best_V, best_W = U, V, W
    
    # Try 7+8 decomposition
    if time.time() < deadline:
        # First try the known 7+8 with schoolbook for 7 (rank 28+28=56, not better)
        # Then search for better 2x2x7
        result7 = _search_2x2x7_low_rank(deadline)
        if result7 is not None:
            U7, V7, W7 = result7
            rank7 = U7.shape[1]
            U_comb, V_comb, W_comb = _build_combined_7_8(n, m, p, U7, V7, W7, rank7)
            if _verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
                total_rank = rank7 + 28
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_U, best_V, best_W = U_comb, V_comb, W_comb
    
    # Check time
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
