import numpy as np
import time

def _verify_solution(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] == 1 for all i,j,k.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    # Reshape to 3D tensors
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute the contraction
    # S[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
    try:
        S = np.einsum('ijs,jks,ksi->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(S, np.ones((n, m, p)), atol=tol)
    except Exception:
        return False

def _get_strassen_222():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, s], V[j*2+k, s], W[k*2+i, s]
    """
    n, m, p = 2, 2, 2
    R = 7
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Strassen's 7 factors
    # Local indices:
    # U: 0:a00, 1:a01, 2:a10, 3:a11
    # V: 0:b00, 1:b01, 2:b10, 3:b11
    # W: 0:c00, 1:c01, 2:c10, 3:c11
    
    factors = [
        # 1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # 2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # 3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # 4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # 5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # 6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # 7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    for s, (u, v, w) in enumerate(factors):
        U[:, s] = u
        V[:, s] = v
        W[:, s] = w
        
    return U, V, W

def _get_laderman_234():
    """
    Returns (U, V, W) for 2x3x4 matrix multiplication with rank 14.
    U: (6, 14), V: (12, 14), W: (8, 14)
    Indices: U[i*3+j, s], V[j*4+k, s], W[k*2+i, s]
    """
    n, m, p = 2, 3, 4
    R = 14
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Laderman's 14 factors for 2x3x4
    # Reference: Standard Laderman decomposition
    # A: 2x3 (6 elements), B: 3x4 (12 elements), C: 4x2 (8 elements)
    
    # We define the 14 factors.
    # Indices for U (A): 0:a00, 1:a01, 2:a02, 3:a10, 4:a11, 5:a12
    # Indices for V (B): 0:b00, 1:b01, 2:b02, 3:b03, 4:b10, 5:b11, 6:b12, 7:b13, 8:b20, 9:b21, 10:b22, 11:b23
    # Indices for W (C): 0:c00, 1:c01, 2:c02, 3:c03, 4:c10, 5:c11, 6:c12, 7:c13
    
    factors = [
        # 1: a00 b00 -> c00
        ([1,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0]),
        # 2: a01 b10 -> c01
        ([0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0]),
        # 3: a02 b20 -> c02
        ([0,0,1,0,0,0], [0,0,0,0,0,0,0,0,1,0,0,0], [0,0,1,0,0,0,0,0]),
        # 4: a10 b01 -> c11
        ([0,0,0,1,0,0], [0,1,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,1,0,0]),
        # 5: a11 b11 -> c12
        ([0,0,0,0,1,0], [0,0,0,0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0]),
        # 6: a12 b21 -> c13
        ([0,0,0,0,0,1], [0,0,0,0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,1]),
        # 7: (a00+a11) (b02+b12) -> c02+c12
        # Wait, let's use the verified Laderman table.
        # A common explicit set:
        # 7: (a00+a11) * (b02+b12) -> c02 + c12 ? No, dimensions.
        # Let's use a reliable source for 2x3x4 rank 14.
        # From "The arithmetic complexity of matrix multiplication" or similar.
        # Actually, let's construct it via Kronecker if we can find a simpler base, 
        # but 2x3x4 is not a Kronecker product of 2x2x2.
        # However, we can verify the factors below.
        
        # Let's use the factors from a known valid 14-rank decomposition.
        # 7: (a00+a11) (b12+b23) -> c02+c03+c12+c13 ?? 
        # Let's try to find the exact 14.
        # If I cannot guarantee the Laderman factors are correct from memory, 
        # I should fall back to a verified smaller block or schoolbook for that part.
        # But 2x3x4 rank 14 is well-known.
        # Let's use this set (verified in literature):
        # 7: (a00+a11) (b02+b12) -> c02 + c12  (Check: a00b02+a00b12+a11b02+a11b12. c02 needs a02b22? No. c02 = sum a0j b_j2. So a00b02+a01b12+a02b22. 
        # My factor 7 contributes a00b02+a00b12+a11b02+a11b12. 
        # c02 gets a00b02. c12 gets a11b12.
        # This seems to mix terms.
        
        # Let's use a safer approach: 
        # If Laderman is uncertain, use Schoolbook for 2x3x4 (rank 24) or 
        # decompose 2x3x4 into 2x2x4 + 2x1x4.
        # 2x2x4 rank? 2x2x2 (7) + 2x2x2 (7) = 14.
        # 2x1x4 rank? 8.
        # Total 22. Still better than 24.
        # But 2x3x4 rank 14 is the theoretical minimum for this size in many contexts.
        
        # To be safe and ensure a score > 1.0 (which is 96), 
        # even a suboptimal but verified decomposition is better than failing.
        # Let's implement a robust recursive builder using ONLY 2x2x2 Strassen blocks.
        # This guarantees correctness.
        # 3x4x8 can be split into:
        # 2x4x8 (using 4 Strassen blocks of 2x2x2? No, 2x4x8 = 2x2x4 + 2x2x4. 
        # 2x2x4 = 2x2x2 + 2x2x2. Rank 7+7=14.
        # So 2x4x8 = 14 + 14 = 28.
        # 1x4x8 = Schoolbook = 32.
        # Total = 60.
        
        # Wait, 2x2x4 rank is 14?
        # 2x2x4 = 2x2x2 (rank 7) + 2x2x2 (rank 7). Yes.
        # 2x4x8 = 2x2x8 + 2x2x8.
        # 2x2x8 = 4 * (2x2x2) = 4 * 7 = 28.
        # So 2x4x8 = 28 + 28 = 56.
        # 1x4x8 = 32.
        # Total = 88.
        
        # This is the same as the current program's intended 88, but derived rigorously.
        # I will implement the recursive Kronecker-style construction using only 2x2x2 blocks.
    ]
    
    # Since I am not 100% confident in the exact Laderman factors from memory without verification,
    # and the goal is a robust "different strategy", I will implement the 
    # **Recursive Strassen Composition** strategy.
    # This uses ONLY the verified 2x2x2 rank-7 block.
    # It will produce a rank-88 solution, which is better than the current 96.
    
    # I will return None to signal "use recursive builder" in the main construct.
    return None

def _construct_recursive(n, m, p, U_base, V_base, W_base, n_base, m_base, p_base):
    """
    Construct U, V, W for n x m x p using the base block n_base x m_base x p_base.
    This assumes n is a multiple of n_base, m of m_base, p of p_base.
    """
    # Determine block counts
    ni = n // n_base
    mi = m // m_base
    pi = p // p_base
    
    rank_base = U_base.shape[1]
    total_rank = (ni * mi * pi) * rank_base
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i_block in range(ni):
        for j_block in range(mi):
            for k_block in range(pi):
                # Local indices
                i_start = i_block * n_base
                j_start = j_block * m_base
                k_start = k_block * p_base
                
                # Iterate through base factors
                for s in range(rank_base):
                    # U
                    for i in range(n_base):
                        for j in range(m_base):
                            u_idx = (i_start + i) * m + (j_start + j)
                            U[u_idx, col] = U_base[i * m_base + j, s]
                            
                    # V
                    for j in range(m_base):
                        for k in range(p_base):
                            v_idx = (j_start + j) * p + (k_start + k)
                            V[v_idx, col] = V_base[j * p_base + k, s]
                            
                    # W
                    for k in range(p_base):
                        for i in range(n_base):
                            w_idx = (k_start + k) * n + (i_start + i)
                            W[w_idx, col] = W_base[k * n_base + i, s]
                            
                    col += 1
                    
    return U, V, W

def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 140
    
    # Get base 2x2x2 Strassen block
    U_base, V_base, W_base = _get_strassen_222()
    
    # Verify base
    if not _verify_solution(U_base, V_base, W_base, 2, 2, 2):
        # Fallback to schoolbook 2x2x2 (rank 8)
        U_base = np.zeros((4, 8))
        V_base = np.zeros((4, 8))
        W_base = np.zeros((4, 8))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = i*4 + j*2 + k
                    U_base[i*2+j, s] = 1
                    V_base[j*2+k, s] = 1
                    W_base[k*2+i, s] = 1
        if not _verify_solution(U_base, V_base, W_base, 2, 2, 2):
            # Last resort: return schoolbook 3x4x8
            rank = 96
            U = np.zeros((12, rank))
            V = np.zeros((32, rank))
            W = np.zeros((24, rank))
            for i in range(3):
                for j in range(4):
                    for k in range(8):
                        s = i*32 + j*8 + k
                        U[i*4+j, s] = 1
                        V[j*8+k, s] = 1
                        W[k*3+i, s] = 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Construct 2x4x8 block using 2x2x2 base
    # n=2, m=4, p=8
    # ni=1, mi=2, pi=4
    U_248, V_248, W_248 = _construct_recursive(2, 4, 8, U_base, V_base, W_base, 2, 2, 2)
    
    # Verify 2x4x8
    if not _verify_solution(U_248, V_248, W_248, 2, 4, 8):
        # Fallback to schoolbook 2x4x8
        rank_248 = 2*4*8
        U_248 = np.zeros((8, rank_248))
        V_248 = np.zeros((32, rank_248))
        W_248 = np.zeros((16, rank_248))
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    s = i*32 + j*8 + k
                    U_248[i*4+j, s] = 1
                    V_248[j*8+k, s] = 1
                    W_248[k*2+i, s] = 1

    # Construct 1x4x8 block (Schoolbook is optimal for 1xMxP)
    rank_148 = 1*4*8
    U_148 = np.zeros((4, rank_148))
    V_148 = np.zeros((32, rank_148))
    W_148 = np.zeros((24, rank_148))
    for j in range(4):
        for k in range(8):
            s = j*8 + k
            U_148[j, s] = 1
            V_148[j*8+k, s] = 1
            W_148[k*3+2, s] = 1 # i=2 for the third row
            
    # Verify 1x4x8
    if not _verify_solution(U_148, V_148, W_148, 1, 4, 8):
        # This should not happen, but handle it
        return {"U": np.zeros((12, 96)).tolist(), "V": np.zeros((32, 96)).tolist(), "W": np.zeros((24, 96)).tolist()}

    # Combine 2x4x8 and 1x4x8 into 3x4x8
    # U_total: (12, rank_248 + rank_148)
    # V_total: (32, rank_248 + rank_148)
    # W_total: (24, rank_248 + rank_148)
    
    rank_total = U_248.shape[1] + U_148.shape[1]
    
    U_total = np.zeros((12, rank_total))
    V_total = np.zeros((32, rank_total))
    W_total = np.zeros((24, rank_total))
    
    # Copy 2x4x8 part (rows 0-7 of U, all V, rows 0-15 of W)
    U_total[:8, :U_248.shape[1]] = U_248
    V_total[:, :V_248.shape[1]] = V_248
    W_total[:16, :W_248.shape[1]] = W_248
    
    # Copy 1x4x8 part (rows 8-11 of U, all V, rows 16-23 of W)
    offset = U_248.shape[1]
    U_total[8:, offset:] = U_148
    V_total[:, offset:] = V_148
    W_total[16:, offset:] = W_148
    
    # Final verification
    if _verify_solution(U_total, V_total, W_total, 3, 4, 8):
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    else:
        # Fallback to schoolbook 3x4x8
        rank = 96
        U = np.zeros((12, rank))
        V = np.zeros((32, rank))
        W = np.zeros((24, rank))
        for i in range(3):
            for j in range(4):
                for k in range(8):
                    s = i*32 + j*8 + k
                    U[i*4+j, s] = 1
                    V[j*8+k, s] = 1
                    W[k*3+i, s] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
