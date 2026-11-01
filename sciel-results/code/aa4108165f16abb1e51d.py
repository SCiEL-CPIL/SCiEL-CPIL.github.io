"""Matrix multiplication tensor decomposition for 2x5 by 5x5 using block decomposition."""
DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    block_sizes_m = [2, 2, 1]
    block_sizes_p = [2, 2, 1]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    total_rank = 0
    for a in range(3):
        for b in range(3):
            sa = block_sizes_m[a]
            sb = block_sizes_p[b]
            if sa == 2 and sb == 2:
                total_rank += 7
            elif sa == 2 and sb == 1:
                total_rank += 4
            elif sa == 1 and sb == 2:
                total_rank += 4
            else:
                total_rank += 2
    
    rank = total_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_offset = 0
    
    for a in range(3):
        for b in range(3):
            sa = block_sizes_m[a]
            sb = block_sizes_p[b]
            m_off = m_offsets[a]
            p_off = p_offsets[b]
            
            if sa == 2 and sb == 2:
                # Strassen for 2x2 by 2x2
                # Local indices: A[0:2][0:2], B[0:2][0:2], C[0:2][0:2]
                # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
                r = r_offset
                U[0*m + m_off+0][r] = 1
                U[1*m + m_off+1][r] = 1
                V[(m_off+0)*p + p_off+0][r] = 1
                V[(m_off+1)*p + p_off+1][r] = 1
                W[(p_off+0)*n + 0][r] = 1
                W[(p_off+1)*n + 1][r] = 1
                r_offset += 1
                
                # M2 = (A10+A11)B00 -> C10+=M2
                r = r_offset
                U[1*m + m_off+0][r] = 1
                U[1*m + m_off+1][r] = 1
                V[(m_off+0)*p + p_off+0][r] = 1
                W[(p_off+0)*n + 1][r] = 1
                r_offset += 1
                
                # M3 = A00(B01-B11) -> C01+=M3
                r = r_offset
                U[0*m + m_off+0][r] = 1
                V[(m_off+0)*p + p_off+1][r] = 1
                V[(m_off+1)*p + p_off+1][r] = -1
                W[(p_off+1)*n + 0][r] = 1
                r_offset += 1
                
                # M4 = A11(B10-B00) -> C00+=M4, C10+=M4
                r = r_offset
                U[1*m + m_off+1][r] = 1
                V[(m_off+1)*p + p_off+0][r] = 1
                V[(m_off+0)*p + p_off+0][r] = -1
                W[(p_off+0)*n + 0][r] = 1
                W[(p_off+0)*n + 1][r] = 1
                r_offset += 1
                
                # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
                r = r_offset
                U[0*m + m_off+0][r] = 1
                U[0*m + m_off+1][r] = 1
                V[(m_off+1)*p + p_off+1][r] = 1
                W[(p_off+0)*n + 0][r] = -1
                W[(p_off+1)*n + 0][r] = 1
                r_offset += 1
                
                # M6 = (A10-A00)(B00+B01) -> C11+=M6
                r = r_offset
                U[1*m + m_off+0][r] = 1
                U[0*m + m_off+0][r] = -1
                V[(m_off+0)*p + p_off+0][r] = 1
                V[(m_off+0)*p + p_off+1][r] = 1
                W[(p_off+1)*n + 1][r] = 1
                r_offset += 1
                
                # M7 = (A01-A11)(B10+B11) -> C00+=M7
                r = r_offset
                U[0*m + m_off+1][r] = 1
                U[1*m + m_off+1][r] = -1
                V[(m_off+1)*p + p_off+0][r] = 1
                V[(m_off+1)*p + p_off+1][r] = 1
                W[(p_off+0)*n + 0][r] = 1
                r_offset += 1
                
            elif sa == 2 and sb == 1:
                # Trivial: 2x2 by 2x1
                r = r_offset
                U[0*m + m_off+0][r] = 1
                V[(m_off+0)*p + p_off][r] = 1
                W[p_off*n + 0][r] = 1
                r_offset += 1
                
                r = r_offset
                U[0*m + m_off+1][r] = 1
                V[(m_off+1)*p + p_off][r] = 1
                W[p_off*n + 0][r] = 1
                r_offset += 1
                
                r = r_offset
                U[1*m + m_off+0][r] = 1
                V[(m_off+0)*p + p_off][r] = 1
                W[p_off*n + 1][r] = 1
                r_offset += 1
                
                r = r_offset
                U[1*m + m_off+1][r] = 1
                V[(m_off+1)*p + p_off][r] = 1
                W[p_off*n + 1][r] = 1
                r_offset += 1
                
            elif sa == 1 and sb == 2:
                # Trivial: 2x1 by 1x2
                r = r_offset
                U[0*m + m_off][r] = 1
                V[m_off*p + p_off+0][r] = 1
                W[(p_off+0)*n + 0][r] = 1
                r_offset += 1
                
                r = r_offset
                U[0*m + m_off][r] = 1
                V[m_off*p + p_off+1][r] = 1
                W[(p_off+1)*n + 0][r] = 1
                r_offset += 1
                
                r = r_offset
                U[1*m + m_off][r] = 1
                V[m_off*p + p_off+0][r] = 1
                W[(p_off+0)*n + 1][r] = 1
                r_offset += 1
                
                r = r_offset
                U[1*m + m_off][r] = 1
                V[m_off*p + p_off+1][r] = 1
                W[(p_off+1)*n + 1][r] = 1
                r_offset += 1
                
            else:
                # sa=1, sb=1
                r = r_offset
                U[0*m + m_off][r] = 1
                V[m_off*p + p_off][r] = 1
                W[p_off*n + 0][r] = 1
                r_offset += 1
                
                r = r_offset
                U[1*m + m_off][r] = 1
                V[m_off*p + p_off][r] = 1
                W[p_off*n + 1][r] = 1
                r_offset += 1
    
    return {"U": U, "V": V, "W": W}
