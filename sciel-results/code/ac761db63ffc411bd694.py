"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 84  # 6 Strassen blocks (7 each) + 42 schoolbook terms
    
    # Initialize matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Identify the 6 disjoint 2x2x2 blocks
    # b_i in {0}, b_j in {0, 1}, b_k in {0, 1, 2}
    blocks = []
    for b_i in range(1):
        for b_j in range(2):
            for b_k in range(3):
                blocks.append((b_i, b_j, b_k))
                
    # Helper to get global indices
    def get_idx(b_i, b_j, b_k, u, v, w):
        i = 2 * b_i + u
        j = 2 * b_j + v
        k = 2 * b_k + w
        return i, j, k

    # Strassen coefficients for output C_uv
    # M1, M2, M3, M4, M5, M6, M7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Map (u, v) to list of (m_index, coeff)
    # m_index 0..6 corresponds to M1..M7
    c_coeffs = {
        (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0, 1): [(2, 1), (4, 1)],
        (1, 0): [(1, 1), (3, 1)],
        (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    # Process each block
    for b_i, b_j, b_k in blocks:
        # For each of the 7 Strassen products
        for m_idx in range(7):
            # Determine U, V factors for this product
            # M1: (A00+A11)(B00+B11)
            # M2: (A10+A11)B00
            # M3: A00(B01-B11)
            # M4: A11(B10-B00)
            # M5: (A00+A01)B11
            # M6: (A10-A00)(B00+B01)
            # M7: (A01-A11)(B10+B11)
            
            # U factors (A side)
            # We need to add 1 or -1 to U[i*m+j] for the involved A elements
            # And 1 or -1 to V[j*p+k] for the involved B elements
            
            # Let's define the linear forms explicitly for clarity and correctness
            # A is n x m, B is m x p.
            # Indices in block: i in {2bi, 2bi+1}, j in {2bj, 2bj+1}, k in {2bk, 2bk+1}
            
            i0, j0, k0 = get_idx(b_i, b_j, b_k, 0, 0, 0)
            i1, j1, k1 = get_idx(b_i, b_j, b_k, 1, 1, 1)
            
            # Get all 4 corners for A and B
            # A indices: (i, j)
            # B indices: (j, k)
            
            # M1: (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
            # M2: (A[i1,j0] + A[i1,j1]) * B[j0,k0]
            # M3: A[i0,j0] * (B[j0,k1] - B[j1,k1])
            # M4: A[i1,j1] * (B[j1,k0] - B[j0,k0])
            # M5: (A[i0,j0] + A[i0,j1]) * B[j1,k1]
            # M6: (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
            # M7: (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
            
            # Note: In M2, A11 is A[i1, j1]? No, standard Strassen:
            # A = [[A11, A12], [A21, A22]]
            # B = [[B11, B12], [B21, B22]]
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B12-B22)
            # M4 = A22(B21-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # Mapping to our indices:
            # A11 = A[i0, j0]
            # A12 = A[i0, j1]
            # A21 = A[i1, j0]
            # A22 = A[i1, j1]
            
            # B11 = B[j0, k0]
            # B12 = B[j0, k1]
            # B21 = B[j1, k0]
            # B22 = B[j1, k1]
            
            if m_idx == 0: # M1
                U[i0*m + j0][col] += 1
                U[i1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                V[j1*p + k1][col] += 1
            elif m_idx == 1: # M2
                U[i1*m + j0][col] += 1
                U[i1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
            elif m_idx == 2: # M3
                U[i0*m + j0][col] += 1
                V[j0*p + k1][col] += 1
                V[j1*p + k1][col] -= 1
            elif m_idx == 3: # M4
                U[i1*m + j1][col] += 1
                V[j1*p + k0][col] += 1
                V[j0*p + k0][col] -= 1
            elif m_idx == 4: # M5
                U[i0*m + j0][col] += 1
                U[i0*m + j1][col] += 1
                V[j1*p + k1][col] += 1
            elif m_idx == 5: # M6
                U[i1*m + j0][col] += 1
                U[i0*m + j0][col] -= 1
                V[j0*p + k0][col] += 1
                V[j0*p + k1][col] += 1
            elif m_idx == 6: # M7
                U[i0*m + j1][col] += 1
                U[i1*m + j1][col] -= 1
                V[j1*p + k0][col] += 1
                V[j1*p + k1][col] += 1
            
            # Now add to W based on c_coeffs
            # The output C_uv receives contribution from this M_idx
            # We need to know which C_uv this M_idx contributes to.
            # Instead of iterating c_coeffs now, we can pre-accumulate W contributions
            # But W depends on k and i.
            # The term is M_l * C_uv.
            # So W[k*n+i] gets coeff * 1.
            
            # We need to add to W for all (u,v) that use this m_idx.
            # Let's just loop through c_coeffs and add to W
            for (u, v), terms in c_coeffs.items():
                for (idx, coeff) in terms:
                    if idx == m_idx:
                        # Global i, k for this output C_uv
                        i_out = 2 * b_i + u
                        k_out = 2 * b_k + v
                        W[k_out*n + i_out][col] += coeff
            
            col += 1
            
    # Now handle the remaining schoolbook terms
    # We need to know which (i,j,k) were covered by the blocks
    covered = set()
    for b_i, b_j, b_k in blocks:
        for u in range(2):
            for v in range(2):
                for w in range(2):
                    i, j, k = get_idx(b_i, b_j, b_k, u, v, w)
                    covered.add((i, j, k))
                    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
