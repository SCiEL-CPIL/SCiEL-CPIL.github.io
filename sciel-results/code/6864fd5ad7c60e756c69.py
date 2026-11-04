"""Matrix multiplication 4x4 * 4x9 — rank-minimal exact decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: split B (4x9) into B1 (4x4) and B2 (4x5)
    # Use Strassen for 4x4 * 4x4 (rank 49)
    # Use schoolbook for 4x4 * 4x5 (rank 80)
    # Total rank: 129
    
    def _strassen_2x2_factors():
        """Return the 7 Strassen rank-1 factors for 2x2 * 2x2.
        Each factor is (u, v, w) where:
          u[i,j] coefficient for A[i,j]
          v[j,k] coefficient for B[j,k]
          w[i,k] coefficient for C[i,k]
        """
        # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        factors = []
        # M1
        factors.append(([[1,0],[1,1]], [[1,1],[1,1]], [[1,0],[0,1]]))
        # M2
        factors.append(([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]))
        # M3
        factors.append(([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[1,0]]))
        # M4
        factors.append(([[0,0],[0,1]], [[-1,0],[1,0]], [[1,0],[0,-1]]))
        # M5
        factors.append(([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[0,1]]))
        # M6
        factors.append(([[0,0],[-1,1]], [[1,1],[0,0]], [[0,0],[0,1]]))
        # M7
        factors.append(([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,1]]))
        return factors
    
    def build_strassen_4x4():
        """Build rank-49 decomposition for 4x4 * 4x4 using two-level Strassen."""
        # Top-level Strassen: 7 sub-problems of 2x2 * 2x2
        # Each 2x2*2x2 uses 7 rank-1 terms
        # Total: 49 terms
        
        strassen_2x2 = _strassen_2x2_factors()
        
        # A-blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3), 
        #           A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
        # B-blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3),
        #           B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
        
        # Top-level Strassen factors (coefficients for A-blocks and B-blocks):
        # M1: A=(A11+A22), B=(B11+B22)
        # M2: A=(A21+A22), B=B11
        # M3: A=A11, B=(B12-B22)
        # M4: A=A22, B=(B21-B11)
        # M5: A=(A11+A12), B=B22
        # M6: A=(A21-A11), B=(B11+B12)
        # M7: A=(A12-A22), B=(B21+B22)
        
        top_factors = [
            ({'A11':1, 'A22':1}, {'B11':1, 'B22':1}),  # M1
            ({'A21':1, 'A22':1}, {'B11':1}),             # M2
            ({'A11':1}, {'B12':1, 'B22':-1}),            # M3
            ({'A22':1}, {'B21':1, 'B11':-1}),            # M4
            ({'A11':1, 'A12':1}, {'B22':1}),             # M5
            ({'A21':1, 'A11':-1}, {'B11':1, 'B12':1}),   # M6
            ({'A12':1, 'A22':-1}, {'B21':1, 'B22':1}),   # M7
        ]
        
        # C-block combinations:
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        c_combos = {
            'C11': [(0,1), (3,1), (4,-1), (6,1)],
            'C12': [(2,1), (4,1)],
            'C21': [(1,1), (3,1)],
            'C22': [(0,1), (1,-1), (2,1), (5,1)],
        }
        
        # Block positions in 4x4:
        a_block_pos = {
            'A11': (0,0), 'A12': (0,2),
            'A21': (2,0), 'A22': (2,2),
        }
        b_block_pos = {
            'B11': (0,0), 'B12': (0,2),
            'B21': (2,0), 'B22': (2,2),
        }
        c_block_pos = {
            'C11': (0,0), 'C12': (0,2),
            'C21': (2,0), 'C22': (2,2),
        }
        
        rank = 49
        U = np.zeros((16, rank))  # 4*4
        V = np.zeros((16, rank))  # 4*4
        W = np.zeros((16, rank))  # 4*4
        
        col = 0
        for t in range(7):
            a_coeffs, b_coeffs = top_factors[t]
            for s in range(7):
                u_s, v_s, w_s = strassen_2x2[s]
                
                # For each element in the 2x2 u_s, v_s, w_s:
                # The effective A coefficient is u_s[li][lj] * a_coeff for each A-block
                # The effective B coefficient is v_s[lj][lk] * b_coeff for each B-block
                # The effective C coefficient is w_s[li][lk] * c_coeff for each C-block
                
                for li in range(2):
                    for lj in range(2):
                        u_val = u_s[li][lj]
                        if u_val == 0:
                            continue
                        for lk in range(2):
                            v_val = v_s[lj][lk]
                            if v_val == 0:
                                continue
                            w_val = w_s[li][lk]
                            if w_val == 0:
                                continue
                            
                            # For each A-block and B-block combination:
                            for a_block, a_coeff in a_coeffs.items():
                                ar, ac = a_block_pos[a_block]
                                for b_block, b_coeff in b_coeffs.items():
                                    br, bc = b_block_pos[b_block]
                                    
                                    # U index: (ar + li, ac + lj) -> (ar + li) * 4 + (ac + lj)
                                    u_idx = (ar + li) * 4 + (ac + lj)
                                    # V index: (br + lj, bc + lk) -> (br + lj) * 4 + (bc + lk)
                                    v_idx = (br + lj) * 4 + (bc + lk)
                                    # W index: (bc + lk) * 4 + (ar + li) ... 
                                    # Actually W[k*n+i] where k is col of B, i is row of A
                                    # Wait, the convention is W[k*n+i, r] for output C[i,k]
                                    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
                                    # So W is indexed by (k*n+i) where k is column of C, i is row of C
                                    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                    # Wait no, that's not right. Let me re-think.
                                    
                                    # The standard tensor decomposition:
                                    # C[i,k] = sum_j A[i,j] * B[j,k]
                                    # = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
                                    # Hmm, that's not quite right either.
                                    
                                    # Actually the standard form is:
                                    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                    # No, that's a 3-way tensor decomposition.
                                    
                                    # Let me use the correct form:
                                    # C[i,k] = sum_r (sum_j u_r[i,j] * A[i,j]) * (sum_j v_r[j,k] * B[j,k]) * w_r[i,k]
                                    # where u_r is 4x4, v_r is 4x4, w_r is 4x4
                                    # And U, V, W are the matrices with columns being the flattened u_r, v_r, w_r.
                                    
                                    # So for this rank-1 term:
                                    # u_r[i,j] = u_val * a_coeff if (i,j) is in A-block
                                    # v_r[j,k] = v_val * b_coeff if (j,k) is in B-block
                                    # w_r[i,k] = w_val * c_coeff if (i,k) is in C-block
                                    
                                    # But we need to sum over C-blocks with their coefficients.
                                    # Let me accumulate W separately.
                                    
                                    U[u_idx, col] += u_val * a_coeff
                                    V[v_idx, col] += v_val * b_coeff
                            
                            # Now for W, we need to sum over all C-blocks:
                            for c_block, (t_idx, c_coeff) in c_combos.items():
                                if t_idx != t:
                                    continue
                                cr, cc = c_block_pos[c_block]
                                # W index: (cc + lk) * 4 + (cr + li) ... 
                                # Wait, W[k*n+i] where k is column of C, i is row of C
                                # C-block position: cr is row start, cc is col start
                                # So for element (cr + li, cc + lk) in C:
                                # i = cr + li, k = cc + lk
                                # W index: k * n + i = (cc + lk) * 4 + (cr + li)
                                w_idx = (cc + lk) * 4 + (cr + li)
                                W[w_idx, col] += w_val * c_coeff
                    
                    col += 1
                    if col >= rank:
                        break
                if col >= rank:
                    break
            if col >= rank:
                break
        
        return U, V, W, col
    
    # Build Strassen 4x4 * 4x4
    U_strassen, V_strassen, W_strassen, rank_strassen = build_strassen_4x4()
    
    # Build schoolbook 4x4 * 4x5
    p2 = 5
    rank_school = n * m * p2
    U_school = np.zeros((16, rank_school))
    V_school = np.zeros((20, rank_school))  # 4*5
    W_school = np.zeros((20, rank_school))  # 5*4
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p2):
                U_school[i * m + j, col] = 1
                V_school[j * p2 + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Combine: total rank = rank_strassen + rank_school
    total_rank = rank_strassen + rank_school
    
    # Full U, V, W for 4x4 * 4x9
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((36, total_rank))  # 4*9
    W_full = np.zeros((36, total_rank))  # 9*4
    
    # Copy Strassen part
    U_full[:, :rank_strassen] = U_strassen
    V_full[:, :rank_strassen] = V_strassen
    W_full[:, :rank_strassen] = W_strassen
    
    # Copy schoolbook part (for columns 4..8 of B, i.e., k=4..8)
    # V_school is for B columns 0..4, but we need B columns 4..8
    # So we shift: V_full[:, rank_strassen:rank_strassen+rank_school] should correspond to B columns 4..8
    # V_school[j*p2+k] where k=0..4 corresponds to B columns 4..8
    # So V_school[j*5+k] -> V_full[j*9 + 4 + k]
    
    for j in range(4):
        for k in range(5):
            v_idx_school = j * 5 + k
            v_idx_full = j * 9 + 4 + k
            V_full[v_idx_full, rank_strassen:rank_strassen+rank_school] = V_school[v_idx_school]
    
    # W_school[k*n+i] where k=0..4 corresponds to C columns 4..8
    # So W_school[k*4+i] -> W_full[(4+k)*4 + i]
    for k in range(5):
        for i in range(4):
            w_idx_school = k * 4 + i
            w_idx_full = (4 + k) * 4 + i
            W_full[w_idx_full, rank_strassen:rank_strassen+rank_school] = W_school[w_idx_school]
    
    U_full[:, rank_strassen:rank_strassen+rank_school] = U_school
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
