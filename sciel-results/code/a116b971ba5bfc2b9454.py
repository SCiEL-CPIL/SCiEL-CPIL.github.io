import time
import sys

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Splits
    n_splits = [1, 2]
    m_splits = [2, 1, 2]
    p_splits = [2, 1, 2]
    
    # Calculate total rank
    # We'll build the matrices column by column
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a rank-1 block
    def add_rank1_block(i_start, i_end, j_start, j_end, k_start, k_end):
        # Block dimensions
        ni = i_end - i_start
        mj = j_end - j_start
        pk = k_end - k_start
        
        # Determine which dimension is 1
        # If mj == 1:
        #   u_i = 1 for i in [i_start, i_end)
        #   v_j = 1 for j in [j_start, j_end) (only one)
        #   w_k = 1 for k in [k_start, k_end)
        
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        if mj == 1:
            j0 = j_start
            for i in range(i_start, i_end):
                u_vec[i * m + j0] = 1
            for k in range(k_start, k_end):
                v_vec[j0 * p + k] = 1
            for k in range(k_start, k_end):
                for i in range(i_start, i_end):
                    w_vec[k * n + i] = 1
        elif ni == 1:
            i0 = i_start
            for j in range(j_start, j_end):
                u_vec[i0 * m + j] = 1
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    v_vec[j * p + k] = 1
            for k in range(k_start, k_end):
                w_vec[k * n + i0] = 1
        elif pk == 1:
            k0 = k_start
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    u_vec[i * m + j] = 1
            for j in range(j_start, j_end):
                v_vec[j * p + k0] = 1
            for i in range(i_start, i_end):
                w_vec[k0 * n + i] = 1
        else:
            # Should not happen for non-Strassen blocks in this specific split
            pass
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Helper to add a Strassen block (2,2,2)
    def add_strassen_block(i_start, j_start, k_start):
        # Indices in global space
        # Local indices 0,1
        # M1..M7
        
        # U vectors
        # M1: (A00+A11) -> u1
        # M2: (A10+A11) -> u2
        # M3: A00 -> u3
        # M4: A11 -> u4
        # M5: (A00+A01) -> u5
        # M6: (A10-A00) -> u6
        # M7: (A01-A11) -> u7
        
        u1 = [0]*(n*m); u1[i_start*m + j_start] = 1; u1[(i_start+1)*m + (j_start+1)] = 1
        u2 = [0]*(n*m); u2[(i_start+1)*m + j_start] = 1; u2[(i_start+1)*m + (j_start+1)] = 1
        u3 = [0]*(n*m); u3[i_start*m + j_start] = 1
        u4 = [0]*(n*m); u4[(i_start+1)*m + (j_start+1)] = 1
        u5 = [0]*(n*m); u5[i_start*m + j_start] = 1; u5[i_start*m + (j_start+1)] = 1
        u6 = [0]*(n*m); u6[(i_start+1)*m + j_start] = 1; u6[i_start*m + j_start] = -1
        u7 = [0]*(n*m); u7[i_start*m + (j_start+1)] = 1; u7[(i_start+1)*m + (j_start+1)] = -1
        
        # V vectors
        # M1: (B00+B11)
        # M2: B00
        # M3: (B10-B11)
        # M4: (B01-B00)
        # M5: B11
        # M6: (B00+B01)
        # M7: (B10+B11)
        
        v1 = [0]*(m*p); v1[j_start*p + k_start] = 1; v1[(j_start+1)*p + (k_start+1)] = 1
        v2 = [0]*(m*p); v2[j_start*p + k_start] = 1
        v3 = [0]*(m*p); v3[(j_start+1)*p + k_start] = 1; v3[(j_start+1)*p + (k_start+1)] = -1
        v4 = [0]*(m*p); v4[j_start*p + (k_start+1)] = 1; v4[j_start*p + k_start] = -1
        v5 = [0]*(m*p); v5[(j_start+1)*p + (k_start+1)] = 1
        v6 = [0]*(m*p); v6[j_start*p + k_start] = 1; v6[j_start*p + (k_start+1)] = 1
        v7 = [0]*(m*p); v7[(j_start+1)*p + k_start] = 1; v7[(j_start+1)*p + (k_start+1)] = 1
        
        # W vectors
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        w1 = [0]*(p*n)
        w1[k_start*n + i_start] = 1
        w1[(k_start+1)*n + (i_start+1)] = 1
        # C00 += M1
        w1[k_start*n + i_start] += 1 # Wait, C00 is at k=i? No.
        # C index is (k, i).
        # C00 corresponds to k_start, i_start.
        # C01 corresponds to k_start+1, i_start.
        # C10 corresponds to k_start, i_start+1.
        # C11 corresponds to k_start+1, i_start+1.
        
        # Let's re-verify C indices.
        # A is n x m. B is m x p. C is n x p?
        # Task says: "output factor follows the official trace convention W[k*n+i,r]"
        # This implies C index is (k, i) where k is row of B (col of A?) and i is row of A?
        # Standard: C_{ik} = sum A_{ij} B_{jk}.
        # If W is indexed by [k*n+i], then W corresponds to C^T or C with swapped indices?
        # Usually C is n x p.
        # If W[k*n+i], then first index is k (0..p-1), second is i (0..n-1).
        # So W represents C^T? Or C with indices (k,i).
        # Let's assume the tensor is T_{i,j,k} = A_{ij} B_{jk} C_{ki}?
        # No, multiplication is A (n x m) * B (m x p) = C (n x p).
        # Term: A_{ij} B_{jk} contributes to C_{ik}.
        # The tensor is sum over i,j,k of A_{ij} B_{jk} C_{ik}.
        # If W is indexed by k*n+i, then W_{k,i} corresponds to C_{i,k}?
        # Let's check the seed code.
        # Seed: W[k*n+i][column] = 1.
        # Loop: for i in range(n): for j in range(m): for k in range(p):
        # U[i*m+j], V[j*p+k], W[k*n+i].
        # This confirms W index is k*n+i.
        # So W corresponds to the matrix with rows k and cols i.
        # This is C^T if C is n x p with C[i,k].
        # Or it's just the convention for the third factor.
        # In Strassen:
        # C_{00} (i=0, k=0) -> W[0*n+0]
        # C_{01} (i=0, k=1) -> W[1*n+0]
        # C_{10} (i=1, k=0) -> W[0*n+1]
        # C_{11} (i=1, k=1) -> W[1*n+1]
        
        # My previous mapping:
        # C00 -> k=0, i=0 -> W[0]
        # C01 -> k=1, i=0 -> W[n]
        # C10 -> k=0, i=1 -> W[1]
        # C11 -> k=1, i=1 -> W[n+1]
        
        # Let's re-map Strassen outputs to W.
        # M1 contributes to C00 (+1), C11 (+1)
        # M2 contributes to C10 (+1), C11 (-1)
        # M3 contributes to C01 (+1), C11 (+1)
        # M4 contributes to C00 (+1), C10 (+1)
        # M5 contributes to C00 (-1), C01 (+1)
        # M6 contributes to C11 (+1)
        # M7 contributes to C00 (+1)
        
        # Wait, let's re-list Strassen formulas for C_{ik}:
        # C_{00} = M1 - M5 + M4 + M7
        # C_{01} = M5 + M3
        # C_{10} = M4 + M2
        # C_{11} = M1 + M6 - M2 + M3
        
        # Check C_{00}: M1(A00+A11)(B00+B11) - M5(A00+A01)B11 + M4 A11(B01-B00) + M7(A01-A11)(B10+B11)
        # = (A00B00 + A00B11 + A11B00 + A11B11) - (A00B11 + A01B11) + (A11B01 - A11B00) + (A01B10 + A01B11 - A11B10 - A11B11)
        # A00B00: 1
        # A00B11: 1 - 1 = 0
        # A11B00: 1 - 1 = 0
        # A11B11: 1 - 1 = 0
        # A01B11: -1 + 1 = 0
        # A11B01: 1
        # A01B10: 1
        # A11B10: -1
        # Result: A00B00 + A11B01 + A01B10 - A11B10.
        # This is NOT C00. C00 = A00B00 + A01B10.
        # My Strassen formulas or mapping is wrong.
        
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Using 0-indexed:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's check C00 again with these.
        # M1 = (A00+A11)(B00+B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M7 = (A01-A11)(B10+B11)
        
        # Sum:
        # M1: A00B00 + A00B11 + A11B00 + A11B11
        # M4: A11B01 - A11B00
        # -M5: -A00B11 - A01B11
        # M7: A01B10 + A01B11 - A11B10 - A11B11
        
        # A00B00: 1
        # A00B11: 1 - 1 = 0
        # A11B00: 1 - 1 = 0
        # A11B11: 1 - 1 = 0
        # A11B01: 1
        # A01B11: -1 + 1 = 0
        # A01B10: 1
        # A11B10: -1
        
        # Result: A00B00 + A11B01 + A01B10 - A11B10.
        # Still not C00 = A00B00 + A01B10.
        
        # There is a mistake in my Strassen formulas or expansion.
        # Let's use the known correct ones.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's check C12 (which is C01 in 0-indexed).
        # C01 = A00B01 + A01B11.
        # M3 = A00(B10-B11)
        # M5 = (A00+A01)B11
        # M3+M5 = A00B10 - A00B11 + A00B11 + A01B11 = A00B10 + A01B11.
        # This is A00B10 + A01B11.
        # But C01 should be A00B01 + A01B11.
        # So M3 should be A00(B01-B00)?
        
        # Correct Strassen (1-indexed):
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's re-verify C12 (0-indexed C01).
        # C12 = A11B12 + A12B22.
        # M3 = A11(B21-B22)
        # M5 = (A11+A12)B22
        # M3+M5 = A11B21 - A11B22 + A11B22 + A12B22 = A11B21 + A12B22.
        # This is A11B21 + A12B22.
        # But C12 is A11B12 + A12B22.
        # So M3 is wrong.
        
        # It seems I am mixing up indices.
        # Let's just use the code from a known source or derive it carefully.
        # Actually, for the purpose of this task, if I can't get Strassen right, I can fall back to schoolbook for the 2x2x2 blocks (rank 8).
        # Rank would be 42 - 4*7 + 4*8 = 42 + 4 = 46.
        # Still much better than 71.
        
        # However, I am fairly confident in the standard Strassen.
        # Let's try one more time.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # C_{11} = A_{11}B_{11} + A_{12}B_{21}.
        # C_{12} = A_{11}B_{12} + A_{12}B_{22}.
        # C_{21} = A_{21}B_{11} + A_{22}B_{21}.
        # C_{22} = A_{21}B_{12} + A_{22}B_{22}.
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # = (A11B11 + A11B22 + A22B11 + A22B22) + (A22B12 - A22B11) - (A11B22 + A12B22) + (A12B21 + A12B22 - A22B21 - A22B22)
        # A11B11: 1
        # A11B22: 1 - 1 = 0
        # A22B11: 1 - 1 = 0
        # A22B22: 1 - 1 = 0
        # A22B12: 1
        # A12B22: -1 + 1 = 0
        # A12B21: 1
        # A22B21: -1
        # Result: A11B11 + A22B12 + A12B21 - A22B21.
        # Target: A11B11 + A12B21.
        # Extra terms: A22B12 - A22B21.
        
        # Okay, I will use Schoolbook for 2x2x2 blocks to be safe.
        # Rank 8 for each.
        # Total rank 46.
        
    # Iterate through blocks
    i_idx = 0
    for i_size in n_splits:
        j_idx = 0
        for j_size in m_splits:
            k_idx = 0
            for k_size in p_splits:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Schoolbook for safety
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                u_vec = [0]*(n*m)
                                v_vec = [0]*(m*p)
                                w_vec = [0]*(p*n)
                                i_glob = i_idx + di
                                j_glob = j_idx + dj
                                k_glob = k_idx + dk
                                u_vec[i_glob*m + j_glob] = 1
                                v_vec[j_glob*p + k_glob] = 1
                                w_vec[k_glob*n + i_glob] = 1
                                U_cols.append(u_vec)
                                V_cols.append(v_vec)
                                W_cols.append(w_vec)
                else:
                    add_rank1_block(i_idx, i_idx + i_size, j_idx, j_idx + j_size, k_idx, k_idx + k_size)
                k_idx += k_size
            j_idx += j_size
        i_idx += i_size
        
    # Convert to matrices
    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        for i in range(n*m):
            if U_cols[c][i] != 0:
                U[i][c] = U_cols[c][i]
        for i in range(m*p):
            if V_cols[c][i] != 0:
                V[i][c] = V_cols[c][i]
        for i in range(p*n):
            if W_cols[c][i] != 0:
                W[i][c] = W_cols[c][i]
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
