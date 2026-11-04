import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=7
    
    # Target rank is 60 based on the decomposition:
    # 2 * M_{2,2,4} (rank 14) + 2 * M_{2,2,3} (rank 9) + 2 * M_{1,2,4} (rank 4) + 2 * M_{1,2,3} (rank 3)
    # 28 + 18 + 8 + 6 = 60
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block decomposition
    def add_block(n_block, m_block, p_block, u_offset, v_offset, w_offset, block_decomp):
        nonlocal col
        for u_idx, v_idx, w_idx, coeff in block_decomp:
            # Map local indices to global
            # u_idx is (i, j) in 0..n_block-1, 0..m_block-1
            # v_idx is (j, k) in 0..m_block-1, 0..p_block-1
            # w_idx is (k, i) in 0..p_block-1, 0..n_block-1
            
            # Global U index: (u_offset_i + i) * m + (u_offset_j + j)
            # We assume u_offset is (i_off, j_off)
            i_off, j_off = u_offset
            i, j = divmod(u_idx, m_block)
            global_u = (i_off + i) * m + (j_off + j)
            
            j_off_v, k_off = v_offset
            j, k = divmod(v_idx, p_block)
            global_v = (j_off_v + j) * p + (k_off + k)
            
            k_off_w, i_off_w = w_offset
            k, i = divmod(w_idx, n_block)
            global_w = (k_off_w + k) * n + (i_off_w + i)
            
            U[global_u][col] = coeff
            V[global_v][col] = coeff
            W[global_w][col] = coeff
            col += 1

    # Strassen decomposition for M_{2,2,2}
    # U indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # V indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # W indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*i + i? No, k*2+i)
    # W local index: k*2 + i
    
    # M1: U: 00+11, V: 00+11, W: 00+11
    # M2: U: 10+11, V: 00, W: 10
    # M3: U: 00, V: 01-11, W: 01
    # M4: U: 11, V: 10-00, W: 10
    # M5: U: 00+01, V: 11, W: 01
    # M6: U: 10-00, V: 00+01, W: 11
    # M7: U: 01-11, V: 10+11, W: 00
    
    strassen_222 = [
        # (u_indices, v_indices, w_indices, coeff)
        # We store as list of (u_idx, v_idx, w_idx, coeff)
        # But we need to sum coefficients for same indices in the vector.
        # It's easier to just list the non-zero entries for U, V, W vectors.
        
        # r=0 (M1)
        ([0, 3], [0, 3], [0, 3], 1),
        # r=1 (M2)
        ([2, 3], [0], [2], 1),
        # r=2 (M3)
        ([0], [1, 3], [1], 1), # V: 01-11 -> 1, -3
        # r=3 (M4)
        ([3], [2, 0], [2], 1), # V: 10-00 -> 2, -0
        # r=4 (M5)
        ([0, 1], [3], [1], 1),
        # r=5 (M6)
        ([2, 0], [0, 1], [3], 1), # U: 10-00 -> 2, -0
        # r=6 (M7)
        ([1, 3], [2, 3], [0], 1), # U: 01-11 -> 1, -3
    ]
    
    # Correcting coefficients for subtraction
    # M3: V is B01 - B11. So V[1]=1, V[3]=-1.
    # M4: V is B10 - B00. So V[2]=1, V[0]=-1.
    # M6: U is A10 - A00. So U[2]=1, U[0]=-1.
    # M7: U is A01 - A11. So U[1]=1, U[3]=-1.
    
    strassen_222 = [
        ([0, 3], [0, 3], [0, 3], 1),
        ([2, 3], [0], [2], 1),
        ([0], [1, 3], [1], 1), # Coeffs: 1, -1
        ([3], [2, 0], [2], 1), # Coeffs: 1, -1
        ([0, 1], [3], [1], 1),
        ([2, 0], [0, 1], [3], 1), # Coeffs: 1, -1
        ([1, 3], [2, 3], [0], 1), # Coeffs: 1, -1
    ]
    
    # We need to handle the coefficients properly.
    # Let's define the vectors explicitly.
    
    def get_strassen_vectors():
        vecs = []
        # M1
        vecs.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2
        vecs.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        # M3
        vecs.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
        # M4
        vecs.append(([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 0]))
        # M5
        vecs.append(([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]))
        # M6
        vecs.append(([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7
        vecs.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return vecs

    strassen_vecs = get_strassen_vectors()
    
    def add_strassen_block(u_off, v_off, w_off, m_block, p_block):
        # m_block=2, p_block=2 for M_{2,2,2}
        for u_vec, v_vec, w_vec in strassen_vecs:
            for i in range(2):
                for j in range(2):
                    if u_vec[i*2+j] != 0:
                        U[(u_off[0]+i)*m + (u_off[1]+j)][col] = u_vec[i*2+j]
            for j in range(m_block):
                for k in range(p_block):
                    if v_vec[j*p_block+k] != 0:
                        V[(v_off[0]+j)*p + (v_off[1]+k)][col] = v_vec[j*p_block+k]
            for k in range(p_block):
                for i in range(2):
                    if w_vec[k*2+i] != 0:
                        W[(w_off[0]+k)*n + (w_off[1]+i)][col] = w_vec[k*2+i]
            col += 1

    # Trivial M_{2,2,1} (Rank 2)
    # U: A00, A10. V: B00, B10. W: C00, C10.
    # Actually M_{2,2,1} is A_{2x2} B_{2x1} = C_{2x1}.
    # C0 = A00 B00 + A01 B10
    # C1 = A10 B00 + A11 B10
    # Trivial:
    # r0: A00, B00, C0
    # r1: A01, B10, C0
    # r2: A10, B00, C1
    # r3: A11, B10, C1
    # Rank 4?
    # Wait, M_{2,2,1} rank is 2?
    # T_{i,j,k}. k is always 0.
    # T_{0,0,0} = 1, T_{0,1,0} = 1, T_{1,0,0} = 1, T_{1,1,0} = 1.
    # This is the tensor of the bilinear form.
    # Rank is 2?
    # Let's use trivial rank 4 for simplicity if needed, or check if 2 is possible.
    # Actually, for M_{n,m,1}, rank is n.
    # M_{2,2,1} rank is 2.
    # Decomp:
    # r0: U: A00+A10, V: B00, W: C0+C1? No.
    # Let's just use trivial rank 4 for M_{2,2,1} to be safe, or 2 if we can.
    # If we use rank 4, M_{2,2,3} becomes 7+4=11. Total 62.
    # If we use rank 2, M_{2,2,3} becomes 7+2=9. Total 60.
    
    # Let's try to construct M_{2,2,1} rank 2.
    # C0 = A00 B00 + A01 B10
    # C1 = A10 B00 + A11 B10
    # r0: U: [1,0,1,0] (A00+A10), V: [1,0,0,0] (B00), W: [1,1,0,0] (C0+C1)
    # r1: U: [0,1,0,1] (A01+A11), V: [0,0,1,0] (B10), W: [1,1,0,0] (C0+C1)
    # Check C0: 1*1*1 + 1*1*1 = 2? No.
    # We need coefficients.
    # This is getting complex. Let's stick to Rank 60 with Trivial M_{2,2,1} (Rank 4) -> Total 62.
    # Or just use Trivial for everything smaller than 2x2x2.
    
    # Let's use Trivial for M_{2,2,1}, M_{1,2,2}, M_{1,2,1}.
    # M_{2,2,1} Rank 4.
    # M_{1,2,2} Rank 2.
    # M_{1,2,1} Rank 1.
    
    # Recalculate Total Rank:
    # M_{2,2,4} = 7 (M222) + 7 (M222) = 14.
    # M_{2,2,3} = 7 (M222) + 4 (M221) = 11.
    # M_{1,2,4} = 2 (M122) + 2 (M122) = 4.
    # M_{1,2,3} = 2 (M122) + 1 (M121) = 3.
    
    # Total = 2*14 + 2*11 + 2*4 + 2*3 = 28 + 22 + 8 + 6 = 64.
    
    # Let's try to get M_{2,2,1} rank 2.
    # If we can, Total = 28 + 18 + 8 + 6 = 60.
    
    # I will implement the rank 64 version first as it is safer.
    
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_trivial_block(n_block, m_block, p_block, u_off, v_off, w_off):
        nonlocal col
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    U[(u_off[0]+i)*m + (u_off[1]+j)][col] = 1
                    V[(v_off[0]+j)*p + (v_off[1]+k)][col] = 1
                    W[(w_off[0]+k)*n + (w_off[1]+i)][col] = 1
                    col += 1

    # 1. M_{2,2,4} (Top-Left A, Top-Left B)
    # A: rows 0,1 cols 0,1. B: rows 0,1 cols 0,3.
    # Split B into cols 0,1 and 2,3.
    add_strassen_block((0,0), (0,0), (0,0), 2, 2) # B cols 0,1
    add_strassen_block((0,0), (0,2), (0,0), 2, 2) # B cols 2,3
    
    # 2. M_{2,2,3} (Top-Right A, Bottom-Left B)
    # A: rows 0,1 cols 2,3. B: rows 0,1 cols 0,3.
    # Split B into cols 0,1 (M222) and col 2 (M221).
    add_strassen_block((0,2), (0,0), (0,0), 2, 2) # B cols 0,1
    add_trivial_block(2, 2, 1, (0,2), (0,2), (0,0)) # B col 2
    
    # 3. M_{1,2,4} (Bottom-Left A, Top-Left B)
    # A: row 2 cols 0,1. B: rows 0,1 cols 0,3.
    # Split B into cols 0,1 (M122) and 2,3 (M122).
    add_trivial_block(1, 2, 2, (2,0), (0,0), (0,0)) # B cols 0,1
    add_trivial_block(1, 2, 2, (2,0), (0,2), (0,0)) # B cols 2,3
    
    # 4. M_{1,2,3} (Bottom-Right A, Bottom-Left B)
    # A: row 2 cols 2,3. B: rows 0,1 cols 0,3.
    # Split B into cols 0,1 (M122) and col 2 (M121).
    add_trivial_block(1, 2, 2, (2,2), (0,0), (0,0)) # B cols 0,1
    add_trivial_block(1, 2, 1, (2,2), (0,2), (0,0)) # B col 2
    
    # 5. M_{2,2,4} (Top-Left A, Bottom-Left B)
    # A: rows 0,1 cols 0,1. B: rows 2,3 cols 0,3.
    add_strassen_block((0,0), (2,0), (0,0), 2, 2) # B cols 0,1
    add_strassen_block((0,0), (2,2), (0,0), 2, 2) # B cols 2,3
    
    # 6. M_{2,2,3} (Top-Right A, Bottom-Left B)
    # A: rows 0,1 cols 2,3. B: rows 2,3 cols 0,3.
    add_strassen_block((0,2), (2,0), (0,0), 2, 2) # B cols 0,1
    add_trivial_block(2, 2, 1, (0,2), (2,2), (0,0)) # B col 2
    
    # 7. M_{1,2,4} (Bottom-Left A, Bottom-Left B)
    # A: row 2 cols 0,1. B: rows 2,3 cols 0,3.
    add_trivial_block(1, 2, 2, (2,0), (2,0), (0,0)) # B cols 0,1
    add_trivial_block(1, 2, 2, (2,0), (2,2), (0,0)) # B cols 2,3
    
    # 8. M_{1,2,3} (Bottom-Right A, Bottom-Left B)
    # A: row 2 cols 2,3. B: rows 2,3 cols 0,3.
    add_trivial_block(1, 2, 2, (2,2), (2,0), (0,0)) # B cols 0,1
    add_trivial_block(1, 2, 1, (2,2), (2,2), (0,0)) # B col 2
    
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
