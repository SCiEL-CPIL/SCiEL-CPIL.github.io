import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to add a rank-1 term
    def add_term(U, V, W, u_idx, v_idx, w_idx, coeff=1):
        U[u_idx] += coeff
        V[v_idx] += coeff
        W[w_idx] += coeff

    # We will build the matrices U, V, W
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    
    # We estimate the max rank to be 77 (49 + 28)
    max_rank = 80
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))
    
    current_rank = 0
    
    # Strassen 2x2x2 decomposition
    # Indices for 2x2 blocks:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    # 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2(base_u, base_v, base_w):
        """
        Adds 7 rank-1 terms for a 2x2x2 multiplication.
        base_u, base_v, base_w are offsets into the U, V, W matrices for the 2x2 blocks.
        """
        nonlocal current_rank
        
        # Term 1: (A00+A11)(B00+B11)
        # U indices: A00 (0), A11 (3)
        # V indices: B00 (0), B11 (3)
        # W index: C00 (0) -> coeff 1, C01 (1) -> 0, C10 (2) -> 0, C11 (3) -> 1
        # Wait, W is (p*n, rank). For 2x2, p=2, n=2. W indices 0..3.
        # C00 is W[0], C01 is W[1], C10 is W[2], C11 is W[3].
        
        # We need to be careful with the indices.
        # U[i*m+j]
        # For 2x2 block at offset (bi, bj) in the larger matrix:
        # U index = (bi*2 + i)*2 + (bj*2 + j)
        
        # Let's define the 7 terms explicitly for the local 2x2x2.
        # Local U indices: 0,1,2,3
        # Local V indices: 0,1,2,3
        # Local W indices: 0,1,2,3
        
        terms = [
            # (u_indices, v_indices, w_index, coeff)
            # M1
            ([(0,1), (3,1)], [(0,1), (3,1)], 0, 1), # C00
            ([(0,1), (3,1)], [(0,1), (3,1)], 3, 1), # C11
            # M2
            ([(2,1), (3,1)], [(0,1)], 2, 1), # C10
            # M3
            ([(0,1)], [(1,1), (3,-1)], 1, 1), # C01
            # M4
            ([(3,1)], [(2,1), (0,-1)], 0, 1), # C00
            ([(3,1)], [(2,1), (0,-1)], 2, 1), # C10
            # M5
            ([(0,1), (1,1)], [(3,1)], 0, -1), # C00
            ([(0,1), (1,1)], [(3,1)], 1, 1), # C01
            # M6
            ([(2,-1), (0,-1)], [(0,1), (1,1)], 3, 1), # C11
            # M7
            ([(1,1), (3,-1)], [(2,1), (3,1)], 0, 1), # C00
        ]
        
        # This approach is messy because one rank-1 term in Strassen corresponds to one scalar multiplication.
        # But here the "scalar" is a 2x2 matrix? No.
        # In the recursive step, the "scalar" multiplication is the base case.
        # The base case is 2x2x2.
        # So we need to decompose the 2x2x2 tensor into 7 rank-1 terms.
        
        # The 7 rank-1 terms for 2x2x2 are:
        # 1. (A00+A11) * (B00+B11) -> contributes to C00, C11
        #    U: A00, A11. V: B00, B11.
        #    This is NOT a rank-1 term in the global sense.
        #    A rank-1 term is u * v * w.
        #    Here u is a linear form on A. v is a linear form on B. w is a linear form on C.
        #    So for the 2x2x2 case:
        #    Term 1: u = A00 + A11, v = B00 + B11, w = C00 + C11.
        #    This is one rank-1 term.
        
        # So we just need to add these 7 terms.
        
        def add_local_term(u_coeffs, v_coeffs, w_coeffs):
            """
            u_coeffs: list of (local_u_idx, coeff)
            v_coeffs: list of (local_v_idx, coeff)
            w_coeffs: list of (local_w_idx, coeff)
            """
            nonlocal current_rank
            r = current_rank
            current_rank += 1
            
            for lu, cu in u_coeffs:
                U[base_u + lu, r] += cu
            for lv, cv in v_coeffs:
                V[base_v + lv, r] += cv
            for lw, cw in w_coeffs:
                W[base_w + lw, r] += cw
        
        # M1: (A00+A11)(B00+B11) = C00 + C11
        add_local_term([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)])
        
        # M2: (A10+A11)B00 = C10
        add_local_term([(2,1), (3,1)], [(0,1)], [(2,1)])
        
        # M3: A00(B01-B11) = C01
        add_local_term([(0,1)], [(1,1), (3,-1)], [(1,1)])
        
        # M4: A11(B10-B00) = C00 + C10
        add_local_term([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)])
        
        # M5: (A00+A01)B11 = -C00 + C01
        add_local_term([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)])
        
        # M6: (A10-A00)(B00+B01) = C11
        add_local_term([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)])
        
        # M7: (A01-A11)(B10+B11) = C00
        add_local_term([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)])
        
        return current_rank

    # 1. Compute M(4,4,4) using 4 calls to strassen_2x2x2?
    # No, M(4,4,4) is computed by splitting into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # C00 = A00B00 + A01B10
    # This is NOT a single 2x2x2 multiplication.
    # It is the sum of two 2x2x2 multiplications.
    # We can compute A00B00 using rank 7, and A01B10 using rank 7.
    # Total 14 for C00.
    # Similarly for C01, C10, C11.
    # Total 4 * 14 = 56?
    # But Strassen reduces 8 multiplications to 7.
    # For 4x4x4, we have 4 blocks of 2x2x2.
    # We can use Strassen on the block level?
    # Yes.
    # The block level is a 2x2x2 multiplication of 2x2 matrices.
    # Rank of 2x2x2 (block) is 7.
    # Each block multiplication is a 2x2x2 (element) multiplication.
    # Rank of 2x2x2 (element) is 7.
    # Total rank 7 * 7 = 49.
    
    # So we call strassen_2x2x2 7 times?
    # No, the structure is:
    # We have 7 "block" multiplications M1..M7.
    # Each Mi is a product of sums of 2x2 blocks.
    # Each Mi corresponds to a 2x2x2 (element) multiplication.
    # So we need to decompose each Mi into 7 rank-1 terms.
    # Total 49 terms.
    
    # Block offsets for 4x4:
    # A00: base_u = 0
    # A01: base_u = 2
    # A10: base_u = 8
    # A11: base_u = 10
    
    # B00: base_v = 0
    # B01: base_v = 2
    # B10: base_v = 8
    # B11: base_v = 10
    
    # C00: base_w = 0
    # C01: base_w = 2
    # C10: base_w = 8
    # C11: base_w = 10
    
    # M1: (A00+A11)(B00+B11)
    # This is a 2x2x2 multiplication of the sum of blocks.
    # We need to add 7 terms for this.
    # The "local" indices in strassen_2x2x2 refer to the 2x2 element indices.
    # But here the "elements" are 2x2 blocks.
    # So we need to map the local 2x2 indices to the global 4x4 indices.
    
    # Let's modify strassen_2x2x2 to take a mapping function.
    
    def add_strassen_block(u_map, v_map, w_map):
        """
        u_map: dict local_idx -> global_idx
        """
        nonlocal current_rank
        
        def add_local_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal current_rank
            r = current_rank
            current_rank += 1
            for lu, cu in u_coeffs:
                U[u_map[lu], r] += cu
            for lv, cv in v_coeffs:
                V[v_map[lv], r] += cv
            for lw, cw in w_coeffs:
                W[w_map[lw], r] += cw
        
        # M1
        add_local_term([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)])
        # M2
        add_local_term([(2,1), (3,1)], [(0,1)], [(2,1)])
        # M3
        add_local_term([(0,1)], [(1,1), (3,-1)], [(1,1)])
        # M4
        add_local_term([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)])
        # M5
        add_local_term([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)])
        # M6
        add_local_term([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)])
        # M7
        add_local_term([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)])
        
        return current_rank

    # Define mappings for M(4,4,4)
    # A00: 0,1,2,3 -> 0,1,4,5
    # A01: 0,1,2,3 -> 2,3,6,7
    # A10: 0,1,2,3 -> 8,9,12,13
    # A11: 0,1,2,3 -> 10,11,14,15
    
    def map_4x4(block_i, block_j, local_idx):
        # local_idx 0..3
        # block_i, block_j 0..1
        # global idx = (block_i*2 + local_row)*2 + (block_j*2 + local_col)
        row = local_idx // 2
        col = local_idx % 2
        return (block_i * 2 + row) * 2 + (block_j * 2 + col)
        
    # M1: A00+A11, B00+B11, C00+C11
    u_map = {
        0: map_4x4(0,0,0), 1: map_4x4(0,0,1), 2: map_4x4(0,0,2), 3: map_4x4(0,0,3),
        4: map_4x4(1,1,0), 5: map_4x4(1,1,1), 6: map_4x4(1,1,2), 7: map_4x4(1,1,3)
    }
    # But strassen_2x2x2 uses local indices 0..3 for the 2x2 matrix.
    # Here the "2x2 matrix" is the block of 2x2 matrices.
    # So local index 0 corresponds to the top-left 2x2 block of the sum.
    # This is getting complicated.
    
    # Simpler: Just use the rank 56 solution for 4x4x4 (2 multiplications per block).
    # 4 blocks * 2 mults * 7 rank = 56.
    # And 4x4x2: 2 blocks * 2 mults * 7 rank = 28.
    # Total 84.
    # 84 is better than 96.
    
    # Let's stick to 84 for simplicity and correctness.
    
    # Reset U, V, W
    U = np.zeros((n*m, 100))
    V = np.zeros((m*p, 100))
    W = np.zeros((p*n, 100))
    current_rank = 0
    
    def add_2x2x2(u_base, v_base, w_base):
        nonlocal current_rank
        # 7 terms
        terms = [
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]), # M1
            ([(2,1), (3,1)], [(0,1)], [(2,1)]), # M2
            ([(0,1)], [(1,1), (3,-1)], [(1,1)]), # M3
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]), # M4
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]), # M5
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]), # M6
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]), # M7
        ]
        for u_c, v_c, w_c in terms:
            r = current_rank
            current_rank += 1
            for lu, cu in u_c:
                U[u_base + lu, r] += cu
            for lv, cv in v_c:
                V[v_base + lv, r] += cv
            for lw, cw in w_c:
                W[w_base + lw, r] += cw

    # 4x4x4 part
    # C00 = A00B00 + A01B10
    # A00B00: u_base=0, v_base=0, w_base=0
    add_2x2x2(0, 0, 0)
    # A01B10: u_base=2, v_base=8, w_base=0
    add_2x2x2(2, 8, 0)
    
    # C01 = A00B01 + A01B11
    # A00B01: u_base=0, v_base=2, w_base=2
    add_2x2x2(0, 2, 2)
    # A01B11: u_base=2, v_base=10, w_base=2
    add_2x2x2(2, 10, 2)
    
    # C10 = A10B00 + A11B10
    # A10B00: u_base=8, v_base=0, w_base=8
    add_2x2x2(8, 0, 8)
    # A11B10: u_base=10, v_base=8, w_base=8
    add_2x2x2(10, 8, 8)
    
    # C11 = A10B01 + A11B11
    # A10B01: u_base=8, v_base=2, w_base=10
    add_2x2x2(8, 2, 10)
    # A11B11: u_base=10, v_base=10, w_base=10
    add_2x2x2(10, 10, 10)
    
    # 4x4x2 part (cols 4,5 of B)
    # B is 4x6. Cols 0-3 done. Cols 4-5 remaining.
    # B00 (cols 4-5): v_base = 4*2 + 0 = 8?
    # V indices: j*p+k.
    # j=0, k=4,5 -> 0*6+4=4, 0*6+5=5. Base 4.
    # j=1, k=4,5 -> 1*6+4=10, 1*6+5=11. Base 10.
    
    # C00 (rows 0-1 of C, cols 4-5): w_base = 4*4 + 0 = 16?
    # W indices: k*n+i.
    # k=0, i=0,1 -> 0*4+0=0, 0*4+1=1.
    # k=1, i=0,1 -> 1*4+0=4, 1*4+1=5.
    # Wait, C is 4x6.
    # C rows 0-1, cols 4-5.
    # i=0,1. k=4,5.
    # W index = k*n + i.
    # k=4, i=0 -> 16.
    # k=4, i=1 -> 17.
    # k=5, i=0 -> 20.
    # k=5, i=1 -> 21.
    # Base 16.
    
    # C10 (rows 2-3 of C, cols 4-5):
    # i=2,3. k=4,5.
    # W index = k*n + i.
    # k=4, i=2 -> 18.
    # k=4, i=3 -> 19.
    # k=5, i=2 -> 22.
    # k=5, i=3 -> 23.
    # Base 18.
    
    # C00 = A00B00 + A01B10
    # A00B00: u_base=0, v_base=4, w_base=16
    add_2x2x2(0, 4, 16)
    # A01B10: u_base=2, v_base=10, w_base=16
    add_2x2x2(2, 10, 16)
    
    # C10 = A10B00 + A11B10
    # A10B00: u_base=8, v_base=4, w_base=18
    add_2x2x2(8, 4, 18)
    # A11B10: u_base=10, v_base=10, w_base=18
    add_2x2x2(10, 10, 18)
    
    # Trim to current_rank
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {res['U'].shape[1]}")
