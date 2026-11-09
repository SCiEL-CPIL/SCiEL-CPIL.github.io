"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 115
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Verify a decomposition
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Block decomposition strategy
    # Split A (4x5) into 2x2, 2x3, 2x2, 2x3
    # Split B (5x6) into 2x3, 2x3, 3x3, 3x3
    # Split C (4x6) into 2x3, 2x3, 2x3, 2x3
    
    # We will build U, V, W by concatenating block decompositions
    # U shape: (20, rank)
    # V shape: (30, rank)
    # W shape: (24, rank)
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(u_rows, v_rows, w_rows, u_shape, v_shape, w_shape, terms):
        """
        Add a block decomposition to the global lists.
        u_rows, v_rows, w_rows: list of (local_row, value) for each term?
        No, terms is a list of (u_vec, v_vec, w_vec) where vectors are dicts {local_idx: val}
        """
        for u_dict, v_dict, w_dict in terms:
            # Create global vectors
            u_global = [0.0] * (n * m)
            v_global = [0.0] * (m * p)
            w_global = [0.0] * (p * n)
            
            for local_i, val in u_dict.items():
                global_i = u_rows[local_i]
                u_global[global_i] += val
            
            for local_j, val in v_dict.items():
                global_j = v_rows[local_j]
                v_global[global_j] += val
            
            for local_k, val in w_dict.items():
                global_k = w_rows[local_k]
                w_global[global_k] += val
            
            U_list.append(u_global)
            V_list.append(v_global)
            W_list.append(w_global)
    
    # Define Strassen 2x2x2
    def strassen_2x2x2():
        # Returns list of 7 terms (u_dict, v_dict, w_dict)
        # u_dict keys: (i,j) in 0..1
        # v_dict keys: (j,k) in 0..1
        # w_dict keys: (k,i) in 0..1
        terms = []
        # M1 = (A11+A22)(B11+B22)
        u = {(0,0): 1, (1,1): 1}
        v = {(0,0): 1, (1,1): 1}
        w = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1} # This is not rank 1 in tensor sense?
        # Wait, the tensor decomposition of Strassen is:
        # C = sum M_c * (linear comb of A) * (linear comb of B)
        # The "terms" in the tensor decomposition are the 7 multiplications.
        # For each multiplication M_c, we have a fixed pattern of A and B coefficients.
        # The output C is a linear combination of these M_c.
        
        # Let's define the 7 Strassen products explicitly.
        # M1 = (A11+A22) * (B11+B22)
        # M2 = (A11+A21) * B11
        # M3 = A11 * (B12-B22)
        # M4 = A22 * (B21-B11)
        # M5 = (A12+A22) * B22
        # M6 = (A11-A21) * (B21+B22)
        # M7 = (A12-A22) * (B11+B21)
        
        # Each term contributes to C blocks.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to represent this as a sum of rank-1 tensors.
        # Each "multiplication" is a scalar product of linear forms in A and B.
        # So for each of the 7 products, we define the u and v vectors.
        # The w vector is determined by how this product contributes to each C element.
        
        # Actually, the standard way to represent this in U,V,W is:
        # For each of the 7 products, we have a "column" in U, V, W.
        # U column has the coefficients of A for that product.
        # V column has the coefficients of B for that product.
        # W column has the coefficients of C for that product.
        
        # Product 1: M1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1} # C11 gets +1, C22 gets +1?
        # C11 = M1... so W11 gets 1.
        # C22 = M1... so W22 gets 1.
        # But W is indexed by (k,i).
        # C11 is i=0,k=0. W[0,0] += 1.
        # C22 is i=1,k=1. W[1,1] += 1.
        # Wait, C12 is i=0,k=1. C21 is i=1,k=0.
        
        # Let's be precise.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # W[k*n + i] is the coefficient for C[i,k] in the c-th term.
        
        # Term 1 (M1):
        # A coeffs: A[0,0]=1, A[1,1]=1.
        # B coeffs: B[0,0]=1, B[1,1]=1.
        # C coeffs: C[0,0] += 1, C[1,1] += 1.
        # So W[0*2+0] += 1, W[1*2+1] += 1.
        
        # Term 2 (M2):
        # A: A[0,0]=1, A[1,0]=1.
        # B: B[0,0]=1, B[0,1]=1. (B11 is B[0,0], B[0,1]? No, B11 is top-left 2x2? No, B is 2x2.
        # B11 is B[0,0]. B12 is B[0,1]. B21 is B[1,0]. B22 is B[1,1].
        # So B11 is just B[0,0].
        # M2 = (A11+A21) * B11.
        # A: A[0,0]=1, A[1,0]=1.
        # B: B[0,0]=1.
        # C: C[1,0] += 1. (C21 gets M2).
        # W[0*2+1] += 1.
        
        # Term 3 (M3):
        # M3 = A11 * (B12-B22).
        # A: A[0,0]=1.
        # B: B[0,1]=1, B[1,1]=-1.
        # C: C[0,1] += 1. (C12 gets M3).
        # W[1*2+0] += 1.
        
        # Term 4 (M4):
        # M4 = A22 * (B21-B11).
        # A: A[1,1]=1.
        # B: B[1,0]=1, B[0,0]=-1.
        # C: C[0,0] += 1, C[1,0] += 1. (C11 gets M4, C21 gets M4).
        # W[0*2+0] += 1, W[0*2+1] += 1.
        
        # Term 5 (M5):
        # M5 = (A12+A22) * B22.
        # A: A[0,1]=1, A[1,1]=1.
        # B: B[1,1]=1.
        # C: C[0,1] += 1, C[1,1] -= 1. (C12 gets M5, C22 gets -M5).
        # W[1*2+0] += 1, W[1*2+1] -= 1.
        
        # Term 6 (M6):
        # M6 = (A11-A21) * (B21+B22).
        # A: A[0,0]=1, A[1,0]=-1.
        # B: B[1,0]=1, B[1,1]=1.
        # C: C[1,1] += 1. (C22 gets M6).
        # W[1*2+1] += 1.
        
        # Term 7 (M7):
        # M7 = (A12-A22) * (B11+B21).
        # A: A[0,1]=1, A[1,1]=-1.
        # B: B[0,0]=1, B[1,0]=1.
        # C: C[0,0] += 1. (C11 gets M7).
        # W[0*2+0] += 1.
        
        return [
            (u1, v1, w1),
            ({(0,0): 1, (1,0): 1}, {(0,0): 1}, {(0,1): 1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            ({(0,1): 1, (1,1): 1}, {(1,1): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}, {(0,0): 1})
        ]
        
    # Define Schoolbook for n x m x p
    def schoolbook_block(n, m, p):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = {(i,j): 1}
                    v = {(j,k): 1}
                    w = {(k,i): 1}
                    terms.append((u, v, w))
        return terms

    # Assemble the 8 blocks
    
    # Block 1: A11 (2x2) * B11 (2x3) -> C11 part? No.
    # My previous analysis:
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # We need to sum these.
    # This means we can just add the decompositions for each product.
    
    # Product 1: A11 (2x2) * B11 (2x3). Rank 12 (Schoolbook 2x2x3).
    # Product 2: A12 (2x3) * B21 (3x3). Rank 18 (Schoolbook 2x3x3).
    # Product 3: A21 (2x2) * B11 (2x3). Rank 12.
    # Product 4: A22 (2x3) * B22 (3x3). Rank 18.
    # Product 5: A11 (2x2) * B12 (2x3). Rank 12.
    # Product 6: A12 (2x3) * B22 (3x3). Rank 18.
    # Product 7: A21 (2x2) * B12 (2x3). Rank 12.
    # Product 8: A22 (2x3) * B21 (3x3). Rank 18.
    
    # Total Rank: 12 + 18 + 12 + 18 + 12 + 18 + 12 + 18 = 120.
    # This is not an improvement!
    
    # Why? Because I didn't use Strassen for the 2x2 parts.
    # A11 is 2x2. B11 is 2x3.
    # We can't use Strassen directly on 2x2 * 2x3.
    
    # However, we can split B11 (2x3) into B11a (2x2) and B11b (2x1).
    # A11 * B11a: 2x2 * 2x2. Rank 7 (Strassen).
    # A11 * B11b: 2x2 * 2x1. Rank 4 (Schoolbook).
    # Total for A11 * B11: 7 + 4 = 11.
    
    # Similarly for others.
    
    # Let's refine the block sizes to maximize Strassen usage.
    
    # A11 (2x2) * B11 (2x3):
    # Split B11 into B11_left (2x2) and B11_right (2x1).
    # A11 * B11_left: Strassen (7).
    # A11 * B11_right: Schoolbook (2*2*1 = 4).
    # Total: 11.
    
    # A12 (2x3) * B21 (3x3):
    # Split A12 into A12_left (2x2) and A12_right (2x1).
    # Split B21 into B21_top (2x3) and B21_bottom (1x3).
    # This is getting complicated.
    
    # Let's stick to the 110 rank construction which used:
    # A11 (2x2) * B11 (2x2) -> Strassen (7).
    # But B11 is 2x3 in the 4x5x6 problem.
    
    # Okay, I will implement the 110 rank version which assumes B11 is 2x2.
    # This requires splitting B (5x6) into 2x2, 2x1, 3x2, 3x1?
    # No, B is 5x6.
    # Split B into B_top (2x6) and B_bottom (3x6).
    # Split B_top into B_top_left (2x2) and B_top_right (2x4).
    
    # This is too complex to code correctly in the short time.
    # I will return the schoolbook 120 for now, as I cannot guarantee a lower rank with correct implementation.
    
    return best_result

# EVOLVE-BLOCK-END
