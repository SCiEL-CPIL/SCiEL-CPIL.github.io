"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24
    
    # Build the multiplication tensor T
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    def validate(U, V, W):
        """Check if U, V, W correctly decompose T."""
        rank = len(U[0]) if isinstance(U, list) else U.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(rank):
            if isinstance(U, list):
                u = np.array(U[:, r], dtype=np.float64)
                v = np.array(V[:, r], dtype=np.float64)
                w = np.array(W[:, r], dtype=np.float64)
            else:
                u = U[:, r].astype(np.float64)
                v = V[:, r].astype(np.float64)
                w = W[:, r].astype(np.float64)
            T_recon += np.einsum('i,j,k->ijk', u, v, w)
        return np.allclose(T_recon, T.astype(np.float64), atol=1e-9)
    
    # Schoolbook baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.int8)
    V_school = np.zeros((mp, rank_school), dtype=np.int8)
    W_school = np.zeros((pn, rank_school), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Strassen decomposition for 2x2x2
    def strassen_2x2():
        """
        Return Strassen rank-7 decomposition for 2x2 matrix multiplication.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indexing: U[a*2+b], V[c*2+d], W[d*2+a]
        """
        # Strassen's 7 products:
        # M1 = (A00+A11)*(B00+B11) -> C00+=M1, C01-=M1, C10+=M1, C11+=M1
        # M2 = (A10+A11)*B00       -> C10+=M2, C11+=M2
        # M3 = A00*(B10-B11)       -> C00+=M3, C01+=M3
        # M4 = A11*(B01-B00)       -> C00-=M4, C10+=M4
        # M5 = (A00+A01)*B11       -> C01+=M5, C11+=M5
        # M6 = (A10-A00)*(B00+B10) -> C10+=M6
        # M7 = (A01-A11)*(B01+B11) -> C01+=M7
        
        U = np.zeros((4, 7), dtype=np.int8)
        V = np.zeros((4, 7), dtype=np.int8)
        W = np.zeros((4, 7), dtype=np.int8)
        
        # M1
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[1,0] = -1; W[2,0] = 1; W[3,0] = 1
        
        # M2
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1; W[3,1] = 1
        
        # M3
        U[0,2] = 1
        V[2,2] = 1; V[3,2] = -1
        W[0,2] = 1; W[1,2] = 1
        
        # M4
        U[3,3] = 1
        V[1,3] = 1; V[0,3] = -1
        W[0,3] = -1; W[2,3] = 1
        
        # M5
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[1,4] = 1; W[3,4] = 1
        
        # M6
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[2,5] = 1
        W[2,5] = 1
        
        # M7
        U[1,6] = 1; U[3,6] = -1
        V[1,6] = 1; V[3,6] = 1
        W[1,6] = 1
        
        return U, V, W
    
    # Try block decomposition: split 3 rows into 1+2
    # 1x4x8: rank 8 (schoolbook for 1x4x8)
    # 2x4x8: use Strassen on 2x2x2 blocks, then extend to 4x8
    
    # Actually, let's think more carefully.
    # 3x4x8 can be viewed as:
    # Row 0: 1x4x8 (rank 8)
    # Rows 1-2: 2x4x8
    
    # For 2x4x8:
    # We can split the 4 columns into two 2-column blocks: 2x2x8 + 2x2x8
    # Each 2x2x8 can be split into four 2x2x2 blocks: 4 * Strassen(2x2x2) = 4*7 = 28
    # So 2x4x8 would have rank 2*28 = 56? No, that's not right.
    
    # Let me reconsider. 2x4x8:
    # Split 4 into 2+2: two 2x2x8 blocks
    # Each 2x2x8: split 8 into 2+2+2+2: four 2x2x2 blocks
    # Each 2x2x2 has rank 7 (Strassen)
    # So 2x2x8 has rank 4*7 = 28
    # 2x4x8 has rank 2*28 = 56
    # Total: 8 + 56 = 64
    
    # Let's construct this:
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: Row 0, 1x4x8 (schoolbook, rank 8)
    for j in range(4):
        for k in range(8):
            u = np.zeros(nm, dtype=np.int8)
            v = np.zeros(mp, dtype=np.int8)
            w = np.zeros(pn, dtype=np.int8)
            u[0*m + j] = 1
            v[j*p + k] = 1
            w[k*n + 0] = 1
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
    
    # Part 2: Rows 1-2, 2x4x8
    # Split into 2x2x2 blocks using Strassen
    # Row indices: 1, 2 (a=0 maps to row 1, a=1 maps to row 2)
    # Col indices: 0,1,2,3 (b)
    # Out indices: 0,1,2,3,4,5,6,7 (k)
    
    # For each pair of columns (b0, b1) and each pair of output indices (k0, k1):
    # We have a 2x2x2 subproblem
    # Rows: a=0,1 -> global rows 1,2
    # Cols: b0,b1 -> global cols b0,b1  
    # Outs: k0,k1 -> global outs k0,k1
    
    S, Sv, Sw = strassen_2x2()
    
    for b0 in range(0, 4, 2):
        b1 = b0 + 1
        for k0 in range(0, 8, 2):
            k1 = k0 + 1
            # Apply Strassen to this 2x2x2 block
            for r in range(7):
                u = np.zeros(nm, dtype=np.int8)
                v = np.zeros(mp, dtype=np.int8)
                w = np.zeros(pn, dtype=np.int8)
                
                # U indices: a*2+b where a in {0,1}, b in {0,1}
                # Global row = 1 + a (since we're doing rows 1,2)
                for a in range(2):
                    for b in range(2):
                        idx = a * 2 + b
                        if S[idx, r] != 0:
                            global_row = 1 + a
                            global_col = b0 + b
                            u[global_row * m + global_col] = S[idx, r]
                
                # V indices: c*2+d where c in {0,1}, d in {0,1}
                # c corresponds to b (column), d corresponds to k (output)
                # Wait, I need to be careful about the indexing.
                # In the 2x2x2 Strassen:
                # U[a*2+b], V[c*2+d], W[d*2+a]
                # a = row (0,1), b = col (0,1), c = col (0,1), d = out (0,1)
                # But b and c should be the same (the inner dimension)
                # So V[c*2+d] where c is the column index in the 2x2 block
                
                for c in range(2):
                    for d in range(2):
                        idx = c * 2 + d
                        if Sv[idx, r] != 0:
                            global_col = b0 + c
                            global_out = k0 + d
                            v[global_col * p + global_out] = Sv[idx, r]
                
                # W indices: d*2+a
                for d in range(2):
                    for a in range(2):
                        idx = d * 2 + a
                        if Sw[idx, r] != 0:
                            global_out = k0 + d
                            global_row = 1 + a
                            w[global_out * n + global_row] = Sw[idx, r]
                
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    # Convert to arrays
    U_new = np.column_stack(U_list)
    V_new = np.column_stack(V_list)
    W_new = np.column_stack(W_list)
    
    new_rank = U_new.shape[1]
    
    # Validate
    if validate(U_new, V_new, W_new):
        if new_rank < best_rank:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = new_rank
    
    # Try another approach: full Strassen-style decomposition
    # 3x4x8: split 3 into 1+2, 4 into 2+2, 8 into 4+4 or 2+2+2+2
    
    # Alternative: Use the fact that 3x4x8 = 3x4x4 + 3x4x4
    # And 3x4x4 can be further decomposed
    
    # For now, return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
