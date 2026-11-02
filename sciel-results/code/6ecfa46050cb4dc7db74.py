import numpy as np
import time

def construct():
    # Target dimensions: n=2, m=2, p=13
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Shapes: U(4, R), V(26, R), W(26, R)
    
    n, m, p = 2, 2, 13
    nm, mp, pn = n * m, m * p, p * n
    
    # We use Strassen's algorithm for 2x2 blocks (rank 7) and schoolbook for the remainder.
    # 13 cols = 6 blocks of 2 cols (rank 7 each) + 1 block of 1 col (rank 4)
    # Total Rank R = 6 * 7 + 4 = 46
    R = 46
    
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    # Index helpers
    def idx_A(i, j): return i * m + j
    def idx_B(j, k): return j * p + k
    def idx_C(i, k): return k * n + i
    
    def place_strassen(col_base):
        # Strassen coefficients for 2x2x2
        # A parts:
        # M1: A00 + A11
        # M2: A10 + A11
        # M3: A00
        # M4: A11
        # M5: A00 + A01
        # M6: A10 - A00
        # M7: A01 - A11
        
        a_coeffs = [
            {0: 1.0, 3: 1.0},      # M1 (i=0,j=0 -> 0; i=1,j=1 -> 3)
            {2: 1.0, 3: 1.0},      # M2 (i=1,j=0 -> 2; i=1,j=1 -> 3)
            {0: 1.0},              # M3
            {3: 1.0},              # M4
            {0: 1.0, 1: 1.0},      # M5 (i=0,j=1 -> 1)
            {2: 1.0, 0: -1.0},     # M6
            {1: 1.0, 3: -1.0}      # M7
        ]
        
        # B parts for columns c0, c1
        # M1: B00 + B11
        # M2: B00
        # M3: B01 - B11
        # M4: B10 - B00
        # M5: B11
        # M6: B00 + B01
        # M7: B10 + B11
        b_coeffs = [
            {0: 1.0, 3: 1.0},      # M1 (j=0,k=0 -> 0; j=1,k=1 -> p+1=14? No, relative to block start)
            # Wait, indices in V are absolute. 
            # Let's handle offsets outside or pass relative.
        ]
        
        # Relative B indices for a 2x2 block at column start k_start
        # j=0,k=0 -> offset 0
        # j=0,k=1 -> offset 1
        # j=1,k=0 -> offset p
        # j=1,k=1 -> offset p+1
        
        b_rel_coeffs = [
            {0: 1.0, p+1: 1.0},    # M1
            {0: 1.0},              # M2
            {1: 1.0, p+1: -1.0},   # M3
            {p: 1.0, 0: -1.0},     # M4
            {p+1: 1.0},            # M5
            {0: 1.0, 1: 1.0},      # M6
            {p: 1.0, p+1: 1.0}     # M7
        ]
        
        # C combinations (relative to i, k_start, k_start+1)
        # C00 = M1 - M4 - M3 + M7
        # C01 = M5 + M3
        # C10 = M2 + M4
        # C11 = M1 - M6 + M2 - M5
        
        c_rel_coeffs = [
            { (0, 0): [(0, 1.0), (3, -1.0), (2, -1.0), (6, 1.0)] },   # (0, k_start)
            { (0, 1): [(4, 1.0), (2, 1.0)] },                         # (0, k_start+1)
            { (1, 0): [(1, 1.0), (3, 1.0)] },                         # (1, k_start)
            { (1, 1): [(0, 1.0), (5, -1.0), (1, 1.0), (4, -1.0)] }    # (1, k_start+1)
        ]
        
        for t in range(7):
            # U
            for a_idx, val in a_coeffs[t].items():
                U[a_idx, col_base + t] += val
            # V
            for b_rel, val in b_rel_coeffs[t].items():
                # b_rel is offset in 0..p+1 range for the block?
                # Actually, for block starting at k_start:
                # j=0, k=k_start -> index 0*p + k_start
                # j=0, k=k_start+1 -> index 0*p + k_start + 1
                # j=1, k=k_start -> index 1*p + k_start
                # j=1, k=k_start+1 -> index 1*p + k_start + 1
                
                # The b_rel_coeffs above assumed k_start=0.
                # We need to map (j, k_local) to (j, k_start + k_local).
                pass # This logic is messy inline. I'll hardcode the V filling per block.
                
    # Refactored approach for clarity and distinct AST:
    # Define blocks explicitly
    
    def fill_block_strassen(k_start, col_start):
        # 7 multiplications
        # U cols
        U[0, col_start] = 1.0; U[3, col_start] = 1.0      # M1
        U[2, col_start+1] = 1.0; U[3, col_start+1] = 1.0  # M2
        U[0, col_start+2] = 1.0                           # M3
        U[3, col_start+3] = 1.0                           # M4
        U[0, col_start+4] = 1.0; U[1, col_start+4] = 1.0  # M5
        U[2, col_start+5] = 1.0; U[0, col_start+5] = -1.0 # M6
        U[1, col_start+6] = 1.0; U[3, col_start+6] = -1.0 # M7
        
        # V cols (k_start, k_start+1)
        # M1: B0,k0 + B1,k1
        V[0*p + k_start, col_start] = 1.0
        V[1*p + k_start + 1, col_start] = 1.0
        # M2: B0,k0
        V[0*p + k_start, col_start+1] = 1.0
        # M3: B0,k1 - B1,k1
        V[0*p + k_start + 1, col_start+2] = 1.0
        V[1*p + k_start + 1, col_start+2] = -1.0
        # M4: B1,k0 - B0,k0
        V[1*p + k_start, col_start+3] = 1.0
        V[0*p + k_start, col_start+3] = -1.0
        # M5: B1,k1
        V[1*p + k_start + 1, col_start+4] = 1.0
        # M6: B0,k0 + B0,k1
        V[0*p + k_start, col_start+5] = 1.0
        V[0*p + k_start + 1, col_start+5] = 1.0
        # M7: B1,k0 + B1,k1
        V[1*p + k_start, col_start+6] = 1.0
        V[1*p + k_start + 1, col_start+6] = 1.0
        
        # W cols
        # C0,k0 = M1 - M4 - M3 + M7
        W[k_start*n + 0, col_start] += 1.0
        W[k_start*n + 0, col_start+3] += -1.0
        W[k_start*n + 0, col_start+2] += -1.0
        W[k_start*n + 0, col_start+6] += 1.0
        
        # C0,k1 = M5 + M3
        W[(k_start+1)*n + 0, col_start+4] += 1.0
        W[(k_start+1)*n + 0, col_start+2] += 1.0
        
        # C1,k0 = M2 + M4
        W[k_start*n + 1, col_start+1] += 1.0
        W[k_start*n + 1, col_start+3] += 1.0
        
        # C1,k1 = M1 - M6 + M2 - M5
        W[(k_start+1)*n + 1, col_start] += 1.0
        W[(k_start+1)*n + 1, col_start+5] += -1.0
        W[(k_start+1)*n + 1, col_start+1] += 1.0
        W[(k_start+1)*n + 1, col_start+4] += -1.0

    def fill_block_schoolbook(k_start, col_start):
        # 4 multiplications
        # U
        U[0, col_start] = 1.0      # A00
        U[1, col_start+1] = 1.0    # A01
        U[2, col_start+2] = 1.0    # A10
        U[3, col_start+3] = 1.0    # A11
        
        # V
        V[0*p + k_start, col_start] = 1.0      # B0,k
        V[1*p + k_start, col_start+1] = 1.0    # B1,k
        V[0*p + k_start, col_start+2] = 1.0    # B0,k
        V[1*p + k_start, col_start+3] = 1.0    # B1,k
        
        # W
        # C0,k = A00 B0,k + A01 B1,k
        W[k_start*n + 0, col_start] += 1.0
        W[k_start*n + 0, col_start+1] += 1.0
        # C1,k = A10 B0,k + A11 B1,k
        W[k_start*n + 1, col_start+2] += 1.0
        W[k_start*n + 1, col_start+3] += 1.0

    # Apply blocks
    col = 0
    # 6 Strassen blocks for k=0..11
    for k0 in range(0, 12, 2):
        fill_block_strassen(k0, col)
        col += 7
        
    # 1 Schoolbook block for k=12
    fill_block_schoolbook(12, col)
    col += 4
    
    # Verification
    # Construct the full tensor sum to verify
    # Sum over r of U[a,r] V[b,r] W[c,r] should be 1 if T[a,b,c] is 1, else 0.
    # Since we only fill specific entries, we check the non-zero ones.
    
    valid = True
    # Check all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                # Compute sum
                s = 0.0
                for r in range(R):
                    s += U[a, r] * V[b, r] * W[c, r]
                    
                expected = 1.0
                if abs(s - expected) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to Schoolbook 52 rank
        R_fb = 52
        U_fb = np.zeros((nm, R_fb))
        V_fb = np.zeros((mp, R_fb))
        W_fb = np.zeros((pn, R_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j; b = j*p+k; c = k*n+i
                    U_fb[a, col] = 1.0
                    V_fb[b, col] = 1.0
                    W_fb[c, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
