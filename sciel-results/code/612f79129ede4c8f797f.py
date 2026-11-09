import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    # U: (n*m, R) = (16, R)
    # V: (m*p, R) = (24, R)  
    # W: (p*n, R) = (24, R)
    
    # Top-level Strassen: 7 multiplications of (2x2) x (2x3)
    # For each (2x2)x(2x3): use inner Strassen for 2x2x2 (7 mults) + schoolbook for 2x2x1 (4 mults) = 11
    # Total rank: 7 * 11 = 77
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Top-level Strassen coefficients
    # A blocks: 0=00, 1=01, 2=10, 3=11
    # B blocks: 0=00, 1=01, 2=10, 3=11
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},  # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # s=6: A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}   # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]
    
    # Inner Strassen for 2x2x2 (same coefficients)
    inner_alpha = alpha
    inner_beta = beta
    inner_recon = recon
    
    col = 0
    for s in range(7):
        # For each top-level Strassen term s:
        # X_s is a 2x2 matrix, Y_s is a 2x3 matrix
        # Compute X_s * Y_s using: inner Strassen for cols 0-1, schoolbook for col 2
        
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            # Inner Strassen term si
            # X_s * (inner combination of Y_s columns 0,1)
            # The inner Strassen combines Y_s[:,0] and Y_s[:,1]
            # Inner A-blocks are rows of X_s: X_s[0,0], X_s[0,1], X_s[1,0], X_s[1,1]
            # Inner B-blocks are columns of Y_s[:,0:2]: Y_s[0,0], Y_s[0,1], Y_s[1,0], Y_s[1,1]
            
            # U part: X_s entries with inner_alpha[si] coefficients
            # X_s[i,j] = sum_r alpha[s][r] * A_r[i,j]
            # Inner Strassen: inner_X[i,j] = sum_a inner_alpha[si][a] * X_s[a//2, a%2]
            # So U[i,j] = sum_a inner_alpha[si][a] * sum_r alpha[s][r] * A_r[i,j]
            
            for a in range(4):
                if inner_alpha[si][a] != 0:
                    # a indexes position in X_s: row=a//2, col=a%2
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            # A_r[x_row, x_col]
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += inner_alpha[si][a] * alpha[s][r]
            
            # V part: Y_s entries with inner_beta[si] coefficients
            # Inner B-blocks: Y_s[0,0], Y_s[0,1], Y_s[1,0], Y_s[1,1]
            # inner_beta[si][b] where b=0: Y_s[0,0], b=1: Y_s[0,1], b=2: Y_s[1,0], b=3: Y_s[1,1]
            # Y_s[i,j] = sum_c beta[s][c] * B_c[i,j]
            # For inner col 0: Y_s[i,0], for inner col 1: Y_s[i,1]
            # b=0: Y_s[0,0] -> i=0, inner_j=0
            # b=1: Y_s[0,1] -> i=0, inner_j=1
            # b=2: Y_s[1,0] -> i=1, inner_j=0
            # b=3: Y_s[1,1] -> i=1, inner_j=1
            
            for b in range(4):
                if inner_beta[si][b] != 0:
                    # b indexes position in Y_s[:,0:2]: row=b//2, col=b%2
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            # B_c[y_row, y_inner_col]
                            # B_c is 2x3, so B_c[row, col] where col is 0,1,2
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += inner_beta[si][b] * beta[s][c]
            
            # W part: output of inner Strassen
            # inner_recon[si] gives (out_row, out_col, gamma)
            # Output is a 2x2 matrix Z, Z[out_row, out_col]
            # Z[out_row, out_col] contributes to C at position determined by recon[s]
            # The inner output Z is 2x2, and it multiplies into the top-level recon
            # Top-level: C_ab gets contribution from X_s * Y_s
            # The inner Strassen computes (X_s) * (Y_s[:,0:2]) which is a 2x2 matrix
            # This 2x2 matrix is the "inner C" which then gets placed via top-level recon
            
            # Actually, let me reconsider the structure.
            # Top-level: C = sum_s recon[s] * (X_s * Y_s)
            # X_s * Y_s is 2x3. Let's call it P_s (2x3).
            # P_s = X_s * Y_s
            # Inner Strassen computes P_s[:, 0:2] = X_s * Y_s[:, 0:2]
            # This is a 2x2 matrix.
            # The inner Strassen output Z (2x2) = X_s * Y_s[:,0:2]
            # Z[i,j] for i in {0,1}, j in {0,1}
            # This corresponds to P_s[i,j] for j in {0,1}
            
            # In the top-level, P_s is a 2x3 block that gets added to C blocks via recon[s]
            # C_ab[u,w] gets gamma * P_s[u,w] where (a,b,gamma) in recon[s]
            # P_s[u,w] for w in {0,1} comes from inner Strassen
            # P_s[u,2] comes from schoolbook
            
            # So for inner Strassen output Z[i,j]:
            # This is P_s[i,j] (i=u, j=w where w=0 or 1)
            # It contributes to C_ab[i, j] with coefficient gamma from recon[s]
            # where (a,b,gamma) in recon[s]
            
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    # za, zb are positions in the 2x2 inner output
                    # This is P_s[za, zb]
                    # It contributes to C block (a,b) at position (za, zb)
                    # C global: row = a*2 + za, col = b*3 + zb
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        # X_s (2x2) * Y_s[:,2] (2x1) -> Z (2x1)
        # Z[0] = X_s[0,0]*Y_s[0,2] + X_s[0,1]*Y_s[1,2]
        # Z[1] = X_s[1,0]*Y_s[0,2] + X_s[1,1]*Y_s[1,2]
        # 4 schoolbook multiplications
        
        for u in range(2):
            for v in range(2):
                # Term: X_s[u,v] * Y_s[v,2]
                
                # U part
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V part
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        # B_c[v, 2]
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W part
                for a, b, gamma in recon[s]:
                    # C_ab[u, 2]
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
                
    # Check for linear dependencies to potentially reduce rank
    # If the columns of U are linearly dependent, we might be able to reduce the rank.
    # However, for a valid decomposition, we must remove columns from U, V, and W simultaneously.
    # A simple heuristic: check if any column is zero.
    # A more robust check: compute the rank of U. If rank(U) < R, we have dependencies.
    # But removing columns requires knowing which ones to remove such that the tensor sum remains invariant.
    # This is generally hard. For Strassen, the columns are typically independent.
    # We will just check for zero columns to be safe.
    
    keep_cols = []
    for k in range(R):
        if np.any(U[:, k] != 0) and np.any(V[:, k] != 0) and np.any(W[:, k] != 0):
            keep_cols.append(k)
            
    if len(keep_cols) < R:
        U = U[:, keep_cols]
        V = V[:, keep_cols]
        W = W[:, keep_cols]
        
    # Convert to lists
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
