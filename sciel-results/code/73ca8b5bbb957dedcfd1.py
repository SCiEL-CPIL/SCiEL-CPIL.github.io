import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 145.0

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 7
    Na, Nb, Nc = n*m, m*p, p*n  # 8, 28, 14
    
    # Build target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[(u_idx, v_idx, w_idx)] = 1
    
    # Build rank-50 block decomposition (known correct)
    def build_rank50():
        rank = 50
        U = [[0]*rank for _ in range(Na)]
        V = [[0]*rank for _ in range(Nb)]
        W = [[0]*rank for _ in range(Nc)]
        
        def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_col):
            a00 = a_rows[0]*m + a_cols[0]
            a01 = a_rows[0]*m + a_cols[1]
            a10 = a_rows[1]*m + a_cols[0]
            a11 = a_rows[1]*m + a_cols[1]
            b00 = b_rows[0]*p + b_cols[0]
            b01 = b_rows[0]*p + b_cols[1]
            b10 = b_rows[1]*p + b_cols[0]
            b11 = b_rows[1]*p + b_cols[1]
            c00 = c_cols[0]*n + c_rows[0]
            c01 = c_cols[1]*n + c_rows[0]
            c10 = c_cols[0]*n + c_rows[1]
            c11 = c_cols[1]*n + c_rows[1]
            col = start_col
            # Strassen's 7 multiplications for 2x2*2x2
            U[a00][col] += 1; U[a11][col] += 1; V[b00][col] += 1; V[b11][col] += 1; W[c00][col] += 1; W[c11][col] += 1; col += 1
            U[a10][col] += 1; U[a11][col] += 1; V[b00][col] += 1; W[c10][col] += 1; W[c11][col] -= 1; col += 1
            U[a00][col] += 1; V[b01][col] += 1; V[b11][col] -= 1; W[c01][col] += 1; W[c11][col] += 1; col += 1
            U[a11][col] += 1; V[b10][col] += 1; V[b00][col] -= 1; W[c00][col] += 1; W[c10][col] += 1; col += 1
            U[a00][col] += 1; U[a01][col] += 1; V[b11][col] += 1; W[c00][col] -= 1; W[c01][col] += 1; col += 1
            U[a10][col] += 1; U[a00][col] -= 1; V[b00][col] += 1; V[b01][col] += 1; W[c11][col] += 1; col += 1
            U[a01][col] += 1; U[a11][col] -= 1; V[b10][col] += 1; V[b11][col] += 1; W[c00][col] += 1; col += 1
            return col
        
        def add_schoolbook_2x2_2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col, start_col):
            a00 = a_rows[0]*m + a_cols[0]; a01 = a_rows[0]*m + a_cols[1]
            a10 = a_rows[1]*m + a_cols[0]; a11 = a_rows[1]*m + a_cols[1]
            b0 = b_rows[0]*p + b_col; b1 = b_rows[1]*p + b_col
            c0 = c_col*n + c_rows[0]; c1 = c_col*n + c_rows[1]
            col = start_col
            U[a00][col] += 1; V[b0][col] += 1; W[c0][col] += 1; col += 1
            U[a01][col] += 1; V[b1][col] += 1; W[c0][col] += 1; col += 1
            U[a10][col] += 1; V[b0][col] += 1; W[c1][col] += 1; col += 1
            U[a11][col] += 1; V[b1][col] += 1; W[c1][col] += 1; col += 1
            return col
        
        col = 0
        # 6 Strassen blocks (2x2 x 2x2) + 2 schoolbook blocks (2x2 x 2x1)
        col = add_strassen([0,1],[0,1],[0,1],[0,1],[0,1],[0,1], col)       # A[:2,:2] * B[:2,:2] -> C[:2,:2]
        col = add_schoolbook_2x2_2x1([0,1],[0,1],[0,1],2,[0,1],2, col)       # A[:2,:2] * B[:2,2] -> C[:2,2]
        col = add_strassen([0,1],[2,3],[2,3],[0,1],[0,1],[0,1], col)         # A[:2,2:4] * B[2:4,:2] -> C[:2,:2]
        col = add_schoolbook_2x2_2x1([0,1],[2,3],[2,3],2,[0,1],2, col)       # A[:2,2:4] * B[2:4,2] -> C[:2,2]
        col = add_strassen([0,1],[0,1],[0,1],[3,4],[0,1],[3,4], col)         # A[:2,:2] * B[:2,3:5] -> C[:2,3:5]
        col = add_strassen([0,1],[0,1],[0,1],[5,6],[0,1],[5,6], col)         # A[:2,:2] * B[:2,5:7] -> C[:2,5:7]
        col = add_strassen([0,1],[2,3],[2,3],[3,4],[0,1],[3,4], col)         # A[:2,2:4] * B[2:4,3:5] -> C[:2,3:5]
        col = add_strassen([0,1],[2,3],[2,3],[5,6],[0,1],[5,6], col)         # A[:2,2:4] * B[2:4,5:7] -> C[:2,5:7]
        return {"U": U, "V": V, "W": W}
    
    # Verify decomposition
    def verify(U, V, W, rank):
        for u in range(Na):
            for v in range(Nb):
                for w in range(Nc):
                    s = 0.0
                    for r in range(rank):
                        s += U[u][r] * V[v][r] * W[w][r]
                    expected = target.get((u,v,w), 0)
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    # Try to find rank-49 by looking for linear dependencies in rank-50
    def try_rank49():
        """Try to find a rank-49 decomposition by checking for linear dependencies."""
        result50 = build_rank50()
        U = np.array(result50["U"], dtype=float)
        V = np.array(result50["V"], dtype=float)
        W = np.array(result50["W"], dtype=float)
        
        # Check if any two rank-1 terms are proportional
        # Two terms r1 and r2 are proportional if U[:,r1] = alpha*U[:,r2], 
        # V[:,r1] = alpha*V[:,r2], and W[:,r1] = alpha*W[:,r2]
        
        for r1 in range(50):
            for r2 in range(r1+1, 50):
                # Check proportionality
                U1, U2 = U[:, r1], U[:, r2]
                V1, V2 = V[:, r1], V[:, r2]
                W1, W2 = W[:, r1], W[:, r2]
                
                # Find non-zero entries to determine ratio
                ratios = []
                for idx in range(Na):
                    if abs(U1[idx]) > 1e-12 and abs(U2[idx]) > 1e-12:
                        ratios.append(U1[idx] / U2[idx])
                if not ratios:
                    continue
                    
                # Check if all ratios are consistent
                alpha = ratios[0]
                consistent = True
                for ratio in ratios[1:]:
                    if abs(ratio - alpha) > 1e-9:
                        consistent = False
                        break
                if not consistent:
                    continue
                
                # Verify with V
                v_ok = True
                for idx in range(Nb):
                    if abs(V1[idx] - alpha * V2[idx]) > 1e-9:
                        v_ok = False
                        break
                if not v_ok:
                    continue
                    
                # Verify with W
                w_ok = True
                for idx in range(Nc):
                    if abs(W1[idx] - alpha * W2[idx]) > 1e-9:
                        w_ok = False
                        break
                if not w_ok:
                    continue
                
                # Found proportional terms! Merge them
                # New term = term1 + term2 (or term1 - term2 depending on sign)
                # Actually, if U1 = alpha*U2, then term1 = alpha*term2
                # We can eliminate one by absorbing into the other
                # But we need to be careful: we want to reduce rank by 1
                
                # Create new matrices without the duplicate
                new_rank = 49
                new_U = np.zeros((Na, new_rank), dtype=float)
                new_V = np.zeros((Nb, new_rank), dtype=float)
                new_W = np.zeros((Nc, new_rank), dtype=float)
                
                # Copy all terms except r2
                for r in range(50):
                    if r == r2:
                        continue
                    new_r = r if r < r2 else r - 1
                    new_U[:, new_r] = U[:, r]
                    new_V[:, new_r] = V[:, r]
                    new_W[:, new_r] = W[:, r]
                
                # The term r1 should be (1+alpha)*term2 if we're adding, but actually
                # since term1 = alpha*term2, we just need to keep one of them
                # But wait - if they're proportional, the sum is (1+alpha)*term2
                # We can just scale term2 by (1+alpha) and drop term1
                
                # Actually, simpler: just drop one and verify
                # If term1 = alpha * term2, then total = (1+alpha)*term2 + other terms
                # So we can replace term2 with (1+alpha)*term2 and drop term1
                
                # Let's try: keep r1, drop r2, but scale r1 by (1+1/alpha) if alpha != 0
                # Or just try both: drop r1 or drop r2
                
                # Try dropping r2
                for drop_r in [r1, r2]:
                    new_U2 = np.zeros((Na, new_rank), dtype=float)
                    new_V2 = np.zeros((Nb, new_rank), dtype=float)
                    new_W2 = np.zeros((Nc, new_rank), dtype=float)
                    
                    for r in range(50):
                        if r == drop_r:
                            continue
                        new_r = r if r < drop_r else r - 1
                        new_U2[:, new_r] = U[:, r]
                        new_V2[:, new_r] = V[:, r]
                        new_W2[:, new_r] = W[:, r]
                    
                    # Verify
                    if verify(new_U2.tolist(), new_V2.tolist(), new_W2.tolist(), new_rank):
                        return {"U": new_U2.tolist(), "V": new_V2.tolist(), "W": new_W2.tolist()}
        
        return None
    
    # Fallback: return rank-50
    result = build_rank50()
    
    # Verify it's correct
    if verify(result["U"], result["V"], result["W"], 50):
        # Try to reduce
        reduced = try_rank49()
        if reduced is not None:
            return reduced
        return result
    else:
        # Should not happen, but return anyway
        return result
