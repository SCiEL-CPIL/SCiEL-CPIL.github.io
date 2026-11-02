"""Deterministic block-decomposition search for 2x5x6 matrix multiplication tensor."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Helper to validate a decomposition (vectorized)
    def is_valid_decomposition(U, V, W, rank):
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Build tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # T should have T[i,j,k] = 1 for all valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True
    
    # Schoolbook baseline: rank = n*m*p = 60
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    best_rank = rank_sb
    
    # STRATTEN-BASED BLOCK DECOMPOSITION
    # For 2x5x6, we can use Strassen's algorithm on 2x2x2 blocks
    # Partition: A is 2x5, B is 5x6
    # Split A into [A1 | A2] where A1 is 2x2, A2 is 2x3
    # Split B into [B1; B2] where B1 is 2x6, B2 is 3x6
    # C = A1*B1 + A2*B2
    
    # Apply Strassen to each 2x2x2 block:
    # A1*B1: rank 7 (Strassen)
    # A2*B2: need to handle 2x3x6
    
    # For 2x3x6, we can further decompose:
    # A2 is 2x3, B2 is 3x6
    # Split B2 into [B21 | B22] where B21 is 3x3, B22 is 3x3
    # A2*B21: 2x3x3, A2*B22: 2x3x3
    
    # For 2x3x3: schoolbook gives 18, but we can do better
    # Actually, let's use a more systematic approach
    
    # Generate candidate rank-1 tensors with Strassen-like patterns
    # Strassen for 2x2x2 uses these 7 rank-1 tensors:
    # M1 = (A11-A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B21-B22)
    # M4 = A22*(B12-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Let's generate a pool of candidate rank-1 tensors with small integer/half-integer entries
    # and use a greedy/conflict-graph approach to find a minimal valid decomposition
    
    num_candidates = 2000
    candidates = []
    
    # Generate Strassen-inspired candidates
    # For each possible "block" structure, generate rank-1 tensors
    
    # First, generate all "standard" rank-1 tensors (one for each (i,j,k))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                candidates.append((u, v, w))
    
    # Generate Strassen-like candidates for 2x2x2 sub-blocks
    # Strassen M1: (A11-A22)*(B11+B22)
    # This corresponds to u[i*m+j] = 1 if (i,j) in {(0,0),(1,1)} with signs, etc.
    
    def generate_strassen_candidates():
        """Generate Strassen-inspired rank-1 tensors for 2x2x2 blocks."""
        strassen_cands = []
        
        # Strassen's 7 multiplications for 2x2x2
        # M1 = (A11-A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B21-B22)
        # M4 = A22*(B12-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # For a 2x2x2 block at positions (i0, j0, k0):
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # C indices: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
        
        for i0 in range(n - 1):  # 0 to 0 (since n=2)
            for j0 in range(m - 1):  # 0 to 3
                for k0 in range(p - 1):  # 0 to 4
                    
                    # M1: (A11-A22)*(B11+B22)
                    # A11 = (i0, j0), A22 = (i0+1, j0+1)
                    # B11 = (j0, k0), B22 = (j0+1, k0+1)
                    # C = (i0, k0) + (i0, k0+1) + (i0+1, k0) + (i0+1, k0+1) with signs
                    u1 = np.zeros(n * m)
                    v1 = np.zeros(m * p)
                    w1 = np.zeros(p * n)
                    u1[i0 * m + j0] = 1
                    u1[(i0+1) * m + (j0+1)] = -1
                    v1[j0 * p + k0] = 1
                    v1[(j0+1) * p + (k0+1)] = 1
                    w1[k0 * n + i0] = 1
                    w1[(k0+1) * n + i0] = 1
                    w1[k0 * n + (i0+1)] = 1
                    w1[(k0+1) * n + (i0+1)] = 1
                    strassen_cands.append((u1, v1, w1))
                    
                    # M2: (A21+A22)*B11
                    u2 = np.zeros(n * m)
                    v2 = np.zeros(m * p)
                    w2 = np.zeros(p * n)
                    u2[(i0+1) * m + j0] = 1
                    u2[(i0+1) * m + (j0+1)] = 1
                    v2[j0 * p + k0] = 1
                    w2[k0 * n + (i0+1)] = 1
                    w2[(k0+1) * n + (i0+1)] = 1
                    strassen_cands.append((u2, v2, w2))
                    
                    # M3: A11*(B21-B22)
                    u3 = np.zeros(n * m)
                    v3 = np.zeros(m * p)
                    w3 = np.zeros(p * n)
                    u3[i0 * m + j0] = 1
                    v3[(j0+1) * p + k0] = 1
                    v3[(j0+1) * p + (k0+1)] = -1
                    w3[k0 * n + i0] = 1
                    w3[(k0+1) * n + i0] = 1
                    strassen_cands.append((u3, v3, w3))
                    
                    # M4: A22*(B12-B11)
                    u4 = np.zeros(n * m)
                    v4 = np.zeros(m * p)
                    w4 = np.zeros(p * n)
                    u4[(i0+1) * m + (j0+1)] = 1
                    v4[j0 * p + (k0+1)] = 1
                    v4[j0 * p + k0] = -1
                    w4[k0 * n + (i0+1)] = 1
                    w4[(k0+1) * n + (i0+1)] = 1
                    strassen_cands.append((u4, v4, w4))
                    
                    # M5: (A11+A12)*B22
                    u5 = np.zeros(n * m)
                    v5 = np.zeros(m * p)
                    w5 = np.zeros(p * n)
                    u5[i0 * m + j0] = 1
                    u5[i0 * m + (j0+1)] = 1
                    v5[(j0+1) * p + (k0+1)] = 1
                    w5[(k0+1) * n + i0] = 1
                    strassen_cands.append((u5, v5, w5))
                    
                    # M6: (A21-A11)*(B11+B12)
                    u6 = np.zeros(n * m)
                    v6 = np.zeros(m * p)
                    w6 = np.zeros(p * n)
                    u6[(i0+1) * m + j0] = 1
                    u6[i0 * m + j0] = -1
                    v6[j0 * p + k0] = 1
                    v6[j0 * p + (k0+1)] = 1
                    w6[k0 * n + (i0+1)] = 1
                    w6[(k0+1) * n + (i0+1)] = 1
                    strassen_cands.append((u6, v6, w6))
                    
                    # M7: (A12-A22)*(B21+B22)
                    u7 = np.zeros(n * m)
                    v7 = np.zeros(m * p)
                    w7 = np.zeros(p * n)
                    u7[i0 * m + (j0+1)] = 1
                    u7[(i0+1) * m + (j0+1)] = -1
                    v7[(j0+1) * p + k0] = 1
                    v7[(j0+1) * p + (k0+1)] = 1
                    w7[k0 * n + i0] = 1
                    w7[(k0+1) * n + i0] = 1
                    strassen_cands.append((u7, v7, w7))
        
        return strassen_cands
    
    strassen_cands = generate_strassen_candidates()
    candidates.extend(strassen_cands)
    
    # Generate random sparse candidates
    random.seed(42)
    for _ in range(1000):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        
        mask_u = np.random.random(n * m) < 0.3
        u[mask_u] = np.random.choice([-1, 1], size=mask_u.sum())
        
        mask_v = np.random.random(m * p) < 0.3
        v[mask_v] = np.random.choice([-1, 1], size=mask_v.sum())
        
        mask_w = np.random.random(p * n) < 0.3
        w[mask_w] = np.random.choice([-1, 1], size=mask_w.sum())
        
        candidates.append((u, v, w))
    
    # Build conflict graph and use greedy maximum independent set
    # Two candidates conflict if they cannot coexist in a valid decomposition
    # For now, use a simpler approach: greedy set cover with conflict detection
    
    # Build the linear system matrix
    num_constraints = n * m * p
    A_mat = np.zeros((num_constraints, len(candidates)))
    b_vec = np.ones(num_constraints)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c_idx, (u, v, w) in enumerate(candidates):
                    A_mat[idx, c_idx] = u[i*m+j] * v[j*p+k] * w[k*n+i]
                idx += 1
    
    # Greedy approach: iteratively select candidates that best reduce residual
    residual = b_vec.copy()
    selected = []
    
    for step in range(min(60, len(candidates))):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Find candidate that best reduces residual
        best_c = -1
        best_reduction = -np.inf
        
        for c in range(len(candidates)):
            if c in selected:
                continue
            
            # Compute how much this candidate reduces the residual
            contrib = A_mat[:, c]
            # We want to add x_c * contrib to approximate residual
            # Optimal x_c = <residual, contrib> / <contrib, contrib>
            dot_rc = np.dot(residual, contrib)
            dot_cc = np.dot(contrib, contrib)
            
            if dot_cc < 1e-10:
                continue
            
            x_c = dot_rc / dot_cc
            
            # Compute reduction in ||residual - x_c * contrib||^2
            new_residual = residual - x_c * contrib
            new_norm = np.dot(new_residual, new_residual)
            old_norm = np.dot(residual, residual)
            reduction = old_norm - new_norm
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_c = c
        
        if best_c == -1 or best_reduction < 1e-10:
            break
        
        selected.append(best_c)
        contrib = A_mat[:, best_c]
        dot_rc = np.dot(residual, contrib)
        dot_cc = np.dot(contrib, contrib)
        x_best = dot_rc / dot_cc
        residual = residual - x_best * contrib
        
        # Check if we've converged
        if np.max(np.abs(residual)) < 1e-8:
            break
    
    # Verify the selected candidates form a valid decomposition
    if len(selected) > 0 and len(selected) < best_rank:
        U_sel = np.zeros((n * m, len(selected)))
        V_sel = np.zeros((m * p, len(selected)))
        W_sel = np.zeros((p * n, len(selected)))
        
        for sel_idx, c_idx in enumerate(selected):
            u, v, w = candidates[c_idx]
            U_sel[:, sel_idx] = u
            V_sel[:, sel_idx] = v
            W_sel[:, sel_idx] = w
        
        if is_valid_decomposition(U_sel, V_sel, W_sel, len(selected)):
            best_U = U_sel.copy()
            best_V = V_sel.copy()
            best_W = W_sel.copy()
            best_rank = len(selected)
    
    # If greedy didn't work well, try the structured Strassen approach directly
    if best_rank >= 55 and time.time() - start_time < time_limit - 10:
        # Build a specific Strassen-based decomposition for 2x5x6
        # This is a hand-crafted approach based on known results
        
        # For 2x5x6, we can achieve rank 54 using block Strassen
        # Let's construct it explicitly
        
        rank_target = 54
        U_strassen = np.zeros((n * m, rank_target))
        V_strassen = np.zeros((m * p, rank_target))
        W_strassen = np.zeros((p * n, rank_target))
        
        col = 0
        
        # Process 2x2x2 blocks using Strassen
        # A is 2x5, B is 5x6
        # Split into: A = [A_left | A_right] where A_left is 2x2, A_right is 2x3
        # B = [B_top; B_bottom] where B_top is 2x6, B_bottom is 3x6
        
        # C = A_left * B_top + A_right * B_bottom
        
        # A_left * B_top: 2x2x6
        # Split B_top into [B_top1 | B_top2] where each is 2x3
        # A_left * B_top1: 2x2x3, A_left * B_top2: 2x2x3
        
        # For 2x2x3: use Strassen on 2x2x2 + standard for 2x2x1
        # Strassen for 2x2x2: rank 7
        # Standard for 2x2x1: rank 2
        # Total: 9 for 2x2x3
        
        # A_right * B_bottom: 2x3x6
        # Split B_bottom into [B_bottom1 | B_bottom2] where each is 3x3
        # A_right * B_bottom1: 2x3x3, A_right * B_bottom2: 2x3x3
        # For 2x3x3: schoolbook gives 18, but we can do better
        # Actually, let's just use schoolbook for now: 18 each, total 36
        
        # Total: 9 + 9 + 36 = 54
        
        # Let's implement this step by step
        
        # Part 1: A_left (2x2) * B_top1 (2x3) using Strassen + standard
        # B_top1 has columns 0,1,2 of B (which is rows 0,1 of B)
        
        # Strassen for A_left * B_top1[:, 0:2] (2x2x2)
        # A_left = [[a00, a01], [a10, a11]]
        # B_top1[:, 0:2] = [[b00, b01], [b10, b11]]
        
        # M1 = (a00-a11)*(b00+b11)
        U_strassen[0, col] = 1  # a00
        U_strassen[1, col] = 0  # a01
        U_strassen[2, col] = 0  # a10
        U_strassen[3, col] = -1  # a11
        V_strassen[0, col] = 1  # b00
        V_strassen[1, col] = 0  # b01
        V_strassen[2, col] = 1  # b10
        V_strassen[3, col] = 1  # b11
        W_strassen[0, col] = 1  # c00
        W_strassen[1, col] = 1  # c01
        W_strassen[2, col] = 1  # c10
        W_strassen[3, col] = 1  # c11
        col += 1
        
        # M2 = (a10+a11)*b00
        U_strassen[0, col] = 0
        U_strassen[1, col] = 0
        U_strassen[2, col] = 1  # a10
        U_strassen[3, col] = 1  # a11
        V_strassen[0, col] = 1  # b00
        V_strassen[1, col] = 0
        V_strassen[2, col] = 0
        V_strassen[3, col] = 0
        W_strassen[0, col] = 0
        W_strassen[1, col] = 0
        W_strassen[2, col] = 1  # c10
        W_strassen[3, col] = 0
        col += 1
        
        # M3 = a00*(b10-b11)
        U_strassen[0, col] = 1  # a00
        U_strassen[1, col] = 0
        U_strassen[2, col] = 0
        U_strassen[3, col] = 0
        V_strassen[0, col] = 0
        V_strassen[1, col] = 0
        V_strassen[2, col] = 1  # b10
        V_strassen[3, col] = -1  # b11
        W_strassen[0, col] = 1  # c00
        W_strassen[1, col] = 0
        W_strassen[2, col] = 0
        W_strassen[3, col] = 0
        col += 1
        
        # M4 = a11*(b01-b00)
        U_strassen[0, col] = 0
        U_strassen[1, col] = 0
        U_strassen[2, col] = 0
        U_strassen[3, col] = 1  # a11
        V_strassen[0, col] = -1  # b00
        V_strassen[1, col] = 1  # b01
        V_strassen[2, col] = 0
        V_strassen[3, col] = 0
        W_strassen[0, col] = 0
        W_strassen[1, col] = 0
        W_strassen[2, col] = 1  # c10
        W_strassen[3, col] = 0
        col += 1
        
        # M5 = (a00+a01)*b11
        U_strassen[0, col] = 1  # a00
        U_strassen[1, col] = 1  # a01
        U_strassen[2, col] = 0
        U_strassen[3, col] = 0
        V_strassen[0, col] = 0
        V_strassen[1, col] = 0
        V_strassen[2, col] = 0
        V_strassen[3, col] = 1  # b11
        W_strassen[0, col] = 0
        W_strassen[1, col] = 1  # c01
        W_strassen[2, col] = 0
        W_strassen[3, col] = 0
        col += 1
        
        # M6 = (a10-a00)*(b00+b01)
        U_strassen[0, col] = -1  # a00
        U_strassen[1, col] = 0
        U_strassen[2, col] = 1  # a10
        U_strassen[3, col] = 0
        V_strassen[0, col] = 1  # b00
        V_strassen[1, col] = 1  # b01
        V_strassen[2, col] = 0
        V_strassen[3, col] = 0
        W_strassen[0, col] = 0
        W_strassen[1, col] = 0
        W_strassen[2, col] = 1  # c10
        W_strassen[3, col] = 0
        col += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U_strassen[0, col] = 0
        U_strassen[1, col] = 1  # a01
        U_strassen[2, col] = 0
        U_strassen[3, col] = -1  # a11
        V_strassen[0, col] = 0
        V_strassen[1, col] = 0
        V_strassen[2, col] = 1  # b10
        V_strassen[3, col] = 1  # b11
        W_strassen[0, col] = 0
        W_strassen[1, col] = 1  # c01
        W_strassen[2, col] = 0
        W_strassen[3, col] = 0
        col += 1
        
        # Standard for A_left * B_top1[:, 2] (2x2x1)
        # c02 = a00*b02 + a01*b12
        # c12 = a10*b02 + a11*b12
        
        U_strassen[0, col] = 1  # a00
        U_strassen[1, col] = 0
        U_strassen[2, col] = 0
        U_strassen[3, col] = 0
        V_strassen[0, col] = 1  # b02
        V_strassen[1, col] = 0
        V_strassen[2, col] = 0
        V_strassen[3, col] = 0
        W_strassen[0, col] = 1  # c02
        W_strassen[1, col] = 0
        W_strassen[2, col] = 0
        W_strassen[3, col] = 0
        col += 1
        
        U_strassen[0, col] = 0
        U_strassen[1, col] = 1  # a01
        U_strassen[2, col] = 0
        U_strassen[3, col] = 0
        V_strassen[0, col] = 0
        V_strassen[1, col] = 1  # b12
        V_strassen[2, col] = 0
        V_strassen[3, col] = 0
        W_strassen[0, col] = 1  # c02
        W_strassen[1, col] = 0
        W_strassen[2, col] = 0
        W_strassen[3, col] = 0
        col += 1
        
        # Part 2: A_left (2x2) * B_top2 (2x3) using Strassen + standard
        # B_top2 has columns 3,4,5 of B (which is rows 0,1 of B)
        # Similar to Part 1 but with different column indices
        
        # Strassen for A_left * B_top2[:, 0:2] (2x2x2)
        # B_top2[:, 0:2] = B[0:2, 3:5]
        
        # M1 = (a00-a11)*(b03+b14)
        U_strassen[0, col] = 1
        U_strassen[1, col] = 0
        U_strassen[2, col] = 0
        U_strassen[3, col] = -1
        V_strassen[0, col] = 1  # b03
        V_strassen[1, col] = 0
        V_strassen[2, col] = 1  # b13
        V_strassen[3, col] = 1  # b14
        W_strassen[0, col] = 1  # c03
        W_strassen[1, col] = 1  # c04
        W_strassen[2, col] = 1  # c13
        W_strassen[3, col] = 1  # c14
        col += 1
        
        # M2 = (a10+a11)*b03
        U_strassen[0, col] = 0
        U_strassen[1, col] = 0
        U_strassen[2, col] = 1
        U_strassen[3, col] = 1
        V_strassen[0, col] = 1  # b03
        V_strassen[1, col] = 0
        V_strassen[2, col] = 0
        V_strassen[3, col] = 0
        W_strassen[0, col] = 0
        W_strassen[1, col] = 0
        W_strassen[2, col] = 1  # c13
        W_strassen[3, col] = 0
        col += 1
        
        # M3 = a00*(b13-b14)
        U_strassen[0, col] = 1
        U_strassen[1, col] = 0
        U_strassen[2, col] = 0
        U_strassen[3, col] = 0
        V_strassen[0, col] = 0
        V_strassen[1, col] = 0
        V_strassen[2, col] = 1  # b13
        V_strassen[3, col] = -1  # b14
        W_strassen[0, col] = 1  # c03
        W_strassen[1, col] = 0
        W_strassen[2, col] = 0
        W_strassen[3, col] = 0
        col += 1
        
        # M4 = a11*(b04-b03)
        U_strassen[0, col] = 0
        U_strassen[1, col] = 0
        U_strassen[2, col] = 0
        U_strassen[3, col] = 1
        V_strassen[0, col] = -1  # b03
        V_strassen[1, col] = 1  # b04
        V_strassen[2, col] = 0
        V_strassen[3, col] = 0
        W_strassen[0, col] = 0
        W_strassen[1, col] = 0
        W_strassen[2, col] = 1  # c13
        W_strassen[3, col] = 0
        col += 1
        
        # M5 = (a00+a01)*b14
        U_strassen[0, col] = 1
        U_strassen[1, col] = 1
        U_strassen[2, col] = 0
        U_strassen[3, col] = 0
        V_strassen[0, col] = 0
        V_strassen[1, col] = 0
        V_strassen[2, col] = 0
        V_strassen[3, col] = 1  # b14
        W_strassen[0, col] = 0
        W_strassen[1, col] = 1  # c04
        W_strassen[2, col] = 0
        W_strassen[3, col] = 0
        col += 1
        
        # M6 = (a10-a00)*(b03+b04)
        U_strassen[0, col] = -1
        U_strassen[1, col] = 0
        U_strassen[2, col] = 1
        U_strassen[3, col] = 0
        V_strassen[0, col] = 1  # b03
        V_strassen[1, col] = 1  # b04
        V_strassen[2, col] = 0
        V_strassen[3, col] = 0
        W_strassen[0, col] = 0
        W_strassen[1, col] = 0
        W_strassen[2, col] = 1  # c13
        W_strassen[3, col] = 0
        col += 1
        
        # M7 = (a01-a11)*(b13+b14)
        U_strassen[0, col] = 0
        U_strassen[1, col] = 1
        U_strassen[2, col] = 0
        U_strassen[3, col] = -1
        V_strassen[0, col] = 0
        V_strassen[1, col] = 0
        V_strassen[2, col] = 1  # b13
        V_strassen[3, col] = 1  # b14
        W_strassen[0, col] = 0
        W_strassen[1, col] = 1  # c04
        W_strassen[2, col] = 0
        W_strassen[3, col] = 0
        col += 1
        
        # Standard for A_left * B_top2[:, 2] (2x2x1)
        # c05 = a00*b05 + a01*b15
        # c15 = a10*b05 + a11*b15
        
        U_strassen[0, col] = 1
        U_strassen[1, col] = 0
        U_strassen[2, col] = 0
        U_strassen[3, col] = 0
        V_strassen[0, col] = 1  # b05
        V_strassen[1, col] = 0
        V_strassen[2, col] = 0
        V_strassen[3, col] = 0
        W_strassen[0, col] = 1  # c05
        W_strassen[1, col] = 0
        W_strassen[2, col] = 0
        W_strassen[3, col] = 0
        col += 1
        
        U_strassen[0, col] = 0
        U_strassen[1, col] = 1
        U_strassen[2, col] = 0
        U_strassen[3, col] = 0
        V_strassen[0, col] = 0
        V_strassen[1, col] = 1  # b15
        V_strassen[2, col] = 0
        V_strassen[3, col] = 0
        W_strassen[0, col] = 1  # c05
        W_strassen[1, col] = 0
        W_strassen[2, col] = 0
        W_strassen[3, col] = 0
        col += 1
        
        # Part 3: A_right (2x3) * B_bottom (3x6) using schoolbook
        # A_right = A[:, 2:5], B_bottom = B[2:5, :]
        # 18 multiplications
        
        for i in range(n):
            for j in range(3):  # j=0,1,2 correspond to original j=2,3,4
                for k in range(p):
                    U_strassen[i * m + (j+2), col] = 1
                    V_strassen[(j+2) * p + k, col] = 1
                    W_strassen[k * n + i, col] = 1
                    col += 1
        
        # Verify the Strassen decomposition
        if is_valid_decomposition(U_strassen, V_strassen, W_strassen, col):
            if col < best_rank:
                best_U = U_strassen[:n*m, :col].copy()
                best_V = V_strassen[:m*p, :col].copy()
                best_W = W_strassen[:p*n, :col].copy()
                best_rank = col
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
