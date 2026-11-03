"""Matrix multiplication 4x5 by 5x6 — Hybrid Strassen/schoolbook decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """
    Build the 2x2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one global decomposition.
    Each decomposition is a tuple: (U_sub, V_sub, W_sub, i_start, j_start, k_start, i_size, j_size, k_size)
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz in decomps:
        rank_sub = len(U_sub[0])
        
        # Map A indices: i in [0, i_sz), j in [0, j_sz)
        for i in range(i_sz):
            for j in range(j_sz):
                u_row_sub = i * j_sz + j
                u_row_glob = (i_s + i) * m + (j_s + j)
                for c in range(rank_sub):
                    U[u_row_glob][current_rank + c] = U_sub[u_row_sub][c]
        
        # Map B indices: j in [0, j_sz), k in [0, k_sz)
        for j in range(j_sz):
            for k in range(k_sz):
                v_row_sub = j * k_sz + k
                v_row_glob = (j_s + j) * p + (k_s + k)
                for c in range(rank_sub):
                    V[v_row_glob][current_rank + c] = V_sub[v_row_sub][c]
        
        # Map C indices: k in [0, k_sz), i in [0, i_sz)
        # Convention: W[k*n + i]
        for k in range(k_sz):
            for i in range(i_sz):
                w_row_sub = k * i_sz + i
                w_row_glob = (k_s + k) * n + (i_s + i)
                for c in range(rank_sub):
                    W[w_row_glob][current_rank + c] = W_sub[w_row_sub][c]
        
        current_rank += rank_sub
    
    return U, V, W

def _build_initial_decomposition(n, m, p):
    """Build initial decomposition using 4x4x4 Kronecker-Strassen + schoolbook blocks."""
    decomps = []
    
    def _build_kronecker_strassen_4x4x4():
        U2, V2, W2 = _build_strassen_2x2x2()
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        for t1 in range(7):
            for t2 in range(7):
                col = t1 * 7 + t2
                # U[i1*2+i2, j1*2+j2] = U2[i1,j1,t1] * U2[i2,j2,t2]
                for i1 in range(2):
                    for j1 in range(2):
                        c1 = U2[i1*2+j1][t1]
                        if c1 == 0: continue
                        for i2 in range(2):
                            for j2 in range(2):
                                c2 = U2[i2*2+j2][t2]
                                if c2 == 0: continue
                                u_row = (i1*2+i2)*4 + (j1*2+j2)
                                U[u_row][col] += c1 * c2
                
                # V[j1*2+j2, k1*2+k2] = V2[j1,k1,t1] * V2[j2,k2,t2]
                for j1 in range(2):
                    for k1 in range(2):
                        c1 = V2[j1*2+k1][t1]
                        if c1 == 0: continue
                        for j2 in range(2):
                            for k2 in range(2):
                                c2 = V2[j2*2+k2][t2]
                                if c2 == 0: continue
                                v_row = (j1*2+j2)*4 + (k1*2+k2)
                                V[v_row][col] += c1 * c2
                
                # W[k1*2+k2, i1*2+i2] = W2[k1,i1,t1] * W2[k2,i2,t2]
                for k1 in range(2):
                    for i1 in range(2):
                        c1 = W2[k1*2+i1][t1]
                        if c1 == 0: continue
                        for k2 in range(2):
                            for i2 in range(2):
                                c2 = W2[k2*2+i2][t2]
                                if c2 == 0: continue
                                w_row = (k1*2+k2)*4 + (i1*2+i2)
                                W[w_row][col] += c1 * c2
        
        return U, V, W

    U4, V4, W4 = _build_kronecker_strassen_4x4x4()
    decomps.append((U4, V4, W4, 0, 0, 0, 4, 4, 4))
    
    # Block 2: 4x4x2 (i=0..3, j=0..3, k=4..5)
    U5, V5, W5 = _build_schoolbook(4, 4, 2)
    decomps.append((U5, V5, W5, 0, 0, 4, 4, 4, 2))
    
    # Block 3: 4x1x6 (i=0..3, j=4, k=0..5)
    U6, V6, W6 = _build_schoolbook(4, 1, 6)
    decomps.append((U6, V6, W6, 0, 4, 0, 4, 1, 6))
    
    return _combine_decompositions(decomps, n, m, p)

def _verify_decomposition(U, V, W, n, m, p, num_tests=5):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rng = np.random.RandomState(42)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for _ in range(num_tests):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        C_expected = A @ B
        
        # Reconstruct C: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # More efficiently: C = sum_r (U_r^T * A) * (B * V_r) * W_r
        # where U_r is column r of U reshaped as (n*m,) -> (n,m)
        # Actually: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient computation using einsum
        # U[i*m+j, r] -> reshape to (n, m, rank)
        U_reshaped = U_np.reshape(n, m, -1)  # (n, m, rank)
        V_reshaped = V_np.reshape(m, p, -1)  # (m, p, rank)
        W_reshaped = W_np.reshape(p, n, -1)  # (p, n, rank)
        
        # C[i,k] = sum_{j,r} U[i,j,r] * V[j,k,r] * W[k,i,r]
        # = sum_r (sum_j U[i,j,r] * V[j,k,r]) * W[k,i,r]
        # Let M[i,k,r] = sum_j U[i,j,r] * V[j,k,r]
        # Then C[i,k] = sum_r M[i,k,r] * W[k,i,r]
        
        # Compute M using einsum
        M = np.einsum('ijr,jkr->ikr', U_reshaped, V_reshaped)  # (n, p, rank)
        C_computed = np.einsum('ikr,kir->ik', M, W_reshaped)  # (n, p)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _reduce_rank(U, V, W, n, m, p, deadline, max_iterations=200):
    """
    Iteratively remove linearly dependent rank-1 columns using SVD-based rank reduction.
    Returns reduced (U, V, W) with lower rank.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    # Track which columns are still active
    active_cols = list(range(rank))
    
    # We'll work with numpy arrays and keep track of active columns
    # Strategy: repeatedly find the column with smallest singular value contribution
    # and try to remove it, then verify.
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        if len(active_cols) <= 1:
            break
        
        # Try to find a removable column
        # For each column, check if removing it still gives a valid decomposition
        # This is expensive, so use a heuristic: remove the column with smallest norm
        # weighted by how "redundant" it is.
        
        # Compute column norms
        col_norms = np.zeros(len(active_cols))
        for idx, col in enumerate(active_cols):
            norm = np.linalg.norm(U_np[:, col]) * np.linalg.norm(V_np[:, col]) * np.linalg.norm(W_np[:, col])
            col_norms[idx] = norm
        
        # Try removing the column with smallest norm first
        order = np.argsort(col_norms)
        
        removed = False
        for idx in order:
            col_to_remove = active_cols[idx]
            
            # Check if this column is linearly dependent with others
            # A column r is removable if the tensor T_r = U_r ⊗ V_r ⊗ W_r
            # lies in the span of other columns.
            
            # Simple check: if U_r, V_r, or W_r is a linear combination of others
            # in a compatible way, we might be able to remove it.
            
            # For now, just try removing and verify
            new_active = [c for c in active_cols if c != col_to_remove]
            if not new_active:
                break
            
            U_sub = U_np[:, new_active]
            V_sub = V_np[:, new_active]
            W_sub = W_np[:, new_active]
            
            # Quick verification with 1 random test
            rng = np.random.RandomState(12345 + iteration)
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C_expected = A @ B
            
            U_reshaped = U_sub.reshape(n, m, -1)
            V_reshaped = V_sub.reshape(m, p, -1)
            W_reshaped = W_sub.reshape(p, n, -1)
            
            M = np.einsum('ijr,jkr->ikr', U_reshaped, V_reshaped)
            C_computed = np.einsum('ikr,kir->ik', M, W_reshaped)
            
            if np.allclose(C_computed, C_expected, atol=1e-8):
                # Column was redundant, remove it
                active_cols = new_active
                removed = True
                break
        
        if not removed:
            # Try a more sophisticated approach: look for pairs of columns
            # that can be combined (like in Winograd-style optimizations)
            # For now, if no single column is removable, try small perturbations
            # and see if we can find a lower-rank equivalent.
            
            # Try random linear combinations of pairs
            # This is a heuristic search
            found_pair = False
            for idx1 in range(len(active_cols)):
                if time.time() > deadline - 5:
                    break
                for idx2 in range(idx1 + 1, len(active_cols)):
                    if time.time() > deadline - 5:
                        break
                    
                    c1 = active_cols[idx1]
                    c2 = active_cols[idx2]
                    
                    # Try replacing columns c1 and c2 with a single column
                    # that is a linear combination: new_col = alpha * col_c1 + beta * col_c2
                    # This only works if the tensor components factor appropriately.
                    
                    # Check if U[:,c1] and U[:,c2] are proportional
                    u1 = U_np[:, c1]
                    u2 = U_np[:, c2]
                    v1 = V_np[:, c1]
                    v2 = V_np[:, c2]
                    w1 = W_np[:, c1]
                    w2 = W_np[:, c2]
                    
                    # If u1 = a*u2, v1 = b*v2, w1 = c*w2 with a*b*c = 1
                    # then col c1 is redundant given col c2 (up to scaling).
                    
                    # Check proportionality of U columns
                    if np.linalg.norm(u2) < 1e-15:
                        continue
                    ratio_u = np.linalg.norm(u1) / np.linalg.norm(u2)
                    if ratio_u < 1e-15:
                        # u1 is zero, try removing c1
                        new_active = [c for c in active_cols if c != c1]
                        if new_active:
                            U_sub = U_np[:, new_active]
                            V_sub = V_np[:, new_active]
                            W_sub = W_np[:, new_active]
                            A = np.random.RandomState(999).rand(n, m)
                            B = np.random.RandomState(998).rand(m, p)
                            C_exp = A @ B
                            U_r = U_sub.reshape(n, m, -1)
                            V_r = V_sub.reshape(m, p, -1)
                            W_r = W_sub.reshape(p, n, -1)
                            M = np.einsum('ijr,jkr->ikr', U_r, V_r)
                            C_comp = np.einsum('ikr,kir->ik', M, W_r)
                            if np.allclose(C_comp, C_exp, atol=1e-8):
                                active_cols = new_active
                                found_pair = True
                                break
                        continue
                    
                    # Check if u1 ≈ ratio_u * u2 (or -ratio_u * u2)
                    u2_unit = u2 / np.linalg.norm(u2)
                    proj = np.dot(u1, u2_unit)
                    residual = u1 - proj * u2_unit
                    if np.linalg.norm(residual) < 1e-10 * np.linalg.norm(u1):
                        # u1 is proportional to u2
                        # Check V and W similarly
                        if np.linalg.norm(v2) < 1e-15 or np.linalg.norm(w2) < 1e-15:
                            continue
                        v2_unit = v2 / np.linalg.norm(v2)
                        proj_v = np.dot(v1, v2_unit)
                        residual_v = v1 - proj_v * v2_unit
                        if np.linalg.norm(residual_v) < 1e-10 * np.linalg.norm(v1):
                            w2_unit = w2 / np.linalg.norm(w2)
                            proj_w = np.dot(w1, w2_unit)
                            residual_w = w1 - proj_w * w2_unit
                            if np.linalg.norm(residual_w) < 1e-10 * np.linalg.norm(w1):
                                # All three are proportional!
                                # Check if the product of ratios is 1
                                r_u = np.dot(u1, u2_unit) / np.linalg.norm(u2)
                                r_v = np.dot(v1, v2_unit) / np.linalg.norm(v2)
                                r_w = np.dot(w1, w2_unit) / np.linalg.norm(w2)
                                if abs(r_u * r_v * r_w - 1.0) < 1e-8:
                                    # Column c1 is redundant
                                    new_active = [c for c in active_cols if c != c1]
                                    if new_active:
                                        active_cols = new_active
                                        found_pair = True
                                        break
                    
                    if found_pair:
                        break
                if found_pair:
                    break
            
            if not found_pair:
                # No more easy reductions
                break
    
    # Convert back to list format
    U_final = U_np[:, active_cols].tolist()
    V_final = V_np[:, active_cols].tolist()
    W_final = W_np[:, active_cols].tolist()
    
    return U_final, V_final, W_final

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Build initial decomposition
    U, V, W = _build_initial_decomposition(n, m, p)
    
    # Verify initial decomposition
    if not _verify_decomposition(U, V, W, n, m, p, num_tests=3):
        # Fallback to full schoolbook
        U, V, W = _build_schoolbook(n, m, p)
    
    # Attempt rank reduction using tensor network contraction approach
    if time.time() < deadline - 15:
        U, V, W = _reduce_rank(U, V, W, n, m, p, deadline)
    
    # Final verification
    if time.time() < deadline - 5:
        if not _verify_decomposition(U, V, W, n, m, p, num_tests=5):
            # If reduction broke correctness, fall back to initial
            U, V, W = _build_initial_decomposition(n, m, p)
            if not _verify_decomposition(U, V, W, n, m, p, num_tests=3):
                U, V, W = _build_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
