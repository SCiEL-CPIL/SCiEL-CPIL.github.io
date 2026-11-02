"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    def build_rank36():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def u_idx(i, j):
            return i * m + j
        def v_idx(j, k):
            return j * p + k
        def w_idx(k, i):
            return k * n + i
        
        col = 0
        
        def strassen_block(y_indices, z_indices, w_offsets):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices
            
            # M1 = (y00+y11)(z00+z11)
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M2 = (y10+y11)z00
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
            col += 1
            
            # M3 = y00(z01-z11)
            U[u_idx(y00[0], y00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M4 = y11(z10-z00)
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] -= 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            col += 1
            
            # M5 = (y00+y01)z11
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y01[0], y01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            col += 1
            
            # M6 = (y10-y00)(z00+z01)
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y00[0], y00[1])][col] -= 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M7 = (y01-y11)(z10+z11)
            U[u_idx(y01[0], y01[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] -= 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            col += 1
        
        # Block 1: A_L B_11
        y1 = [(0,0), (0,1), (1,0), (1,1)]
        z1 = [(0,0), (0,1), (1,0), (1,1)]
        w1 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y1, z1, w1)
        
        # Block 2: A_R B_21
        y2 = [(0,2), (0,3), (1,2), (1,3)]
        z2 = [(2,0), (2,1), (3,0), (3,1)]
        w2 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y2, z2, w2)
        
        # Block 3: A_L B_12
        y3 = [(0,0), (0,1), (1,0), (1,1)]
        z3 = [(0,2), (0,3), (1,2), (1,3)]
        w3 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y3, z3, w3)
        
        # Block 4: A_R B_22
        y4 = [(0,2), (0,3), (1,2), (1,3)]
        z4 = [(2,2), (2,3), (3,2), (3,3)]
        w4 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y4, z4, w4)
        
        # Block 5: A_L B_15 - schoolbook 2x2x1
        U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        
        # Block 6: A_R B_25 - schoolbook 2x2x1
        U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] += 1; col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        rank = len(U[0])
        T_reconstructed = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_reconstructed[i, j, k] += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
        T_target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, T_target, atol=1e-10)
    
    def try_reduce_rank(result, target_rank):
        """Try to reduce the rank by finding linear dependencies among the terms.
        
        Strategy: For each pair of columns (a, b), check if their combined contribution
        can be represented by modifying only columns a and b (keeping others fixed).
        This is a 2-variable nonlinear system that we can solve iteratively.
        
        More practically: check if column b is in the 'span' of column a in the 
        tensor sense, i.e., if there exist scalars such that the tensor equation
        holds with b eliminated.
        """
        U, V, W = result["U"], result["V"], result["W"]
        rank = len(U[0])
        
        if rank <= target_rank:
            return result
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        best_rank = rank
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        
        # Strategy: Try to find if any column can be expressed as a linear combination
        # of others by solving a linear system on the tensor entries.
        # 
        # For a rank-r decomposition, the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # If we fix all columns except column 'rem', we need:
        #   sum_{r != rem} c_r * U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 
        #     U[i*m+j,rem]*V[j*p+k,rem]*W[k*n+i,rem] + (correction from other columns)
        #
        # This is nonlinear. Instead, use a different approach:
        # Check if the column vectors (U[:,r], V[:,r], W[:,r]) have a dependency.
        #
        # Practical approach: Use the fact that for matrix multiplication, we can
        # sometimes combine Strassen terms with schoolbook terms.
        
        # Try removing each column and see if the remaining can be adjusted
        for rem in range(rank):
            if time.time() > deadline - 30:
                break
            
            # Build the "residual" that column rem contributes
            residual = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i, j, k] = U_arr[i*m+j, rem] * V_arr[j*p+k, rem] * W_arr[k*n+i, rem]
            
            # Check if residual is zero (column is already redundant)
            if np.allclose(residual, 0, atol=1e-10):
                # Remove this column
                mask = [i for i in range(rank) if i != rem]
                new_rank = rank - 1
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U = np.delete(U_arr, rem, axis=1)
                    best_V = np.delete(V_arr, rem, axis=1)
                    best_W = np.delete(W_arr, rem, axis=1)
                continue
            
            # Try to find if residual can be expressed as a combination of other columns
            # This requires solving: sum_{r != rem} c_r * (U[:,r] * V[:,r] * W[:,r]) = residual
            # which is nonlinear. Instead, try a greedy approach:
            # For each other column r, check if it has "overlap" with residual
            
            # Check if any single other column can absorb the residual
            for r in range(rank):
                if r == rem:
                    continue
                if time.time() > deadline - 30:
                    break
                
                # Column r contributes: U[:,r] * V[:,r] * W[:,r]
                contrib = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            contrib[i, j, k] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                
                # Check if residual is proportional to contrib
                if np.allclose(contrib, 0, atol=1e-10):
                    continue
                
                # Find scalar c such that residual ≈ c * contrib
                nonzero_mask = np.abs(contrib) > 1e-10
                if np.sum(nonzero_mask) == 0:
                    continue
                
                c_vals = residual[nonzero_mask] / contrib[nonzero_mask]
                c = np.median(c_vals)
                
                if np.allclose(residual, c * contrib, atol=1e-10):
                    # We can merge: modify column r to include the contribution
                    # New column r: U_new = U[:,r] + c' * U[:,rem], etc.
                    # Actually, if residual = c * contrib, then:
                    # U[:,rem]*V[:,rem]*W[:,rem] = c * U[:,r]*V[:,r]*W[:,r]
                    # This means we can set column rem to zero and adjust column r
                    # But we can't simply add because (U1+U2)(V1+V2)(W1+W2) != U1V1W1 + U2V2W2
                    # 
                    # However, if the vectors are aligned in a specific way...
                    # For now, mark as potential merge but don't actually merge
                    pass
        
        # Try a different strategy: look for structural simplifications
        # In the Strassen decomposition, some terms might be combinable
        
        # Check for columns with identical support patterns that might be mergeable
        for a in range(rank):
            if time.time() > deadline - 20:
                break
            for b in range(a+1, rank):
                # Check if U[:,a] and U[:,b] have the same non-zero pattern
                # and similarly for V and W
                Ua_nonzero = np.nonzero(U_arr[:,a])[0]
                Ub_nonzero = np.nonzero(U_arr[:,b])[0]
                Va_nonzero = np.nonzero(V_arr[:,a])[0]
                Vb_nonzero = np.nonzero(V_arr[:,b])[0]
                Wa_nonzero = np.nonzero(W_arr[:,a])[0]
                Wb_nonzero = np.nonzero(W_arr[:,b])[0]
                
                if set(Ua_nonzero) == set(Ub_nonzero) and \
                   set(Va_nonzero) == set(Vb_nonzero) and \
                   set(Wa_nonzero) == set(Wb_nonzero):
                    # Same support - check if they can be combined
                    # If U[:,a] = alpha * U[:,b] and V[:,a] = beta * V[:,b] and W[:,a] = gamma * W[:,b]
                    # with alpha*beta*gamma = 1, then they represent the same "direction"
                    
                    # Check proportionality
                    Ua = U_arr[:,a]
                    Ub = U_arr[:,b]
                    Va = V_arr[:,a]
                    Vb = V_arr[:,b]
                    Wa = W_arr[:,a]
                    Wb = W_arr[:,b]
                    
                    if not np.allclose(Ub, 0) and not np.allclose(Vb, 0) and not np.allclose(Wb, 0):
                        # Try to find alpha, beta, gamma
                        U_ratio = Ua[Ua != 0] / Ub[Ua != 0] if np.any(Ua != 0) else np.array([1])
                        V_ratio = Va[Va != 0] / Vb[Va != 0] if np.any(Va != 0) else np.array([1])
                        W_ratio = Wa[Wa != 0] / Wb[Wa != 0] if np.any(Wa != 0) else np.array([1])
                        
                        if len(U_ratio) > 0 and len(V_ratio) > 0 and len(W_ratio) > 0:
                            alpha = np.median(U_ratio)
                            beta = np.median(V_ratio)
                            gamma = np.median(W_ratio)
                            
                            if np.allclose(Ua, alpha * Ub, atol=1e-10) and \
                               np.allclose(Va, beta * Vb, atol=1e-10) and \
                               np.allclose(Wa, gamma * Wb, atol=1e-10):
                                # Check if alpha*beta*gamma ≈ 1 (same tensor direction)
                                if abs(alpha * beta * gamma - 1) < 1e-10:
                                    # These two columns represent the same tensor direction
                                    # We can combine them: new column = (1+1) * column_b
                                    # But this changes the contribution from 2 to 4, not 1
                                    # Actually, we need to be more careful
                                    pass
        
        # Convert back to list format if we found a better rank
        if best_rank < rank:
            return {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
        
        return result
    
    # Try rank 36 first (known to work)
    result36 = build_rank36()
    if verify_decomposition(result36["U"], result36["V"], result36["W"]):
        # Try iterative rank reduction
        current_result = result36
        for _ in range(5):  # Try up to 5 reduction steps
            if time.time() > deadline - 40:
                break
            reduced = try_reduce_rank(current_result, len(current_result["U"][0]) - 1)
            if len(reduced["U"][0]) < len(current_result["U"][0]):
                if verify_decomposition(reduced["U"], reduced["V"], reduced["W"]):
                    current_result = reduced
                else:
                    break
            else:
                break
        return current_result
    
    # If rank 36 doesn't verify (shouldn't happen), fall back to schoolbook
    rank = n * m * p  # 40
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
# EVOLVE-BLOCK-END
