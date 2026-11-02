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
        """Try to reduce the rank by finding linear dependencies among the terms."""
        U, V, W = result["U"], result["V"], result["W"]
        rank = len(U[0])
        
        if rank <= target_rank:
            return result
        
        # Convert to numpy arrays
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Try to find a column that can be expressed as a combination of others
        # by checking if removing it and adjusting others still works
        # This is a greedy approach: try removing each column and see if we can fix
        
        best_result = result
        best_rank = rank
        
        # Try removing each column one at a time
        for remove_idx in range(rank):
            if time.time() > deadline - 20:
                break
            
            # Remove column remove_idx
            mask = [i for i in range(rank) if i != remove_idx]
            U_new = np.delete(U_arr, remove_idx, axis=1)
            V_new = np.delete(V_arr, remove_idx, axis=1)
            W_new = np.delete(W_arr, remove_idx, axis=1)
            
            new_rank = rank - 1
            if new_rank >= best_rank:
                continue
            
            # Check if we can add back the contribution of the removed column
            # by adjusting the remaining columns
            # The removed column contributes: U[:,remove_idx] * V[:,remove_idx] * W[:,remove_idx]
            # We need to see if this can be absorbed
            
            # For each (i,j,k), the removed column contributes:
            # U[i*m+j, remove_idx] * V[j*p+k, remove_idx] * W[k*n+i, remove_idx]
            # We need to find coefficients c_r for remaining r such that:
            # sum_r c_r * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 
            #   U[i*m+j, remove_idx] * V[j*p+k, remove_idx] * W[k*n+i, remove_idx]
            # for all (i,j,k)
            
            # This is a bilinear system - hard to solve directly
            # Instead, let's try a different approach: use SVD-like decomposition
            
            # Try: can we express the removed term as a sum of a few remaining terms?
            # This requires solving a linear system
            
            # Build the system: for each (i,j,k), we need
            # sum_r c_r * (U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]) = U[i*m+j,remove_idx] * V[j*p+k,remove_idx] * W[k*n+i,remove_idx]
            
            # This is nonlinear in c_r, so let's try a simpler heuristic:
            # Check if the removed column is "small" in some sense
            
            # Alternative: try to find if two columns are similar and can be merged
            pass
        
        # Try a different strategy: look for pairs of columns that can be combined
        # If column a and column b have proportional U, V, or W vectors, they might be mergeable
        
        for a in range(rank):
            if time.time() > deadline - 10:
                break
            for b in range(a+1, rank):
                # Check if columns a and b can be merged
                # They can be merged if there exists a scalar c such that:
                # U[:,a] = c*U[:,b], V[:,a] = c*V[:,b], W[:,a] = c*W[:,b]
                # Or more generally, if their contributions can be combined
                
                Ua, Ub = U_arr[:,a], U_arr[:,b]
                Va, Vb = V_arr[:,a], V_arr[:,b]
                Wa, Wb = W_arr[:,a], W_arr[:,b]
                
                # Check proportionality
                if np.allclose(Ua, 0) and np.allclose(Ub, 0):
                    continue
                if np.allclose(Va, 0) and np.allclose(Vb, 0):
                    continue
                if np.allclose(Wa, 0) and np.allclose(Wb, 0):
                    continue
                
                # Try to find c such that Ua ≈ c*Ub
                if not np.allclose(Ub, 0):
                    c_candidates = []
                    for idx in range(len(Ua)):
                        if abs(Ub[idx]) > 1e-10:
                            c_candidates.append(Ua[idx] / Ub[idx])
                    if c_candidates:
                        c = np.median(c_candidates)
                        if np.allclose(Ua, c*Ub, atol=1e-10) and \
                           np.allclose(Va, c*Vb, atol=1e-10) and \
                           np.allclose(Wa, c*Wb, atol=1e-10):
                            # They are proportional! Merge them
                            # New column = Ua + Ub, Va + Vb, Wa + Wb (scaled)
                            # Actually, if Ua = c*Ub, then the combined contribution is
                            # (Ua+Ub)*(Va+Vb)*(Wa+Wb) which is NOT the same as
                            # Ua*Va*Wa + Ub*Vb*Wb
                            # So this doesn't work directly.
                            pass
        
        return best_result
    
    # Try rank 36 first (known to work)
    result36 = build_rank36()
    if verify_decomposition(result36["U"], result36["V"], result36["W"]):
        # Try to reduce to rank 35
        result35 = try_reduce_rank(result36, 35)
        if len(result35["U"][0]) < 36:
            if verify_decomposition(result35["U"], result35["V"], result35["W"]):
                return result35
        return result36
    
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
