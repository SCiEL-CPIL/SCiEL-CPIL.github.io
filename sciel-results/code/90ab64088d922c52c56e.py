"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify the tensor decomposition is exact using NumPy."""
        R = len(U[0])
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)

        # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # and check it equals 1.
        for i in range(n):
            for j in range(m):
                u_row = U_np[i*m+j]
                for k in range(p):
                    s = np.dot(u_row * V_np[j*p+k], W_np[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def dual_catalecticant_analysis():
        """
        Compute mode-1 and mode-2 flattening ranks to establish lower bounds.
        
        Mode-1 flattening: T_{(i,j),k} -> matrix of shape (n*m) x p
        Mode-2 flattening: T_{j,(i,k)} -> matrix of shape m x (n*p)
        
        The rank of any flattening is a lower bound on the tensor rank.
        """
        # Build the full tensor T[i,j,k] = 1 for all valid indices (it's the identity tensor)
        # Actually, the multiplication tensor has T[i,j,k] = 1 for all (i,j,k)
        # since (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        
        # Mode-1 flattening: reshape to (n*m, p)
        # T[i,j,k] -> M1[i*m+j, k]
        M1 = np.ones((nm, p))  # All ones since T[i,j,k]=1
        # Wait, that's not right. The tensor is:
        # T[i,j,k] = 1 for all i,j,k (the standard matmul tensor)
        # Mode-1: fix (i,j), vary k -> M1[i*m+j, k] = T[i,j,k] = 1
        # So M1 is all ones, rank 1. That's not useful.
        
        # Let me reconsider. The tensor T is defined by:
        # T[i,j,k] such that C_{i,k} = sum_{j,r} U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # The target is C_{i,k} = sum_j A_{i,j} * B_{j,k}
        # So T[i,j,k] = 1 for all i,j,k.
        
        # Actually for rank lower bounds, we need to be more careful.
        # The mode-1 flattening has entries T_{(i,j),k} = delta_{i,k} if we're doing identity...
        # No, for matrix multiplication, T[i,j,k] = 1 for ALL i,j,k.
        
        # Let me just compute the flattening ranks directly.
        # M1 shape: (n*m, p) = (4, 4)
        # M1[i*m+j, k] = T[i,j,k] = 1
        M1_flat = np.ones((nm, p))
        rank_m1 = np.linalg.matrix_rank(M1_flat)
        
        # M2 shape: (m, n*p) = (2, 8)  
        # M2[j, k*n+i] = T[i,j,k] = 1
        M2_flat = np.ones((m, n*p))
        rank_m2 = np.linalg.matrix_rank(M2_flat)
        
        # M3 shape: (p*n, m) = (8, 2)
        # M3[k*n+i, j] = T[i,j,k] = 1
        M3_flat = np.ones((p*n, m))
        rank_m3 = np.linalg.matrix_rank(M3_flat)
        
        lower_bound = max(rank_m1, rank_m2, rank_m3)
        return lower_bound, rank_m1, rank_m2, rank_m3

    def try_strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B) - standard Strassen
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B) - standard Strassen
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    def try_merge_rank13():
        """
        Attempt rank-13 by merging one pair of Strassen terms across blocks.
        
        Key insight: Both blocks share the same A matrix. The Strassen products
        that use the same A-combination can potentially be merged.
        
        For example, M1_left uses (a00+a11) and M1_right uses (a00+a11).
        If we could combine (a00+a11)(b00+b11) and (a00+a11)(b02+b13) into
        a single product (a00+a11)(b00+b11+b02+b13) and then split the output,
        we'd save one rank. But this doesn't work directly because we need
        separate outputs.
        
        However, we can try: replace two terms with one term that uses a combined
        B factor, and adjust W to extract the correct outputs.
        
        Strategy: Merge terms that share the same U-row pattern.
        Looking at U vectors:
        - r=0: U=[1,0,0,1] (a00+a11)
        - r=7: U=[1,0,0,1] (a00+a11) -- SAME!
        
        So we can try to merge r=0 and r=7.
        r=0: V=[1,0,0,0,0,1,0,0], W=[1,0,0,1,0,0,0,0]
        r=7: V=[0,0,1,0,0,0,0,1], W=[0,0,0,0,1,0,0,1]
        
        If we replace with a single term:
        V_new = V0 + V7 = [1,0,1,0,0,1,0,1]
        W_new = W0 + W7 = [1,0,0,1,1,0,0,1]
        This gives (a00+a11)(b00+b11+b02+b13)
        But we need separate outputs for left and right blocks.
        
        This won't work directly. Let's try a different approach:
        Use the null space of the flattening to find linear dependencies.
        
        Alternative: Try all possible single-term merges and verify.
        """
        if time.time() > deadline - 5:
            return None
            
        # Get the rank-14 decomposition
        r14 = try_strassen_rank14()
        if r14 is None:
            return None
            
        U = [row[:] for row in r14['U']]
        V = [row[:] for row in r14['V']]
        W = [row[:] for row in r14['W']]
        R = 14
        
        # Try merging pairs (a, b) where a < b
        # We try: U_new = Ua + Ub, V_new = Va + Vb, W_new = Wa + Wb
        # But this changes the contribution. We need to verify.
        # Actually, merging means replacing two rank-1 terms with one.
        # The new term contributes Ua*Va*Wa + Ub*Vb*Wb to each (i,j,k).
        # We need this to equal the original sum.
        # If we set U_new = Ua, V_new = Va, W_new = Wa + Wb, that doesn't work either.
        
        # The correct way: we need to find if there exist vectors u,v,w such that
        # u*V_a*W_a + u*V_b*W_b = u_new * v_new * w_new (as tensors)
        # This is generally not possible for arbitrary pairs.
        
        # Let's try a different approach: search for rank-13 directly.
        # Use the fact that the tensor has structure.
        
        # Actually, let's try a more systematic approach:
        # For each pair (a,b), try all combinations of signs for the merge.
        # The idea: replace terms a and b with a single term where
        # U_new[r] = sa*U[a][r] + sb*U[b][r] for some signs
        # This is getting complex. Let's just try a few specific merges.
        
        # Merge r=0 and r=7 (both use a00+a11):
        # Try: single term with V = V0+V7, W = W0+W7
        U_test = [row[:] for row in U[:13]]  # Remove term 7 (index 7)
        V_test = [row[:] for row in V[:13]]
        W_test = [row[:] for row in W[:13]]
        
        # Modify term 0 to absorb term 7
        for idx in range(8):
            V_test[0][idx] = V[0][idx] + V[7][idx]
            W_test[0][idx] = W[0][idx] + W[7][idx]
        
        if verify(U_test, V_test, W_test):
            return {"U": U_test, "V": V_test, "W": W_test}
        
        # Try other merge strategies with sign flips
        for sa in [-1, 1]:
            for sb in [-1, 1]:
                U_test = [row[:] for row in U[:13]]
                V_test = [row[:] for row in V[:13]]
                W_test = [row[:] for row in W[:13]]
                
                # Merge term 7 into term 0 with signs
                for idx in range(4):
                    U_test[idx][0] = U[idx][0] + sa * sb * U[idx][7]
                for idx in range(8):
                    V_test[idx][0] = V[idx][0] + sa * V[idx][7]
                    W_test[idx][0] = W[idx][0] + sb * W[idx][7]
                
                if verify(U_test, V_test, W_test):
                    return {"U": U_test, "V": V_test, "W": W_test}
        
        # Try merging r=1 and r=8 (both use a10+a11):
        for sa in [-1, 1]:
            for sb in [-1, 1]:
                U_test = [row[:] for row in U[:13]]
                V_test = [row[:] for row in V[:13]]
                W_test = [row[:] for row in W[:13]]
                
                # Remove term 8 (index 8), merge into term 1
                for idx in range(4):
                    U_test[idx][1] = U[idx][1] + sa * sb * U[idx][8]
                for idx in range(8):
                    V_test[idx][1] = V[idx][1] + sa * V[idx][8]
                    W_test[idx][1] = W[idx][1] + sb * W[idx][8]
                
                if verify(U_test, V_test, W_test):
                    return {"U": U_test, "V": V_test, "W": W_test}
        
        # Try merging r=2 and r=9 (both use a00):
        for sa in [-1, 1]:
            for sb in [-1, 1]:
                U_test = [row[:] for row in U[:13]]
                V_test = [row[:] for row in V[:13]]
                W_test = [row[:] for row in W[:13]]
                
                # Remove term 9 (index 9), merge into term 2
                for idx in range(4):
                    U_test[idx][2] = U[idx][2] + sa * sb * U[idx][9]
                for idx in range(8):
                    V_test[idx][2] = V[idx][2] + sa * V[idx][9]
                    W_test[idx][2] = W[idx][2] + sb * W[idx][9]
                
                if verify(U_test, V_test, W_test):
                    return {"U": U_test, "V": V_test, "W": W_test}
        
        # Try merging r=3 and r=10 (both use a11):
        for sa in [-1, 1]:
            for sb in [-1, 1]:
                U_test = [row[:] for row in U[:13]]
                V_test = [row[:] for row in V[:13]]
                W_test = [row[:] for row in W[:13]]
                
                # Remove term 10 (index 10), merge into term 3
                for idx in range(4):
                    U_test[idx][3] = U[idx][3] + sa * sb * U[idx][10]
                for idx in range(8):
                    V_test[idx][3] = V[idx][3] + sa * V[idx][10]
                    W_test[idx][3] = W[idx][3] + sb * W[idx][10]
                
                if verify(U_test, V_test, W_test):
                    return {"U": U_test, "V": V_test, "W": W_test}
        
        # Try merging r=4 and r=11 (both use a00+a01):
        for sa in [-1, 1]:
            for sb in [-1, 1]:
                U_test = [row[:] for row in U[:13]]
                V_test = [row[:] for row in V[:13]]
                W_test = [row[:] for row in W[:13]]
                
                # Remove term 11 (index 11), merge into term 4
                for idx in range(4):
                    U_test[idx][4] = U[idx][4] + sa * sb * U[idx][11]
                for idx in range(8):
                    V_test[idx][4] = V[idx][4] + sa * V[idx][11]
                    W_test[idx][4] = W[idx][4] + sb * W[idx][11]
                
                if verify(U_test, V_test, W_test):
                    return {"U": U_test, "V": V_test, "W": W_test}
        
        # Try merging r=5 and r=12 (both use a10-a00):
        for sa in [-1, 1]:
            for sb in [-1, 1]:
                U_test = [row[:] for row in U[:13]]
                V_test = [row[:] for row in V[:13]]
                W_test = [row[:] for row in W[:13]]
                
                # Remove term 12 (index 12), merge into term 5
                for idx in range(4):
                    U_test[idx][5] = U[idx][5] + sa * sb * U[idx][12]
                for idx in range(8):
                    V_test[idx][5] = V[idx][5] + sa * V[idx][12]
                    W_test[idx][5] = W[idx][5] + sb * W[idx][12]
                
                if verify(U_test, V_test, W_test):
                    return {"U": U_test, "V": V_test, "W": W_test}
        
        # Try merging r=6 and r=13 (both use a01-a11):
        for sa in [-1, 1]:
            for sb in [-1, 1]:
                U_test = [row[:] for row in U[:13]]
                V_test = [row[:] for row in V[:13]]
                W_test = [row[:] for row in W[:13]]
                
                # Remove term 13 (index 13), merge into term 6
                for idx in range(4):
                    U_test[idx][6] = U[idx][6] + sa * sb * U[idx][13]
                for idx in range(8):
                    V_test[idx][6] = V[idx][6] + sa * V[idx][13]
                    W_test[idx][6] = W[idx][6] + sb * W[idx][13]
                
                if verify(U_test, V_test, W_test):
                    return {"U": U_test, "V": V_test, "W": W_test}
        
        return None

    # First, perform the dual-catalecticant analysis
    lower_bound, r1, r2, r3 = dual_catalecticant_analysis()
    
    # Try the structured rank-14 decomposition first
    result = try_strassen_rank14()
    if result is not None:
        # If lower bound suggests rank 13 might be possible, try to merge
        if lower_bound <= 13:
            merged = try_merge_rank13()
            if merged is not None:
                return merged
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
