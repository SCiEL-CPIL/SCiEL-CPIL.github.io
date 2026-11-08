import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 145.0
    
    n, m, p = 3, 4, 4
    Nm, Mp, Pn = n*m, m*p, p*n  # 12, 16, 12
    
    # Build the multiplication tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((Nm, Mp, Pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify_fast(U, V, W):
        R = U.shape[1]
        # T_rec[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        UV = U[:, None, :] * V[None, :, :]  # (Nm, Mp, R)
        UV_flat = UV.reshape(-1, R)
        T_rec_flat = UV_flat @ W.T  # (Nm*Mp, Pn)
        T_rec = T_rec_flat.reshape(Nm, Mp, Pn)
        return np.allclose(T_rec, T, atol=1e-8)
    
    # Build Strassen 2x2x2 decomposition
    def strassen_2x2():
        """Return U, V, W for 2x2x2 matrix multiplication, rank 7"""
        R = 7
        U = np.zeros((4, R))
        V = np.zeros((4, R))
        W = np.zeros((4, R))
        
        # Strassen's 7 products
        # M1 = (A00+A11)*(B00+B11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        
        # M2 = A11*(B10-B00)
        U[3,1]=1
        V[2,1]=1; V[0,1]=-1
        W[2,1]=1; W[3,1]=1
        
        # M3 = A00*(B01-B11)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        
        # M4 = (A10+A00)*B00
        U[2,3]=1; U[0,3]=1
        V[0,3]=1
        W[0,3]=1; W[2,3]=1
        
        # M5 = (A01+A11)*B11
        U[1,4]=1; U[3,4]=1
        V[3,4]=1
        W[1,4]=1; W[3,4]=1
        
        # M6 = (A10-A00)*(B00+B01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[2,5]=1; W[3,5]=1
        
        # M7 = (A01-A10)*(B10+B11)
        U[1,6]=1; U[2,6]=-1
        V[2,6]=1; V[3,6]=1
        W[1,6]=1; W[2,6]=1
        
        return U, V, W
    
    # Build 2x4x4 decomposition using Strassen on 2x2 column blocks
    def build_2x4x4():
        """
        Decompose 2x4 matrix * 4x4 matrix = 2x4 result
        Using 2 Strassen 2x2x2 blocks (columns 0-1 and 2-3)
        Each block: 2x2 * 2x2 -> 2x2
        Total rank: 2 * 7 = 14
        """
        R = 14
        U = np.zeros((8, R))   # 2 rows x 4 cols = 8 entries
        V = np.zeros((16, R))  # 4 rows x 4 cols = 16 entries
        W = np.zeros((8, R))   # 2 rows x 4 cols = 8 entries
        
        U_s, V_s, W_s = strassen_2x2()
        
        # Block 1: columns 0-1
        # U: rows 0-1, cols 0-1 -> local indices 0-3
        for r in range(2):
            for c in range(2):
                U[r*4+c, 0:7] += U_s[r*2+c, :]
        
        # V: rows 0-1, cols 0-1 -> local indices 0-3
        for r in range(2):
            for c in range(2):
                V[r*4+c, 0:7] += V_s[r*2+c, :]
        
        # W: rows 0-1, cols 0-1 -> local indices 0-3
        for r in range(2):
            for c in range(2):
                W[r*4+c, 0:7] += W_s[r*2+c, :]
        
        # Block 2: columns 2-3
        for r in range(2):
            for c in range(2):
                U[r*4+(c+2), 7:14] += U_s[r*2+c, :]
        
        for r in range(2):
            for c in range(2):
                V[r*4+(c+2), 7:14] += V_s[r*2+c, :]
        
        for r in range(2):
            for c in range(2):
                W[r*4+(c+2), 7:14] += W_s[r*2+c, :]
        
        return U, V, W
    
    U_24, V_24, W_24 = build_2x4x4()
    
    # Now build 3x4x4 by combining two 2x4x4 decompositions
    # Block A: rows 0-1 of A (indices 0-7 in U)
    # Block B: rows 1-2 of A (indices 4-11 in U)
    # They overlap in row 1 of A
    
    # Strategy: 
    # - Use block A for rows 0-1, but only use its contribution to row 0 of output
    # - Use block B for rows 1-2, but only use its contribution to row 2 of output
    # - For row 1 of output, we need contributions from both blocks
    # 
    # Actually, let's think differently. The output C has 3 rows x 4 cols.
    # C[0,:] = A[0,:] * B
    # C[1,:] = A[1,:] * B  
    # C[2,:] = A[2,:] * B
    #
    # Block A (rows 0-1 of A) gives us C[0,:] and C[1,:]
    # Block B (rows 1-2 of A) gives us C[1,:] and C[2,:]
    #
    # If we use both blocks fully, we get C[1,:] computed twice (sum of both)
    # We need C[1,:] computed once.
    #
    # Solution: Use block A for C[0,:] and part of C[1,:], 
    #            Use block B for C[2,:] and the rest of C[1,:]
    #
    # But Strassen doesn't split cleanly this way.
    #
    # Alternative: Just use 3 separate 1x4x4 decompositions
    # Each 1x4x4 can be done with rank 8 (two 2x2 Strassen blocks, but only 1 row)
    # Actually 1x4 * 4x4 = 1x4, which is 4 dot products, rank 4
    # Total: 3 * 4 = 12? No, that's not right either.
    #
    # Let me just build it directly with the block approach and verify.
    
    def build_3x4x4_v1():
        """
        Use two overlapping 2x4x4 blocks.
        Block A: A rows 0-1 -> C rows 0-1
        Block B: A rows 1-2 -> C rows 1-2
        Total rank: 2 * 14 = 28
        But C row 1 gets double-counted. Need to fix.
        
        Actually, let's just use the full 2x4x4 for rows 0-1 and 
        a 1x4x4 for row 2.
        1x4x4: rank 4 (4 independent dot products)
        Total: 14 + 4 = 18
        """
        R_total = 14 + 4
        U = np.zeros((Nm, R_total))
        V = np.zeros((Mp, R_total))
        W = np.zeros((Pn, R_total))
        
        # Block A: 2x4x4 for A rows 0-1
        U_a, V_a, W_a = build_2x4x4()
        # U_a is (8, 14): 2 rows x 4 cols
        # Map to global: A row i, col j -> index i*4+j
        for r in range(2):
            for c in range(4):
                U[r*4+c, 0:14] += U_a[r*4+c, :]
        
        V[0:16, 0:14] += V_a
        
        # W_a is (8, 14): 2 rows x 4 cols
        # C row i, col k -> index k*3+i
        for r in range(2):
            for c in range(4):
                W[c*3+r, 0:14] += W_a[r*4+c, :]
        
        # Block B: 1x4x4 for A row 2
        # C[2,:] = A[2,:] * B
        # This is 4 dot products: for each col k of C, sum_j A[2,j]*B[j,k]
        # Rank 4: one term per output column
        for k in range(4):
            idx = 14 + k
            # U: A[2, j] for j=0..3
            for j in range(4):
                U[2*4+j, idx] += 1.0
            # V: B[j, k] for j=0..3
            for j in range(4):
                V[j*4+k, idx] += 1.0
            # W: C[2, k]
            W[k*3+2, idx] += 1.0
        
        return U, V, W
    
    U1, V1, W1 = build_3x4x4_v1()
    
    if verify_fast(U1, V1, W1):
        best_U, best_V, best_W = U1, V1, W1
        best_rank = 18
    else:
        # Fallback to direct
        R = 48
        U = np.zeros((Nm, R))
        V = np.zeros((Mp, R))
        W = np.zeros((Pn, R))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        best_U, best_V, best_W = U, V, W
        best_rank = 48
    
    # Try to improve by searching for lower rank
    np.random.seed(42)
    vals = np.array([-2, -1, -0.5, 0.0, 0.5, 1, 2], dtype=float)
    
    for target_rank in range(best_rank - 1, 10, -1):
        if time.time() - start_time > timeout - 20:
            break
        
        found = False
        max_attempts = 50 if target_rank >= 15 else 100
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout - 15:
                break
            
            U = np.random.choice(vals, size=(Nm, target_rank))
            V = np.random.choice(vals, size=(Mp, target_rank))
            W = np.random.choice(vals, size=(Pn, target_rank))
            
            if verify_fast(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue
    
    # Try column removal from best solution
    if best_rank > 12:
        U_try = best_U.copy()
        V_try = best_V.copy()
        W_try = best_W.copy()
        r = 0
        while r < U_try.shape[1]:
            if time.time() - start_time > timeout - 10:
                break
            U_test = np.delete(U_try, r, axis=1)
            V_test = np.delete(V_try, r, axis=1)
            W_test = np.delete(W_try, r, axis=1)
            if verify_fast(U_test, V_test, W_test):
                U_try, V_try, W_try = U_test, V_test, W_test
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank -= 1
                # Don't increment r, check same position again
            else:
                r += 1
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
