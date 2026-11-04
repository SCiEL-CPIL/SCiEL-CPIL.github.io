import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n  # 16, 36, 36

    # Build the matmul tensor T as a 3D array T[a, b, c] where
    # a = i*m+j (0..15), b = j*p+k (0..35), c = k*n+i (0..35)
    # T[a, b, c] = 1 if a%4==b//9 and b%9==c//4 and c%4==a//4 else 0
    # Actually: a = i*4+j, b = j*9+k, c = k*4+i
    # So a%4=j, b//9=j, b%9=k, c//4=k, c%4=i, a//4=i
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Reshape T into a matrix of shape (nm*mp, pn) for rank analysis
    # T_mat[a*mp+b, c] = T[a,b,c]
    T_mat = T.reshape(nm * mp, pn)

    # Schoolbook: rank 144
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.int64)
        V = np.zeros((mp, rank), dtype=np.int64)
        W = np.zeros((pn, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_s, V_s, W_s = schoolbook()
    best_rank = n * m * p  # 144
    best_U, best_V, best_W = U_s, V_s, W_s

    def validate(U, V, W):
        """Check that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c."""
        # Efficient check using einsum
        # result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        rank = U.shape[1]
        # Compute via matrix multiplication: (U @ V^T) gives (nm, mp) but that's not right
        # We need T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # This is equivalent to: for each r, outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Sum over r
        # Efficient: for each a, compute (V * W[:,a].T) @ U[:,a].T ... no
        # Let's do: result = np.einsum('ar,br,cr->abc', U, V, W)
        # But that's O(nm*mp*pn*rank) which is 16*36*36*rank = 20736*rank
        # For rank=100, that's ~2M ops. Fine.
        result = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.array_equal(result, T)

    # Try to find a better decomposition
    # Strategy: Use the known 4x4x4 rank-49 decomposition and extend to 4x4x9
    # 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
    # For 4x4x1: rank 16 (each output is a separate bilinear form)
    # Total: 49 + 49 + 16 = 114
    
    # Let's try to construct 4x4x4 rank-49 properly using two-level Strassen
    def build_4x4x4_strassen():
        """Build rank-49 decomposition for 4x4x4 using two levels of Strassen."""
        n4, m4, p4 = 4, 4, 4
        nm4, mp4, pn4 = n4*m4, m4*p4, p4*n4  # 16, 16, 16
        
        # Strassen 2x2x2: rank 7
        # For 2x2 matrices: A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
        # C = AB, C[i,k] = sum_j A[i,j]*B[j,k]
        # 
        # The 7 Strassen products (standard formulation):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00+B01)  
        # M3 = (A00-A10)(B10+B11)
        # M4 = A11(B11-B00)
        # M5 = A00(B01-B11)
        # M6 = (A10+A11)B00
        # M7 = (A00+A01)B11
        #
        # C00 = M1 - M4 + M6 + M7
        # C01 = M3 + M5 - M5 + M7  -> Let me use the correct formulas
        #
        # Actually, let me just verify a known correct set of 7 terms for 2x2x2.
        # I'll use the formulation where we explicitly define (u, v, w) for each of 7 terms.
        # u[i*2+j], v[j*2+k], w[k*2+i]
        
        # Known correct Strassen 2x2x2 decomposition (verified):
        # I'll use a different approach: solve for it directly.
        # For 2x2x2, T has shape (4,4,4) with T[i*2+j, j*2+k, k*2+i] = 1
        # Find 7 vectors (u_r, v_r, w_r) in R^4 such that sum_r u_r[a]*v_r[b]*w_r[c] = T[a,b,c]
        
        # Let me just hardcode a verified Strassen decomposition.
        # After careful derivation, here's a valid rank-7 decomposition for 2x2x2:
        
        # Using the standard Strassen with correct reconstruction:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)(b00+b01)
        # M3 = (a00-a10)(b10+b11)
        # M4 = a11(b11-b00)
        # M5 = a00(b01-b11)
        # M6 = (a10+a11)b00
        # M7 = (a00+a01)b11
        #
        # C00 = M1 - M4 + M6 + M7
        # C01 = M3 + M5
        # C10 = M6 - M4 + M1 - M7  -- wait, let me re-derive
        # C10 = a10*b00 + a11*b10
        # C11 = a10*b01 + a11*b11
        #
        # Let me verify C00 = M1 - M4 + M6 + M7:
        # M1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
        # M4 = a11*b11 - a11*b00
        # M6 = a10*b00 + a11*b00
        # M7 = a00*b11 + a01*b11
        # M1 - M4 + M6 + M7 = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b11 + a11*b00 + a10*b00 + a11*b00 + a00*b11 + a01*b11
        # = a00*b00 + 2*a00*b11 + 3*a11*b00 + a10*b00 + a01*b11
        # This is NOT equal to C00 = a00*b00 + a01*b10. So this is wrong.
        
        # I'm having trouble remembering the exact Strassen formulas. Let me just use
        # a computational approach: find a rank-7 decomposition for 2x2x2 by solving.
        
        # For 2x2x2: T[a,b,c] where a=i*2+j, b=j*2+k, c=k*2+i
        # T has 8 non-zero entries (one for each (i,j,k) in {0,1}^3)
        # We want sum_r u_r[a]*v_r[b]*w_r[c] = T[a,b,c]
        
        # This is a system of 64 equations (4*4*4) with 7*4*3=84 unknowns.
        # But most equations are trivially satisfied (T=0).
        # The 8 non-trivial equations: for each (i,j,k), T[i*2+j, j*2+k, k*2+i] = 1
        # Plus 56 equations where T=0.
        
        # Let me use a greedy approach: pick u_r and v_r, then solve for w_r.
        # Actually, for a rank-r decomposition, we can fix U and V and solve for W
        # via a linear system, or vice versa.
        
        # Alternative: just use the schoolbook rank-4 for 2x2x2 and build 4x4x4 with rank 16*4=64?
        # No, that's worse than 49.
        
        # Let me try a different approach: use the known fact that M(4)=49 and construct
        # it via the "doubling" method.
        
        # Actually, let me just try to find ANY valid decomposition with rank < 144 for 4x4x9.
        # The easiest improvement: use the fact that 4x4x9 can be split as 4x4x4 + 4x4x4 + 4x4x1
        # with ranks 49+49+16 = 114. But I need valid 4x4x4 rank-49 decompositions.
        
        # Let me try yet another approach: use the "border rank" or "approximate" methods?
        # No, the task requires exact decomposition.
        
        # OK, let me just try to find a valid low-rank decomposition by random search + refinement.
        # Start with random integer vectors and use ALS (alternating least squares) to refine.
        
        return None  # Placeholder

    # Let's try a computational approach: random initialization + ALS refinement
    def try_rank_search(max_rank, time_budget):
        """Try to find a valid decomposition with rank <= max_rank using random init + ALS."""
        start_time = time.time()
        np.random.seed(42)
        
        for attempt in range(50):
            if time.time() - start_time > time_budget:
                break
            
            r = max_rank
            # Random integer initialization
            U = np.random.randint(-2, 3, size=(nm, r))
            V = np.random.randint(-2, 3, size=(mp, r))
            W = np.random.randint(-2, 3, size=(pn, r))
            
            # ALS refinement: fix two, solve for the third
            for iteration in range(200):
                if time.time() - start_time > time_budget:
                    return None
                
                # Fix V and W, solve for U
                # For each a: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For fixed a: sum_r U[a,r] * (sum_c V[b,r]*W[c,r]) = T[a,b,c] for all b,c
                # This is: U[a,:] @ (V * W.T) ... not quite.
                # Let's define M_b[c] = sum_r V[b,r]*W[c,r] for fixed b
                # Then T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For fixed a and b: T[a,b,:] = sum_r U[a,r]*V[b,r]*W[:,r]
                # This is a linear system in U[a,:] for each (a,b).
                
                # More efficient: for each a, we have equations for all (b,c):
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Let A_a[b,c,r] = V[b,r]*W[c,r], then T[a,:,:] = sum_r U[a,r] * A_a[:,:,r]
                # This is: vec(T[a,:,:]) = (A_a_reshaped) @ U[a,:]
                # where A_a_reshaped has shape (mp*pn, r)
                
                # For each a:
                for a in range(nm):
                    # Build the system: for all (b,c), T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                    # Matrix form: T[a].reshape(-1) = M_a @ U[a]
                    # where M_a[b*pn+c, r] = V[b,r]*W[c,r]
                    M_a = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, r)
                    rhs = T[a].reshape(-1)
                    # Solve least squares
                    U[a], _, _, _ = np.linalg.lstsq(M_a, rhs, rcond=None)
                
                # Similarly for V and W
                for b in range(mp):
                    M_b = np.einsum('ar,cr->acr', U, W).reshape(nm*pn, r)
                    rhs = T[:, b, :].reshape(-1)
                    V[b], _, _, _ = np.linalg.lstsq(M_b, rhs, rcond=None)
                
                for c in range(pn):
                    M_c = np.einsum('ar,br->abr', U, V).reshape(nm*mp, r)
                    rhs = T[:, :, c].reshape(-1)
                    W[c], _, _, _ = np.linalg.lstsq(M_c, rhs, rcond=None)
                
                # Round to nearest integers/half-integers and check
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Check if the rounded version is valid
                if validate(U_int, V_int, W_int):
                    return U_int, V_int, W_int
                
                # Also check if the unrounded version is close enough
                # (for exact match, we need integer/half-integer)
        
        return None

    # Try various ranks
    for target_rank in [100, 110, 114, 120, 130, 140]:
        if time.time() > deadline - 30:
            break
        result = try_rank_search(target_rank, 10)
        if result is not None:
            U_r, V_r, W_r = result
            if target_rank < best_rank:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break  # Found a valid one, stop
    
    # Also try a structured approach: split 9 into 4+4+1
    # 4x4x4 + 4x4x4 + 4x4x1
    # For 4x4x1, rank is 16 (each of the 16 products i*4+j and k is separate)
    # For 4x4x4, use schoolbook rank 16 each, total 16+16+16 = 48 for the two 4x4x4 blocks
    # Wait, 4x4x4 schoolbook is rank 16 (4*4*4/4 = 16? No, 4*4*4 = 64? No.)
    # 4x4x4: n=4, m=4, p=4, rank = n*m*p = 64 for schoolbook? No, rank = sum over i,j,k of 1 = 4*4*4 = 64.
    # Wait, schoolbook rank for n x m x p is n*m*p? No, it's n*m*p only if we count each (i,j,k) triple.
    # Actually, schoolbook: for each (i,j,k), one term. Number of terms = n*m*p.
    # For 4x4x4: 4*4*4 = 64. For 4x4x9: 4*4*9 = 144. For 4x4x1: 4*4*1 = 16.
    # So 4x4x4 + 4x4x4 + 4x4x1 = 64 + 64 + 16 = 144. Same as direct schoolbook. No improvement.
    
    # The only way to improve is to find a non-schoolbook decomposition.
    # Given the time constraints and difficulty, let's just return the best we found.
    
    # Convert to list format as required
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    # Ensure entries are integer or half-integer
    # Convert to Fraction for safety, then back to float/int
    def clean_matrix(M):
        cleaned = []
        for row in M:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                v = round(val * 2) / 2
                if abs(v - round(v)) < 1e-9:
                    new_row.append(int(round(v)))
                else:
                    new_row.append(v)
            cleaned.append(new_row)
        return cleaned
    
    U_out = clean_matrix(U_out)
    V_out = clean_matrix(V_out)
    W_out = clean_matrix(W_out)
    
    return {"U": U_out, "V": V_out, "W": W_out}
