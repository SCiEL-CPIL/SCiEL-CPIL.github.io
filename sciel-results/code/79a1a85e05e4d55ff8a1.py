"""Matrix multiplication 3,3,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Verification function
    def verify(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], np.multiply(V[j*p+k], W[k*n+i]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def is_half_integer(arr, tol=1e-9):
        arr = np.array(arr, dtype=float)
        doubled = arr * 2
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=tol)

    # Schoolbook fallback (rank 72)
    def schoolbook():
        rank = n * m * p
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

    # Strategy 1: Block-circulant embedding
    # Embed M_{3,3,8} into M_{4,4,8} by padding with a zero row/column
    # Use Strassen's rank-16 for M_{4,4,4} on first 4 cols + rank-8 for remaining 4 cols
    # Project back to 3x3 subspace
    
    def strassen_4x4x4():
        """Return Strassen decomposition factors for 4x4 matrix multiplication.
        Rank 16 decomposition."""
        # Strassen's 7-multiply algorithm for 2x2, embedded in 4x4
        # For 4x4x4, use block structure: split 4x4 into four 2x2 blocks
        # M_{4,4,4} = direct sum of four M_{2,2,2} with shared multipliers
        # Actually, use the known rank-16 for 4x4x4
        
        # The 4x4x4 tensor has rank at most 16 (4x4x4/4 * 4 = 16 via block Strassen)
        # Let's construct it explicitly using 2x2 Strassen blocks
        
        # 2x2 Strassen multipliers:
        # m1 = (A11+A22)(B11+B22)
        # m2 = (A21+A22)B11
        # m3 = A11(B12-B22)
        # m4 = A22(B21-B11)
        # m5 = (A11+A12)B22
        # m6 = (A21-A11)(B11+B12)
        # m7 = (A12-A22)(B21+B22)
        # C11 = m1+m4-m5+m7
        # C12 = m3+m5
        # C21 = m2+m4
        # C22 = m1-m2+m3+m6
        
        # For 4x4, we use 4 blocks of 2x2, so we need 4*7 = 28 multipliers
        # But we can share some. Actually rank of M_{4,4,4} is known to be 16.
        # Let me use a different approach: the rank-16 decomposition of 4x4x4.
        
        # Actually, let's just use the block approach: 
        # 4x4 = [[A11, A12], [A21, A22]] where each block is 2x2
        # This gives 4 sub-multiplications of 2x2x2, each of rank 7
        # Total rank = 4*7 = 28 for the block diagonal part
        # But M_{4,4,4} rank is actually 16? No, it's at most 16 only with specific structure.
        
        # Let me reconsider. The rank of M_{4,4,4} is 16 (Alon et al.).
        # I'll use a known construction.
        
        # For now, let's use a simpler approach: 
        # The 4x4x4 tensor can be decomposed with rank 16 using 
        # the "circulant" embedding trick.
        
        # Actually, let me just implement a direct search for a low-rank 
        # decomposition of the full 3x3x8 tensor using the structural hint.
        
        pass
    
    # Let's use a more direct approach:
    # Split the 8 columns of B into two groups of 4
    # M_{3,3,8} = M_{3,3,4} (for first 4 cols) ⊕ M_{3,3,4} (for last 4 cols)
    # Each M_{3,3,4} can be viewed as embedding into M_{4,4,4}
    
    # For M_{3,3,4}: pad A to 4x4 with zero row/col, pad B to 4x4
    # Use rank-16 Strassen for M_{4,4,4}, project back to get rank ≤ 16 for each half
    # Total rank ≤ 32 for M_{3,3,8}
    # But we need to project carefully.
    
    # Actually, the projection from 4x4x4 to 3x3x4 reduces rank.
    # Let's think differently.
    
    # Direct approach: For each group of 4 output columns, we need 
    # C[:, 4j:4j+4] = A @ B[:, 4j:4j+4]
    # This is a 3x3 times 3x4 multiplication.
    
    # The tensor M_{3,3,4} has rank at most 12 (schoolbook) but possibly less.
    # Let's try to find a rank-8 decomposition for M_{3,3,4} using Strassen-like tricks.
    
    # M_{3,3,4}: We can embed into M_{4,4,4} by padding.
    # If we have a rank-16 decomposition of M_{4,4,4}, projecting gives rank ≤ 16 for M_{3,3,4}.
    # Two such projections give rank ≤ 32 for M_{3,3,8}.
    
    # But we can do better: note that M_{3,3,4} can be decomposed as:
    # C[:,0:2] = A @ B[:,0:2] (rank ≤ 3*2*2=12 schoolbook, or less)
    # C[:,2:4] = A @ B[:,2:4] (rank ≤ 12)
    # This doesn't help directly.
    
    # Let me try a concrete construction for rank-32:
    # Use Strassen's algorithm for 2x2 blocks, applied to the 3x3x4 case.
    
    # Actually, the simplest improvement over schoolbook:
    # Use the fact that 3x3 matrix multiplication can be done in rank 23 (Lader 2007)
    # or rank 24. For 3x3x8, we can split into three 3x3x3 + one 3x3x2.
    # M_{3,3,8} = M_{3,3,3} ⊕ M_{3,3,3} ⊕ M_{3,3,2}
    # Rank ≤ rank(M_{3,3,3}) + rank(M_{3,3,3}) + rank(M_{3,3,2})
    # If rank(M_{3,3,3}) = 23 and rank(M_{3,3,2}) ≤ 6 (schoolbook is 18, but 3x3x2 might be less)
    # Actually rank(M_{3,3,2}) ≤ 3*3*2 = 18 schoolbook, but likely less.
    
    # For M_{3,3,2}: C = A @ B where A is 3x3, B is 3x2
    # This is 6 scalar multiplications needed (each C[i,j] = sum_k A[i,k]*B[k,j])
    # So 2 outputs, each requiring 3 mults = 6 mults total. Rank = 6.
    # Wait, no. The tensor M_{3,3,2} has 3*3*2 = 18 entries, rank at most 18.
    # But structurally, C[:,0] = A @ B[:,0] and C[:,1] = A @ B[:,1]
    # Each is a 3x3x1 tensor of rank at most 9 (schoolbook) but actually rank 9 for 3x3x1?
    # No, M_{3,3,1} is just a linear map, rank = 9 (since it's 3x3 matrix times a vector, 
    # the tensor has 9 independent multiplications).
    # So M_{3,3,2} = M_{3,3,1} ⊕ M_{3,3,1}, rank ≤ 18.
    # But actually M_{3,3,1} rank is 9 (each output C[i] = sum_j A[i,j]*B[j], 9 mults).
    
    # So M_{3,3,8} rank ≤ 9+9+9+9+9+9+9+9 = 72 (schoolbook)
    # With Strassen for 3x3x3: rank 23 each, so 23+23+18 = 64
    
    # Let's try to find rank-64 decomposition using rank-23 for 3x3x3 blocks.
    # I need the actual rank-23 factors for 3x3x3.
    
    # Known: The rank of 3x3 matrix multiplication is 23 (Lader 2007, but this is disputed; 
    # some say it's 23, others prove lower bounds of 19).
    # Actually, the best known upper bound is 23 (Strassen's algorithm gives 19 for some variants?).
    # No, Strassen's algorithm for 2x2 gives rank 7. For 3x3, the best known is 19 (Strassen 1985) 
    # or 23. Let me check: Strassen showed rank ≤ 19 for 3x3x3.
    
    # Strassen's 3x3x3 rank-19 algorithm:
    # This is complex. Let me use a simpler known decomposition.
    
    # Alternative: Use the "block" approach with 2x2 Strassen inside 3x3.
    # Split 3x3 into:
    # [A11 A12 A13]
    # [A21 A22 A23]  
    # [A31 A32 A33]
    
    # One approach: use 2x2 Strassen on the top-left 2x2 block, treating 
    # the third row/col separately.
    
    # Let me just do a computational search for a good decomposition.
    
    # Given the complexity, let me implement a structured search:
    # 1. Try to find a rank-64 decomposition by splitting into three 3x3x3 (rank 23 each) 
    #    and one 3x3x2 (rank 6 each? No, rank 18).
    #    23+23+23+18 = 87 > 72. That's worse!
    
    # Wait, I made an error. 3x3x8 = 3x3x3 + 3x3x3 + 3x3x2.
    # Rank ≤ 23 + 23 + 18 = 64. That's better than 72!
    
    # So if I can find rank-23 factors for 3x3x3 and rank-18 for 3x3x2 (trivial),
    # I get rank 64.
    
    # Let me implement the rank-18 for 3x3x2 (trivial schoolbook for that part)
    # and search for rank-23 for 3x3x3.
    
    # For 3x3x3, let me try to construct a rank-19 or 23 decomposition.
    # Strassen's 3x3x3 rank-19:
    # 
    # Actually, let me use a known result. The 3x3x3 tensor can be decomposed 
    # with rank 19 using Strassen's method. Here's one construction:
    # 
    # A = [a11 a12 a13; a21 a22 a23; a31 a32 a33]
    # B = [b11 b12 b13; b21 b22 b23; b31 b32 b33]
    # 
    # Define 19 products:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)(b21)
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)(b22)
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # m8 = a13(b11+b22+b33)
    # m9 = a23(b11+b22+b33)
    # m10 = a33(b11+b22+b33)
    # m11 = a13(b21+b31)
    # m12 = a23(b21+b31)
    # m13 = a33(b21+b31)
    # m14 = a13(b32+b33)  -- need to be more careful
    # ...
    
    # This is getting too complex to get right without reference. Let me use a 
    # computational approach: search for a low-rank decomposition of 3x3x3.
    
    # For 3x3x3, the tensor has 27 entries. We want to express it as sum of r rank-1 tensors.
    # Each rank-1 tensor is u ⊗ v ⊗ w where u ∈ R^9, v ∈ R^9, w ∈ R^9.
    
    # Let me do a random search for 3x3x3 with rank 19, 20, 21, 22, 23, 24.
    
    def find_3x3x3_decomposition(max_rank=24, num_trials=200):
        """Search for a low-rank decomposition of 3x3x3 matrix multiplication."""
        nn, mm, pp = 3, 3, 3
        
        def verify_3x3x3(U, V, W, rank):
            U = np.array(U, dtype=float)
            V = np.array(V, dtype=float)
            W = np.array(W, dtype=float)
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        val = np.dot(U[i*mm+j], np.multiply(V[j*pp+k], W[k*nn+i]))
                        if abs(val - 1.0) > 1e-9:
                            return False
            return True
        
        for r in range(max_rank, 18, -1):
            for trial in range(num_trials):
                if time.time() > deadline:
                    return None
                
                # Generate random half-integer U and V
                U = np.zeros((nn*mm, r))
                V = np.zeros((mm*pp, r))
                
                for c in range(r):
                    U[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=nn*mm)
                    V[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=mm*pp)
                    if np.all(U[:, c] == 0):
                        U[np.random.randint(nn*mm), c] = 1
                    if np.all(V[:, c] == 0):
                        V[np.random.randint(mm*pp), c] = 1
                
                W = np.zeros((pp*nn, r))
                valid = True
                
                for k in range(pp):
                    for i in range(nn):
                        M_i = np.zeros((mm, r))
                        for j in range(mm):
                            for c in range(r):
                                M_i[j, c] = U[i*mm+j, c] * V[j*pp+k, c]
                        
                        try:
                            w_min, res, rank_sol, sv = np.linalg.lstsq(M_i, np.ones(mm), rcond=None)
                            w_rounded = np.round(w_min * 2) / 2
                            check = M_i @ w_rounded
                            if np.allclose(check, np.ones(mm), atol=1e-9):
                                W[k*nn+i, :] = w_rounded
                            else:
                                # Try null space perturbation
                                if rank_sol < r:
                                    _, s, Vt = np.linalg.svd(M_i)
                                    null_space = Vt[rank_sol:, :].T
                                    found = False
                                    for _ in range(30):
                                        coeffs = np.random.choice([-1, -0.5, 0, 0.5, 1], size=r - rank_sol)
                                        z = null_space @ coeffs
                                        w_cand = w_min + z
                                        w_r2 = np.round(w_cand * 2) / 2
                                        if np.allclose(M_i @ w_r2, np.ones(mm), atol=1e-9):
                                            W[k*nn+i, :] = w_r2
                                            found = True
                                            break
                                    if not found:
                                        valid = False
                                        break
                                else:
                                    valid = False
                                    break
                        except:
                            valid = False
                            break
                    
                    if not valid:
                        break
                
                if valid and verify_3x3x3(U.tolist(), V.tolist(), W.tolist(), r):
                    if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                        return (U.tolist(), V.tolist(), W.tolist(), r)
        
        return None
    
    # Try to find 3x3x3 decomposition
    result_333 = find_3x3x3_decomposition(max_rank=24, num_trials=100)
    
    if result_333 is not None:
        U333, V333, W333, r333 = result_333
        
        # Now assemble 3x3x8 = M_{3,3,3} ⊕ M_{3,3,3} ⊕ M_{3,3,2}
        # For the 3x3x2 part, use schoolbook (rank 18)
        # Total rank = 2*r333 + 18
        
        # Actually, let me be more careful about the assembly.
        # M_{3,3,8}: A is 3x3, B is 3x8, C = A@B is 3x8.
        # Split B into B1 (3x3), B2 (3x3), B3 (3x2).
        # C1 = A@B1 (3x3), C2 = A@B2 (3x3), C3 = A@B3 (3x2).
        
        # For C1 and C2, use the 3x3x3 decomposition (rank r333 each).
        # For C3, use schoolbook 3x3x2 (rank 18).
        
        total_rank = 2 * r333 + 18
        
        U_full = np.zeros((n*m, total_rank))
        V_full = np.zeros((m*p, total_rank))
        W_full = np.zeros((p*n, total_rank))
        
        # Fill in the first 3x3x3 block (columns 0-2 of B)
        col_offset = 0
        for c in range(r333):
            U_full[:, col_offset + c] = U333[c] if isinstance(U333[c], list) else U333[:, c]
        
        # Actually, let me be more careful with the indexing.
        # U333 has shape (9, r333), V333 has shape (9, r333), W333 has shape (9, r333)
        # W333[k*n+i, c] corresponds to output C[i, k] for the 3x3x3 case.
        
        U_full = np.zeros((n*m, total_rank))
        V_full = np.zeros((m*p, total_rank))
        W_full = np.zeros((p*n, total_rank))
        
        # Block 1: columns 0-2 of B (first 3x3x3)
        for c in range(r333):
            U_full[:, c] = U333[:, c]
            # V: for columns 0-2 of B, V[j*p+k, c] should be V333[j*3+k, c] for k=0,1,2
            for j in range(m):
                for k in range(3):
                    V_full[j*p + k, c] = V333[j*3 + k, c]
            # W: W333[k*3+i, c] for k=0,1,2, i=0,1,2
            for k in range(3):
                for i in range(n):
                    W_full[k*n + i, c] = W333[k*3 + i, c]
        
        # Block 2: columns 3-5 of B (second 3x3x3)
        col_offset = r333
        for c in range(r333):
            U_full[:, col_offset + c] = U333[:, c]
            for j in range(m):
                for k in range(3):
                    V_full[j*p + (3 + k), col_offset + c] = V333[j*3 + k, c]
            for k in range(3):
                for i in range(n):
                    W_full[(3 + k)*n + i, col_offset + c] = W333[k*3 + i, c]
        
        # Block 3: columns 6-7 of B (3x3x2, schoolbook rank 18)
        col_offset = 2 * r333
        col = col_offset
        for i in range(n):
            for j in range(m):
                for k in range(2):
                    U_full[i*m + j, col] = 1
                    V_full[j*p + (6 + k), col] = 1
                    W_full[(6 + k)*n + i, col] = 1
                    col += 1
        
        # Verify
        if verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), total_rank):
            if is_half_integer(U_full) and is_half_integer(V_full) and is_half_integer(W_full):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: try direct search on 3x3x8
    def direct_search_338(max_rank=68, num_trials=200):
        """Direct search for low-rank decomposition of 3x3x8."""
        for r in range(max_rank, 60, -1):
            for trial in range(num_trials):
                if time.time() > deadline:
                    return None
                
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                
                for c in range(r):
                    U[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=n*m)
                    V[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=m*p)
                    if np.all(U[:, c] == 0):
                        U[np.random.randint(n*m), c] = 1
                    if np.all(V[:, c] == 0):
                        V[np.random.randint(m*p), c] = 1
                
                W = np.zeros((p*n, r))
                valid = True
                
                for k in range(p):
                    for i in range(n):
                        M_i = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                M_i[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        
                        try:
                            w_min, res, rank_sol, sv = np.linalg.lstsq(M_i, np.ones(m), rcond=None)
                            w_rounded = np.round(w_min * 2) / 2
                            check = M_i @ w_rounded
                            if np.allclose(check, np.ones(m), atol=1e-9):
                                W[k*n+i, :] = w_rounded
                            else:
                                if rank_sol < r:
                                    _, s, Vt = np.linalg.svd(M_i)
                                    null_space = Vt[rank_sol:, :].T
                                    found = False
                                    for _ in range(30):
                                        coeffs = np.random.choice([-1, -0.5, 0, 0.5, 1], size=r - rank_sol)
                                        z = null_space @ coeffs
                                        w_cand = w_min + z
                                        w_r2 = np.round(w_cand * 2) / 2
                                        if np.allclose(M_i @ w_r2, np.ones(m), atol=1e-9):
                                            W[k*n+i, :] = w_r2
                                            found = True
                                            break
                                    if not found:
                                        valid = False
                                        break
                                else:
                                    valid = False
                                    break
                        except:
                            valid = False
                            break
                    
                    if not valid:
                        break
                
                if valid and verify(U.tolist(), V.tolist(), W.tolist(), r):
                    if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                        return (U.tolist(), V.tolist(), W.tolist(), r)
        
        return None
    
    result_direct = direct_search_338(max_rank=68, num_trials=100)
    
    if result_direct is not None:
        U_d, V_d, W_d, r_d = result_direct
        return {"U": U_d, "V": V_d, "W": W_d}
    
    # Final fallback: schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
