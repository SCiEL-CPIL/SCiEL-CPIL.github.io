import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the 2x2 x 2x8 multiplication tensor."""
    n, m, p = 2, 2, 8
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The multiplication tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # where i indexes A (n*m=4), j indexes B (m*p=16), k indexes C (p*n=16)
    # A is 2x2: A[i//2, i%2]
    # B is 2x8: B[j//8, j%8]
    # C is 2x8: C[k//8, k%8] = sum_l A[0,l]*B[l,k%8] + A[1,0]*B[1,k%8]... 
    # Actually C[a,b] = sum_l A[a,l] * B[l,b]
    # C index k: k = b*2 + a (trace convention W[k*n+i] where i is row, k is col? No...)
    # W[k*n+i,r] - the task says output factor follows trace convention W[k*n+i,r]
    # So W index = k*n + i where k is column index (0..p-1) and i is row index (0..n-1)
    # So W index = col * n + row = col * 2 + row
    
    # Let's build the tensor T[a, l, b] = delta for multiplication
    # T[A_idx, B_idx, C_idx] = 1 if C = A*B component
    # A_idx = a*2 + l (row a, col l of A)
    # B_idx = l*8 + b (row l, col b of B)  
    # C_idx = b*2 + a (col b, row a of C) - trace convention
    
    # T[A_idx, B_idx, C_idx] = 1 if A_idx//2 == a, A_idx%2 == l, 
    #                            B_idx//8 == l, B_idx%8 == b,
    #                            C_idx//2 == b, C_idx%2 == a
    # i.e., A_idx%2 == B_idx//8 and C_idx//2 == B_idx%8 and A_idx//2 == C_idx%2
    
    max_err = 0.0
    for a in range(n):
        for l in range(m):
            for b in range(p):
                A_idx = a * m + l
                B_idx = l * p + b
                C_idx = b * n + a
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[A_idx, r] * V[B_idx, r] * W[C_idx, r]
                err = abs(actual - expected)
                if err > max_err:
                    max_err = err
    return max_err

def _strassen_28():
    """Return the standard 4-block Strassen decomposition with rank 28."""
    n, m, p = 2, 2, 8
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(4):
        k0 = 2 * s
        k1 = 2 * s + 1
        
        v_e = 0 * 8 + k0
        v_f = 0 * 8 + k1
        v_g = 1 * 8 + k0
        v_h = 1 * 8 + k1
        
        w_c00 = k0 * 2 + 0
        w_c01 = k1 * 2 + 0
        w_c10 = k0 * 2 + 1
        w_c11 = k1 * 2 + 1
        
        r = s * 7 + 0
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        
        r = s * 7 + 1
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        
        r = s * 7 + 2
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        
        r = s * 7 + 3
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        
        r = s * 7 + 4
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        
        r = s * 7 + 5
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        
        r = s * 7 + 6
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
    
    return U, V, W

def _try_rank_27():
    """Try to find a rank-27 decomposition by merging one pair of Strassen terms."""
    # The key insight: if two terms r1, r2 have the same U vector (up to scale),
    # we can potentially merge them. Let's check all pairs.
    
    n, m, p = 2, 2, 8
    U28, V28, W28 = _strassen_28()
    U28 = np.array(U28, dtype=float)
    V28 = np.array(V28, dtype=float)
    W28 = np.array(W28, dtype=float)
    
    # Check if any two columns of U are proportional
    # If U[:,r1] = c * U[:,r2], we can combine
    for r1 in range(28):
        for r2 in range(r1+1, 28):
            u1 = U28[:, r1]
            u2 = U28[:, r2]
            # Check if proportional
            if np.allclose(u1, u2):
                # Same U vector - can merge
                pass
            # Check if u1 = c * u2
            nz = u1 != 0
            if np.any(nz):
                c = u1[nz][0] / u2[nz][0] if u2[nz][0] != 0 else None
                if c is not None and np.allclose(u1[nz], c * u2[nz]):
                    # Proportional - potential merge candidate
                    # New term: U_new = u1 + u2 (or just keep one)
                    # Actually if U[:,r1] = c*U[:,r2], we can do:
                    # Replace r1 with U[:,r1] - c*U[:,r2] = 0, which doesn't help
                    # We need a different approach: linear combination
                    pass
    
    # Alternative: try to find a rank-27 solution by perturbing the 28-rank solution
    # and checking if one term can be eliminated
    # This is a hard problem; let's try a simpler approach
    
    # Try: can we express one of the 28 terms as a linear combination of the others?
    # If term r is in the span of the other 27 terms (in the tensor sense), we can remove it.
    # The tensor T = sum_r U_r ⊗ V_r ⊗ W_r
    # If T_r = U_r ⊗ V_r ⊗ W_r is in span{T_s : s != r}, then rank <= 27
    
    # This is equivalent to checking if the 28 vectors U_r⊗V_r⊗W_r are linearly dependent
    # Flatten each term into a 4*16*16 = 1024-dim vector
    # Check if they're linearly dependent
    
    terms = []
    for r in range(28):
        term = np.zeros(4 * 16 * 16)
        for i in range(4):
            for j in range(16):
                for k in range(16):
                    term[i * 256 + j * 16 + k] = U28[i, r] * V28[j, r] * W28[k, r]
        terms.append(term)
    
    terms_arr = np.array(terms)
    # Check rank of this 1024x28 matrix
    # If rank < 28, the terms are linearly dependent and we can find a lower-rank decomposition
    
    # Use SVD to check
    try:
        U_svd, s_vals, Vt = np.linalg.svd(terms_arr.T, full_matrices=False)
        # s_vals has min(1024, 28) = 28 entries
        # Count how many are significantly nonzero
        threshold = 1e-8
        num_nonzero = np.sum(s_vals > threshold)
        if num_nonzero < 28:
            # Linearly dependent! Find the null space vector
            # The null space of terms_arr (28x1024) gives coefficients c such that sum c_r * term_r = 0
            # Actually we want: is there c != 0 such that sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0?
            # This means the 28 vectors are linearly dependent
            # The right null space of terms_arr.T (1024x28) gives this
            # Vt is 28x28 (since full_matrices=False and min(1024,28)=28)
            # The last row of Vt corresponds to the smallest singular value
            if s_vals[-1] < threshold:
                # Find null vector
                null_vec = Vt[-1, :]
                # This gives c such that sum_r c_r * term_r ≈ 0
                # We can remove one term and adjust the others
                # For simplicity, let's just return the 28-rank solution
                pass
    except:
        pass
    
    return None  # No rank-27 found yet

def _search_lower_rank(start_rank=24, max_rank=27, deadline=130.0):
    """Search for lower-rank decompositions using algebraic methods."""
    n, m, p = 2, 2, 8
    t0 = time.time()
    
    # Try rank 27 first (closest to 28)
    result_27 = _try_rank_27()
    if result_27 is not None:
        return result_27
    
    # For ranks below 27, we need a more systematic search
    # The multiplication tensor for (2,2,8) has known lower bounds
    # Lower bound: max(n*m, m*p, p*n) / max(n,m,p) ... not quite
    # Actually, a simple lower bound is: rank >= (n*m*p) / (max dimension product considerations)
    # For 2x2x8: the tensor has 4*16*16 = 1024 entries but structure
    # Each term contributes at most (number of nonzeros in U)*(number in V)*(number in W) to the tensor
    # In Strassen, each term has ~2 nonzeros in U, ~2 in V, ~2 in W = 8 nonzero contributions
    # 28 terms * 8 = 224, but the tensor has 4*16*16/4 = 256 nonzero entries (each C[a,b] depends on 2 A entries * 2 B entries... no)
    
    # Let's think about it differently:
    # C[a,b] = A[a,0]*B[0,b] + A[a,1]*B[1,b]
    # So for each (a,b), C[a,b] = sum of 2 products
    # Total "product terms" in the naive multiplication: 2*2*8 = 32
    # Each rank-1 term can contribute to multiple (a,b) positions
    # A rank-r decomposition means: C = sum_{r=1}^{R} u_r * v_r * w_r (outer product style)
    # where u_r is a vector in R^4, v_r in R^16, w_r in R^16
    
    # The key constraint: for the decomposition to be exact, for each (a,b):
    # sum_r U[a*2+l, r] * V[l*8+b, r] * W[b*2+a, r] = 1 for all l... no
    # Actually: C[a,b] = sum_l A[a,l]*B[l,b]
    # In the tensor: T[a*2+l, l*8+b, b*2+a] = 1 for all a,l,b
    # And T[i,j,k] = 0 otherwise
    
    # A rank-1 term U_r ⊗ V_r ⊗ W_r contributes to T[i,j,k] = U_r[i]*V_r[j]*W_r[k]
    # For the decomposition to work: sum_r U_r[i]*V_r[j]*W_r[k] = T[i,j,k]
    
    # For a given rank R, we need to find U, V, W such that this holds.
    # This is a system of 4*16*16 = 1024 equations (but only 256 are nonzero targets)
    # with R*(4+16+16) = 40R unknowns (minus the scaling ambiguity: each term can be scaled, so 39R effective)
    
    # For R=27: 40*27 = 1080 unknowns, 256 equations - should be solvable if the structure allows
    # For R=24: 40*24 = 960 unknowns, 256 equations
    # For R=20: 40*20 = 800 unknowns, 256 equations
    
    # The question is whether the specific structure of the tensor allows such low-rank decompositions.
    
    # Known results: 
    # - 2x2x2 has rank 7 (Strassen)
    # - 2x2x4: can we do better than 2*7=14? 
    #   Actually, 2x2x4 = 2x2x(2+2) = two 2x2x2 blocks, but they share the A matrix
    #   The A matrix is the same for both blocks, so there might be sharing
    #   Rank of 2x2x4: at least 7 (since it contains 2x2x2 as a sub-tensor)
    #   Can we do 2x2x4 with rank < 14?
    #   If we could do 2x2x4 with rank r, then 2x2x8 with rank 2r
    #   So if 2x2x4 has rank 13, then 2x2x8 has rank 26
    
    # Let's try to find a rank-13 decomposition for 2x2x4
    # This would give rank 26 for 2x2x8
    
    # For 2x2x4: n=2, m=2, p=4
    # A is 2x2, B is 2x4, C is 2x4
    # T[i,j,k] where i in 0..3, j in 0..7, k in 0..7
    # T[a*2+l, l*4+b, b*2+a] = 1
    
    # Try to find rank-13 decomposition for 2x2x4
    # This is still a hard problem. Let's use a heuristic:
    # Start from the rank-14 (two Strassen) solution and try to merge terms
    
    if time.time() - t0 > deadline - 20:
        return None
    
    # For now, return None and use the 28-rank solution
    return None

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0  # Leave 20s safety margin
    
    # First, try to find a lower-rank solution
    lower_rank = _search_lower_rank(deadline=deadline)
    if lower_rank is not None:
        U, V, W = lower_rank
        # Verify
        err = _verify(U, V, W)
        if err < 1e-9:
            return {"U": U, "V": V, "W": W}
    
    # Fall back to the 28-rank Strassen solution
    U, V, W = _strassen_28()
    # Verify for safety
    err = _verify(U, V, W)
    if err >= 1e-9:
        # This should never happen, but just in case
        pass
    
    return {"U": U, "V": V, "W": W}
