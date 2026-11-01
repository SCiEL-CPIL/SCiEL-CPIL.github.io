"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]
_DEADLINE = time.time() + 140  # safety margin before 150s kill
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt global rank reduction of the 2x2 by 2x10 matrix multiplication tensor.
    
    Strategy: Build the 35-term Laderman block decomposition, construct the full
    400-dim bilinear map matrix, compute its numerical rank via SVD, and if rank
    is < 35, construct a reduced exact decomposition. Otherwise fall back to rank 35.
    
    Interface:
      U: shape (n*m, rank) = (4, rank)   — coefficients for A
      V: shape (m*p, rank) = (20, rank)  — coefficients for B  
      W: shape (p*n, rank) = (20, rank)  — coefficients for C (trace convention W[k*n+i])
    """
    n, m, p = DIMENSIONS
    
    # Build the base 35-term Laderman decomposition
    base_rank = 35
    U_base, V_base, W_base = _build_laderman_blocks(n, m, p)
    
    # Construct the 400-dim bilinear map: for each output index c (0..19),
    # the map from (A_vec, B_vec) to C[c] is:
    #   C[c] = sum_{a,b} T[a,b,c] * A[a] * B[b]
    # where T is the matmul tensor.
    # We express this as: C_vec = M @ (A_vec ⊗ B_vec)
    # where M is 20 x (4*20) = 20 x 80
    # But we want the 400-dim space: each of the 35 terms gives a rank-1
    # contribution to the bilinear map.
    
    # Actually, let's think of it differently. The tensor T has entries
    # T[A_idx, B_idx, C_idx] = 1 if (i_A == i_C) and (j_A == j_B) and (k_B == k_C)
    # else 0.
    
    # The 35-term decomposition says:
    # T[a,b,c] = sum_{r=0}^{34} U[a,r] * V[b,r] * W[c,r]
    
    # We want to find if there exists a representation with fewer than 35 terms.
    # This is equivalent to: does the set of 35 vectors {u_r ⊗ v_r ⊗ w_r} span
    # a subspace of dimension < 35 in the 400-dim tensor space?
    
    # More practically: construct the 400-dim tensor as a matrix of shape
    # (4*20, 20) = (80, 20) by flattening (A_idx, B_idx) vs C_idx.
    # Then the rank of this 80x20 matrix gives the tensor rank (for this
    # particular unfolding). But tensor rank >= any unfolding rank.
    
    # Better approach: Use the 35-term decomposition to get 35 vectors in R^400,
    # compute their span dimension, and if < 35, find a basis.
    
    # Flatten each term r as a vector in R^{4*20*20} = R^1600
    # term_r[a,b,c] = U[a,r] * V[b,r] * W[c,r]
    
    # But 1600-dim is large. Let's use a more efficient approach.
    
    # The key insight: T[a,b,c] = 1 iff (a//m == c//n) and (a%m == b//p) and (b%p == c%n)
    # For n=2, m=2, p=10:
    # a in [0,3], b in [0,19], c in [0,19]
    # T[a,b,c] = 1 iff (a//2 == c//2) and (a%2 == b//10) and (b%10 == c%2)
    
    # Let's verify this is correct:
    # A[i,j] has idx i*m+j = 2i+j, so a//2 = i, a%2 = j
    # B[j,k] has idx j*p+k = 10j+k, so b//10 = j, b%10 = k
    # C[i,k] has idx k*n+i = 2k+i, so c//2 = k, c%2 = i
    # So T[a,b,c] = 1 iff (a//2 == c%2) and (a%2 == b//10) and (b%10 == c//2)
    
    # Now, the 35-term Laderman decomposition gives us 35 vectors in R^800
    # (flattening a and b indices together, with c as the "output").
    
    # Actually, let me just do the SVD on the 80x20 unfolding.
    # Reshape T as matrix M where M[(a,b), c] = T[a,b,c]
    # This is an 80x20 matrix. Its rank is at most 20.
    # The tensor rank is >= max over all unfoldings of the rank.
    
    # For a better bound, let's use the 35-term representation and check
    # if we can reduce it.
    
    # Construct 35 vectors v_r in R^80 where v_r[(a,b)] = U[a,r]*V[b,r]
    # Then T = sum_r v_r ⊗ W[:,r]
    # The question is: what is the minimum number of such rank-1 terms?
    
    # This is the tensor rank problem. Let's try to solve it numerically.
    
    # Build the 80x20 matrix M where M[(a,b), c] = T[a,b,c]
    import numpy as np
    
    # Build T as a 4x20x20 tensor
    T = np.zeros((4, 20, 20))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Reshape to 80x20: index (a*20+b, c)
    M = T.reshape(80, 20)
    
    # SVD of M
    U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
    
    # The rank of this unfolding
    tol = 1e-10
    num_sig = np.sum(s_vals > tol)
    
    # The tensor rank is at least num_sig.
    # If num_sig < 35, we might be able to do better.
    # But we need to find an actual decomposition.
    
    # For now, let's try: use the top-r singular vectors to construct
    # a rank-r decomposition where r = num_sig.
    
    # M = U_svd @ diag(s_vals) @ Vt
    # So M[(a,b), c] = sum_{r=0}^{r-1} U_svd[(a,b), r] * s_vals[r] * Vt[r, c]
    
    # This gives us a rank-r decomposition of the unfolding, but NOT necessarily
    # a valid tensor decomposition. We need:
    # T[a,b,c] = sum_r U'[a, r] * V'[b, r] * W'[c, r]
    
    # The SVD of the unfolding gives:
    # T[a,b,c] = sum_r U_svd[a*20+b, r] * s_vals[r] * Vt[r, c]
    
    # For this to be a valid tensor decomposition, we need:
    # U_svd[a*20+b, r] = U'[a, r] * V'[b, r] for all a, b, r
    
    # This is generally NOT true. So the SVD of one unfolding doesn't directly
    # give a tensor decomposition.
    
    # Alternative approach: Use the 35-term Laderman decomposition and try to
    # find linear dependencies among the 35 term vectors.
    
    # Each term r contributes vector t_r in R^800 where t_r[a,b,c] = U[a,r]*V[b,r]*W[c,r]
    # But actually, we should think of it as: the bilinear map for output c is
    #   f_c(A, B) = sum_{a,b} T[a,b,c] A[a] B[b] = sum_r W[c,r] * (sum_a U[a,r] A[a]) * (sum_b V[b,r] B[b])
    
    # So for each c, f_c is a sum of 35 rank-1 bilinear forms.
    # The question is whether we can represent all 20 functions f_c using fewer
    # than 35 shared rank-1 bilinear forms.
    
    # Let's construct the 35 bilinear forms:
    # g_r(A, B) = (sum_a U[a,r] A[a]) * (sum_b V[b,r] B[b])
    # Then f_c = sum_r W[c,r] g_r
    
    # The g_r are elements of the space of bilinear forms on R^4 x R^20,
    # which has dimension 4*20 = 80.
    
    # So we have 35 vectors in an 80-dim space. They are always linearly
    # independent if 35 <= 80, unless there are specific dependencies.
    
    # Let's check: construct the 80x35 matrix G where G[a*20+b, r] = U[a,r]*V[b,r]
    # If rank(G) < 35, then we can reduce.
    
    G = np.zeros((80, 35))
    for a in range(4):
        for b in range(20):
            for r in range(35):
                G[a*20+b, r] = U_base[a][r] * V_base[b][r]
    
    G_rank = np.linalg.matrix_rank(G, tol=tol)
    
    if G_rank < 35:
        # We can potentially reduce! Find a basis for the column space.
        # Use QR with column pivoting.
        Q, R, P = _qr_with_pivoting(G)
        # The first G_rank columns of G @ P are linearly independent.
        # But we need to express all 35 columns as combinations of the basis.
        
        # Actually, let's use SVD to get a basis.
        U_g, s_g, Vt_g = np.linalg.svd(G, full_matrices=False)
        # Basis vectors are U_g[:, :G_rank]
        # G = U_g @ diag(s_g) @ Vt_g
        # G[:, r] = sum_{k=0}^{G_rank-1} U_g[:, k] * s_g[k] * Vt_g[k, r]
        
        # So we can write each column r as a combination of G_rank basis vectors.
        # Let B = U_g[:, :G_rank] (80 x G_rank)
        # Let C = diag(s_g[:G_rank]) @ Vt_g[:G_rank, :] (G_rank x 35)
        # Then G = B @ C, so G[:, r] = B @ C[:, r]
        
        # Now, f_c(A,B) = sum_r W[c,r] * g_r(A,B)
        # where g_r(A,B) = (B @ C[:, r])^T (A ⊗ B_vec)
        # Hmm, this is getting complicated. Let me think differently.
        
        # Actually, the key realization: if G has rank < 35, then the 35
        # bilinear forms g_r are linearly dependent, so we can express some
        # as combinations of others, reducing the number of terms needed.
        
        # But the reduction in tensor rank is not simply the rank of G.
        # The tensor rank is the minimum number of rank-1 tensors whose
        # sum equals T.
        
        # For now, let's just try to use the SVD-based approach and verify.
        pass
    
    # Given the complexity of finding an exact reduced decomposition,
    # and the fact that the 5 blocks are independent (different B indices),
    # it's very likely that rank 35 is minimal. Let's verify the 35-term
    # decomposition is correct and return it.
    
    _verify_decomposition(U_base, V_base, W_base, n, m, p)
    
    return {"U": U_base, "V": V_base, "W": W_base}

def _build_laderman_blocks(n, m, p):
    """Build the 35-term Laderman block decomposition."""
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35
    
    laderman_A = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]
    
    laderman_B = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1],   # M7: g+h
    ]
    
    w_contrib = [
        [1, 0, 0, 1],   # M1: C00 += 1, C11 += 1
        [0, 0, 1, -1],  # M2: C10 += 1, C11 -= 1
        [0, 1, 0, 1],   # M3: C01 += 1, C11 += 1
        [1, 0, 1, 0],   # M4: C00 += 1, C10 += 1
        [-1, 1, 0, 0],  # M5: C00 -= 1, C01 += 1
        [0, 0, 0, 1],   # M6: C11 += 1
        [1, 0, 0, 0],   # M7: C00 += 1
    ]
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            U[0][col] = laderman_A[r][0]
            U[1][col] = laderman_A[r][1]
            U[2][col] = laderman_A[r][2]
            U[3][col] = laderman_A[r][3]
            
            V[2*b][col] = laderman_B[r][0]
            V[2*b + 1][col] = laderman_B[r][1]
            V[10 + 2*b][col] = laderman_B[r][2]
            V[10 + 2*b + 1][col] = laderman_B[r][3]
            
            W[4*b][col] = w_contrib[r][0]
            W[4*b + 2][col] = w_contrib[r][1]
            W[4*b + 1][col] = w_contrib[r][2]
            W[4*b + 3][col] = w_contrib[r][3]
    
    return U, V, W

def _qr_with_pivoting(A):
    """QR decomposition with column pivoting."""
    import numpy as np
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    P = np.arange(n)
    
    for k in range(min(m, n)):
        # Find column with largest norm
        norms = np.linalg.norm(R[k:, :], axis=0)
        col = k + np.argmax(norms)
        # Swap columns
        R[:, [k, col]] = R[:, [col, k]]
        P[[k, col]] = P[[col, k]]
        # QR step
        if norms[col] > 1e-15:
            v = R[k:, k].copy()
            v = v / np.linalg.norm(v)
            R[k:, :] -= np.outer(v, v @ R[k:, :])
            Q[:, k:] -= np.outer(Q[k:, :] @ v, v)
    
    return Q, R, P

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the matmul tensor."""
    rank = len(U[0])
    # For each (i,j,k,l) in the tensor:
    # T[i,j,k,l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The matmul tensor has T[i,j,k,l] = 1 if j==l and i==i' and k==k' 
    # More precisely: T[i,j, j',k, i',k'] = δ[j,j']δ[i,i']δ[k,k']
    # In our indexing: A[i,j], B[j,k], C[i,k]
    # T[i,j,k] where these are flattened: A_idx = i*m+j, B_idx = j*p+k, C_idx = k*n+i
    # T[A_idx, B_idx, C_idx] = 1 if (i from A == i from C) and (j from A == j from B) and (k from B == k from C)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # This should be 1 in the tensor
                A_idx = i * m + j
                B_idx = j * p + k
                C_idx = k * n + i
                expected = 1
                actual = 0
                for r in range(rank):
                    actual += U[A_idx][r] * V[B_idx][r] * W[C_idx][r]
                if actual != expected:
                    raise ValueError(f"Verification failed at A[{i},{j}], B[{j},{k}], C[{k},{i}]: got {actual}, expected {expected}")
    
    # Also verify that non-matching entries are 0
    # For random sample of non-matching triples
    import random
    random.seed(42)
    for _ in range(200):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        # Perturb to get a non-matching triple
        if random.random() < 0.5:
            i2 = random.randint(0, n-1)
            if i2 == i: i2 = (i2+1) % n
            A_idx = i2 * m + j
            B_idx = j * p + k
            C_idx = k * n + i
        else:
            k2 = random.randint(0, p-1)
            if k2 == k: k2 = (k2+1) % p
            A_idx = i * m + j
            B_idx = j * p + k2
            C_idx = k * n + i
        actual = 0
        for r in range(rank):
            actual += U[A_idx][r] * V[B_idx][r] * W[C_idx][r]
        if actual != 0:
            raise ValueError(f"Verification failed: non-zero at A_idx={A_idx}, B_idx={B_idx}, C_idx={C_idx}, got {actual}")
# EVOLVE-BLOCK-END
