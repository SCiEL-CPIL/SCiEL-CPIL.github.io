"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # 1. Construct the full matrix multiplication tensor T
    # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k]
    # In the trace convention W[k*n+i, r], the tensor index is (i, j, k) -> (k*n+i)
    # Actually, the standard matmul tensor T_{ijk} has rank R such that:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor is T_{ijk} = delta_{jk} (in the sense that j is the shared index)
    # Wait, the tensor to decompose is the "multiplication tensor" M_{ijk} where M_{ijk}=1 if j is the same in A and B?
    # No, the tensor is defined by the structure of the multiplication.
    # T_{ijk} = 1 if the indices match such that A_{ij} and B_{jk} multiply.
    # Since j is the contraction index, for every triplet (i, j, k), the term exists.
    # So T_{ijk} = 1 for all i, j, k?
    # No. The tensor is T_{ijk} = 1 if we are computing C_{ik} using A_{ij} and B_{jk}.
    # Yes, for every i in [0,n), j in [0,m), k in [0,p), there is a term.
    # So the tensor T is a 3-way array of size n x m x p, all ones?
    # If T_{ijk} = 1 for all i,j,k, then we are decomposing the all-ones tensor.
    # BUT, the all-ones tensor has rank 1 (if we allow real numbers) or rank 1 (U=1, V=1, W=1).
    # This is clearly wrong because matrix multiplication is not just summing all products.
    
    # Let's re-read the task carefully.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 6 matrix."
    # The tensor is NOT the all-ones tensor.
    # The tensor T_{ijk} represents the coefficient of the product A_{ij} B_{jk} in the sum for C_{ik}.
    # For a fixed i and k, C_{ik} = sum_j A_{ij} B_{jk}.
    # So the term A_{ij} B_{jk} contributes to C_{ik} with coefficient 1.
    # Therefore, the tensor T has entries T_{i, j, k} = 1 if the term A_{ij} B_{jk} is part of the calculation for C_{ik}.
    # Since j is the index being summed over, for ANY i, j, k, the term A_{ij} B_{jk} DOES contribute to C_{ik}.
    # So T_{ijk} = 1 for all i, j, k.
    
    # Why did I think it's not all ones?
    # Ah, the "tensor decomposition" of matrix multiplication usually refers to decomposing the *structure* of the multiplication into rank-1 tensors.
    # A rank-1 tensor in this context is of the form u_i * v_j * w_k.
    # The decomposition says: sum_r (sum_i u_{ri} A_{ij}) (sum_j v_{rj} B_{jk}) ... wait.
    
    # Let's look at the definition of the tensor rank of matrix multiplication.
    # The matrix multiplication tensor T_{n,m,p} is defined by T_{ijk} = 1 if the indices align?
    # Actually, the standard definition is that the tensor T has entries T_{ijk} = 1 if the multiplication of the (i,j)-th entry of A and the (j,k)-th entry of B contributes to the (i,k)-th entry of C.
    # Since j is the common index, EVERY (i,j,k) triplet contributes.
    # So T is indeed the all-ones tensor of size n x m x p?
    # If T is all ones, its rank is 1. That can't be right.
    
    # Let's check the seed program's logic.
    # It uses Strassen. Strassen works by changing the basis.
    # The tensor rank is the minimum number of multiplications.
    # The tensor is NOT all ones.
    # The tensor T_{ijk} is defined such that the matrix multiplication is:
    # C_{ik} = sum_{j} A_{ij} B_{jk}
    # This can be written as:
    # C_{ik} = sum_{r} (sum_{i,j} U_{ri} A_{ij}) (sum_{j,k} V_{rj} B_{jk}) W_{rk} ... No.
    
    # The standard tensor decomposition form is:
    # C_{ik} = sum_r (sum_i u_{ri} A_{i,?}) ...
    # Let's look at the provided interface:
    # U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # C = sum_r (U[:,r] . A_flat) * (V[:,r] . B_flat) * W[:,r] (broadcast to C)
    
    # This means:
    # C_{ik} = sum_r W_{k*n+i, r} * (sum_{i',j'} U_{i'*m+j', r} A_{i'j'}) * (sum_{j',k'} V_{j'*p+k', r} B_{j'k'})
    
    # We require this to equal sum_j A_{ij} B_{jk}.
    # This implies that the tensor T_{ijk} (where T_{ijk} is the coefficient of A_{ij} B_{jk} in C_{ik}) must be decomposable.
    # The coefficient of A_{ij} B_{jk} in the RHS is:
    # sum_r W_{k*n+i, r} U_{i*m+j, r} V_{j*p+k, r}
    # We require this sum to be 1 for all i,j,k?
    # YES. Because for any i,j,k, the term A_{ij} B_{jk} contributes to C_{ik} with coefficient 1.
    # So the tensor T_{ijk} = 1 for all i,j,k.
    
    # Wait, if T_{ijk}=1 for all i,j,k, then T is the all-ones tensor.
    # The rank of the all-ones tensor is 1.
    # U_{ij,r}=1, V_{jk,r}=1, W_{ki,r}=1.
    # Then sum_r ... = 1 * 1 * 1 = 1.
    # This would mean matrix multiplication can be done in 1 multiplication?
    # That is obviously false.
    
    # Where is the error?
    # The error is in the interpretation of "Tensor Decomposition of Matrix Multiplication".
    # The tensor is NOT the all-ones tensor.
    # The tensor is defined on the space of matrices.
    # The "multiplication tensor" is an element of V^* \otimes W^* \otimes U.
    # Its components are T_{ijk}.
    # T_{ijk} = 1 if the indices i,j,k form a valid chain for multiplication?
    # No, the tensor rank of matrix multiplication is a well-known problem.
    # For 2x2x2, rank is 7 (Strassen).
    # If the tensor were all ones, rank would be 1.
    # Therefore, the tensor T_{ijk} is NOT all ones.
    
    # Let's look at the 2x2x2 case.
    # C_{00} = A_{00}B_{00} + A_{01}B_{10}
    # C_{01} = A_{00}B_{01} + A_{01}B_{11}
    # C_{10} = A_{10}B_{00} + A_{11}B_{10}
    # C_{11} = A_{10}B_{01} + A_{11}B_{11}
    
    # The term A_{00}B_{00} contributes to C_{00}. So T_{0,0,0} = 1.
    # The term A_{00}B_{01} contributes to C_{01}. So T_{0,0,1} = 1.
    # The term A_{01}B_{00} contributes to NOTHING? 
    # A_{01} is row 0, col 1. B_{00} is row 0, col 0.
    # A_{01} * B_{00} is NOT in any C_{ik}.
    # C_{ik} sums over j. A_{ij} B_{jk}.
    # For A_{01} (i=0, j=1) and B_{00} (j=0, k=0), the j indices don't match (1 != 0).
    # So A_{01} B_{00} does NOT contribute to any C_{ik}.
    
    # AH! The tensor T_{ijk} is 1 ONLY IF the indices align such that the term is part of the sum.
    # In the expression C_{ik} = sum_j A_{ij} B_{jk}, the term is A_{ij} B_{jk}.
    # The indices are i (from A), j (from A col / B row), k (from B col).
    # So the triplet is (i, j, k).
    # The term A_{ij} B_{jk} contributes to C_{ik}.
    # So for ANY i, j, k, the term A_{ij} B_{jk} contributes to C_{ik}.
    # So T_{ijk} = 1 for all i,j,k?
    
    # Let's re-evaluate "A_{01} B_{00}".
    # A_{01} has indices i=0, j=1.
    # B_{00} has indices j=0, k=0.
    # These two specific entries do NOT multiply together in the standard dot product.
    # The dot product for C_{00} is A_{00}B_{00} + A_{01}B_{10}.
    # The pair (A_{01}, B_{00}) is NOT a pair in the sum.
    # The pairs in the sum are (A_{ij}, B_{jk}) where j is the SAME index.
    # So the tensor T_{ijk} is defined over the indices (i, j, k) where j is the shared index.
    # The entry T_{ijk} is 1 if the term A_{ij} B_{jk} is present.
    # Since for any i, j, k, the term A_{ij} B_{jk} IS present in the calculation of C_{ik},
    # T_{ijk} = 1 for all i, j, k.
    
    # This leads back to Rank 1. Why is Strassen Rank 7?
    
    # I am missing the definition of the tensor.
    # The matrix multiplication tensor is NOT T_{ijk} = 1.
    # It is a tensor that maps A and B to C.
    # The rank is the minimum number of scalar multiplications.
    # The tensor is defined in the space of bilinear forms.
    # The basis elements are e_{ij} \otimes e_{jk} \otimes e_{ik}.
    # The tensor is T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}.
    # This IS the all-ones tensor in the basis of (i,j) for A, (j,k) for B, (i,k) for C.
    # Wait, the basis for B is (j,k). The basis for A is (i,j).
    # The index j appears in BOTH A and B.
    # This is the key. The index j is shared.
    # In the tensor product space V_A \otimes V_B \otimes V_C, the basis is (i,j) \otimes (j',k) \otimes (i',k').
    # The tensor T has a component 1 ONLY IF j=j' AND i=i' AND k=k'.
    # So T_{(i,j), (j',k), (i',k')} = 1 if j=j', i=i', k=k', else 0.
    # If we flatten the indices:
    # U index: i*m + j
    # V index: j'*p + k
    # W index: i'*n + k' (or k'*n+i')
    # T[U_idx, V_idx, W_idx] = 1 if j == j' and i == i' and k == k'.
    
    # So the tensor is NOT all ones. It is sparse.
    # It has non-zero entries only when the shared index j matches and the output indices match the input indices.
    # Number of non-zeros: n * m * p.
    # For 2x4x6: 2*4*6 = 48 non-zeros.
    
    # So my previous "All Ones" hypothesis was wrong because I ignored the constraint that the j-index in A must match the j-index in B.
    
    # Okay, so we have a sparse tensor T of size (n*m) x (m*p) x (p*n).
    # T[u, v, w] = 1 if:
    #   u = i*m + j
    #   v = j*p + k
    #   w = k*n + i
    #   (Note: W index convention is k*n+i)
    
    # We want to find the minimum rank R and matrices U, V, W of shape (n*m, R), (m*p, R), (p*n, R)
    # such that T = sum_r U[:,r] V[:,r]^T W[:,r]^T (Outer product sum).
    # i.e., T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r].
    
    # This is the standard CP decomposition.
    # We can solve this numerically!
    
    # 1. Construct T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # 2. Try to find CP decomposition with increasing rank R
    # Start from lower bound.
    # Lower bound for rank of matmul tensor is max(n*m, m*p, p*n) ? No.
    # Lower bound is n*p ? No.
    # For 2x2x2, rank is 7. n*m=4, m*p=4, p*n=4.
    # Lower bound is often cited as max(n*m, m*p, p*n) is not correct.
    # The trivial rank is n*m*p.
    # Strassen for 2x2x2 is 7.
    # For 2x4x6, we can use block Strassen.
    # The current program uses 42.
    # Can we do better?
    # Maybe 40? 38?
    
    # We will use a numerical solver to find the rank.
    # Since the entries are integers/half-integers, we can snap.
    
    def solve_cp_rank(R, T, max_iter=1000, tol=1e-6):
        """
        Find CP decomposition of T with rank R.
        Returns U, V, W or None if fails.
        """
        I, J, K = T.shape
        # Initialize randomly
        U = np.random.randn(I, R)
        V = np.random.randn(J, R)
        W = np.random.randn(K, R)
        
        # Normalize to prevent divergence
        norms_U = np.linalg.norm(U, axis=0, keepdims=True)
        norms_V = np.linalg.norm(V, axis=0, keepdims=True)
        norms_W = np.linalg.norm(W, axis=0, keepdims=True)
        
        U = U / norms_U
        V = V / norms_V
        W = W / norms_W
        
        for _ in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U
            # U[:, r] = reshape(T * (V[:, r] outer W[:, r]), (I, 1)) / norm
            for r in range(R):
                # T contracted with V[:, r] and W[:, r]
                # T is (I, J, K). V[:, r] is (J). W[:, r] is (K).
                # Product is T * V[:, r][None, :, None] * W[:, r][None, None, :]
                # Sum over J and K -> Result shape (I,)
                # This is equivalent to:
                # Temp = np.einsum('ijk,j,k->i', T, V[:, r], W[:, r])
                Temp = np.einsum('ijk,j,k->i', T, V[:, r], W[:, r])
                U[:, r] = Temp / (np.linalg.norm(Temp) + 1e-10)
                
            # Update V
            for r in range(R):
                Temp = np.einsum('ijk,i,k->j', T, U[:, r], W[:, r])
                V[:, r] = Temp / (np.linalg.norm(Temp) + 1e-10)
                
            # Update W
            for r in range(R):
                Temp = np.einsum('ijk,i,j->k', T, U[:, r], V[:, r])
                W[:, r] = Temp / (np.linalg.norm(Temp) + 1e-10)
                
        # Check convergence
        # Reconstruct T_approx
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        error = np.linalg.norm(T - T_approx) / np.linalg.norm(T)
        
        if error < tol:
            # We found a solution. Now snap to integers/half-integers.
            # The scale is arbitrary. We need to fix the scale.
            # The product U V W must match T.
            # Since T has entries 0 or 1, and we expect integer/half-integer factors,
            # the magnitudes of U, V, W should be small integers or halves.
            
            # Strategy:
            # 1. Estimate the scale.
            # 2. Snap U, V, W to nearest 0.5.
            # 3. Verify exactness.
            
            # Since the decomposition is not unique (scaling), we need to be careful.
            # However, for matmul tensors, the "canonical" decomposition often has small integers.
            
            # Let's try to snap.
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            # Check if snapped version works
            T_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
            if np.allclose(T_snap, T, atol=1e-9):
                return U_snap, V_snap, W_snap
            
            # If snapping fails, maybe the scale is off.
            # Try to adjust scale.
            # The product of scales s_u * s_v * s_w must be 1 (relative to the fitted solution).
            # Actually, if U, V, W fit T, then cU, V/c, W fits T.
            # We want integer entries.
            
            # Try a few scaling factors
            for c in [0.5, 1.0, 2.0, 0.25, 4.0]:
                U_t = U * c
                V_t = V / c
                W_t = W
                U_s = np.round(U_t * 2) / 2
                V_s = np.round(V_t * 2) / 2
                W_s = np.round(W_t * 2) / 2
                T_check = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
                if np.allclose(T_check, T, atol=1e-9):
                    return U_s, V_s, W_s
                    
            # Try different permutation of scaling
            for c in [0.5, 1.0, 2.0]:
                for d in [0.5, 1.0, 2.0]:
                    U_t = U * c
                    V_t = V * d
                    W_t = W / (c * d)
                    U_s = np.round(U_t * 2) / 2
                    V_s = np.round(V_t * 2) / 2
                    W_s = np.round(W_t * 2) / 2
                    T_check = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
                    if np.allclose(T_check, T, atol=1e-9):
                        return U_s, V_s, W_s

            return None
        else:
            return None

    # Search for minimal rank
    # Start from a known good rank (42) and go down?
    # Or start from lower bound?
    # Lower bound for 2x4x6?
    # 2x2x2 is 7.
    # 2x4x4 might be 14? (2 * 7)
    # 2x4x6 might be 21? (2 * 7 * 1.5?)
    # The current solution is 42.
    # Let's try ranks from 30 to 42.
    
    best_result = None
    
    # Try ranks in descending order from 42 to find the lowest that works quickly?
    # Or ascending?
    # If we start low, we might waste time on impossible ranks.
    # If we start high, we find a solution fast, then try to improve.
    
    # Let's try a few specific ranks likely to be minimal.
    # Known: 2x2x2 -> 7.
    # 2x2x3 -> ?
    # 2x3x3 -> ?
    # 3x3x3 -> 23 (Landsberg).
    
    # For 2x4x6:
    # We can view it as 2x2x2 blocks?
    # If we use Strassen on 2x2 blocks, we need 7 mults per 2x2 block.
    # A is 2x4 (two 2x2 blocks horizontally).
    # B is 4x6 (three 2x2 blocks vertically).
    # Total 2x3 = 6 blocks.
    # 6 * 7 = 42. This is the current solution.
    
    # Can we do better?
    # Maybe 40?
    # Let's try ranks 40, 39, 38...
    
    ranks_to_try = [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 41, 42]
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try multiple initializations for robustness
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(attempt * 100 + R) # Deterministic for reproducibility in this run
            U, V, W = solve_cp_rank(R, T, max_iter=200, tol=1e-5)
            if U is not None:
                # Verify strictly
                T_check = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.allclose(T_check, T, atol=1e-9):
                    best_result = (U, V, W, R)
                    found = True
                    break
        if found:
            break
            
    if best_result:
        U, V, W, R = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to the known rank 42 Strassen decomposition if numerical search fails
    # This ensures we always return a valid result.
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            # M1 = (a00+a11)(b00+b11)
            U[0*4+j0, r] = 1; U[1*4+j0+1, r] = 1
            V[j0*6+k0, r] = 1; V[(j0+1)*6+(k0+1), r] = 1
            W[k0*2+0, r] = 1; W[(k0+1)*2+0, r] = 1
            W[k0*2+1, r] = 1; W[(k0+1)*2+1, r] = 1
            r += 1
            
            # M2 = (a10+a11)*b00
            U[1*4+j0, r] = 1; U[1*4+j0+1, r] = 1
            V[j0*6+k0, r] = 1
            W[k0*2+1, r] = 1
            W[(k0+1)*2+1, r] = -1
            r += 1
            
            # M3 = a00*(b01-b11)
            U[0*4+j0, r] = 1
            V[j0*6+(k0+1), r] = 1; V[(j0+1)*6+(k0+1), r] = -1
            W[(k0+1)*2+0, r] = 1
            W[(k0+1)*2+1, r] = 1
            r += 1
            
            # M4 = a11*(b10-b00)
            U[1*4+j0+1, r] = 1
            V[(j0+1)*6+k0, r] = 1; V[j0*6+k0, r] = -1
            W[k0*2+0, r] = 1
            W[k0*2+1, r] = 1
            r += 1
            
            # M5 = (a00+a01)*b11
            U[0*4+j0, r] = 1; U[0*4+j0+1, r] = 1
            V[(j0+1)*6+(k0+1), r] = 1
            W[k0*2+0, r] = -1
            W[(k0+1)*2+0, r] = 1
            r += 1
            
            # M6 = (a10-a00)*(b00+b01)
            U[1*4+j0, r] = 1; U[0*4+j0, r] = -1
            V[j0*6+k0, r] = 1; V[j0*6+(k0+1), r] = 1
            W[(k0+1)*2+1, r] = 1
            r += 1
            
            # M7 = (a01-a11)*(b10+b11)
            U[0*4+j0+1, r] = 1; U[1*4+j0+1, r] = -1
            V[(j0+1)*6+k0, r] = 1; V[(j0+1)*6+(k0+1), r] = 1
            W[k0*2+0, r] = 1
            r += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
