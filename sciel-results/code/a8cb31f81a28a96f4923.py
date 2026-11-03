import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 3, 4, 6
    
    def build_matrix(U, V, W, rank):
        # M shape: (rank, n*m * m*p * p*n)
        # M[c, (i, k, l)] = U[i,c] * V[k,c] * W[l,c]
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        # Use einsum to compute the tensor for each rank term
        # 'ic,kc,lc->cilk'
        T = np.einsum('ic,kc,lc->cilk', Ua, Va, Wa)
        return T.reshape(rank, -1)

    def prune_decomposition(U, V, W, max_iter=50):
        rank = len(U[0])
        if rank == 0:
            return U, V, W, 0
        
        # Build the constraint matrix M
        # M[c, :] = vec(U[:,c] * V[:,c] * W[:,c])
        # We want to find the minimum number of columns in M that span the same space.
        # This is equivalent to finding the rank of M.
        # If rank(M) < number of rows, we can remove rows.
        
        current_U, current_V, current_W = U, V, W
        current_rank = rank
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if current_rank <= 0:
                break
                
            M = build_matrix(current_U, current_V, current_W, current_rank)
            
            # Compute SVD of M to find numerical rank
            try:
                s = np.linalg.svd(M, compute_uv=False)
            except np.linalg.LinAlgError:
                break
                
            if len(s) == 0:
                break
                
            # Determine numerical rank
            # Use a relative tolerance
            tol = 1e-9 * max(s[0], 1.0)
            num_rank = int(np.sum(s > tol))
            
            if num_rank >= current_rank:
                break
                
            # We know there are current_rank - num_rank dependent rows.
            # We need to identify which rows to remove.
            # A simple heuristic: remove the row with the smallest singular value contribution?
            # Actually, SVD doesn't directly give row removal.
            # We can use QR with column pivoting on M.T to find a basis.
            # Or, we can iterate: try removing each row and check if the rank drops.
            
            removed_any = False
            # Try removing each row one by one
            # To speed up, we can check rows with small norm first?
            # Let's just iterate through all rows. If rank(current_rank) is small, this is fast.
            # If rank is large (e.g., 70), 70 SVDs might be slow.
            # Optimization: Only try to remove rows if current_rank is small enough or time allows.
            
            # Better approach: Use the fact that if rank(M) < current_rank, 
            # there is a linear dependency. 
            # We can find a null vector of M.T? No, M is (rank, dim). 
            # We want to find a subset of rows that are linearly independent.
            
            # Let's use a greedy approach:
            # Start with an empty set of kept rows.
            # Add rows one by one if they increase the rank.
            # This is O(r^2 * dim) which might be slow for r=70, dim=5000.
            # 70^2 * 5000 = 24.5M ops. SVD is O(r * dim^2) or similar.
            # SVD of 70x5184 is fast.
            
            # Let's just try removing one row at a time and re-checking rank.
            # We'll try to remove the row that is "most redundant".
            # Heuristic: The row with the smallest norm?
            # Or just iterate.
            
            candidates = list(range(current_rank))
            random.shuffle(candidates) # Randomize to escape local minima
            
            for rem_idx in candidates:
                if time.time() > deadline - 5.0:
                    break
                # Try removing rem_idx
                U_new = [row for i, row in enumerate(current_U) if i != rem_idx]
                V_new = [row for i, row in enumerate(current_V) if i != rem_idx]
                W_new = [row for i, row in enumerate(current_W) if i != rem_idx]
                
                new_rank = current_rank - 1
                if new_rank == 0:
                    # Check if the resulting zero matrix is valid (it is, if tensor is 0, but it's not)
                    # Actually if we remove all, we have rank 0.
                    pass
                
                # Check if the new set still spans the same tensor
                # We need to verify that the tensor decomposed by U_new, V_new, W_new is the same.
                # This is equivalent to checking if the removed row was in the span of the others.
                # We can check this by seeing if the rank of [M; M_new] is the same as rank(M).
                # But we already know rank(M) = num_rank.
                # If removing row rem_idx keeps the rank at num_rank, then it was redundant.
                
                M_new = build_matrix(U_new, V_new, W_new, new_rank) if new_rank > 0 else np.zeros((0, M.shape[1]))
                
                # Check if rank(M_new) == num_rank
                if new_rank == 0:
                    if num_rank == 0:
                        current_U, current_V, current_W = U_new, V_new, W_new
                        current_rank = 0
                        removed_any = True
                        break
                    else:
                        continue
                
                try:
                    s_new = np.linalg.svd(M_new, compute_uv=False)
                    tol_new = 1e-9 * max(s_new[0], 1.0) if len(s_new) > 0 else 0
                    num_rank_new = int(np.sum(s_new > tol_new))
                except:
                    continue
                    
                if num_rank_new == num_rank:
                    # Row rem_idx was redundant
                    current_U, current_V, current_W = U_new, V_new, W_new
                    current_rank = new_rank
                    removed_any = True
                    break # Restart the loop to recalculate M and dependencies
                    
            if not removed_any:
                break
                
        return current_U, current_V, current_W, current_rank

    def verify(U, V, W):
        # Exact verification using integers/floats
        # T[i,k,l] = sum_c U[i,c] V[k,c] W[l,c]
        # Target T[i,k,l] = 1 if i==k and k==l (wait, no)
        # Matrix mult: C = A * B
        # A is n x m, B is m x p. C is n x p.
        # C[i, l] = sum_k A[i, k] B[k, l]
        # Tensor T[i, k, l] is 1 if this term contributes to C[i, l].
        # So T[i, k, l] = 1 for all i, k, l? No.
        # The tensor of matrix multiplication is defined by T_{i,k,l} = 1 if we are computing C_{i,l} using A_{i,k} and B_{k,l}.
        # Actually, the tensor T has indices (i, k, l) where i in [0,n), k in [0,m), l in [0,p).
        # T[i, k, l] = 1.
        # Wait, the decomposition is U[i, c] V[k, c] W[l, c].
        # The sum over c gives C[i, l] = sum_k A[i, k] B[k, l].
        # So the tensor T[i, k, l] must be 1 for all i, k, l?
        # No, the "matrix multiplication tensor" is the set of coefficients of the bilinear map.
        # The bilinear map is (A, B) -> A B.
        # The tensor T has entries T[i, k, l] = 1 if the term A[i, k] B[k, l] appears in the sum for C[i, l].
        # Yes, T[i, k, l] = 1 for all valid i, k, l.
        
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Compute reconstructed tensor
        # R[i, k, l] = sum_c U[i, c] V[k, c] W[l, c]
        R = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
        
        # Target tensor
        # T[i, k, l] = 1
        # Note: The problem statement says "output factor follows the official trace convention W[k*n+i,r]".
        # This refers to the indexing of W.
        # W is of shape (p*n, rank).
        # The index for W is k*n + i?
        # Let's check the seed program:
        # W[C00][col] += 1 where C00 = c_c * n + c_r.
        # c_c is the column index of C (which is l in my notation? No, C is n x p).
        # In the seed:
        # A is n x m. Indices: a_r (row), a_c (col).
        # B is m x p. Indices: b_r (row), b_c (col).
        # C is n x p. Indices: c_r (row), c_c (col).
        # C[c_r, c_c] = sum_b A[a_r, b] B[b, c_c] where a_r = c_r.
        # So the tensor term is A[c_r, b] B[b, c_c].
        # The indices are (c_r, b, c_c).
        # U index: c_r * m + b.
        # V index: b * p + c_c.
        # W index: c_c * n + c_r.
        # So T[c_r, b, c_c] = 1.
        
        # My R is computed as R[i, k, l] where i is U index part, k is V index part, l is W index part.
        # U[i, c] corresponds to A index i = c_r * m + b.
        # V[k, c] corresponds to B index k = b * p + c_c.
        # W[l, c] corresponds to C index l = c_c * n + c_r.
        
        # So we need to check if R[i, k, l] == 1 for all i, k, l?
        # No, the tensor T is sparse? No, T[i, k, l] is 1 for all combinations?
        # Let's check:
        # C[c_r, c_c] = sum_{b=0}^{m-1} A[c_r, b] B[b, c_c].
        # The bilinear form is sum_{c_r, c_c, b} A[c_r, b] B[b, c_c] C[c_r, c_c] (in the sense of contraction).
        # The tensor T is defined such that the multiplication is sum_{c_r, c_c, b} T[c_r, b, c_c] A[c_r, b] B[b, c_c].
        # So T[c_r, b, c_c] = 1 for all c_r, b, c_c.
        
        # So the target tensor is all ones?
        # Let's verify with the seed code.
        # Seed adds to U, V, W such that the product reconstructs the tensor.
        # If T is all ones, then rank is n*m*p? No, rank of all-ones tensor is 1?
        # No, the tensor is not "all ones" in the standard basis of the vector space of matrices.
        # The tensor T is an element of A* \otimes B* \otimes C*.
        # T = sum_{c_r, b, c_c} e_{c_r, b} \otimes e_{b, c_c} \otimes e_{c_r, c_c}.
        # So yes, T[i, k, l] = 1 if the indices are consistent?
        # Wait, the indices i, k, l in my R array correspond to:
        # i = c_r * m + b
        # k = b * p + c_c
        # l = c_c * n + c_r
        # Are these independent?
        # i determines c_r and b.
        # k determines b and c_c.
        # l determines c_c and c_r.
        # For a fixed i and k, b is fixed. Then c_r is fixed by i, and c_c is fixed by k.
        # Then l is determined by c_c and c_r.
        # So T is not "all ones" in the flattened array of size (n*m * m*p * p*n).
        # Only the entries where the indices are consistent (i.e., the b in i matches the b in k, and the c_r in i matches the c_r in l, etc.) are 1.
        # Actually, for ANY i, k, l, there is a unique c_r, b, c_c.
        # So T[i, k, l] is 1 if the indices i, k, l are consistent with a single (c_r, b, c_c).
        # But i, k, l are just indices.
        # Let's check consistency:
        # i = c_r * m + b  => c_r = i // m, b = i % m
        # k = b * p + c_c  => b' = k // p, c_c = k % p
        # l = c_c * n + c_r => c_r' = l % n, c_c' = l // n
        # The term is non-zero only if b == b' and c_r == c_r' and c_c == c_c'.
        # So T[i, k, l] = 1 if i%m == k//p and i//m == l%n and k%p == l//n.
        # Otherwise 0.
        
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    target[i, k, l] = 1.0
                    
        # Check if R matches target
        # R is (n*m, m*p, p*n)
        # Note: R[i, k, l] = sum_c U[i,c] V[k,c] W[l,c]
        # We need R == target
        
        if not np.allclose(R, target, atol=1e-9):
            return False
        return True

    def get_schoolbook():
        # Rank n*m*p = 72
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        
        col = 0
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    U[i][col] = 1
                    V[k][col] = 1
                    W[l][col] = 1
                    col += 1
        return U, V, W

    def get_strassen_66():
        # Reconstruct the 66 rank solution from the seed
        rank = 66
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1

            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1

            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1

            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1

            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1

        add_1x2_2x2(0, 0, 0, 0, 0, 0)
        add_1x2_2x2(0, 2, 2, 0, 0, 0)
        add_1x2_2x2(0, 0, 0, 2, 0, 2)
        add_1x2_2x2(0, 2, 2, 2, 0, 2)
        add_1x2_2x2(0, 0, 0, 4, 0, 4)
        add_1x2_2x2(0, 2, 2, 4, 0, 4)

        add_strassen(1, 0, 0, 0, 1, 0)
        add_strassen(1, 2, 2, 0, 1, 0)
        add_strassen(1, 0, 0, 2, 1, 2)
        add_strassen(1, 2, 2, 2, 1, 2)
        add_strassen(1, 0, 0, 4, 1, 4)
        add_strassen(1, 2, 2, 4, 1, 4)
        
        return U, V, W

    best_U, best_V, best_W, best_rank = None, None, None, float('inf')
    
    # Generate candidates
    candidates = []
    
    # 1. The 66-rank solution
    U66, V66, W66 = get_strassen_66()
    candidates.append((U66, V66, W66))
    
    # 2. The 72-rank schoolbook solution
    U72, V72, W72 = get_schoolbook()
    candidates.append((U72, V72, W72))
    
    # 3. Random perturbations of 66? 
    # Maybe mix 66 and 72?
    # Let's try pruning 72.
    
    # Prune each candidate
    for U, V, W in candidates:
        if time.time() > deadline - 10.0:
            break
            
        U_p, V_p, W_p, r_p = prune_decomposition(U, V, W)
        
        # Verify the pruned decomposition
        if verify(U_p, V_p, W_p):
            if r_p < best_rank:
                best_rank = r_p
                best_U, best_V, best_W = U_p, V_p, W_p
        else:
            # If pruning broke it (due to numerical issues), fall back to original if valid
            if verify(U, V, W):
                r_orig = len(U[0])
                if r_orig < best_rank:
                    best_rank = r_orig
                    best_U, best_V, best_W = U, V, W
            # Also try to prune again with different random seeds?
            # For now, just accept the failure.

    # If we didn't find anything better than 66, return 66
    if best_U is None:
        best_U, best_V, best_W = get_strassen_66()
        best_rank = 66

    return {"U": best_U, "V": best_V, "W": best_W}
