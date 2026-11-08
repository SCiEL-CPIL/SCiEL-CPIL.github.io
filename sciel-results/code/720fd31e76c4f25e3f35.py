import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 2, 4, 7
    
    # Build the matrix multiplication tensor
    # T[i,j,k] = 1 means output (i,k) gets contribution from A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should sum to T[i,j,k]
    
    # Schoolbook baseline: rank = n*m*p = 56
    def schoolbook_decomposition():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    U_school, V_school, W_school = schoolbook_decomposition()
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if total != 1:
                        return False
        return True
    
    def strassen_222_block(U, V, W, col_offset, i_off, j_off, k_off):
        """
        Add Strassen decomposition for (2x2)*(2x2) block.
        i_off: row offset in A (0 or 1 for the 2 rows)
        j_off: col offset in A / row offset in B (for the 2 middle indices)
        k_off: col offset in B (for the 2 output cols)
        Returns new col_offset after adding 7 terms.
        """
        # Strassen for [[a,b],[c,d]] * [[e,f],[g,h]]
        # M1 = (a+b)(e+h)
        # M2 = (a+b)(f-h)
        # M3 = (c-d)(e+g)
        # M4 = d(h-g)
        # M5 = (a+c)(h)
        # M6 = (b-d)(e+g)
        # M7 = (a-c)(f+h)
        # C11 = M1+M4-M5-M7
        # C12 = M2+M3+M5-M6
        # C21 = M2-M3+M4+M7
        # C22 = M1-M2+M3+M6
        
        # For our tensor: A[i,j]*B[j,k] where i,j,k are local indices
        # We need U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        
        # M1 = (A[i0,j0]+A[i1,j0]+A[i0,j1]+A[i1,j1]) * (B[j0,k0]+B[j0,k1]+B[j1,k0]+B[j1,k1])
        # Actually, let me be more careful.
        # A has entries A[i,j] for i in {0,1}, j in {j_off, j_off+1}
        # B has entries B[j,k] for j in {j_off, j_off+1}, k in {k_off, k_off+1}
        # C[i,k] = A[i,j0]*B[j0,k] + A[i,j1]*B[j1,k]
        
        # Strassen factors:
        # S1 = A[i0,j0] + A[i1,j0] + A[i0,j1] + A[i1,j1]  -> this is sum of all 4 A entries
        # Actually no. Let me redo.
        
        # For C[i0,k0] = A[i0,j0]*B[j0,k0] + A[i0,j1]*B[j1,k0]
        # For C[i0,k1] = A[i0,j0]*B[j0,k1] + A[i0,j1]*B[j1,k1]
        # For C[i1,k0] = A[i1,j0]*B[j0,k0] + A[i1,j1]*B[j1,k0]
        # For C[i1,k1] = A[i1,j0]*B[j0,k1] + A[i1,j1]*B[j1,k1]
        
        # Strassen:
        # M1 = (A[i0,j0]+A[i0,j1]+A[i1,j0]+A[i1,j1]) * (B[j0,k0]+B[j0,k1]+B[j1,k0]+B[j1,k1])
        # Hmm, that's not quite right either. Let me use the standard formulation.
        
        # Standard Strassen for [[a,b],[c,d]] * [[e,f],[g,h]]:
        # where a=A[i0,j0], b=A[i0,j1], c=A[i1,j0], d=A[i1,j1]
        # e=B[j0,k0], f=B[j0,k1], g=B[j1,k0], h=B[j1,k1]
        
        # M1 = (a+b)(e+h)
        # M2 = (a+b)(f-h)
        # M3 = (c-d)(e+g)
        # M4 = d(h-g)
        # M5 = (a+c)(h)
        # M6 = (b-d)(e+g)
        # M7 = (a-c)(f+h)
        
        # C11 = M1+M4-M5-M7
        # C12 = M2+M3+M5-M6
        # C21 = M2-M3+M4+M7
        # C22 = M1-M2+M3+M6
        
        # Each M is a product of a linear combination of A entries and a linear combination of B entries.
        # For M1: A_part = a+b = A[i0,j0]+A[i0,j1], B_part = e+h = B[j0,k0]+B[j1,k1]
        # U contribution: U[i0*m+j0, c] += 1, U[i0*m+j1, c] += 1
        # V contribution: V[j0*p+k0, c] += 1, V[j1*p+k1, c] += 1
        # W contribution: W[k0*n+i0, c] += 1, W[k1*n+i0, c] += 1, W[k0*n+i1, c] += 1, W[k1*n+i1, c] += 1
        # Wait, that's not right. W should only have one non-zero per term for the output.
        
        # Actually, the way tensor decomposition works:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For a single product term U[idx_u, c] * V[idx_v, c] * W[idx_w, c] = 1:
        # This contributes to T[i,j,k] where i*m+j = idx_u, j*p+k = idx_v, k*n+i = idx_w
        # So i,j come from idx_u, j,k come from idx_v, k,i come from idx_w
        # All three must be consistent: same i, j, k
        
        # So for M1 = (A[i0,j0]+A[i0,j1]) * (B[j0,k0]+B[j1,k1]):
        # This is actually 4 products: A[i0,j0]*B[j0,k0], A[i0,j0]*B[j1,k1], A[i0,j1]*B[j0,k0], A[i0,j1]*B[j1,k1]
        # Each of these contributes to a different (i,j,k) tuple.
        
        # Hmm, I think I'm overcomplicating this. Let me use a different approach.
        
        # The key insight: each "product" in the decomposition is of the form:
        # (linear_combination_of_A) * (linear_combination_of_B)
        # where the linear combinations are over specific indices.
        
        # For the full tensor, T[i,j,k] = 1 for all valid (i,j,k).
        # A decomposition term c with U, V, W vectors contributes:
        # sum over (i,j,k) of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * delta(T[i,j,k])
        # We need this sum to equal 1 for each (i,j,k) where T=1, and 0 otherwise.
        # But T[i,j,k] is always 1 for valid indices, so we need:
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # This is a system of bilinear equations. Let me just construct the Strassen decomposition directly.
        
        # For M1 = (a+b)(e+h) where a=A[i0,j0], b=A[i0,j1], e=B[j0,k0], h=B[j1,k1]:
        # This term contributes to:
        # (i0,j0,k0): U[i0*m+j0]*V[j0*p+k0]*W[k0*n+i0] should get +1
        # (i0,j0,k1): U[i0*m+j0]*V[j0*p+k1]*W[k1*n+i0] should get... wait
        
        # I think the issue is that a single Strassen "product" like (a+b)(e+h) 
        # is not a single rank-1 term in the tensor decomposition. It's a product of 
        # sums, which expands to multiple rank-1 terms.
        
        # Let me reconsider. In tensor decomposition for matrix multiplication,
        # each rank-1 term corresponds to a single product A[i,j]*B[j,k].
        # Strassen reduces the number of such products by using linear combinations.
        
        # Actually, I think the correct way to think about it:
        # A rank-R decomposition has R terms, each of the form:
        # u_c * v_c * w_c where u_c, v_c, w_c are vectors
        # The contribution to T[i,j,k] is sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
        
        # For Strassen, we're not changing the structure of individual terms,
        # we're changing which (i,j,k) tuples get non-zero contributions.
        
        # I think I need to just directly construct the decomposition and verify it.
        
        # Let me use a cleaner approach: define the Strassen factors explicitly.
        
        col = col_offset
        # M1 = (A[i0,j0]+A[i0,j1]) * (B[j0,k0]+B[j1,k1])
        # This means: for the A side, we have non-zeros at (i0,j0) and (i0,j1)
        # For the B side, we have non-zeros at (j0,k0) and (j1,k1)
        # For the C side (W), we need to figure out which (k,i) pairs get contributions
        
        # Actually, I realize the W matrix encodes the output. For a product A[i,j]*B[j,k],
        # the output is C[i,k], so W[k*n+i] gets the coefficient.
        
        # For the Strassen term M1 = (a+b)(e+h):
        # This represents the sum of 4 products: a*e, a*h, b*e, b*h
        # = A[i0,j0]*B[j0,k0] + A[i0,j0]*B[j1,k1] + A[i0,j1]*B[j0,k0] + A[i0,j1]*B[j1,k1]
        # These contribute to:
        # C[i0,k0] from A[i0,j0]*B[j0,k0]: W[k0*n+i0] += 1
        # C[i0,k1] from A[i0,j0]*B[j1,k1]: W[k1*n+i0] += 1
        # C[i0,k0] from A[i0,j1]*B[j0,k0]: W[k0*n+i0] += 1
        # C[i0,k1] from A[i0,j1]*B[j1,k1]: W[k1*n+i0] += 1
        
        # So W[k0*n+i0] += 2, W[k1*n+i0] += 2
        # But that's not right because in the final Strassen formula, we subtract other terms.
        
        # I think the cleanest way is to just enumerate all the rank-1 terms that make up 
        # the Strassen decomposition and add them to U, V, W.
        
        # Let me abandon this complex approach and just use a direct search.
        
        return col_offset
    
    # Let me try a different approach: use the known fact that we can get rank 50
    # by using Strassen on 2x2x2 blocks.
    
    # Actually, let me just try to find a good decomposition using random search 
    # with a smarter initialization based on the block structure.
    
    import random
    
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 40 to 55
    for target_rank in range(40, 56):
        if time.time() > deadline - 15:
            break
        
        found = False
        num_attempts = 200 if target_rank >= 45 else 100
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with sparse random entries
            U_t = np.zeros((n*m, target_rank), dtype=float)
            V_t = np.zeros((m*p, target_rank), dtype=float)
            W_t = np.zeros((p*n, target_rank), dtype=float)
            
            # For each column, pick a random pattern
            for c in range(target_rank):
                # Randomly select which (i,j) pairs are active in U
                # and which (j,k) pairs are active in V
                # The constraint is that for each active (i,j) in U and (j,k) in V,
                # the corresponding (k,i) must be active in W with the right sign.
                
                # Simple approach: pick random non-zero patterns
                u_nz = set()
                v_nz = set()
                w_nz = set()
                
                # Pick 2-4 non-zeros for U
                num_u = random.randint(2, 4)
                for _ in range(num_u):
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    u_nz.add((i, j))
                
                # Pick 2-4 non-zeros for V
                num_v = random.randint(2, 4)
                for _ in range(num_v):
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    v_nz.add((j, k))
                
                # For consistency, W should have non-zeros at (k,i) where
                # there exists j such that (i,j) in u_nz and (j,k) in v_nz
                for (i, j) in u_nz:
                    for (j2, k) in v_nz:
                        if j == j2:
                            w_nz.add((k, i))
                
                if not w_nz:
                    continue
                
                # Assign random signs
                for (i, j) in u_nz:
                    U_t[i*m+j, c] = random.choice([-1, 1])
                for (j, k) in v_nz:
                    V_t[j*p+k, c] = random.choice([-1, 1])
                for (k, i) in w_nz:
                    W_t[k*n+i, c] = random.choice([-1, 1])
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for c in range(target_rank):
                            total += U_t[i*m+j, c] * V_t[j*p+k, c] * W_t[k*n+i, c]
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to lists
                best_rank = target_rank
                best_U = U_t.tolist()
                best_V = V_t.tolist()
                best_W = W_t.tolist()
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
