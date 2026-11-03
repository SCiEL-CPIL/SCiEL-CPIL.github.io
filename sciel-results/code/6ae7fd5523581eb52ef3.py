"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = 3, 3, 5
    # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (15, rank)  
    # W shape: (p*n, rank) = (15, rank)
    # Convention: W[k*n+i, r]
    
    # Use rank-23 Strassen decomposition for 3x3x3 core (columns 0,1,2)
    # Then add terms for columns 3,4
    
    # Strassen 23-term decomposition for 3x3x3:
    # Each term r has U[r] (9-vector), V[r] (9-vector for 3x3 part), W[r] (9-vector)
    # U indexed by (i,j) -> i*3+j
    # V indexed by (j,k) -> j*3+k  
    # W indexed by (k,i) -> k*3+i
    
    # The 23 Strassen terms for 3x3x3:
    # Term: (U_mask, V_mask, W_mask) where each mask is a 9-element vector
    # U_mask[i*3+j] = coefficient for A[i,j]
    # V_mask[j*3+k] = coefficient for B[j,k]
    # W_mask[k*3+i] = coefficient for C[k,i] (note: output convention)
    
    # Strassen 23 terms (i,j,k indices 0-2):
    # I'll use a verified set of 23 terms.
    
    # Standard Strassen for 3x3 uses these 23 bilinear forms:
    # Let me define them explicitly.
    
    # Term definitions: (list of (i,j) pairs for U, (j,k) pairs for V, (k,i) pairs for W)
    # Each entry is a coefficient (can be -1, 0, 1, or fractions)
    
    # Using a known rank-23 decomposition:
    # I'll encode each term as three 9-vectors.
    
    def make_u(ij_list):
        """Create U vector from list of (i,j,coef) tuples"""
        v = [0]*9
        for i,j,c in ij_list:
            v[i*3+j] += c
        return v
    
    def make_v(jk_list):
        """Create V vector from list of (j,k,coef) tuples"""
        v = [0]*9
        for j,k,c in jk_list:
            v[j*3+k] += c
        return v
    
    def make_w(ik_list):
        """Create W vector from list of (k,i,coef) tuples - note k*n+i convention"""
        v = [0]*9
        for k,i,c in ik_list:
            v[k*3+i] += c
        return v
    
    # 23 Strassen terms for 3x3x3
    # Each term: (U_list, V_list, W_list)
    # where lists contain (index1, index2, coefficient)
    
    strassen_terms = [
        # Term 0: M0 = A[0,0]*B[0,0]
        ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
        # Term 1: M1 = A[0,1]*B[1,0]
        ([(0,1,1)], [(1,0,1)], [(0,1,1)]),
        # Term 2: M2 = A[1,0]*B[0,1]
        ([(1,0,1)], [(0,1,1)], [(1,0,1)]),
        # Term 3: M3 = A[1,1]*B[1,1]
        ([(1,1,1)], [(1,1,1)], [(1,1,1)]),
        # Term 4: M4 = A[2,0]*B[0,2]
        ([(2,0,1)], [(0,2,1)], [(2,0,1)]),
        # Term 5: M5 = A[2,1]*B[1,2]
        ([(2,1,1)], [(1,2,1)], [(2,1,1)]),
        # Term 6: M6 = A[0,2]*B[2,0]
        ([(0,2,1)], [(2,0,1)], [(0,2,1)]),
        # Term 7: M7 = A[1,2]*B[2,1]
        ([(1,2,1)], [(2,1,1)], [(1,2,1)]),
        # Term 8: M8 = A[2,2]*B[2,2]
        ([(2,2,1)], [(2,2,1)], [(2,2,1)]),
        # Term 9: M9 = (A[0,0]+A[0,1])*(B[1,0]+B[1,1])
        ([(0,0,1),(0,1,1)], [(1,0,1),(1,1,1)], [(0,0,-1),(0,1,-1),(1,0,-1),(1,1,-1)]),
        # Term 10: M10 = (A[1,0]+A[1,1])*(B[0,1]+B[1,1])
        ([(1,0,1),(1,1,1)], [(0,1,1),(1,1,1)], [(0,1,-1),(1,0,-1),(0,1,-1),(1,1,-1)]),
        # Term 11: M11 = (A[0,0]+A[1,0])*(B[0,0]+B[0,1])
        ([(0,0,1),(1,0,1)], [(0,0,1),(0,1,1)], [(0,0,-1),(1,0,-1),(0,1,-1),(1,1,-1)]),
        # Term 12: M12 = (A[1,1]+A[2,1])*(B[1,1]+B[1,2])
        ([(1,1,1),(2,1,1)], [(1,1,1),(1,2,1)], [(1,1,-1),(1,2,-1),(2,1,-1),(2,2,-1)]),
        # Term 13: M13 = (A[1,1]+A[1,2])*(B[1,1]+B[2,1])
        ([(1,1,1),(1,2,1)], [(1,1,1),(2,1,1)], [(1,1,-1),(2,1,-1),(1,2,-1),(2,2,-1)]),
        # Term 14: M14 = (A[1,1]+A[2,1])*(B[1,1]+B[2,2])
        ([(1,1,1),(2,1,1)], [(1,1,1),(2,2,1)], [(1,1,-1),(2,2,-1),(1,2,-1),(2,1,-1)]),
        # Term 15: M15 = (A[0,2]+A[1,2])*(B[2,0]+B[2,1])
        ([(0,2,1),(1,2,1)], [(2,0,1),(2,1,1)], [(0,2,-1),(1,2,-1),(0,1,-1),(1,0,-1)]),
        # Term 16: M16 = (A[2,0]+A[2,1])*(B[0,2]+B[1,2])
        ([(2,0,1),(2,1,1)], [(0,2,1),(1,2,1)], [(2,0,-1),(2,1,-1),(0,0,-1),(1,1,-1)]),
        # Term 17: M17 = (A[0,0]+A[0,2])*(B[2,0]+B[2,2])
        ([(0,0,1),(0,2,1)], [(2,0,1),(2,2,1)], [(0,0,-1),(0,2,-1),(2,0,-1),(2,2,-1)]),
        # Term 18: M18 = (A[2,2]+A[2,0])*(B[0,2]+B[0,0])
        ([(2,2,1),(2,0,1)], [(0,2,1),(0,0,1)], [(2,2,-1),(0,0,-1),(0,2,-1),(2,0,-1)]),
        # Term 19: M19 = (A[0,1]+A[2,1])*(B[1,0]+B[1,2])
        ([(0,1,1),(2,1,1)], [(1,0,1),(1,2,1)], [(0,1,-1),(2,1,-1),(0,0,-1),(2,2,-1)]),
        # Term 20: M20 = (A[1,0]+A[1,2])*(B[0,1]+B[2,1])
        ([(1,0,1),(1,2,1)], [(0,1,1),(2,1,1)], [(1,0,-1),(1,2,-1),(0,1,-1),(2,1,-1)]),
        # Term 21: M21 = (A[0,0]+A[2,2])*(B[0,0]+B[2,2])
        ([(0,0,1),(2,2,1)], [(0,0,1),(2,2,1)], [(0,0,-1),(2,2,-1),(0,2,-1),(2,0,-1)]),
        # Term 22: M22 = (A[0,1]+A[1,0]+A[1,1]+A[2,2])*(B[0,1]+B[1,0]+B[1,1]+B[2,2])
        # This is getting complex. Let me use a simpler approach.
        ([(0,1,1),(1,0,1),(1,1,1),(2,2,1)], [(0,1,1),(1,0,1),(1,1,1),(2,2,1)], 
         [(0,1,-1),(1,0,-1),(1,1,-1),(2,2,-1),(0,0,-1),(1,1,-1),(1,0,-1),(2,2,-1)]),
    ]
    
    # Verify the 23 terms actually work for 3x3x3
    # Check: for each (i,k), sum over r of U[r][i*3+j]*V[r][j*3+k] should give W[r][k*3+i] when summed
    
    # Let me verify and if not, fall back to schoolbook
    
    U_core = []
    V_core = []
    W_core = []
    
    for term in strassen_terms:
        u_list, v_list, w_list = term
        U_core.append(make_u(u_list))
        V_core.append(make_v(v_list))
        W_core.append(make_w(w_list))
    
    # Verify: C[i,k] = sum_j A[i,j]*B[j,k]
    # For each output (i,k): sum_r sum_j U[r][i*3+j] * V[r][j*3+k] should equal sum_r W[r][k*3+i]
    
    def verify_core():
        for i in range(3):
            for k in range(3):
                lhs = 0
                # LHS: sum over j of A[i,j]*B[j,k] represented as sum over r
                # Actually we need: sum_r U[r][i*3+j] * V[r][j*3+k] summed over j
                # = sum_j (sum_r U[r][i*3+j] * V[r][j*3+k])
                # This should equal sum_r W[r][k*3+i]
                # But this is a bilinear identity, so we check symbolically
                # For each (i,j,k): coefficient of A[i,j]*B[j,k] in LHS should be 1
                # and in RHS should be 1
                
                # Coefficient of A[i,j]*B[j,k] in LHS:
                lhs_coeff = 0
                for r in range(23):
                    lhs_coeff += U_core[r][i*3+j] * V_core[r][j*3+k]
                
                # Coefficient in RHS:
                rhs_coeff = 0
                for r in range(23):
                    rhs_coeff += W_core[r][k*3+i]
                
                # For the identity to hold, for each (i,j,k):
                # sum_r U[r][i*3+j]*V[r][j*3+k] should be 1 if this is the (i,j,k) term
                # But that's not right either. The identity is:
                # sum_j A[i,j]*B[j,k] = sum_r (sum_j U[r][i*3+j]*A[i,j]) * (sum_k' V[r][j*3+k']*B[j,k'])
                # Wait, I'm confusing myself. Let me re-derive.
                
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # In tensor form: C[i,k] = sum_{j,r} U[r][i*3+j] * V[r][j*3+k] * (A[i,j]*B[j,k] placeholder)
                # No. The decomposition is:
                # C[i,k] = sum_r (sum_j U[r][i*3+j] * A[i,j]) * (sum_j V[r][j*3+k] * B[j,k])
                # Wait, that's not right either. Let me be precise.
                
                # The tensor decomposition says:
                # M[i,j,k] = sum_r U[r][i*3+j] * V[r][j*3+k] * W[r][k*3+i]
                # where M[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
                # Actually M[i,j,k] is the multiplication tensor: M[i,j,k] = 1 for all valid (i,j,k)
                # No, M[i,j,k] = delta (it's 1 for each (i,j,k) triple in the product)
                
                # The multiplication tensor T has T[i,j,k] = 1 for all i,j,k in range
                # (since C[i,k] = sum_j A[i,j]*B[j,k], each (i,j,k) contributes)
                
                # So we need: T[i,j,k] = sum_r U[r][i*3+j] * V[r][j*3+k] * W[r][k*3+i]
                # For all i,j,k in {0,1,2}
                
                # So for each (i,j,k): sum_r U[r][i*3+j] * V[r][j*3+k] * W[r][k*3+i] = 1
                
                # Let me check this.
                
                total = 0
                for r in range(23):
                    total += U_core[r][i*3+j] * V_core[r][j*3+k] * W_core[r][k*3+i]
                # Should be 1 for all i,j,k
                if abs(total - 1) > 1e-9:
                    return False
        return True
    
    # Actually the check above is wrong. Let me redo.
    # T[i,j,k] = 1 for all i,j,k
    # T[i,j,k] = sum_r U[r][i*3+j] * V[r][j*3+k] * W[r][k*3+i]
    
    def verify_tensor():
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = 0
                    for r in range(23):
                        total += U_core[r][i*3+j] * V_core[r][j*3+k] * W_core[r][k*3+i]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    core_valid = verify_tensor()
    
    if not core_valid:
        # Fall back to schoolbook rank 45
        rank = 45
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
    
    # Core is valid. Now extend to handle columns 3 and 4.
    # We need additional terms such that for k=3,4:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # In tensor form: T[i,j,k] = 1 for k=3,4
    # T[i,j,k] = sum_{r=0}^{22} U[r][i*3+j]*V[r][j*3+k]*W[r][k*3+i] + sum_{r=23}^{R-1} ...
    # But V[r][j*3+k] for k=3,4 is out of range for the core V vectors (which are 9-dim for 3x3)
    # So we need to extend V to 15-dim (3x5)
    
    # Extend the core terms: V vectors become 15-dim, with entries for k=3,4 set to 0
    # Then add new terms for the extra columns.
    
    # For the new terms (r >= 23), we need:
    # sum_{r=23}^{R-1} U[r][i*3+j] * V[r][j*3+k] * W[r][k*3+i] = 1 for k=3,4
    # and = 0 for k=0,1,2 (to not disturb the core)
    
    # Since V[r][j*3+k] for k=0,1,2 can be nonzero, we need to be careful.
    # Actually, for the new terms, we can set V[r][j*3+k] = 0 for k=0,1,2
    # and only use k=3,4. Then they won't disturb the core.
    
    # So for new terms: V[r][j*3+k] = 0 for k in {0,1,2}, and we only care about k in {3,4}
    # Then: sum_{r=23}^{R-1} U[r][i*3+j] * V[r][j*3+k] * W[r][k*3+i] = 1 for k=3,4
    
    # This is equivalent to: for k=3 and k=4, we need a rank-r decomposition of
    # the 3x3x2 tensor where T[i,j,k] = 1.
    
    # The 3x3x2 tensor has rank at most 6 (schoolbook: 3*3*2=18, but we can do better)
    # Actually rank(3,3,2) <= 6? Let me think.
    # C[i,3] = sum_j A[i,j]*B[j,3] and C[i,4] = sum_j A[i,j]*B[j,4]
    # This is like computing two dot products of length 3 for each i.
    # Rank is at most 3*2 = 6? No, it's at most 3*3*2 = 18 schoolbook.
    # But we can do: for each j, use one term that handles B[j,3] and B[j,4] together?
    # No, because W depends on k separately.
    
    # Simple approach: use 18 terms (schoolbook for 3x3x2).
    # But we can do better. Let me use 9 terms: for each (i,j), one term.
    # Term for (i,j): U has 1 at position i*3+j, V has 1 at j*3+3 and j*3+4, W has... 
    # But W[k*3+i] needs to be separate for k=3 and k=4.
    # So we can't combine k=3 and k=4 in one term easily.
    
    # Use 6 terms: for each (j, k_extra) where k_extra in {3,4}, and we handle all i.
    # No, that doesn't work either because U depends on i.
    
    # Schoolbook for 3x3x2: 18 terms.
    # But we can use Strassen-like tricks. Let me just use 18 terms for safety.
    
    # Actually, let me try to use fewer. The 3x3x2 tensor:
    # T[i,j,k] = 1 for i,j in 0..2, k in 3,4
    # We can write this as: for each j, the slice T[:,j,:] is a 3x2 all-ones matrix.
    # Rank of a 3x2 all-ones matrix is 1.
    # So T = sum_j (e_j in A-space) outer (row vector in B-space) outer (column vector in C-space)
    # More precisely: T[i,j,k] = sum_j delta[i,j] * 1 * 1 where the sum is over j
    # No, T[i,j,k] = 1 for all i,j,k (in range).
    
    # Let me just use 18 schoolbook terms for the 3x3x2 part.
    # Total rank: 23 + 18 = 41.
    
    # But wait, can we do better? Let me try rank 40.
    # If I can find a rank-17 decomposition for 3x3x2, total is 40.
    
    # For now, use 18 terms. Total rank 41.
    
    extra_rank = 18
    total_rank = 23 + extra_rank
    
    # Build full U, V, W
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill core terms (r = 0..22)
    for r in range(23):
        for idx in range(9):
            U[idx][r] = U_core[r][idx]
            # V for core: only k=0,1,2
            V[idx][r] = V_core[r][idx]
            W[idx][r] = W_core[r][idx]
        # V for k=3,4 is 0 (already initialized to 0)
    
    # Fill extra terms (r = 23..40) for k=3,4
    extra_col = 0
    for i in range(3):
        for j in range(3):
            for k_extra in [3, 4]:
                r = 23 + extra_col
                U[i * 3 + j][r] = 1
                V[j * 3 + k_extra][r] = 1
                W[k_extra * 3 + i][r] = 1
                extra_col += 1
    
    # Verify the full decomposition
    def verify_full():
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    total = 0
                    for r in range(total_rank):
                        total += U[i*3+j][r] * V[j*3+k][r] * W[k*3+i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    if not verify_full():
        # Fallback to schoolbook
        rank = 45
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
