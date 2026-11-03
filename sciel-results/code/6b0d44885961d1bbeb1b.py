import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def get_schoolbook():
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

    def build_rank_28_decomposition():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_strassen_block(rank_offset, i0, j0, k0):
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            c00_idx = k0 * n + i0
            c01_idx = (k0 + 1) * n + i0
            c10_idx = k0 * n + (i0 + 1)
            c11_idx = (k0 + 1) * n + (i0 + 1)
            
            r0 = rank_offset
            r1 = rank_offset + 1
            r2 = rank_offset + 2
            r3 = rank_offset + 3
            r4 = rank_offset + 4
            r5 = rank_offset + 5
            r6 = rank_offset + 6
            
            U[a_idx][r0] += 1; U[d_idx][r0] += 1
            V[e_idx][r0] += 1; V[h_idx][r0] += 1
            W[c00_idx][r0] += 1; W[c11_idx][r0] += 1
            
            U[c_idx][r1] += 1; U[d_idx][r1] += 1
            V[e_idx][r1] += 1
            W[c10_idx][r1] += 1
            
            U[a_idx][r2] += 1
            V[f_idx][r2] += 1; V[h_idx][r2] -= 1
            W[c01_idx][r2] += 1; W[c11_idx][r2] -= 1
            
            U[d_idx][r3] += 1
            V[g_idx][r3] += 1; V[e_idx][r3] -= 1
            W[c00_idx][r3] -= 1; W[c10_idx][r3] += 1
            
            U[a_idx][r4] += 1; U[b_idx][r4] += 1
            V[h_idx][r4] += 1
            W[c01_idx][r4] += 1; W[c11_idx][r4] += 1
            
            U[c_idx][r5] += 1; U[a_idx][r5] -= 1
            V[e_idx][r5] += 1; V[f_idx][r5] += 1
            W[c00_idx][r5] += 1
            
            U[b_idx][r6] += 1; U[d_idx][r6] -= 1
            V[g_idx][r6] += 1; V[h_idx][r6] += 1
            W[c00_idx][r6] -= 1; W[c11_idx][r6] += 1

        add_strassen_block(0, 0, 0, 0)
        add_strassen_block(7, 0, 0, 2)
        
        col = 14
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
        
        for i in range(2):
            for k in range(5):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        
        assert col == rank
        return {"U": U, "V": V, "W": W}

    def validate_float(U, V, W, tol=1e-9):
        rank = len(U[0])
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        U_3d = U_mat.reshape(n, m, rank)
        V_3d = V_mat.reshape(m, p, rank)
        W_3d = W_mat.reshape(p, n, rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T_recon, 1.0, atol=tol)

    def validate_exact(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    def try_rank_27_search():
        """
        Attempt to find a rank-27 decomposition using structured search.
        Uses the insight that we might merge one schoolbook term with Strassen
        structure by finding a linear dependency.
        
        Strategy: Start from rank-28 and try to eliminate one rank by finding
        a non-trivial linear combination that preserves the tensor identity.
        
        We parameterize the last few terms with small integer coefficients and
        use polynomial identity testing to verify candidates.
        """
        if time.time() > deadline - 20:
            return None
        
        # Start from the rank-28 decomposition
        base = build_rank_28_decomposition()
        U_base = [row[:] for row in base["U"]]
        V_base = [row[:] for row in base["V"]]
        W_base = [row[:] for row in base["W"]]
        
        # The rank-28 has 10 schoolbook terms in the last block (cols 14-27)
        # Block 3: cols 14-17 (4 terms): A[0:2,0:2]*B[0:2,4]
        # Block 4: cols 18-27 (10 terms): A[0:2,2]*B[2,0:5]
        
        # Try to merge one term from Block 4 into the Strassen structure
        # by finding if any of the 10 schoolbook terms can be expressed
        # as a linear combination of the existing 18 terms.
        
        # For each candidate rank to eliminate (from the last block),
        # check if the corresponding tensor component is already spanned.
        
        # The tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # For rank 27, we need T to be representable with 27 terms.
        
        # Key insight: The 10 terms in Block 4 are all of the form
        # A[i,2]*B[2,k] -> C[i,k]
        # These are "outer product" type terms. If any one can be written
        # as a combination of the other 27 terms, we can eliminate it.
        
        # Try eliminating each of the 10 Block-4 terms one at a time
        for elim_col in range(18, 28):
            if time.time() > deadline - 15:
                break
            
            # Build a 27-rank candidate by removing column elim_col
            # and checking if the tensor identity still holds
            U_27 = []
            for row in U_base:
                new_row = row[:elim_col] + row[elim_col+1:]
                U_27.append(new_row)
            
            V_27 = []
            for row in V_base:
                new_row = row[:elim_col] + row[elim_col+1:]
                V_27.append(new_row)
            
            W_27 = []
            for row in W_base:
                new_row = row[:elim_col] + row[elim_col+1:]
                W_27.append(new_row)
            
            # Quick float check
            if not validate_float(U_27, V_27, W_27, tol=1e-8):
                continue
            
            # Exact check
            U_frac = [[Fraction(x) for x in row] for row in U_27]
            V_frac = [[Fraction(x) for x in row] for row in V_27]
            W_frac = [[Fraction(x) for x in row] for row in W_27]
            
            if validate_exact(U_frac, V_frac, W_frac):
                return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        # Try a different approach: modify one Strassen block to absorb
        # a schoolbook term. For example, in Block 3 (4 terms for A[0:2,0:2]*B[0:2,4]),
        # try to use a 3-rank decomposition instead of 4.
        
        # A 2x2 * 2x1 matrix product has rank at most 4 (schoolbook).
        # Can we do it in 3? The tensor is T[i,j,k] where i in {0,1}, j in {0,1}, k=4.
        # This is a 2x2x1 tensor. The rank of a 2x2x1 tensor is at most 4,
        # but could be 3 if there's a special structure.
        
        # The 2x2x1 tensor T[i,j,4] = A[i,j]*B[j,4]
        # As a bilinear map: (A, B) -> C where C[i] = sum_j A[i,j]*B[j,4]
        # This is just matrix-vector multiplication: C = A * b where b = [B[0,4], B[1,4]]
        # The rank of this tensor is 4 (it's the standard 2x2 matrix-vector product).
        
        # Actually, for 2x2 * 2x1, the tensor rank is 4 (no known improvement).
        # So we can't reduce Block 3.
        
        # Try merging across blocks: Can we combine one term from Block 3
        # with one from Block 4 using a shared Strassen-like structure?
        
        # Alternative: Try a completely different partition.
        # What if we use Strassen on a 2x2 block that spans the boundary?
        
        # Partition: A[0:2, 0:2] and A[0:2, 2:3]
        # B: rows 0,1,2; cols 0,1,2,3,4
        
        # What if we group B columns differently?
        # B cols {0,2,4} and {1,3}? No, that doesn't help with Strassen.
        
        # Let's try: use a 3-rank decomposition for a 2x1 * 1x2 sub-product
        # embedded in the larger structure.
        
        # Actually, the most promising approach: try all permutations of
        # which schoolbook term to try to eliminate, and also try small
        # integer coefficient modifications.
        
        # Try modifying the last block terms with small coefficients
        for delta in range(-2, 3):
            if delta == 0:
                continue
            if time.time() > deadline - 10:
                break
            
            for mod_col in range(14, 28):
                if time.time() > deadline - 10:
                    break
                
                U_mod = [row[:] for row in U_base]
                V_mod = [row[:] for row in V_base]
                W_mod = [row[:] for row in W_base]
                
                # Add delta to one entry in the modified column
                # Try adding to U, V, or W entries
                for mat, mod_mat in [('U', U_mod), ('V', V_mod), ('W', W_mod)]:
                    if time.time() > deadline - 10:
                        break
                    dims = {'U': n*m, 'V': m*p, 'W': p*n}[mat]
                    for idx in range(dims):
                        if mod_mat[idx][mod_col] == 0:
                            continue
                        # Skip if this would make it zero (already tried removal)
                        if mod_mat[idx][mod_col] == -delta:
                            continue
                        
                        orig = mod_mat[idx][mod_col]
                        mod_mat[idx][mod_col] = orig + delta
                        
                        if validate_float(mod_mat if mat=='U' else U_mod, 
                                         mod_mat if mat=='V' else V_mod,
                                         mod_mat if mat=='W' else W_mod, tol=1e-8):
                            U_frac = [[Fraction(x) for x in row] for row in U_mod]
                            V_frac = [[Fraction(x) for x in row] for row in V_mod]
                            W_frac = [[Fraction(x) for x in row] for row in W_mod]
                            if validate_exact(U_frac, V_frac, W_frac):
                                return {"U": U_frac, "V": V_frac, "W": W_frac}
                        
                        mod_mat[idx][mod_col] = orig
        
        return None

    # First try rank 27
    result_27 = try_rank_27_search()
    if result_27 is not None:
        return result_27
    
    # Fall back to rank 28
    result_28 = build_rank_28_decomposition()
    if validate_float(result_28["U"], result_28["V"], result_28["W"]):
        U_frac = [[Fraction(x) for x in row] for row in result_28["U"]]
        V_frac = [[Fraction(x) for x in row] for row in result_28["V"]]
        W_frac = [[Fraction(x) for x in row] for row in result_28["W"]]
        if validate_exact(U_frac, V_frac, W_frac):
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return get_schoolbook()
# EVOLVE-BLOCK-END
