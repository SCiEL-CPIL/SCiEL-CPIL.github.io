"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_np(U, V, W):
        """Fast numpy-based verification using einsum."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        # Reshape: U[4,R], V[8,R], W[8,R]
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Build the tensor via broadcasting
        R = Ua.shape[1]
        # For each r, the contribution is outer product of U[:,r], V[:,r], W[:,r]
        # But we need to map indices correctly
        # U index: i*m+j -> (i, j)
        # V index: j*p+k -> (j, k)
        # W index: k*n+i -> (k, i)
        # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for r in range(R):
            u = Ua[:, r].reshape(n, m)
            v = Va[:, r].reshape(m, p)
            w = Wa[:, r].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # This is: for each i,j,k: u[i,j]*v[j,k]*w[k,i]
            # = np.einsum('ij,jk,ki->ijk', u, v, w)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        # Check T[i,j,k] == 1 for all i,j,k
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def get_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        Returns (U, V, W) as numpy arrays of shape (4,14), (8,14), (8,14).
        """
        R = 14
        U = np.zeros((4, R), dtype=int)
        V = np.zeros((8, R), dtype=int)
        W = np.zeros((8, R), dtype=int)

        # Left block (cols 0,1 of B): Strassen on A * B_left where B_left is 2x2
        # B_left[0,0]=b00, B_left[0,1]=b01, B_left[1,0]=b10, B_left[1,1]=b11
        # V indices: j*p+k -> j*4+k
        #   b00 -> V[0], b01 -> V[1], b10 -> V[4], b11 -> V[5]
        # W indices: k*n+i -> k*2+i
        #   C[0,0] -> W[0], C[0,1] -> W[1], C[1,0] -> W[2], C[1,1] -> W[3]

        # L1: (a00+a11)(b00+b11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[5,0] = 1
        W[0,0] = 1; W[3,0] = 1

        # L2: (a10+a11)b00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[1,1] = 1; W[3,1] = -1

        # L3: a00(b01-b11)
        U[0,2] = 1
        V[1,2] = 1; V[5,2] = -1
        W[2,2] = 1; W[3,2] = 1

        # L4: a11(b10-b00)
        U[3,3] = 1
        V[4,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[1,3] = 1

        # L5: (a00+a01)b11
        U[0,4] = 1; U[1,4] = 1
        V[5,4] = 1
        W[0,4] = -1; W[2,4] = 1

        # L6: (a10-a00)(b00+b01)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1

        # L7: (a01-a11)(b10+b11)
        U[1,6] = 1; U[3,6] = -1
        V[4,6] = 1; V[5,6] = 1
        W[0,6] = 1

        # Right block (cols 2,3 of B): Strassen on A * B_right where B_right is 2x2
        # B_right[0,0]=b02, B_right[0,1]=b03, B_right[1,0]=b12, B_right[1,1]=b13
        # V indices:
        #   b02 -> V[2], b03 -> V[3], b12 -> V[6], b13 -> V[7]
        # W indices:
        #   C[0,2] -> W[4], C[0,3] -> W[5], C[1,2] -> W[6], C[1,3] -> W[7]

        # R1: (a00+a11)(b02+b13)
        U[0,7] = 1; U[3,7] = 1
        V[2,7] = 1; V[7,7] = 1
        W[4,7] = 1; W[7,7] = 1

        # R2: (a10+a11)b02
        U[2,8] = 1; U[3,8] = 1
        V[2,8] = 1
        W[5,8] = 1; W[7,8] = -1

        # R3: a00(b03-b13)
        U[0,9] = 1
        V[3,9] = 1; V[7,9] = -1
        W[6,9] = 1; W[7,9] = 1

        # R4: a11(b12-b02)
        U[3,10] = 1
        V[6,10] = 1; V[2,10] = -1
        W[4,10] = 1; W[5,10] = 1

        # R5: (a00+a01)b13
        U[0,11] = 1; U[1,11] = 1
        V[7,11] = 1
        W[4,11] = -1; W[6,11] = 1

        # R6: (a10-a00)(b02+b03)
        U[2,12] = 1; U[0,12] = -1
        V[2,12] = 1; V[3,12] = 1
        W[7,12] = 1

        # R7: (a01-a11)(b12+b13)
        U[1,13] = 1; U[3,13] = -1
        V[6,13] = 1; V[7,13] = 1
        W[4,13] = 1

        return U, V, W

    def try_strassen_rank14():
        """Try the structured rank-14 decomposition."""
        U, V, W = get_rank14()
        if verify_np(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    def try_merge_rank13():
        """
        Try to merge two terms from the rank-14 decomposition to get rank 13.
        For each pair (r1, r2), check if their sum can be expressed as a
        single rank-1 term with coefficients in {-1, 0, 1}.
        """
        if time.time() > deadline - 15:
            return None

        U14, V14, W14 = get_rank14()
        R14 = 14

        # For each pair of terms to merge, check if the combined tensor
        # is rank 1 (i.e., can be written as u ⊗ v ⊗ w)
        for r1 in range(R14):
            for r2 in range(r1 + 1, R14):
                if time.time() > deadline - 5:
                    return None

                # Build the 13-term decomposition by replacing terms r1,r2 with their sum
                # The sum tensor is: sum of two rank-1 tensors
                # We need to check if this sum tensor has rank 1
                # i.e., if there exist u, v, w such that
                # u[i]*v[j]*w[k] = U14[i,r1]*V14[j,r1]*W14[k,r1] + U14[i,r2]*V14[j,r2]*W14[k,r2]
                # for all i, j, k (where i in {0,1}, j in {0,1}, k in {0,1,2,3})
                # But wait - the tensor indices are:
                # U index: i*m+j = i*2+j (4 values)
                # V index: j*p+k = j*4+k (8 values)
                # W index: k*n+i = k*2+i (8 values)
                # The "tensor" T has entries T[i,j,k] for i in 0..1, j in 0..1, k in 0..3
                # T[i,j,k] = sum_r U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r]

                # For the merged term, we need:
                # U_new[i*2+j] * V_new[j*4+k] * W_new[k*2+i]
                # = U14[i*2+j, r1]*V14[j*4+k, r1]*W14[k*2+i, r1]
                #   + U14[i*2+j, r2]*V14[j*4+k, r2]*W14[k*2+i, r2]
                # for all i, j, k

                # This is a system of 16 equations. Let's check if a rank-1 solution exists.
                # A rank-1 tensor T[i,j,k] = u[i,j]*v[j,k]*w[k,i] where:
                # u has 4 entries (indexed by i*2+j)
                # v has 8 entries (indexed by j*4+k)
                # w has 8 entries (indexed by k*2+i)
                # But the constraint is that u[i*2+j] depends only on (i,j),
                # v[j*4+k] depends only on (j,k), w[k*2+i] depends only on (k,i).

                # Let's build the 16x? system. For each (i,j,k), we have:
                # u[i*2+j] * v[j*4+k] * w[k*2+i] = target[i,j,k]
                # where target[i,j,k] = sum of the two rank-1 contributions.

                # This is a nonlinear system. Let's try a different approach:
                # Check if the sum of two rank-1 tensors is itself rank-1.
                # A sum of two rank-1 tensors is rank-1 iff they are "parallel"
                # in a specific sense.

                # Actually, let's just try to solve it.
                # The target tensor has 16 entries. We need to find u (4 vals), v (8 vals), w (8 vals)
                # such that u[i*2+j] * v[j*4+k] * w[k*2+i] = target[i,j,k] for all 16 combos.
                # But u, v, w have 4+8+8 = 20 unknowns and 16 equations.
                # However, the system is highly structured.

                # Let's compute the target tensor first.
                target = np.zeros((2, 2, 4))
                for i in range(2):
                    for j in range(2):
                        for k in range(4):
                            target[i,j,k] = (U14[i*2+j, r1] * V14[j*4+k, r1] * W14[k*2+i, r1]
                                          + U14[i*2+j, r2] * V14[j*4+k, r2] * W14[k*2+i, r2])

                # Now check if target is rank-1 in the structured sense.
                # target[i,j,k] = u[i,j] * v[j,k] * w[k,i]
                # where u has 4 entries, v has 8, w has 8.

                # Strategy: Pick a reference entry to fix scaling, then solve.
                # If any target entry is non-zero, we can use it.
                # Let's find non-zero entries.
                nz = np.argwhere(target != 0)
                if len(nz) == 0:
                    # Both terms cancel everywhere - this means the merged term is zero
                    # which is fine, we just drop both terms
                    # Build 12-term decomposition
                    U_new = np.delete(U14, [r1, r2], axis=1)
                    V_new = np.delete(V14, [r1, r2], axis=1)
                    W_new = np.delete(W14, [r1, r2], axis=1)
                    # Add a zero term to make it 13
                    U_new = np.column_stack([U_new, np.zeros(4)])
                    V_new = np.column_stack([V_new, np.zeros(8)])
                    W_new = np.column_stack([W_new, np.zeros(8)])
                    if verify_np(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                    continue

                # Try to solve for rank-1 structure.
                # Pick a non-zero entry (i0,j0,k0) to fix u[i0,j0] = 1 (scaling).
                i0, j0, k0 = nz[0]
                t0 = target[i0,j0,k0]

                # From target[i0,j0,k0] = u[i0,j0]*v[j0,k0]*w[k0,i0] = t0
                # Set u[i0,j0] = 1, then v[j0,k0]*w[k0,i0] = t0

                # From target[i0,j0,k] = u[i0,j0]*v[j0,k]*w[k,i0] = v[j0,k]*w[k,i0]
                # So v[j0,k]*w[k,i0] = target[i0,j0,k] for all k
                # And target[i0,j0,k0] = t0, so v[j0,k0]*w[k0,i0] = t0

                # From target[i,j0,k0] = u[i,j0]*v[j0,k0]*w[k0,i]
                # u[i,j0]*v[j0,k0]*w[k0,i] = target[i,j0,k0]

                # Let's parameterize:
                # Let a_k = v[j0,k] for k=0..3
                # Let b_i = w[k0,i] for i=0,1
                # Then a_k * b_i = target[i0,j0,k] ... wait, that's not right.
                # target[i0,j0,k] = u[i0,j0]*v[j0,k]*w[k,i0] = v[j0,k]*w[k,i0]
                # So for fixed i0, j0: target[i0,j0,k] = v[j0,k]*w[k,i0]
                # This means for each k, v[j0,k]*w[k,i0] is determined.
                # But w[k,i0] depends on k, so we can't separate easily.

                # Alternative approach: Use the fact that for a rank-1 tensor
                # T[i,j,k] = u[i,j]*v[j,k]*w[k,i], we have:
                # T[i,j,k]*T[i',j',k'] = T[i,j',k']*T[i',j,k] when the indices are consistent
                # More precisely, for any i,i',j,j',k,k':
                # T[i,j,k]*T[i',j',k'] = T[i,j',k']*T[i',j,k] * (w[k,i]*w[k',i'])/(w[k',i]*w[k,i'])
                # This is getting complicated. Let me just try a direct solve.

                # Direct approach: Assume u[i0,j0] = 1.
                # Then for all (i,j,k):
                # u[i,j]*v[j,k]*w[k,i] = target[i,j,k]
                # We have 20 unknowns, 16 equations. Let's fix some scalings.
                # Set u[i0,j0] = 1.
                # For j=j0: u[i,j0]*v[j0,k]*w[k,i] = target[i,j0,k] for all i,k
                # For i=i0: u[i0,j]*v[j,k]*w[k,i0] = target[i0,j,k] for all j,k

                # Let's try: set all u[i,j] = 1 for now, and see if we can find v, w.
                # Then v[j,k]*w[k,i] = target[i,j,k] for all i,j,k.
                # This is a system of 16 equations in 16 unknowns (8 v + 8 w).
                # But it's nonlinear (products of v and w).

                # Alternative: Take ratios.
                # target[i,j,k]/target[i,j,k'] = w[k,i]/w[k',i] for fixed i,j
                # target[i,j,k]/target[i',j,k] = (u[i,j]*w[k,i])/(u[i',j]*w[k,i']) ... messy

                # Let me try a completely different approach:
                # Just check if the 16x16 matrix formed by the tensor
                # has rank 1 in the appropriate sense.

                # Actually, the simplest check: if target[i,j,k] can be written as
                # u[i,j]*v[j,k]*w[k,i], then for any fixed i:
                # target[i,j,k] = u[i,j]*v[j,k]*w[k,i]
                # The matrix M_i[j,k] = target[i,j,k] should have the property that
                # M_i[j,k]*M_i[j',k'] = M_i[j,k']*M_i[j',k] * (w[k,i]*w[k',i])/(w[k',i]*w[k,i])
                # which is always true. So that's not a useful check.

                # Let me just try to solve numerically.
                # Fix u[i0,j0] = 1.
                # For each j, k: from target[i0,j,k] = u[i0,j]*v[j,k]*w[k,i0]
                # We have 8 equations (j in 0,1; k in 0,1,2,3) with 16 unknowns.
                # Not enough constraints.

                # Let me try yet another approach: enumerate possible u, v, w with entries in {-1,0,1}.
                # u has 4 entries, v has 8, w has 8. Total 20 entries, each in {-1,0,1}.
                # That's 3^20 = 3.5 billion possibilities. Too many.

                # But we can be smarter. The target tensor has at most 16 non-zero entries.
                # For a rank-1 tensor, the support structure is constrained.

                # Let me just try a small number of candidate solutions.
                # Candidate 1: u[i,j] in {-1,1}, v[j,k] in {-1,1}, w[k,i] in {-1,1}
                # with the constraint that u[i0,j0] = 1.
                # That's 2^3 * 2^8 * 2^8 = 2^19 = 524288 possibilities. Feasible!

                found = False
                for u_vals in product([-1, 1], repeat=4):
                    if u_vals[i0*2+j0] != 1:
                        continue
                    for v_vals in product([-1, 1], repeat=8):
                        for w_vals in product([-1, 1], repeat=8):
                            ok = True
                            for i in range(2):
                                for j in range(2):
                                    for k in range(4):
                                        val = u_vals[i*2+j] * v_vals[j*4+k] * w_vals[k*2+i]
                                        if val != target[i,j,k]:
                                            ok = False
                                            break
                                    if not ok:
                                        break
                                if not ok:
                                    break
                            if ok:
                                # Found a valid rank-1 representation!
                                # Build the 13-term decomposition
                                U_new = np.delete(U14, [r1, r2], axis=1)
                                V_new = np.delete(V14, [r1, r2], axis=1)
                                W_new = np.delete(W14, [r1, r2], axis=1)
                                U_new = np.column_stack([U_new, np.array(u_vals, dtype=int)])
                                V_new = np.column_stack([V_new, np.array(v_vals, dtype=int)])
                                W_new = np.column_stack([W_new, np.array(w_vals, dtype=int)])
                                if verify_np(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                                    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

        return None

    def try_random_rank13():
        """
        More targeted random search for rank 13, starting near the rank-14 solution.
        """
        if time.time() > deadline - 10:
            return None

        U14, V14, W14 = get_rank14()
        R14 = 14

        np.random.seed(123)
        for attempt in range(500):
            if time.time() > deadline - 3:
                break

            R = 13
            # Start from rank-14 and perturb
            U = U14[:, :R].copy()
            V = V14[:, :R].copy()
            W = W14[:, :R].copy()

            # Randomly modify some entries
            for _ in range(3):
                # Pick a random row and column to flip or zero out
                mat = np.random.choice([U, V, W])
                row = np.random.randint(mat.shape[0])
                col = np.random.randint(R)
                if mat[row, col] != 0:
                    # Flip sign or zero out
                    if np.random.random() < 0.5:
                        mat[row, col] = -mat[row, col]
                    else:
                        mat[row, col] = 0
                else:
                    mat[row, col] = np.random.choice([-1, 1])

            # Also try adding a new random term in the last column
            if attempt % 2 == 0:
                # Replace last column with a random sparse term
                a_mask = np.random.choice([0,1], 4, p=[0.4, 0.6])
                b_mask = np.random.choice([0,1], 8, p=[0.4, 0.6])
                c_mask = np.random.choice([0,1], 8, p=[0.4, 0.6])
                U[:, R-1] = 0
                V[:, R-1] = 0
                W[:, R-1] = 0
                U[a_mask.astype(bool), R-1] = np.random.choice([-1, 1], a_mask.sum())
                V[b_mask.astype(bool), R-1] = np.random.choice([-1, 1], b_mask.sum())
                W[c_mask.astype(bool), R-1] = np.random.choice([-1, 1], c_mask.sum())

            if verify_np(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return None

    def try_structured_rank13():
        """
        Try specific structured rank-13 decompositions.
        Key idea: share one term between the two Strassen blocks.
        """
        if time.time() > deadline - 10:
            return None

        # Strategy: The left block uses terms L1-L7, right block uses R1-R7.
        # L1 = (a00+a11)(b00+b11), R1 = (a00+a11)(b02+b13)
        # If we merge L1 and R1 into one term: (a00+a11)(b00+b11+b02+b13)
        # This contributes to C[0,0], C[1,1], C[0,2], C[1,3]
        # But we need specific coefficients. Let's check if this works.

        # Try: shared term S = (a00+a11)(b00+b11+b02+b13)
        # S contributes:
        # C[0,0] += a00*b00 + a00*b11 + a00*b02 + a00*b13  (wrong! should be a00*b00 + a01*b10)
        # This doesn't work directly.

        # Alternative: Try different sharing patterns.
        # What if we use a non-Strassen decomposition for one block?

        # Let me try: 6-term decomposition for the left block + 7 for right = 13
        # But 2x2*2x2 has rank 7, so we can't do 6.

        # What about: use 7 terms for left, 7 for right, but one term is shared?
        # The shared term must contribute to both blocks.
        # A term u ⊗ v ⊗ w where v has non-zero entries in both {b00,b01,b10,b11}
        # and {b02,b03,b12,b13} would contribute to both blocks.

        # Let's try: shared term with v = (1,0,1,0,0,0,0,0) i.e., v[b00]=1, v[b02]=1
        # And u = (1,0,0,1) i.e., u[a00]=1, u[a11]=1
        # And w = (1,0,1,0,1,0,1,0) i.e., w[C[0,0]]=1, w[C[0,2]]=1, w[C[1,0]]=1, w[C[1,2]]=1
        # This term: (a00+a11)(b00+b02) contributes to C[0,0], C[0,2], C[1,0], C[1,2]
        # C[0,0] += a00*b00 + a11*b00  (but we need a00*b00 + a01*b10)
        # Doesn't match.

        # Let me try a more systematic approach.
        # Enumerate all possible shared terms and check if the remaining
        # 12 terms can complete the decomposition.

        # This is complex. Let me try a few specific candidates.

        # Candidate: Use the standard rank-14 but merge L2 and R2.
        # L2 = (a10+a11)*b00, R2 = (a10+a11)*b02
        # Merged: (a10+a11)*(b00+b02)
        # This contributes to: C[1,0] += (a10+a11)*b00, C[1,2] += (a10+a11)*b02
        # In the original: L2 contributes C[1,0] += a10*b00+a11*b00 and C[1,1] += a10*b00+a11*b00 (wait, no)
        # Let me recheck L2: (a10+a11)*b00
        # U: a10, a11 -> U[2]=1, U[3]=1
        # V: b00 -> V[0]=1
        # W: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
        # L2: W[1]=1, W[3]=-1
        # So L2 contributes:
        # C[0,1] += (a10+a11)*b00*1 = a10*b00+a11*b00
        # C[1,1] += (a10+a11)*b00*(-1) = -(a10*b00+a11*b00)
        # Wait, that doesn't seem right for Strassen. Let me re-verify.

        # Actually, I think my Strassen implementation might have errors.
        # Let me just verify the rank-14 first.
        U14, V14, W14 = get_rank14()
        if not verify_np(U14.tolist(), V14.tolist(), W14.tolist()):
            # The rank-14 is wrong, return None
            return None

        # OK so rank-14 is correct. Now try merging.
        # The merge approach in try_merge_rank13 should handle this.
        return None

    # Try rank-13 approaches
    result = try_merge_rank13()
    if result is not None:
        return result

    result = try_structured_rank13()
    if result is not None:
        return result

    result = try_random_rank13()
    if result is not None:
        return result

    # Try the structured rank-14 decomposition
    result = try_strassen_rank14()
    if result is not None:
        return result

    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
