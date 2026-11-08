import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 15.0
    deadline = start_time + time_limit - safety_margin

    n, m, p = 2, 3, 9
    nm = n * m   # 6
    mp = m * p   # 27
    pn = p * n   # 18

    # The multiplication tensor T has entries T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # for some i in {0,1}, j in {0,1,2}, k in {0,...,8}, else 0.
    # We want a rank-r decomposition: T[a,b,c] = sum_{r=1}^R U[a,r]*V[b,r]*W[c,r]

    def make_schoolbook():
        """Schoolbook decomposition: rank = n*m*p = 54"""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return U, V, W

    def make_strassen_based():
        """
        Use Strassen-style decomposition for the 2x2 blocks of the first 8 columns of B,
        plus schoolbook for the 9th column.
        
        A is 2x3, B is 3x9.
        Split B into B1 (3x8) and B2 (3x1).
        Split A into A1 (2x2) and A2 (2x1).
        
        For A1 * B1: treat as block multiplication where we process 2x2 blocks.
        Actually, let's think of it differently.
        
        C = A * B where A is 2x3, B is 3x9.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        Let's group the 9 columns of B. For each pair of columns (k, k+1) for k=0,2,4,6,
        we have a 2x2 sub-problem if we also group rows of A appropriately.
        
        Actually, the standard approach for small matrix multiplication:
        C (2x9) = A (2x3) * B (3x9)
        
        We can view this as: for each j, A[:,j] (2x1) * B[j,:] (1x9) = outer product.
        Sum over j=0,1,2.
        
        Rank 54 schoolbook: each (i,j,k) triple is one rank-1 term.
        
        For Strassen: we can group the 3 rows of A and 3 rows of B differently.
        
        Let's try: split the 3 middle dimension into 2+1.
        A = [A_left (2x2) | A_right (2x1)]
        B = [B_top (2x9); B_bottom (1x9)]
        
        C = A_left * B_top + A_right * B_bottom
        
        A_left * B_top is (2x2)*(2x9). We can use Strassen for the 2x2 part...
        but B_top is 2x9, not 2x2.
        
        Alternative: process column pairs.
        For k = 0,2,4,6: columns k and k+1 of B form a 2-column block.
        A * B[:, k:k+2] is (2x3)*(3x2) = 2x2 matrix.
        This can be done with rank 7 (Strassen) instead of rank 4 (schoolbook).
        
        For k = 8: single column, rank 3 (schoolbook: 3 terms for 2x3 * 3x1).
        
        Total: 4 * 7 + 3 = 31? Wait, let me recalculate.
        
        Actually for (2x3)*(3x2):
        C[i,k'] = sum_{j=0}^{2} A[i,j] * B[j,k']
        where k' in {0,1} corresponds to columns k, k+1.
        
        This is 4 output entries, each a sum of 3 products. Schoolbook rank = 4*3 = 12? No.
        Schoolbook: each (i,j,k') triple. There are 2*3*2 = 12 triples. Rank 12.
        
        Strassen for 2x2: if we could do (2x2)*(2x2) in rank 7, but we have (2x3)*(3x2).
        
        Let me think more carefully. The tensor for (2x3)*(3x2) multiplication:
        T[i, j, k'] where i in {0,1}, j in {0,1,2}, k' in {0,1}
        T[i, j, k'] = 1 (this is the "multiplication tensor" entry)
        
        Actually the multiplication tensor has T[a,b,c] where a indexes A, b indexes B, c indexes C.
        a = i*m + j = i*3 + j (but m=3 here for the sub-problem... wait)
        
        Let me just build it properly.
        
        For the full problem:
        a = i*3 + j, i in {0,1}, j in {0,1,2}
        b = j*9 + k, j in {0,1,2}, k in {0,...,8}
        c = k*2 + i, k in {0,...,8}, i in {0,1}
        
        T[a,b,c] = 1 iff there exist i,j,k such that a=i*3+j, b=j*9+k, c=k*2+i.
        
        For the sub-problem with columns k0, k0+1 (a 2-column block):
        a = i*3 + j
        b = j*9 + k where k in {k0, k0+1}
        c = k*2 + i
        
        This sub-tensor has rank at most 2*3*2 = 12 (schoolbook).
        Can we do better?
        
        Let's use Strassen on the 2x2 block formed by rows {0,1} of A and columns {k0, k0+1} of B,
        but we have 3 rows in A and 3 rows in B connecting them.
        
        Actually, let me try a different decomposition.
        
        C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        
        For fixed i, this is a dot product of A[i,:] (length 3) with B[:,k] (length 3).
        There are 2*9 = 18 such dot products, each of length 3.
        
        Total schoolbook rank = 18*3 = 54.
        
        To reduce rank, we need to share multiplications across different (i,k) pairs.
        
        Strassen idea: for 2x2 matrices, (A00 A01; A10 A11) * (B00 B01; B10 B11)
        can be done with 7 products instead of 8.
        
        Let's try to apply this. Group the 9 columns into 4 pairs + 1 single.
        For each pair (k0, k0+1), we compute C[:, k0:k0+2] = A * B[:, k0:k0+2].
        
        A is 2x3, B[:, k0:k0+2] is 3x2.
        
        Let A = [A0 | A1 | A2] where Aj is a 2x1 column vector.
        Let B_block = [B0; B1; B2] where Bj is a 1x2 row vector.
        
        C_block = A0*B0 + A1*B1 + A2*B2 (each term is 2x2 outer product)
        
        This is sum of 3 rank-1 (in the tensor sense) 2x2 matrices... 
        Actually each Aj*Bj is a 2x2 matrix, and we're summing 3 of them.
        
        The tensor for this sub-problem: T_sub[a', b', c'] where
        a' = i*1 + 0 (just i, since A is 2x1 in each column) -- no, this isn't quite right.
        
        Let me just try to find a good decomposition empirically.
        
        For now, let's use the schoolbook rank 54 and try to reduce it.
        """
        return make_schoolbook()

    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        R = U.shape[1]
        # Check all valid triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        # Check some invalid triples should be zero
        import random
        rng = random.Random(42)
        for _ in range(100):
            a = rng.randint(0, nm - 1)
            b = rng.randint(0, mp - 1)
            c = rng.randint(0, pn - 1)
            # Check if valid
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            if i_a == i_c and j_a == j_b and k_b == k_c:
                continue
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val) > 1e-6:
                return False
        return True

    def try_remove_columns(U, V, W, max_rounds=10, time_deadline=None):
        """Try to remove columns one at a time."""
        R = U.shape[1]
        for _ in range(max_rounds):
            if time_deadline and time.time() > time_deadline:
                break
            removed = False
            for col in range(R):
                if time_deadline and time.time() > time_deadline:
                    break
                mask = [i for i in range(R) if i != col]
                if not mask:
                    continue
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                if verify(U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    R = U.shape[1]
                    removed = True
                    break
            if not removed:
                break
        return U, V, W

    def try_remove_pairs(U, V, W, max_rounds=5, time_deadline=None):
        """Try to remove two columns at once."""
        R = U.shape[1]
        for _ in range(max_rounds):
            if time_deadline and time.time() > time_deadline:
                break
            removed = False
            for c1, c2 in combinations(range(R), 2):
                if time_deadline and time.time() > time_deadline:
                    break
                mask = [i for i in range(R) if i != c1 and i != c2]
                if len(mask) < R - 2:
                    continue
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                if verify(U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    R = U.shape[1]
                    removed = True
                    break
            if not removed:
                break
        return U, V, W

    # Start with schoolbook
    U, V, W = make_schoolbook()
    best_rank = U.shape[1]

    # Try column removal
    U, V, W = try_remove_columns(U, V, W, max_rounds=10, time_deadline=deadline)
    best_rank = min(best_rank, U.shape[1])

    # Try pair removal
    U, V, W = try_remove_pairs(U, V, W, max_rounds=3, time_deadline=deadline)
    best_rank = min(best_rank, U.shape[1])

    # Final verification
    if not verify(U, V, W):
        # Fallback to schoolbook
        U, V, W = make_schoolbook()
        best_rank = 54

    # Convert to lists
    def to_list(matrix):
        result = []
        for row in matrix:
            row_list = []
            for val in row:
                r = round(val * 2) / 2
                if abs(r - round(r)) < 1e-9:
                    r = round(r)
                row_list.append(r)
            result.append(row_list)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
