"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product as iproduct

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    def time_left():
        return deadline - time.time()

    # Build target tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_rec - T)
        return err < 1e-9

    def make_strassen_42():
        """Create the standard Strassen-42 decomposition."""
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # Strassen 2x2x2 factors for local indices
        prods = [
            # M1: (a00+a11)(b00+b11) -> c00+=1, c11+=1
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
            # M2: (a10+a11)b00 -> c10+=1
            ([(2,1), (3,1)], [(0,1)], [(2,1)]),
            # M3: a00(b01-b11) -> c01+=1
            ([(0,1)], [(1,1), (3,-1)], [(1,1)]),
            # M4: a11(b10-b00) -> c00+=1, c10+=1
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
            # M5: (a00+a01)b11 -> c00-=1, c01+=1
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
            # M6: (a10-a00)(b00+b01) -> c11+=1
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
            # M7: (a01-a11)(b10+b11) -> c00-=1, c11+=1
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,-1), (3,1)]),
        ]

        for t in range(6):
            for s in range(7):
                col = 7 * t + s
                u_entries, v_entries, w_entries = prods[s]

                for idx, coeff in u_entries:
                    U[idx, col] = coeff

                for idx, coeff in v_entries:
                    j = idx // 2
                    k_loc = idx % 2
                    k = 2 * t + k_loc
                    V[j * p + k, col] = coeff

                for idx, coeff in w_entries:
                    k_loc = idx // 2
                    i = idx % 2
                    k = 2 * t + k_loc
                    W[k * n + i, col] = coeff

        return U, V, W, rank

    U42, V42, W42, rank42 = make_strassen_42()
    
    # Verify baseline
    if not verify(U42, V42, W42):
        # Fallback to schoolbook
        rank_fb = 48
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a, col] = 1
                    V_fb[b, col] = 1
                    W_fb[c, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_U, best_V, best_W = U42, V42, W42
    best_rank = 42

    # ------------------------------------------------------------------
    # NEW STRATEGY: Decompose 2x2x12 into 3 blocks of 2x2x4.
    # The 12 columns of B and 12 rows of C can be grouped by k mod 3?
    # No, the index k in c=k*n+i and b=j*p+k is linear.
    # However, if we view the 12-length dimension as a tensor product or
    # simply split it into 3 segments of length 4, we have 3 independent
    # 2x2x4 multiplications?
    # Let's check the indices.
    # b = j*12 + k. c = k*2 + i.
    # If we split k into 3 groups of 4: k = 4*t + k', t in {0,1,2}, k' in {0,1,2,3}.
    # Then b = j*12 + 4*t + k'. c = (4*t + k')*2 + i = 8*t + 2*k' + i.
    # The term c depends on t. So the blocks are NOT independent in the standard
    # tensor product sense unless we can factor out the t-dependence.
    #
    # Wait, the tensor T is defined by the multiplication structure.
    # C = A B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Here k is 0..11.
    # If we split B into 3 blocks B_t of width 4, and C into 3 blocks C_t of width 4.
    # C_t[i, k'] = sum_j A[i, j] B_t[j, k'].
    # This IS independent for each t!
    # Because the multiplication C = A B does not mix different k indices.
    # C[i, k] only depends on B[j, k] for the same k.
    # So we have 3 independent 2x2x4 matrix multiplications.
    #
    # We need a rank-10 decomposition for 2x2x4 matrix multiplication.
    # Is there one?
    # Rank of 2x2x4 is at least 2*2*4 / max(2,2,4)? No.
    # Lower bound: ceil(2*2*4 / 4) = 4? No.
    # Known rank for 2x2x4:
    # Schoolbook: 8.
    # Can we do better?
    # Strassen-like for 2x2x2 is 7.
    # For 2x2x4, we can view it as 2 blocks of 2x2x2? Rank 14.
    # Or use the fact that 2x2x4 is equivalent to multiplying a 2x2 matrix by a 2x4 matrix.
    # This is equivalent to 2 independent 2x2x2 multiplications? No, the A is shared.
    #
    # Let's try to find a rank 10 decomposition for 2x2x4.
    # If we can find rank 10, total rank is 30.
    #
    # Let's define the 2x2x4 tensor.
    # n=2, m=2, p=4.
    # a = i*2+j (0..3)
    # b = j*4+k (0..7)
    # c = k*2+i (0..7)
    #
    # We will attempt to construct a rank 10 decomposition using a structured search
    # or a known algebraic identity.
    #
    # Actually, let's just try to find *any* decomposition with rank < 42.
    # We can try rank 36 (6 blocks of 2x2x2 with rank 6? No, 2x2x2 min rank is 7).
    # So 6*7 = 42 is the baseline.
    #
    # If we group into 3 blocks of 2x2x4, and use rank 10 for each, total 30.
    # If we use rank 11, total 33.
    # If we use rank 12, total 36.
    #
    # Let's try to find a rank 10 decomposition for 2x2x4.
    # We can use a randomized search over integers in [-2, 2].
    # ------------------------------------------------------------------

    def find_2x2x4_decomp(rank_target):
        """
        Try to find a rank-rank_target decomposition for 2x2x4.
        Returns U, V, W matrices for the 2x2x4 problem.
        """
        if time_left() < 5:
            return None

        n_sub, m_sub, p_sub = 2, 2, 4
        T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    T_sub[a, b, c] = 1.0

        # Initialize with schoolbook (rank 8) and add 2 random terms
        U = np.zeros((n_sub * m_sub, rank_target))
        V = np.zeros((m_sub * p_sub, rank_target))
        W = np.zeros((p_sub * n_sub, rank_target))

        # Fill first 8 with schoolbook
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 8 # This is a bug in the original, fixing it here for correctness of the seed if used, but we won't use it.
        
        # Correct schoolbook init
        U = np.zeros((n_sub * m_sub, rank_target))
        V = np.zeros((m_sub * p_sub, rank_target))
        W = np.zeros((p_sub * n_sub, rank_target))
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1

        # Add random terms
        np.random.seed(42)
        for r in range(8, rank_target):
            U[:, r] = np.random.randint(-2, 3, size=n_sub * m_sub)
            V[:, r] = np.random.randint(-2, 3, size=m_sub * p_sub)
            W[:, r] = np.random.randint(-2, 3, size=p_sub * n_sub)

        best_err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T_sub)
        
        # Simple local search
        for _ in range(2000):
            if time_left() < 2:
                break
            r = np.random.randint(0, rank_target)
            for delta in [-1, 1]:
                # Perturb U
                idx = np.random.randint(0, n_sub * m_sub)
                orig = U[idx, r]
                U[idx, r] += delta
                err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T_sub)
                if err < best_err:
                    best_err = err
                else:
                    U[idx, r] = orig
                # Perturb V
                idx = np.random.randint(0, m_sub * p_sub)
                orig = V[idx, r]
                V[idx, r] += delta
                err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T_sub)
                if err < best_err:
                    best_err = err
                else:
                    V[idx, r] = orig
                # Perturb W
                idx = np.random.randint(0, p_sub * n_sub)
                orig = W[idx, r]
                W[idx, r] += delta
                err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T_sub)
                if err < best_err:
                    best_err = err
                else:
                    W[idx, r] = orig
            
            if best_err < 1e-9:
                break

        if best_err < 1e-9:
            return U, V, W

        return None

    # Try to find a rank 10 or 11 decomposition for 2x2x4
    for r_sub in [10, 11, 12]:
        if time_left() < 10:
            break
        U_sub, V_sub, W_sub = find_2x2x4_decomp(r_sub)
        if U_sub is not None:
            # Construct full 2x2x12 decomposition
            rank_total = 3 * r_sub
            U_full = np.zeros((n * m, rank_total))
            V_full = np.zeros((m * p, rank_total))
            W_full = np.zeros((p * n, rank_total))
            
            for t in range(3):
                # Map sub-block indices to full indices
                # Sub-block t corresponds to columns 4*t to 4*t+3 of B
                # And rows 4*t to 4*t+3 of C (in terms of k)
                
                for r in range(r_sub):
                    col = t * r_sub + r
                    
                    # U is the same for all blocks (A is 2x2)
                    U_full[:, col] = U_sub[:, r]
                    
                    # V: b = j*p + k. In sub, b_sub = j*4 + k_sub.
                    # k = 4*t + k_sub
                    for j in range(2):
                        for k_sub in range(4):
                            b_sub = j * 4 + k_sub
                            b_full = j * p + (4 * t + k_sub)
                            V_full[b_full, col] = V_sub[b_sub, r]
                            
                    # W: c = k*n + i. In sub, c_sub = k_sub*2 + i.
                    # k = 4*t + k_sub
                    for k_sub in range(4):
                        for i in range(2):
                            c_sub = k_sub * 2 + i
                            c_full = (4 * t + k_sub) * n + i
                            W_full[c_full, col] = W_sub[c_sub, r]
                            
            if verify(U_full, V_full, W_full):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = rank_total
                break

    # If we found a better rank, return it
    if best_rank < 42:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }

    # Otherwise, return the Strassen-42 solution
    return {
        "U": U42.tolist(),
        "V": V42.tolist(),
        "W": W42.tolist()
    }
# EVOLVE-BLOCK-END
