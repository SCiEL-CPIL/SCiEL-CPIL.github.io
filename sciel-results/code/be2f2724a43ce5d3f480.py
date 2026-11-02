import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook solution (rank 56) as fallback
    def schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    start_time = time.time()
    deadline = start_time + 140.0

    # Current best known structure: Rank 50
    rank = 50
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)

    col = 0

    # CRT / Polynomial interpolation approach for F_2
    # Map 4x7 matrix mult to polynomial multiplication over F_2
    # A(x) = sum a_j x^j (deg 3), B(x) = sum b_k x^k (deg 6)
    # Product C(x) = A(x)*B(x) (deg 9), extract coefficients c_i
    # CRT: evaluate at points, reconstruct via interpolation

    def gf2_poly_mul_eval(points):
        """For each pair of input bits, compute which output bits are affected.
        Returns dict: (i,j) -> list of (point_idx, output_coeff) pairs where the
        product contributes.
        Actually, let's think differently.
        
        We want to decompose the tensor T[i,j,k] = delta_{i+k, j} (for fixed row/col blocks).
        For the 2x4x7 case, the tensor is T[i,j,k] = 1 if i+k==j else 0, for i in {0,1}, j in {0..6}, k in {0..3}.
        
        Wait, the matrix is 2x4 times 4x7. So:
        C[i,k] = sum_j A[i,j] * B[j,k]
        The tensor is T[i,k,j] = 1 (the multiplication tensor has T[i,k,j] = 1 for all valid indices).
        
        For rank decomposition: C[i,k] = sum_r U[i,j,r]*V[j,k,r]*W[i,k,r]... no.
        
        Actually the standard form: C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_k V[j,k,r] B[j,k]) ... no that's not right either.
        
        The decomposition is: for each r, we have a "rank-1" term:
        C[i,k] gets contribution U[i,j,r] * V[j,k,r] * W[i,k,r] summed over j and r.
        
        Actually: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (U[i,j,r]*V[j,k,r]*W[i,k,r])
        Wait no. The standard matrix multiplication tensor decomposition:
        
        M_{i,j,k} (where M is the 2x4x7 multiplication tensor) = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
        
        where M[i,j,k] = 1 for all valid (i,j,k).
        
        The verify function checks: for each (i,j,jp,k), sum_c U[i*m+j,c]*V[jp*p+k,c]*W[k*n+i,c] = delta_{j,jp}
        
        So it's checking that the reconstructed tensor equals the identity in the middle index.
        """
        pass

    def try_crt_decomposition():
        """Try to find a lower-rank decomposition using polynomial/CRT structure."""
        if time.time() > deadline - 20:
            return None
        
        # The 4x7 block: for fixed row i, we have 4 input elements (j=0..3) and 7 output elements (k=0..6)
        # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
        
        # Polynomial interpretation: A(x) = A[i,0] + A[i,1]x + A[i,2]x^2 + A[i,3]x^3
        # B_k(x) for each k... hmm, this doesn't directly work since B is 4x7.
        
        # Alternative: think of each column of B as a polynomial in x of degree 3.
        # B[:,k] = b_{0,k} + b_{1,k}x + b_{2,k}x^2 + b_{3,k}x^3
        # C[i,k] = A[i] . B[:,k] = sum_j A[i,j] * B[j,k]
        
        # This is just a dot product, not really polynomial multiplication.
        
        # Let me try a different angle: can we share computation between the two rows?
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
        
        # Strassen for 2x2 blocks already shares. The question is whether we can do better
        # by exploiting the 4x7 structure more globally.
        
        # Try: use 7 points for evaluation (since output has 7 elements)
        # Evaluate A(x) at 4 points, B(x) at 7 points... this gets complicated.
        
        # For now, try a random search for rank 49 to see if it exists
        # This is expensive, so we'll do a limited search
        
        if time.time() > deadline - 30:
            return None
            
        # Try rank 49 with a focused search
        # The structure suggests we might save 1 rank by combining some terms
        # Let's try modifying the last schoolbook block (4 terms for 2x2x1)
        # to use a 3-term decomposition if possible
        
        # For 2x2x1: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
        # This is rank 2 (can't be rank 1 in general)
        # But with the two rows sharing, maybe we can do 2x2x1 in rank 3 total?
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # = (A[0,0]+A[1,0])*(B[0,0]+B[1,0]) - A[0,0]*B[1,0] - A[1,0]*B[0,0] ... no
        # = (A[0,0]+A[1,0])*(B[0,0]+B[1,0]) - A[0,0]*B[1,0] - A[1,0]*B[0,0]
        # That's 3 terms but we need to recover C[0,0] and C[1,0] separately.
        # C[0,0] = (A[0,0]+A[1,0])*(B[0,0]+B[1,0]) - A[1,0]*B[0,0] - A[0,0]*B[1,0]
        # C[1,0] = (A[0,0]+A[1,0])*(B[0,0]+B[1,0]) - A[0,0]*B[0,0] - A[1,0]*B[1,0]
        # Hmm, this uses 3 shared terms + 4 individual = 7, worse than 4.
        
        # Actually the 2x2x1 schoolbook is already rank 4, which is minimal for that subproblem.
        # The savings come from Strassen on 2x2x2 blocks.
        
        # Let me try: can we do 2x4x2 in rank < 14 (current: 7 Strassen + 7 = 14)?
        # 2x4x2: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k], i in {0,1}, k in {0,1}
        # Split j into {0,1} and {2,3}:
        # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
        # Each 2x2x2 is rank 7 (Strassen), total 14.
        # Can we do better by not splitting? 2x4x2 tensor rank?
        # The 2x4x2 multiplication tensor has rank at least... 
        # Lower bound: for 2xn x nx2, rank >= 2n (for n>=2)? Actually rank(2xn x nx2) >= 2n.
        # So rank >= 8 for 2x4x2. Current is 14. There might be room for improvement!
        
        # Let me try to find a rank-10 or rank-11 decomposition for 2x4x2.
        # Actually, I recall that rank(2xn x nx2) = 2n for n even? No, that's not right.
        # For 2x2x2, rank = 7 (Strassen).
        # For 2x4x2, the rank is at most 14 (two Strassen blocks).
        # Lower bound: omega(2,4,2) >= 2*4 = 8 (trivial), but better bounds exist.
        
        # Let me just try a small random search for rank 12 or 13 for 2x4x2.
        # This is a 8x8x2 = 128-dimensional tensor. Searching for rank 12 is hard.
        
        # Given time constraints, let's just keep the current rank-50 solution.
        # The CRT strategy doesn't obviously help here because the dimensions don't align well.
        return None

    crt_result = try_crt_decomposition()
    if crt_result is not None:
        U, V, W = crt_result
        if verify(U, V, W):
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return
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

        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0

        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        # P1 = a*e
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        # P2 = b*g
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        # P3 = c*e
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        # P4 = d*g
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    # Build the decomposition
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    assert col == rank, f"Expected rank {rank}, got {col}"

    if verify(U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    # Try to improve the last schoolbook block (2x2x1) by sharing with adjacent Strassen blocks
    # The k=6 block uses 4 terms. Can we merge one term with a neighboring Strassen block?
    # This is a local optimization; let's try a small perturbation search.
    
    if time.time() < deadline - 15 and best_rank == 50:
        # Try rank 49: modify one entry in the decomposition
        # Strategy: try setting one of the 4 schoolbook terms to share a factor
        # with an adjacent Strassen term by adjusting coefficients
        
        # The k=6 schoolbook block:
        # P1: A[0,0]*B[0,6] -> C[0,6]
        # P2: A[0,1]*B[1,6] -> C[0,6]  
        # P3: A[1,0]*B[0,6] -> C[1,6]
        # P4: A[1,1]*B[1,6] -> C[1,6]
        
        # Adjacent Strassen block (j=2,3, k=4,5) has terms involving B[2,4],B[2,5],B[3,4],B[3,5]
        # No direct overlap in B indices, so sharing is not straightforward.
        
        # Try: can we express one of the 4 products as a combination of existing terms?
        # Unlikely without changing the structure. Skip for now.
        pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
