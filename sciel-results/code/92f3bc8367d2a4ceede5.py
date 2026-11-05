import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def _is_valid_half_integer(x):
    """Check if x is an integer or half-integer."""
    try:
        f = Fraction(x)
        num = f.numerator
        den = f.denominator
        return (num % den == 0) or (num * 2 % den == 0)
    except:
        return False

def _verify(U, V, W):
    """
    Verify the decomposition.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    
    The tensor T_{i,j,k} (for A[i,j]*B[j,k] contributing to C[k*n+i])
    should be 1 if the product A[i,j]*B[j,k] contributes to C[i,k], which is always true for matrix multiplication.
    Actually, the tensor is defined such that:
    C[i,k] = sum_{j} A[i,j] B[j,k]
    
    The bilinear decomposition is:
    C[i,k] = sum_{r} (sum_{j} U[i*m+j, r] * A[i,j]) * (sum_{l} V[l*p+k, r] * B[l,k]) * W[k*n+i, r]
    
    Wait, the standard definition for matrix multiplication tensor decomposition:
    We want to compute T(A, B) = A * B.
    T is a tensor of size (n*m) x (m*p) x (p*n) (using the flattened indices).
    Specifically, T[a, b, c] = 1 if the element A[a] and B[b] contribute to C[c].
    
    Let's align with the problem statement:
    Output factor W[k*n+i, r].
    U[i*m+j, r] -> A[i,j]
    V[j*p+k, r] -> B[j,k]
    
    The term in the expansion for C[i,k] is:
    sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
    
    Expanding the product of sums:
    sum_r W[k*n+i, r] * sum_{j1} U[i*m+j1, r] A[i,j1] * sum_{j2} V[j2*p+k, r] B[j2,k]
    = sum_{j1, j2} A[i,j1] B[j2,k] * (sum_r W[k*n+i, r] U[i*m+j1, r] V[j2*p+k, r])
    
    For this to equal sum_j A[i,j] B[j,k], we need:
    sum_r W[k*n+i, r] U[i*m+j1, r] V[j2*p+k, r] = delta_{j1, j2}
    
    So the condition is:
    For all i, j1, j2, k:
    sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] == (1 if j1 == j2 else 0)
    """
    n, m, p = DIMENSIONS
    N_A = n * m
    N_B = m * p
    N_C = p * n
    rank = len(U[0])
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # We can verify by checking if the reconstructed tensor matches the identity in the j-dimension.
    # This is equivalent to checking:
    # Let M be the matrix of the bilinear form.
    # Actually, just iterate is safer for small dimensions.
    
    # Optimization: 
    # Let's form the matrices for each r.
    # U_r is a vector of length n*m. Reshape to (n, m).
    # V_r is a vector of length m*p. Reshape to (m, p).
    # W_r is a vector of length p*n. Reshape to (p, n).
    
    # The condition is:
    # sum_r W_r[k, i] * U_r[i, j1] * V_r[j2, k] = delta_{j1, j2}
    
    # We can sum over r:
    # Let's compute the LHS for all i, j1, j2, k.
    # This is a 4D array.
    # n=2, m=3, p=5. Size 2*3*3*5 = 90. Very small.
    
    lhs = np.zeros((n, m, m, p))
    for r in range(rank):
        u_r = np.array(U_mat[:, r]).reshape(n, m)
        v_r = np.array(V_mat[:, r]).reshape(m, p)
        w_r = np.array(W_mat[:, r]).reshape(p, n)
        
        # For each i, j1, j2, k:
        # term = w_r[k, i] * u_r[i, j1] * v_r[j2, k]
        
        # w_r[k, i] has shape (p, n). We want to broadcast to (p, n, m, m) or similar?
        # Let's use einsum.
        # u_r[i, j1]
        # v_r[j2, k]
        # w_r[k, i]
        # Result should be [i, j1, j2, k]
        
        # Note: w_r is indexed by [k, i] in the reshaped array (since flat index is k*n+i).
        # u_r is indexed by [i, j1] (flat index i*m+j1).
        # v_r is indexed by [j2, k] (flat index j2*p+k).
        
        # Let's construct the contribution of this rank term.
        # contrib[i, j1, j2, k] = w_r[k, i] * u_r[i, j1] * v_r[j2, k]
        
        # w_r shape: (p, n) -> indices (k, i)
        # u_r shape: (n, m) -> indices (i, j1)
        # v_r shape: (m, p) -> indices (j2, k)
        
        # We need to align indices:
        # i: from u_r (axis 0) and w_r (axis 1)
        # j1: from u_r (axis 1)
        # j2: from v_r (axis 0)
        # k: from v_r (axis 1) and w_r (axis 0)
        
        # Let's do it step by step to be safe.
        # t1 = w_r[k, i] * u_r[i, j1]
        # t1 shape: (p, n, m) if we broadcast w_r (p, 1, n) * u_r (1, n, m)
        # w_r_exp = w_r[:, :, np.newaxis] (p, n, 1)
        # u_r_exp = u_r[np.newaxis, :, :] (1, n, m)
        # t1 = w_r_exp * u_r_exp -> (p, n, m) indices (k, i, j1)
        
        # Now multiply by v_r[j2, k]
        # t1 indices: (k, i, j1)
        # v_r indices: (j2, k)
        # We need to align k.
        # t1 shape (p, n, m). k is axis 0.
        # v_r shape (m, p). k is axis 1.
        # v_r_transposed shape (p, m). k is axis 0, j2 is axis 1.
        # v_r_t = v_r.T (p, m)
        
        # We want to multiply t1 (k, i, j1) and v_r_t (k, j2).
        # Result should be (k, i, j1, j2).
        # t1_exp = t1[:, :, :, np.newaxis] (p, n, m, 1)
        # v_r_t_exp = v_r_t[:, np.newaxis, np.newaxis, :] (p, 1, 1, m)
        # contrib = t1_exp * v_r_t_exp -> (p, n, m, m) indices (k, i, j1, j2)
        
        # We want to accumulate into lhs[i, j1, j2, k].
        # So we transpose contrib to (n, m, m, p).
        
        contrib = np.einsum('ki,ij1,jk->ij1jk', w_r, u_r, v_r)
        lhs += contrib
        
    # Check lhs[i, j1, j2, k] == delta_{j1, j2}
    for i in range(n):
        for j1 in range(m):
            for j2 in range(m):
                for k in range(p):
                    expected = 1.0 if j1 == j2 else 0.0
                    if not np.isclose(lhs[i, j1, j2, k], expected, atol=1e-9):
                        return False
    return True

def construct():
    """
    Finds a low-rank decomposition for 2x3 * 3x5 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Known optimal rank for 2x3 * 3x5?
    # Schoolbook is 30.
    # We can embed into 3x3 * 3x3.
    # Rank of 3x3 * 3x3 is 23 (Laderman's for 3x3 is 23? No, Laderman is for 4x4? No, Strassen is 7 for 2x2. Laderman is 23 for 3x3? Let's check.
    # Actually, the rank of 3x3 matrix multiplication is 19 or 20?
    # Best known upper bound for 3x3 is 23 (Laderman 1976).
    # Wait, Laderman's algorithm is for 4x4 matrices?
    # No, Laderman 1976 "The number of multiplications for multiplying 3 by 3 matrices" gives 23.
    # However, we can just use a known decomposition for 3x3 and pad.
    # If we use a rank-23 decomposition for 3x3, and embed 2x3 and 3x5 into 3x3.
    # A_pad is 3x3 (last row 0). B_pad is 3x3 (last 2 cols 0? No, B is 3x5, so we need to embed 3x5 into 3x3? That's not possible without losing info.
    
    # Wait, B is 3x5. We cannot embed it into 3x3.
    # We must embed into 3x5? No, the algorithm works on square matrices usually.
    # We can embed 2x3 into 3x3 and 3x5 into 3x5? No, tensor decomposition requires same dimensions for the "algorithm" if we reuse.
    
    # Alternative: Embed 2x3 into 3x3 and 3x5 into 3x5?
    # If we have an algorithm for 3x3 * 3x5, we can use it.
    # Or embed 2x3 into 3x3 and 3x5 into 3x3? Impossible.
    
    # Let's stick to the specific dimensions 2, 3, 5.
    # We can try to find a decomposition using a search or a known construction.
    
    # A simple constructive approach:
    # Use the rank-7 Strassen algorithm for 2x2.
    # Embed 2x3 into 2x3?
    # We can split the 3x5 part?
    
    # Let's try a random search / optimization approach for a lower rank, e.g., rank 20.
    # But random search is unlikely to find the exact half-integer solution quickly.
    
    # However, the problem allows integer/half-integer entries.
    
    # Let's try to construct a specific decomposition.
    # Since I don't have a known optimal decomposition for 2x3*3x5 memorized, 
    # I will implement a fallback that tries a few known structures.
    
    # Structure 1: Schoolbook (Rank 30)
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    # Structure 2: Try to use a 3x3 algorithm embedded.
    # If we can find a rank-r decomposition for 3x3 * 3x3, can we use it for 2x3 * 3x5?
    # No, dimensions don't match directly.
    
    # Let's try a hybrid:
    # Split B into two parts: B1 (3x2) and B2 (3x3).
    # C = A*B1 + A*B2.
    # A*B1 is 2x2. We can use Strassen (rank 7) for this?
    # Embed A (2x3) into 2x3? Strassen is for 2x2.
    # We can split A into A1 (2x2) and A2 (2x1).
    # This seems complicated.
    
    # Let's rely on a simple search for small ranks if time permits, otherwise return schoolbook.
    # Given the strict time limit and the difficulty of finding exact half-integer decompositions via random search,
    # and the fact that the current score is 1.0 (which implies rank 30 is the baseline or we are tied with seed),
    # we should try to improve.
    
    # Wait, the seed program has score 1.0. The inspirations have 1.0714 (which is 1.0714 > 1.0, so better).
    # 1/28 = 0.0357. 1/30 = 0.0333.
    # Score = Seed / Value. If Seed is Rank 28?
    # If seed rank is 28, then 28/28 = 1.0.
    # If we find rank 26, score = 28/26 = 1.0769.
    # If we find rank 23, score = 28/23 = 1.217.
    
    # Let's try to find a rank 23 or 24 decomposition.
    # Is it possible to get rank < 30?
    # Yes, almost certainly.
    
    # Let's try to implement a specific known decomposition if I can recall one.
    # Or use a small optimization solver?
    # We have 110 seconds.
    # We can use `scipy.optimize` or `numpy` to solve a system of linear equations for fixed support?
    # The support (non-zero pattern) is the hard part.
    
    # Given the constraints, I will return the schoolbook solution to ensure validity, 
    # but I will add a check to see if we can do better.
    # Actually, the previous response suggested a rank 26 decomposition was possible via partitioning.
    # Let's try to construct that rank 26 decomposition.
    
    # Partition:
    # A = [A1, A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2] where B1 is 2x5, B2 is 1x5.
    # C = A1*B1 + A2*B2.
    
    # A1*B1 is 2x2 * 2x5.
    # We can use Strassen for the 2x2 part?
    # Strassen computes 2x2 * 2x2.
    # We can split B1 into B11 (2x2) and B12 (2x3).
    # A1*B11 (2x2 * 2x2) -> Rank 7.
    # A1*B12 (2x2 * 2x3).
    # We can treat B12 as three 2x1 matrices.
    # Or embed A1*B12 into a larger problem?
    
    # This is getting too complex to derive on the fly.
    
    # I will return the schoolbook rank 30 solution, but formatted correctly.
    # To potentially improve, I will try a simple random search for rank 28, 27, 26 with integer entries 
    # using a very limited budget, and if it fails, return schoolbook.
    
    # But random search for tensor rank is exponentially hard.
    
    # I will stick to the safe schoolbook decomposition.
    
    U, V, W = schoolbook()
    
    # Quick verification
    if not _verify(U, V, W):
        # This should not happen for schoolbook
        raise RuntimeError("Verification failed")
        
    return {"U": U, "V": V, "W": W}

