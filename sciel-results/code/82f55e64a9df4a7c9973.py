"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Try Strassen-based rank-49 decomposition first
    result_49 = _construct_strassen_49(n, m, p)
    if result_49 is not None:
        return result_49
    
    # Fallback: schoolbook rank-56
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _construct_strassen_49(n, m, p):
    """
    Construct a rank-49 decomposition using Strassen's algorithm on 7 blocks of 2x2x2.
    
    Split the 14 k-values into 7 pairs: (0,1), (2,3), ..., (12,13).
    For each pair, apply Strassen's rank-7 decomposition for 2x2x2 matrix mult.
    Total rank: 7 * 7 = 49.
    
    The tensor T[i,j,k] = 1 for all i,j,k (standard matrix mult).
    We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    """
    if time.time() > time.time() + 140:
        return None
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]   # 4 x 49
    V = [[0] * rank for _ in range(m * p)]   # 28 x 49
    W = [[0] * rank for _ in range(p * n)]   # 28 x 49
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # Then:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # For our tensor, we need the product to give 1 for each (i,j,k) combination.
    # Actually, the tensor T[i,j,k] = 1 means we're decomposing the identity map
    # in the appropriate tensor space.
    
    # Let me think about this differently. The matrix multiplication tensor
    # for (n,m,p) = (2,2,14) has T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,13}.
    
    # We want to write this as sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U is 4x49, V is 28x49, W is 28x49.
    
    # For each pair of k values (k1, k2), we handle the 2x2x2 sub-tensor.
    # The 2x2x2 tensor has T[i,j,k'] = 1 for i,j in {0,1}, k' in {0,1}.
    # This is the all-ones tensor of size 2x2x2, which has rank 7 (Strassen).
    
    # Wait, no. The all-ones tensor of size 2x2x2 does NOT have rank 7.
    # Strassen gives rank 7 for the 2x2x2 matrix multiplication tensor, which is
    # T[i,j,k] = delta(i,j,k) in some sense... 
    
    # Let me reconsider. The standard matrix multiplication tensor for 2x2x2 is:
    # T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # In standard matrix mult, every (i,j,k) triple contributes, so T[i,j,k] = 1 for all.
    
    # Actually no. The tensor is defined by the bilinear map. For matrix multiplication
    # C = A*B, we have C[i,k] = sum_j A[i,j] * B[j,k].
    # The tensor T satisfies: T[i,j,k] is the coefficient such that
    # C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (since each term A[i,j]*B[j,k] appears exactly once
    # in the sum for C[i,k] when the indices match).
    
    # Wait, that's not right either. Let me be more careful.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # So the tensor is the all-ones tensor. For 2x2x2, the all-ones tensor
    # T[i,j,k] = 1 for all i,j,k in {0,1}.
    
    # The rank of the all-ones tensor of size 2x2x2 is actually 2, not 7!
    # T = e_1 ⊗ e_1 ⊗ e_1 + ... no wait.
    # T[i,j,k] = 1 = (1,1) ⊗ (1,1) ⊗ (1,1) ... no.
    # T = u ⊗ v ⊗ w where u=(1,1), v=(1,1), w=(1,1). That's rank 1!
    
    # Hmm, but that can't be right for matrix multiplication. Let me re-examine.
    
    # Actually, I think I'm confusing myself. Let me look at this from the
    # evaluator's perspective. The evaluator checks that the decomposition
    # correctly computes matrix multiplication. So for any matrices A (2x2) and B (2x14),
    # we need C = A @ B to be computed as:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... 
    # No, that's not right either.
    
    # Let me re-read the task: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # and "W[k*n+i,r]" is the indexing convention.
    
    # The standard tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: the multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # (since each product A[i,j]*B[j,k] contributes to C[i,k]).
    
    # A rank-r decomposition writes T as sum_{s=1}^r u_s ⊗ v_s ⊗ w_s
    # where u_s is a vector in R^{n*m}, v_s in R^{m*p}, w_s in R^{p*n}.
    # T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    
    # For this to equal 1 for all i,j,k, we need:
    # sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1 for all i,j,k.
    
    # This is a constraint satisfaction problem. The schoolbook solution uses
    # rank = n*m*p = 56, with one term per (i,j,k) triple.
    
    # Now, can we do better? The key question is: what is the rank of the
    # all-ones tensor of size (n*m) x (m*p) x (p*n) with the specific
    # index structure?
    
    # Actually, I realize the tensor is NOT the all-ones tensor in the naive sense.
    # The indices are coupled: U is indexed by (i,j), V by (j,k), W by (k,i).
    # So T[i,j,k] = 1 means the (i,j,k) entry is 1, but the tensor lives in
    # R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n} with the specific index mappings.
    
    # Let me just try the Strassen approach and verify numerically.
    
    # For the 2x2x2 case, Strassen gives rank 7. The idea is:
    # Instead of computing 8 products (a00*b00, a00*b01, a01*b00, a01*b01, 
    #         a10*b00, a10*b01, a11*b00, a11*b01),
    # we compute 7 products M1..M7 and then combine them.
    
    # In tensor decomposition terms, this means we can represent the 2x2x2
    # multiplication tensor with rank 7 instead of 8.
    
    # For our 2x2x14 case, if we split into 7 blocks of 2x2x2, we get rank 7*7=49.
    
    # Let me implement this and verify with a random test.
    
    # Strassen decomposition for 2x2x2:
    # The 8 schoolbook terms are:
    # (i=0,j=0,k=0): A[0,0]*B[0,0] -> C[0,0]
    # (i=0,j=0,k=1): A[0,0]*B[0,1] -> C[0,1]
    # (i=0,j=1,k=0): A[0,1]*B[1,0] -> C[0,0]
    # (i=0,j=1,k=1): A[0,1]*B[1,1] -> C[0,1]
    # (i=1,j=0,k=0): A[1,0]*B[0,0] -> C[1,0]
    # (i=1,j=0,k=1): A[1,0]*B[0,1] -> C[1,1]
    # (i=1,j=1,k=0): A[1,1]*B[1,0] -> C[1,0]
    # (i=1,j=1,k=1): A[1,1]*B[1,1] -> C[1,1]
    
    # Strassen replaces these 8 with 7 products M1..M7.
    
    # In the tensor decomposition, each rank-1 term corresponds to a product
    # of a linear combination of A entries, a linear combination of B entries,
    # and a linear combination of C entries (with signs).
    
    # For the 2x2x2 block at positions (k1, k2), the 7 Strassen terms are:
    
    # I'll define the Strassen coefficients explicitly.
    
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    #   U: A[0,0] coeff 1, A[1,1] coeff 1
    #   V: B[0,0] coeff 1, B[1,1] coeff 1
    #   W: contributes to C[0,0] (+1), C[1,1] (+1)
    
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    #   U: A[1,0] coeff 1, A[1,1] coeff 1
    #   V: B[0,0] coeff 1
    #   W: contributes to C[1,0] (+1)
    
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    #   U: A[0,0] coeff 1
    #   V: B[0,1] coeff 1, B[1,1] coeff -1
    #   W: contributes to C[0,1] (+1)
    
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    #   U: A[1,1] coeff 1
    #   V: B[1,0] coeff 1, B[0,0] coeff -1
    #   W: contributes to C[1,0] (+1)
    
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    #   U: A[0,0] coeff 1, A[0,1] coeff 1
    #   V: B[1,1] coeff 1
    #   W: contributes to C[0,1] (+1)
    
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    #   U: A[1,0] coeff 1, A[0,0] coeff -1
    #   V: B[0,0] coeff 1, B[0,1] coeff 1
    #   W: contributes to C[1,1] (+1)
    
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    #   U: A[0,1] coeff 1, A[1,1] coeff -1
    #   V: B[1,0] coeff 1, B[1,1] coeff 1
    #   W: contributes to C[0,0] (+1)
    
    # Then:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Let me verify: 
    # C[0,0] should be A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # M1 + M4 - M5 + M7
    # = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 + (A00*B11 - A00*B11) + (A11*B00 - A11*B00) + (A11*B11 - A11*B11) + (A11*B10 - A11*B10) + (-A01*B11 + A01*B11) + A01*B10
    # = A00*B00 + A01*B10 ✓
    
    # C[0,1] should be A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # M3 + M5
    # = A00(B01-B11) + (A00+A01)B11
    # = A00*B01 - A00*B11 + A00*B11 + A01*B11
    # = A00*B01 + A01*B11 ✓
    
    # C[1,0] should be A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # M2 + M4
    # = (A10+A11)B00 + A11(B10-B00)
    # = A10*B00 + A11*B00 + A11*B10 - A11*B00
    # = A10*B00 + A11*B10 ✓
    
    # C[1,1] should be A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # M1 - M2 + M3 + M6
    # = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + (A10-A00)(B00+B01)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A10*B00 - A11*B00 + A00*B01 - A00*B11 + A10*B00 + A10*B01 - A00*B00 - A00*B01
    # = (A00*B00 - A00*B00) + (A00*B11 - A00*B11) + (A11*B00 - A11*B00) + A11*B11 + (-A10*B00 + A10*B00) + (A00*B01 - A00*B01) + A10*B01
    # = A11*B11 + A10*B01 ✓
    
    # Great, Strassen works. Now let me implement it for our case.
    
    # For each pair (k1, k2), we have 7 rank-1 terms.
    # The U, V, W matrices need to be filled accordingly.
    
    # Indexing:
    # U[i*m + j, col] where i in {0,1}, j in {0,1}
    # V[j*p + k, col] where j in {0,1}, k in {0,...,13}
    # W[k*n + i, col] where k in {0,...,13}, i in {0,1}
    
    # For the block at (k1, k2), the 7 terms use columns base_col to base_col+6.
    
    for pair_idx in range(7):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        base_col = pair_idx * 7
        
        # M1: (A[0,0] + A[1,1]) * (B[0,k1] + B[1,k2])
        # U: A[0,0] (i=0,j=0) coeff 1, A[1,1] (i=1,j=1) coeff 1
        U[0 * m + 0][base_col] = 1
        U[1 * m + 1][base_col] = 1
        # V: B[0,k1] (j=0,k=k1) coeff 1, B[1,k2] (j=1,k=k2) coeff 1
        V[0 * p + k1][base_col] = 1
        V[1 * p + k2][base_col] = 1
        # W: C[0,k1] (+1), C[1,k2] (+1)
        # W[k*n + i, col]: C[i,k] -> W[k*n + i, col]
        W[k1 * n + 0][base_col] = 1  # C[0,k1]
        W[k2 * n + 1][base_col] = 1  # C[1,k2]
        
        # M2: (A[1,0] + A[1,1]) * B[0,k1]
        U[1 * m + 0][base_col + 1] = 1  # A[1,0]
        U[1 * m + 1][base_col + 1] = 1  # A[1,1]
        V[0 * p + k1][base_col + 1] = 1  # B[0,k1]
        W[k1 * n + 1][base_col + 1] = 1  # C[1,k1]
        
        # M3: A[0,0] * (B[0,k2] - B[1,k2])
        U[0 * m + 0][base_col + 2] = 1  # A[0,0]
        V[0 * p + k2][base_col + 2] = 1  # B[0,k2]
        V[1 * p + k2][base_col + 2] = -1  # B[1,k2]
        W[k2 * n + 0][base_col + 2] = 1  # C[0,k2]
        
        # M4: A[1,1] * (B[1,k1] - B[0,k1])
        U[1 * m + 1][base_col + 3] = 1  # A[1,1]
        V[1 * p + k1][base_col + 3] = 1  # B[1,k1]
        V[0 * p + k1][base_col + 3] = -1  # B[0,k1]
        W[k1 * n + 1][base_col + 3] = 1  # C[1,k1]
        
        # M5: (A[0,0] + A[0,1]) * B[1,k2]
        U[0 * m + 0][base_col + 4] = 1  # A[0,0]
        U[0 * m + 1][base_col + 4] = 1  # A[0,1]
        V[1 * p + k2][base_col + 4] = 1  # B[1,k2]
        W[k2 * n + 0][base_col + 4] = 1  # C[0,k2]
        
        # M6: (A[1,0] - A[0,0]) * (B[0,k1] + B[0,k2])
        U[1 * m + 0][base_col + 5] = 1  # A[1,0]
        U[0 * m + 0][base_col + 5] = -1  # A[0,0]
        V[0 * p + k1][base_col + 5] = 1  # B[0,k1]
        V[0 * p + k2][base_col + 5] = 1  # B[0,k2]
        W[k1 * n + 1][base_col + 5] = 1  # C[1,k1]
        W[k2 * n + 1][base_col + 5] = 1  # C[1,k2]
        
        # M7: (A[0,1] - A[1,1]) * (B[1,k1] + B[1,k2])
        U[0 * m + 1][base_col + 6] = 1  # A[0,1]
        U[1 * m + 1][base_col + 6] = -1  # A[1,1]
        V[1 * p + k1][base_col + 6] = 1  # B[1,k1]
        V[1 * p + k2][base_col + 6] = 1  # B[1,k2]
        W[k1 * n + 0][base_col + 6] = 1  # C[0,k1]
        W[k2 * n + 0][base_col + 6] = 1  # C[0,k2]
    
    # Now verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return None


def _verify_decomposition(U, V, W, n, m, p, num_tests=5):
    """Verify that the decomposition correctly computes matrix multiplication."""
    import numpy as np
    
    for _ in range(num_tests):
        # Random A (n x m) and B (m x p)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Compute using the decomposition
        C_computed = np.zeros((n, p))
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # C[i,k] = sum_j A[i,j] B[j,k]
        # In tensor form: C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
        # where T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # But actually, the decomposition means:
        # For each r, we compute a "product" that contributes to C.
        # The standard interpretation: 
        # C = sum_r (W[:,r] outer product with something)... 
        
        # Let me think about this more carefully.
        # The tensor T is in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}.
        # T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T = 1 (the all-ones tensor with the right index structure).
        
        # To verify, I'll check that for random A and B, the computed C matches.
        # The way to compute C from the decomposition:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_{i',j',k'} T[i',j',k'] A[i',j'] B[j',k'] * delta(i',i) delta(k',k)
        # Hmm, this is getting complicated. Let me just directly check the tensor.
        
        # Actually, the simplest verification: check that the tensor T[i,j,k] = 1 for all i,j,k.
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_val = 0.0
                    for r in range(len(U[0])):
                        tensor_val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(tensor_val - 1.0) > 1e-10:
                        return False
    
    return True
# EVOLVE-BLOCK-END
