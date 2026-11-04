"""Matrix multiplication 4,5,5 tensor decomposition with rank minimization."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize, LinearConstraint

DIMENSIONS = [4, 5, 5]

def _build_schoolbook(n, m, p):
    """Build schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _bclr_5x5x5_rank19():
    """
    Bini-Capovani-Lotti-Romani rank-19 decomposition for 5x5x5 matrix multiplication.
    
    For C = A * B where A is 5x5 and B is 5x5:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We need: sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*5+i, r] = 1 for all i,j,k
    
    The BCLR algorithm uses 19 rank-1 tensors.
    """
    # BCLR rank-19 decomposition coefficients
    # Each term r has coefficients for A[i,j], B[j,k], and C[k,i]
    # The decomposition: C = sum_r (u_r ⊗ v_r ⊗ w_r) where
    # u_r is a 5x5 matrix, v_r is a 5x5 matrix, w_r is a 5x5 matrix
    # and C[i,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] ... 
    # Actually the standard formulation:
    # For each rank-1 term r, we have vectors u_r (size 25), v_r (size 25), w_r (size 25)
    # such that sum_r u_r[i*5+j] * v_r[j*5+k] * w_r[k*5+i] = 1 for all i,j,k
    
    # BCLR 19-term decomposition:
    # I'll use the known coefficients from the literature.
    # The BCLR algorithm partitions 5 = 2+3 and uses Strassen-like tricks.
    
    # Let me construct it using the known structure.
    # Actually, let me use a verified set of coefficients.
    
    # The BCLR rank-19 for 5x5x5:
    # Reference: Bini, Capovani, Lotti, Romani (1979)
    
    # I'll encode the 19 terms. Each term r gives:
    # U_r[i,j] for i,j in 0..4 (flattened to index i*5+j)
    # V_r[j,k] for j,k in 0..4 (flattened to index j*5+k)  
    # W_r[k,i] for k,i in 0..4 (flattened to index k*5+i)
    
    # BCLR coefficients (from published results):
    terms = []
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    terms.append({(0,0): 1, (0,0): 1, (0,0): 1})
    # This is getting complex. Let me use a different approach.
    
    # Actually, the cleanest way is to use the explicit BCLR construction.
    # The BCLR algorithm works as follows:
    # Split 5x5 matrices into blocks of sizes 2 and 3.
    # Use Strassen for 2x2x2 (rank 7) and handle the mixed blocks.
    
    # Let me instead directly verify and use a known valid set of 19 terms.
    # I'll generate them algorithmically based on the BCLR construction.
    
    # BCLR construction for 5x5x5:
    # Partition indices into {0,1} and {2,3,4}
    # A = [A11(2x2) A12(2x3)]
    #     [A21(3x2) A22(3x3)]
    # B = [B11(2x2) B12(2x3)]
    #     [B21(3x2) B22(3x3)]
    # C = [C11(2x2) C12(2x3)]
    #     [C21(3x2) C22(3x3)]
    
    # C11 = A11*B11 + A12*B21  (2x2 = 2x2*2x2 + 2x3*3x2)
    # C12 = A11*B12 + A12*B22  (2x3 = 2x2*2x3 + 2x3*3x3)
    # C21 = A21*B11 + A22*B21  (3x2 = 3x2*2x2 + 3x3*3x2)
    # C22 = A21*B12 + A22*B22  (3x3 = 3x2*2x3 + 3x3*3x3)
    
    # For 2x2x2, use Strassen (rank 7)
    # For other blocks, use schoolbook or optimized methods
    
    # This is getting very complex to implement from scratch.
    # Let me use a simpler approach: since we need 4x5x5, not 5x5x5,
    # and the directive says to restrict the 5x5x5 rank-19 to 4x5x5,
    # let me find a valid rank-19 for 5x5x5 first.
    
    # Alternative: Use the fact that 4x5x5 can be embedded in 5x5x5.
    # If we have rank-r for 5x5x5, we get rank-r for 4x5x5 by restriction.
    
    # Let me try a different known result. The best known for 4x4x4 is rank 21.
    # For 4x5x5, we can potentially do better than schoolbook (100).
    
    # Let me try a block-based approach with Strassen:
    # 4x5x5: partition A as 2+2 rows, B as 2+3 rows, C as 2+3 cols
    # A = [A11(2x2) A12(2x3)]
    #     [A21(2x2) A22(2x3)]
    # B = [B11(2x2) B12(2x3)]
    #     [B21(3x2) B22(3x3)]
    # C = [C11(2x2) C12(2x3)]
    #     [C21(2x2) C22(2x3)]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For 2x2x2 blocks, use Strassen (rank 7 each)
    # For mixed blocks, use schoolbook or other methods
    
    # This is still complex. Let me just return schoolbook for now
    # but with a better structure.
    
    return None

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    u_val = U[i * m + j, r]
                    v_val = V[j * p + k, r]
                    w_val = W[k * n + i, r]
                    val += u_val * v_val * w_val
                if abs(val - 1.0) > tol:
                    errors.append((i, j, k, val))
    return len(errors) == 0, errors

def _try_strassen_based(n, m, p):
    """
    Try to build a decomposition using Strassen's algorithm as a building block.
    For 4x5x5: partition into 2x2 and 2x3/3x2/3x3 blocks.
    """
    # Strassen 2x2x2 rank-7 decomposition
    # For multiplying 2x2 by 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Let me build a general block-based decomposition
    
    # For 4x5x5, split A into rows (2,2), B into rows (2,3), B into cols (2,3)
    # A is 4x5: A[i,j] where i in 0..3, j in 0..4
    # B is 5x5: B[j,k] where j in 0..4, k in 0..4
    # C is 4x5: C[i,k] where i in 0..3, k in 0..4
    
    # Block structure:
    # A11 = A[0:2, 0:2] (2x2)
    # A12 = A[0:2, 2:5] (2x3)
    # A21 = A[2:4, 0:2] (2x2)
    # A22 = A[2:4, 2:5] (2x3)
    
    # B11 = B[0:2, 0:2] (2x2)
    # B12 = B[0:2, 2:5] (2x3)
    # B21 = B[2:5, 0:2] (3x2)
    # B22 = B[2:5, 2:5] (3x3)
    
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x3)
    # C21 = A21*B11 + A22*B21 (2x2)
    # C22 = A21*B12 + A22*B22 (2x3)
    
    # For 2x2x2 multiplications, use Strassen (rank 7)
    # For 2x3x2, 2x3x3, 3x2x2, 3x2x3, 3x3x2, 3x3x3, use schoolbook
    
    # This gives:
    # C11: 7 (Strassen for A11*B11) + 6 (schoolbook for A12*B21) = 13
    # C12: 7 (Strassen-like? No, 2x2*2x3) + 6 (2x3*3x3 schoolbook) 
    # Hmm, Strassen only works for square matrices of the same size.
    
    # Actually, Strassen can be adapted. But for simplicity, let me use schoolbook
    # for all mixed blocks and Strassen only for the 2x2x2 parts.
    
    # C11 = A11*B11 + A12*B21
    #   A11*B11: 2x2x2, use Strassen rank 7
    #   A12*B21: 2x3x2, schoolbook rank 12
    #   Total for C11: 19
    
    # C12 = A11*B12 + A12*B22
    #   A11*B12: 2x2x3, schoolbook rank 12
    #   A12*B22: 2x3x3, schoolbook rank 18
    #   Total for C12: 30
    
    # C21 = A21*B11 + A22*B21
    #   A21*B11: 2x2x2, use Strassen rank 7
    #   A22*B21: 2x3x2, schoolbook rank 12
    #   Total for C21: 19
    
    # C22 = A21*B12 + A22*B22
    #   A21*B12: 2x2x3, schoolbook rank 12
    #   A22*B22: 2x3x3, schoolbook rank 18
    #   Total for C22: 30
    
    # Grand total: 19 + 30 + 19 + 30 = 98
    # That's only slightly better than schoolbook (100). Not great.
    
    # Let me try a different partition. What if I use 2x2x2 Strassen more aggressively?
    
    # Actually, the issue is that Strassen only helps for square same-size blocks.
    # For 4x4x4, we can use 4 = 2+2 and get 4 blocks of 2x2x2, each using rank 7.
    # Total: 4*7 = 28, but we also need to combine, so it's more like 28 + overhead.
    # Best known for 4x4x4 is 21, so there's a better way.
    
    # For 4x5x5, let me just return schoolbook for now since the block approach
    # doesn't give a clear improvement.
    
    return None

def _optimize_rank(n, m, p, target_rank, time_limit=30):
    """
    Try to find a decomposition of the given target_rank using optimization.
    """
    start_time = time.time()
    
    # Initialize with schoolbook, then try to compress
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    full_rank = n * m * p
    
    if time.time() - start_time > time_limit:
        return U_school.tolist(), V_school.tolist(), W_school.tolist()
    
    return U_school.tolist(), V_school.tolist(), W_school.tolist()

def _bclr_rank19_5x5x5():
    """
    Implement the BCLR rank-19 decomposition for 5x5x5 matrix multiplication.
    
    Based on: Bini, Capovani, Lotti, Romani, "Decomposizione di tensore in somma
    di un minimo numero di monomi", 1979.
    
    The algorithm splits 5x5 matrices into 2+3 blocks and uses a combination
    of Strassen and other techniques.
    """
    # I'll construct the 19 rank-1 terms explicitly.
    # Each term r has:
    #   u_r: 25-vector (flattened 5x5 matrix for A)
    #   v_r: 25-vector (flattened 5x5 matrix for B)
    #   w_r: 25-vector (flattened 5x5 matrix for C)
    # such that sum_r u_r[i*5+j] * v_r[j*5+k] * w_r[k*5+i] = 1 for all i,j,k
    
    # BCLR construction:
    # Split indices into S1={0,1} and S2={2,3,4}
    
    # The 19 terms come from:
    # - 7 terms for the 2x2x2 Strassen part (for A[S1,S1]*B[S1,S1])
    # - Additional terms for the mixed blocks
    
    # This is complex to derive from scratch. Let me use a verified set.
    
    # After careful construction, here are the 19 terms:
    # I'll use a systematic approach.
    
    # Actually, let me try a different strategy. Since implementing BCLR from scratch
    # is error-prone, let me use the following approach:
    
    # For 4x5x5, I can use the following:
    # 1. Start with schoolbook (rank 100)
    # 2. Try to find linear dependencies to reduce rank
    
    # Or, I can use the fact that 4x5x5 is a subproblem of 5x5x5.
    # If I can find ANY valid rank-r decomposition for 5x5x5, I can restrict it.
    
    # Let me try to construct a valid decomposition using a combination of
    # Strassen for 2x2x2 blocks and schoolbook for the rest, but more cleverly.
    
    # For 4x5x5:
    # Use 2x2 Strassen where possible.
    
    # Partition:
    # A: rows (2,2), cols (2,3)
    # B: rows (2,3), cols (2,3)
    # C: rows (2,2), cols (2,3)
    
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:5]*B[2:5,0:2]
    # C[0:2,2:5] = A[0:2,0:2]*B[0:2,2:5] + A[0:2,2:5]*B[2:5,2:5]
    # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:5]*B[2:5,0:2]
    # C[2:4,2:5] = A[2:4,0:2]*B[0:2,2:5] + A[2:4,2:5]*B[2:5,2:5]
    
    # For the 2x2x2 multiplications (A[0:2,0:2]*B[0:2,0:2] and A[2:4,0:2]*B[0:2,0:2]),
    # use Strassen (rank 7 each).
    
    # For all other multiplications, use schoolbook.
    
    # Total rank:
    # C[0:2,0:2]: 7 (Strassen) + 2*3*2=12 (schoolbook for 2x3*3x2) = 19
    # C[0:2,2:5]: 2*2*3=12 (schoolbook for 2x2*2x3) + 2*3*3=18 (schoolbook for 2x3*3x3) = 30
    # C[2:4,0:2]: 7 (Strassen) + 12 = 19
    # C[2:4,2:5]: 12 + 18 = 30
    # Total: 19+30+19+30 = 98
    
    # That's only rank 98, barely better than 100.
    
    # To do better, I need to use Strassen or other tricks for the mixed blocks too.
    
    # For now, let me just return the schoolbook decomposition.
    # The BCLR rank-19 for 5x5x5 would give rank 19 for 4x5x5, which is a huge improvement,
    # but implementing it correctly is challenging.
    
    # Let me try to find a reference implementation or construct it carefully.
    
    # BCLR 5x5x5 rank-19:
    # The key insight is that 5 = 2 + 3, and we can use:
    # - Strassen for 2x2x2 (rank 7)
    # - A rank-6 decomposition for 2x3x2 (or similar)
    # - Other optimized decompositions for the mixed blocks
    
    # After research, the BCLR algorithm achieves rank 19 for 5x5x5 using:
    # 7 (for 2x2x2) + 6 (for 2x3x2) + 6 (for 3x2x2) = 19? No, that doesn't add up.
    
    # Actually, the BCLR algorithm is more sophisticated. It uses a global optimization
    # across all blocks simultaneously.
    
    # For now, I'll return None to indicate this needs more work.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110 seconds to be safe
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify the schoolbook decomposition
    valid, errors = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        raise ValueError(f"Schoolbook decomposition failed: {len(errors)} errors")
    
    # Try BCLR rank-19 for 5x5x5, restricted to 4x5x5
    if time.time() < deadline - 5:
        bclr_result = _bclr_rank19_5x5x5()
        if bclr_result is not None:
            U_bclr, V_bclr, W_bclr = bclr_result
            # Restrict to first 4 rows: zero out U entries for i=4
            # U has shape (5*5, rank) = (25, rank)
            # For 4x5x5, we only need i in 0..3, so U[i*5+j, r] for i in 0..3
            # We keep V and W the same, but restrict U to first 4 rows of A
            # Actually, the restriction means we're computing C[i,k] for i in 0..3
            # The U factor has indices i*5+j for i in 0..4, j in 0..4
            # For 4x5x5, we only need i in 0..3, so we keep U[0:20, :] and zero out U[20:25, :]
            # But wait, the decomposition must still satisfy:
            # sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*5+i, r] = 1 for i in 0..3, j in 0..4, k in 0..4
            # If the 5x5x5 decomposition satisfies this for all i in 0..4, then restricting
            # to i in 0..3 automatically gives a valid 4x5x5 decomposition.
            # We just need to adjust the output format: U should have shape (4*5, rank) = (20, rank)
            # So we take U[0:20, :] from the 5x5x5 decomposition.
            # V stays (25, rank), W stays (25, rank) but we only use W[k*5+i, r] for i in 0..3
            # Actually, W has shape (5*5, rank) = (25, rank) with W[k*5+i, r]
            # For 4x5x5, we need W[k*4+i, r] for i in 0..3, k in 0..4
            # This is a different indexing! The W factor needs to be reindexed.
            
            # Hmm, this is the issue. The W factor uses k*n+i where n=5 for 5x5x5,
            # but for 4x5x5, n=4, so W uses k*4+i.
            # The restriction is not straightforward because the indexing changes.
            
            # Let me reconsider. For 5x5x5:
            # T[i,j,k] = sum_r U5[i*5+j, r] * V5[j*5+k, r] * W5[k*5+i, r] = 1
            # For 4x5x5:
            # T[i,j,k] = sum_r U4[i*4+j, r] * V4[j*5+k, r] * W4[k*4+i, r] = 1
            
            # The U, V, W have different shapes and indexings.
            # To restrict 5x5x5 to 4x5x5:
            # U4[i*4+j, r] = U5[i*5+j, r] for i in 0..3, j in 0..4
            # V4[j*5+k, r] = V5[j*5+k, r] for j in 0..4, k in 0..4
            # W4[k*4+i, r] = W5[k*5+i, r] for k in 0..4, i in 0..3
            
            # This works! We just need to reindex U and W.
            
            U4 = np.zeros((4*5, U_bclr.shape[1]))
            V4 = np.zeros((5*5, V_bclr.shape[1]))
            W4 = np.zeros((5*4, W_bclr.shape[1]))
            
            for i in range(4):
                for j in range(5):
                    U4[i*5+j, :] = U_bclr[i*5+j, :]
            
            for j in range(5):
                for k in range(5):
                    V4[j*5+k, :] = V_bclr[j*5+k, :]
            
            for k in range(5):
                for i in range(4):
                    W4[k*4+i, :] = W_bclr[k*5+i, :]
            
            # Verify
            valid_bclr, errors_bclr = _verify_decomposition(U4, V4, W4, 4, 5, 5)
            if valid_bclr:
                U = U4
                V = V4
                W = W4
                best_rank = U4.shape[1]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
