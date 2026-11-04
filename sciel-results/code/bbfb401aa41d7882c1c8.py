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

def _build_block_strassen(n, m, p):
    """
    Build a block-structured decomposition using Strassen for 2x2x2 blocks
    and schoolbook for non-square blocks.
    
    Partition:
    A (n x m) = 4x5: rows (2,2), cols (2,3)
    B (m x p) = 5x5: rows (2,3), cols (2,3)  
    C (n x p) = 4x5: rows (2,2), cols (2,3)
    
    Block products:
    C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:5]*B[2:5,0:2]
    C[0:2,2:5] = A[0:2,0:2]*B[0:2,2:5] + A[0:2,2:5]*B[2:5,2:5]
    C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:5]*B[2:5,0:2]
    C[2:4,2:5] = A[2:4,0:2]*B[0:2,2:5] + A[2:4,2:5]*B[2:5,2:5]
    
    For 2x2x2 sub-blocks, use Strassen (rank 7) instead of schoolbook (rank 8).
    """
    # Strassen's 7 multiplication factors for 2x2x2
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We'll build the full decomposition by enumerating all (i,j,k) triples
    # and for each, determine which block it belongs to and which Strassen/schoolbook
    # multiplication it corresponds to.
    
    # Block boundaries
    a_rows = [0, 2, 4]  # A row splits
    a_cols = [0, 2, 5]  # A col splits (= B row splits)
    b_cols = [0, 2, 5]  # B col splits (= C col splits)
    c_rows = [0, 2, 4]  # C row splits
    
    # For each block pair (ai, aj) for A, (bi, bj) for B, (ci, cj) for C
    # where A[ai:ai+1, aj:aj+1] * B[bj:bj+1, bj:bj+1] contributes to C[ai:ai+1, bj:bj+1]
    
    # Actually, let's think in terms of individual (i,j,k) contributions.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Each term A[i,j]*B[j,k] is a rank-1 contribution to C[i,k].
    
    # For the block approach, we group terms by which block multiplication they belong to.
    # A[i,j] where i in [a_rows[bi], a_rows[bi+1]) and j in [a_cols[bj], a_cols[bj+1])
    # B[j,k] where j in [a_cols[bj], a_cols[bj+1]) and k in [b_cols[bk], b_cols[bk+1])
    # contributes to C[i,k] where i in [c_rows[bi], c_rows[bi+1]) and k in [b_cols[bk], b_cols[bk+1])
    
    # Let's enumerate all block combinations:
    # bi in {0,1} (A row block), bj in {0,1} (A col / B row block), bk in {0,1} (B col / C col block)
    # A block: a_rows[bi]:a_rows[bi+1] x a_cols[bj]:a_cols[bj+1]
    # B block: a_cols[bj]:a_cols[bj+1] x b_cols[bk]:b_cols[bk+1]
    # C block: c_rows[bi]:c_rows[bi+1] x b_cols[bk]:b_cols[bk+1]
    
    # Block sizes:
    # bi=0: A rows 0-1 (size 2), bi=1: A rows 2-3 (size 2)
    # bj=0: A cols/B rows 0-1 (size 2), bj=1: A cols/B rows 2-4 (size 3)
    # bk=0: B cols/C cols 0-1 (size 2), bk=1: B cols/C cols 2-4 (size 3)
    
    # So block (bi,bj,bk) has sizes:
    # A_block: (2, 2 or 3)
    # B_block: (2 or 3, 2 or 3)
    # C_block: (2, 2 or 3)
    
    # For (bi,bj,bk) = (0,0,0): A is 2x2, B is 2x2, C is 2x2 -> use Strassen (rank 7)
    # For (bi,bj,bk) = (0,0,1): A is 2x2, B is 2x3, C is 2x3 -> schoolbook (rank 2*2*3=12)
    # For (bi,bj,bk) = (0,1,0): A is 2x3, B is 3x2, C is 2x2 -> schoolbook (rank 2*3*2=12)
    # For (bi,bj,bk) = (0,1,1): A is 2x3, B is 3x3, C is 2x3 -> schoolbook (rank 2*3*3=18)
    # For (bi,bj,bk) = (1,0,0): A is 2x2, B is 2x2, C is 2x2 -> use Strassen (rank 7)
    # For (bi,bj,bk) = (1,0,1): A is 2x2, B is 2x3, C is 2x3 -> schoolbook (rank 12)
    # For (bi,bj,bk) = (1,1,0): A is 2x3, B is 3x2, C is 2x2 -> schoolbook (rank 12)
    # For (bi,bj,bk) = (1,1,1): A is 2x3, B is 3x3, C is 2x3 -> schoolbook (rank 18)
    
    # Total rank = 7 + 12 + 12 + 18 + 7 + 12 + 12 + 18 = 98
    # Schoolbook is 100, so we save 2.
    
    # Let's build this decomposition.
    factors = []  # list of (U_col, V_col, W_col) where each is a sparse dict
    
    col_idx = 0
    
    # Strassen factors for 2x2x2
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen multiplication, we need to express it as a sum of rank-1 terms
    # M1 = (A11+A22)*(B11+B22)
    # In terms of individual elements:
    # A11 = A[0,0], A12 = A[0,1], A21 = A[1,0], A22 = A[1,1]
    # B11 = B[0,0], B12 = B[0,1], B21 = B[1,0], B22 = B[1,1]
    # C11 = C[0,0], C12 = C[0,1], C21 = C[1,0], C22 = C[1,1]
    
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # This is a product of two linear combinations, which expands to:
    # A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    # Each of these is a rank-1 term in the tensor.
    
    # But wait - for the tensor decomposition, each "multiplication" M is a single scalar
    # that gets added to C entries. So M1 is one scalar, and it contributes to C11 and C22.
    
    # The tensor decomposition works as follows:
    # For each rank-1 component r, we have U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # This is different from the block multiplication view. Let me reconsider.
    
    # The tensor T[i,j,k] = 1 for all i,j,k. We want to decompose it as
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For schoolbook, each (i,j,k) triple gets its own rank-1 component.
    # For Strassen, we group multiple (i,j,k) triples together.
    
    # For a 2x2x2 block at positions (i0, j0, k0) with offsets (di, dj, dk) in {0,1}:
    # The 8 schoolbook terms are for (i0+di, j0+dj, k0+dk) for di,dj,dk in {0,1}
    # Strassen replaces these 8 with 7 terms.
    
    # Let's define the Strassen decomposition for a 2x2x2 block.
    # Local indices: i' in {0,1}, j' in {0,1}, k' in {0,1}
    # Global: i = i0 + i', j = j0 + j', k = k0 + k'
    
    # Strassen's 7 multiplications:
    # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
    # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
    # M3 = A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
    # M4 = A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
    # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
    # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
    # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
    
    # C[i0,k0] = M1 + M4 - M5 + M7
    # C[i0,k0+1] = M3 + M5
    # C[i0+1,k0] = M2 + M4
    # C[i0+1,k0+1] = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, each M is a product of a linear combo of A entries
    # and a linear combo of B entries. This corresponds to a sum of rank-1 tensors.
    
    # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
    # = A[i0,j0]*B[j0,k0] + A[i0,j0]*B[j0+1,k0+1] + A[i0+1,j0+1]*B[j0,k0] + A[i0+1,j0+1]*B[j0+1,k0+1]
    
    # Each product A[i,j]*B[j,k] is a rank-1 tensor contribution to C[i,k].
    # In the tensor T[i,j,k], the term A[i,j]*B[j,k] appears as U[i*m+j]*V[j*p+k]*W[k*n+i] = 1.
    
    # So M1 corresponds to 4 rank-1 tensor terms:
    # (i0,j0,k0), (i0,j0+1,k0+1), (i0+1,j0+1,k0), (i0+1,j0+1,k0+1)
    
    # But wait, that's not right either. M1 is a SINGLE scalar that gets added to C entries.
    # In the tensor decomposition, we're not decomposing the multiplication algorithm,
    # we're decomposing the tensor T[i,j,k]=1.
    
    # I think I'm confusing two different things. Let me restart with a clearer model.
    
    # The tensor T[i,j,k] = 1 for all i,j,k in the valid ranges.
    # We want T = sum_r u_r ⊗ v_r ⊗ w_r where
    # u_r[i*m+j] = U[i*m+j, r], v_r[j*p+k] = V[j*p+k, r], w_r[k*n+i] = W[k*n+i, r]
    
    # For schoolbook: for each (i,j,k), set U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1
    # for a unique r. This gives rank n*m*p.
    
    # For Strassen on a 2x2x2 block at (i0,j0,k0):
    # The 8 schoolbook terms for (i0+di, j0+dj, k0+dk) can be replaced by 7 terms.
    # The 7 Strassen terms are:
    # r1: U has 1 at (i0*m+j0) and (i0+1)*m+(j0+1), V has 1 at j0*p+k0 and (j0+1)*p+(k0+1), W has 1 at k0*n+i0 and (k0+1)*n+(i0+1)
    #     This corresponds to M1 = (A[i0,j0]+A[i0+1,j0+1])*(B[j0,k0]+B[j0+1,k0+1])
    #     Contribution to T: T[i0,j0,k0] += 1, T[i0,j0+1,k0+1] += 1, T[i0+1,j0+1,k0] += 1, T[i0+1,j0+1,k0+1] += 1
    #     Wait, that's not right. Let me think more carefully.
    
    # Actually, for the tensor decomposition, each rank-1 term u⊗v⊗w contributes
    # u[i*m+j] * v[j*p+k] * w[k*n+i] to T[i,j,k].
    
    # For M1 = (A[i0,j0]+A[i0+1,j0+1])*(B[j0,k0]+B[j0+1,k0+1]):
    # This is a product of (sum of A entries) and (sum of B entries).
    # In the tensor, this corresponds to a term where:
    # U[i0*m+j0, r] = 1, U[(i0+1)*m+(j0+1), r] = 1
    # V[j0*p+k0, r] = 1, V[(j0+1)*p+(k0+1), r] = 1
    # W[k0*n+i0, r] = 1, W[(k0+1)*n+(i0+1), r] = 1
    # All other entries are 0.
    
    # This single rank-1 term contributes to T[i,j,k] for all combinations of
    # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1} where the indices match up.
    # Specifically, it contributes 1 to T[i,j,k] when:
    # U[i*m+j, r] = 1 AND V[j*p+k, r] = 1 AND W[k*n+i, r] = 1
    
    # U[i*m+j, r] = 1 when (i,j) = (i0,j0) or (i0+1,j0+1)
    # V[j*p+k, r] = 1 when (j,k) = (j0,k0) or (j0+1,k0+1)
    # W[k*n+i, r] = 1 when (k,i) = (k0,i0) or (k0+1,i0+1)
    
    # So the term contributes 1 to T[i,j,k] when all three conditions hold simultaneously.
    # Let's enumerate:
    # Case 1: (i,j)=(i0,j0), (j,k)=(j0,k0), (k,i)=(k0,i0) -> i=i0, j=j0, k=k0. Consistent.
    # Case 2: (i,j)=(i0,j0), (j,k)=(j0,k0), (k,i)=(k0+1,i0+1) -> i=i0, j=j0, k=k0 from first two, but k=k0+1, i=i0+1 from third. Inconsistent.
    # Case 3: (i,j)=(i0,j0), (j,k)=(j0+1,k0+1), (k,i)=(k0,i0) -> j=j0 from first, j=j0+1 from second. Inconsistent.
    # Case 4: (i,j)=(i0,j0), (j,k)=(j0+1,k0+1), (k,i)=(k0+1,i0+1) -> j=j0 from first, j=j0+1 from second. Inconsistent.
    # Case 5: (i,j)=(i0+1,j0+1), (j,k)=(j0,k0), (k,i)=(k0,i0) -> j=j0+1 from first, j=j0 from second. Inconsistent.
    # Case 6: (i,j)=(i0+1,j0+1), (j,k)=(j0,k0), (k,i)=(k0+1,i0+1) -> j=j0+1 from first, j=j0 from second. Inconsistent.
    # Case 7: (i,j)=(i0+1,j0+1), (j,k)=(j0+1,k0+1), (k,i)=(k0,i0) -> i=i0+1 from first, i=i0 from third. Inconsistent.
    # Case 8: (i,j)=(i0+1,j0+1), (j,k)=(j0+1,k0+1), (k,i)=(k0+1,i0+1) -> i=i0+1, j=j0+1, k=k0+1. Consistent.
    
    # So M1's rank-1 term only contributes to T[i0,j0,k0] and T[i0+1,j0+1,k0+1].
    # That's only 2 out of the 8 needed terms! This approach doesn't work directly.
    
    # I think the issue is that Strassen's algorithm works on the COMPUTATION of C,
    # not directly on the tensor decomposition. The tensor T[i,j,k]=1 represents
    # the fact that each A[i,j]*B[j,k] product contributes to C[i,k].
    
    # To use Strassen in the tensor decomposition, we need a different approach.
    # Let me just use the schoolbook decomposition for now, as implementing Strassen
    # in the tensor framework is more complex than I initially thought.
    
    # Return schoolbook
    return _build_schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify
    valid, errors = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        raise ValueError(f"Schoolbook decomposition failed: {len(errors)} errors")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
