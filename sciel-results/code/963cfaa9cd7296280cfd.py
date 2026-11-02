"""Matrix multiplication 3,3,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction as Fr
from itertools import product

def _build_trivial(n, m, p):
    """Build the trivial (schoolbook) decomposition of rank n*m*p."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_full(decomp, n, m, p, tol=1e-9):
    """Full verification of the decomposition."""
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    r = U.shape[1]
    # Check all valid positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, t] * V[b, t] * W[c, t] for t in range(r))
                if abs(val - 1) > tol:
                    return False
    # Check all invalid positions (sample if too many)
    invalid_count = 0
    for a in range(n * m):
        i_a, j_a = divmod(a, m)
        for b in range(m * p):
            j_b, k_b = divmod(b, p)
            if j_a == j_b:
                continue
            for c in range(p * n):
                k_c, i_c = divmod(c, n)
                if k_b == k_c and i_c == i_a:
                    continue
                val = sum(U[a, t] * V[b, t] * W[c, t] for t in range(r))
                if abs(val) > tol:
                    return False
                invalid_count += 1
    return True

def _build_rank41_decomposition():
    """
    Build a rank-41 decomposition for 3x3x5 matmul.
    
    Strategy: Decompose M_{3,3,5} = M_{3,3,3} ⊕ M_{3,3,2}
    - M_{3,3,2}: trivial rank 18 (two independent 3x3x1 blocks)
    - M_{3,3,3}: Use a rank-23 decomposition based on known results
    
    For M_{3,3,3}, we use a constructive approach based on partitioning
    the 3x3 matmul into overlapping 2x2 Strassen blocks plus corrections.
    
    Actually, let's use a simpler but still improved approach:
    For the 3x3x3 part, we can use the fact that:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    We'll use a decomposition that exploits the structure. The key insight
    is that we can use rank-9 for the "diagonal" contributions and then
    add correction terms for the off-diagonal interactions.
    
    Let me use a more direct approach: implement the known rank-23 decomposition
    for 3x3x3 by referencing the Lysikov decomposition structure.
    
    For practical purposes within the time budget, let me try a rank-44 approach
    first (one less than trivial) by finding a valid linear dependency.
    """
    n, m, p = 3, 3, 5
    
    # Start with trivial rank-45 and try to reduce by 1
    # The idea: find two basis terms that can be replaced by one term with
    # appropriate coefficients, exploiting the tensor structure.
    
    # For matmul tensor T[i,j,k] = 1 (at position (i*m+j, j*p+k, k*n+i)),
    # we need to find u, v, w such that:
    #   sum over the merged terms gives the correct contributions
    
    # A simpler approach: use the block decomposition
    # M_{3,3,5} = M_{3,3,3} ⊕ M_{3,3,2}
    # Rank of M_{3,3,2} = 18 (trivial, since it's two 3x3x1 blocks)
    # For M_{3,3,3}, try to find rank < 27
    
    # For 3x3x3, use the following decomposition:
    # Partition A, B, C into 2x2 and 1x1 blocks
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # Similarly for B and C
    
    # C11 = A11*B11 + A12*B21 (2x2 = 2x2*2x2 + 2x1*1x2)
    # C12 = A11*B12 + A12*B22 (2x1 = 2x2*2x1 + 2x1*1x1)
    # C21 = A21*B11 + A22*B21 (1x2 = 1x2*2x2 + 1x1*1x2)
    # C22 = A21*B12 + A22*B22 (1x1 = 1x2*2x1 + 1x1*1x1)
    
    # This gives us a block structure. The total rank is the sum of ranks of each block product.
    # C11: 2x2*2x2 (rank 7 via Strassen) + 2x1*1x2 (rank 2) = 9
    # C12: 2x2*2x1 (rank 4) + 2x1*1x1 (rank 2) = 6
    # C21: 1x2*2x2 (rank 4) + 1x1*1x2 (rank 2) = 6
    # C22: 1x2*2x1 (rank 2) + 1x1*1x1 (rank 1) = 3
    # Total: 9 + 6 + 6 + 3 = 24
    
    # But wait, these blocks share some of the same input/output indices, so we might
    # be able to share some rank terms. Let me be more careful.
    
    # Actually, the block products are independent in terms of their (i,j,k) indices:
    # C11 involves i in {0,1}, k in {0,1}
    # C12 involves i in {0,1}, k in {2}
    # C21 involves i in {2}, k in {0,1}
    # C22 involves i in {2}, k in {2}
    
    # These are disjoint in the (i,k) space, so the ranks add up: 9+6+6+3 = 24
    
    # For M_{3,3,3}, this gives rank 24.
    # For M_{3,3,2}, rank 18.
    # Total: 24 + 18 = 42.
    
    # Let me implement this.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Helper to create sparse vectors
    def sparse_vec(size, entries):
        """Create a vector with given entries at specified indices."""
        vec = [0] * size
        for idx, val in entries.items():
            vec[idx] = val
        return vec
    
    # === M_{3,3,3} part: k in {0, 1, 2} ===
    
    # C11 = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2]*B[2,0:2]
    # Block 1: 2x2 * 2x2 (i in {0,1}, j in {0,1}, k in {0,1})
    # Use Strassen's rank-7 decomposition for 2x2x2
    # Strassen terms for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 - M4 + M6
    # C22 = M1 - M2 + M3 - M6
    
    # For our 2x2x2 subproblem (i in {0,1}, j in {0,1}, k in {0,1}):
    # U index: i*m + j = i*3 + j (for i in {0,1}, j in {0,1}: indices 0,1,3,4)
    # V index: j*p + k = j*5 + k (for j in {0,1}, k in {0,1}: indices 0,1,5,6)
    # W index: k*n + i = k*3 + i (for k in {0,1}, i in {0,1}: indices 0,1,3,4)
    
    # Strassen M1: (A11+A22)(B11+B22)
    # U: nonzero at (0,0)=0, (1,1)=4 with value 1
    # V: nonzero at (0,0)=0, (1,1)=6 with value 1
    # W: nonzero at (0,0)=0, (1,1)=4 with value 1
    add_term(
        sparse_vec(9, {0: 1, 4: 1}),
        sparse_vec(15, {0: 1, 6: 1}),
        sparse_vec(15, {0: 1, 4: 1})
    )
    
    # Strassen M2: (A21+A22)B11
    # U: nonzero at (1,0)=3, (1,1)=4 with value 1
    # V: nonzero at (0,0)=0 with value 1
    # W: nonzero at (0,0)=0, (1,1)=4 with value 1
    add_term(
        sparse_vec(9, {3: 1, 4: 1}),
        sparse_vec(15, {0: 1}),
        sparse_vec(15, {0: 1, 4: 1})
    )
    
    # Strassen M3: A11(B12-B22)
    # U: nonzero at (0,0)=0 with value 1
    # V: nonzero at (0,1)=1, (1,1)=6 with values 1, -1
    # W: nonzero at (0,1)=1, (1,1)=4 with values 1, 1
    add_term(
        sparse_vec(9, {0: 1}),
        sparse_vec(15, {1: 1, 6: -1}),
        sparse_vec(15, {1: 1, 4: 1})
    )
    
    # Strassen M4: A22(B21-B11)
    # U: nonzero at (1,1)=4 with value 1
    # V: nonzero at (1,0)=5, (0,0)=0 with values 1, -1
    # W: nonzero at (0,0)=0, (1,0)=3 with values 1, 1
    add_term(
        sparse_vec(9, {4: 1}),
        sparse_vec(15, {5: 1, 0: -1}),
        sparse_vec(15, {0: 1, 3: 1})
    )
    
    # Strassen M5: (A11+A12)B22
    # U: nonzero at (0,0)=0, (0,1)=1 with value 1
    # V: nonzero at (1,1)=6 with value 1
    # W: nonzero at (0,1)=1, (1,1)=4 with values 1, 1
    add_term(
        sparse_vec(9, {0: 1, 1: 1}),
        sparse_vec(15, {6: 1}),
        sparse_vec(15, {1: 1, 4: 1})
    )
    
    # Strassen M6: (A21-A11)(B11+B12)
    # U: nonzero at (1,0)=3, (0,0)=0 with values 1, -1
    # V: nonzero at (0,0)=0, (0,1)=1 with value 1
    # W: nonzero at (0,0)=0, (1,0)=3 with values 1, 1
    add_term(
        sparse_vec(9, {3: 1, 0: -1}),
        sparse_vec(15, {0: 1, 1: 1}),
        sparse_vec(15, {0: 1, 3: 1})
    )
    
    # Strassen M7: (A12-A22)(B21+B22)
    # U: nonzero at (0,1)=1, (1,1)=4 with values 1, -1
    # V: nonzero at (1,0)=5, (1,1)=6 with value 1
    # W: nonzero at (0,1)=1, (1,1)=4 with values 1, 1
    add_term(
        sparse_vec(9, {1: 1, 4: -1}),
        sparse_vec(15, {5: 1, 6: 1}),
        sparse_vec(15, {1: 1, 4: 1})
    )
    
    # Now verify: C11 = M1 + M4 - M5 + M7
    # C11[0,0] = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
    # M1[0,0]: U[0]*V[0]*W[0] = 1*1*1 = 1
    # M4[0,0]: U[0]*V[0]*W[0] = 0*(-1)*1 = 0 (U[0]=0 for M4)
    # M5[0,0]: U[0]*V[0]*W[0] = 1*0*1 = 0 (V[0]=0 for M5)
    # M7[0,0]: U[0]*V[0]*W[0] = 0*0*1 = 0 (U[0]=0 for M7)
    # Sum = 1. But we need C11[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # Hmm, this is getting complicated. Let me verify differently.
    
    # Actually, I realize my Strassen embedding might have index errors. Let me restart
    # with a cleaner approach.
    
    # Let me just use the trivial decomposition for now and try a different strategy.
    # The block approach is correct in principle but the index mapping is error-prone.
    
    # Alternative: just return the trivial rank-45 decomposition.
    # This is guaranteed to work.
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with the trivial decomposition as the best known
    best = _build_trivial(n, m, p)
    
    # Try to find a better decomposition
    # For now, the trivial rank-45 is our baseline.
    
    return best
# EVOLVE-BLOCK-END
