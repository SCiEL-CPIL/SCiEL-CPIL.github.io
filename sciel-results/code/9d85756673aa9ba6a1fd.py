"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Construct rank-minimal (or near-minimal) tensor decomposition for
    multiplying a 2x3 matrix by a 3x10 matrix.
    
    Output: {'U': (6, R), 'V': (30, R), 'W': (20, R)} where R is the rank.
    The decomposition satisfies: for all i,j,k: sum_c U[i*3+j, c] * V[j*10+k, c] * W[k*2+i, c] = delta(i,j,k)
    i.e., the tensor T[i*3+j, j*10+k, k*2+i] = sum_c U[i*3+j, c] * V[j*10+k, c] * W[k*2+i, c]
    should equal 1 when (i,j,k) is a valid triple (which they all are in 0..n-1, 0..m-1, 0..p-1)
    and 0 otherwise. Wait, actually the tensor is:
    T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for valid i,j,k, else 0.
    
    Actually, the matrix multiplication tensor for (n x m) * (m x p) is:
    T[i, j, k] = 1 (scalar product gives sum over j: A[i,j]*B[j,k])
    The tensor has indices (i, j, k) with values 1.
    
    In the flattened form used here:
    U has shape (n*m, R) = (6, R)
    V has shape (m*p, R) = (30, R)
    W has shape (p*n, R) = (20, R)
    
    The decomposition requires:
    For each (i, j, k) in [0,n) x [0,m) x [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    For all other index combinations (which don't correspond to valid (i,j,k)), the sum should be 0.
    
    But wait - the indices of U, V, W are:
    U index: i*m + j where i in [0,n), j in [0,m) -> 6 values
    V index: j*p + k where j in [0,m), k in [0,p) -> 30 values
    W index: k*n + i where k in [0,p), i in [0,n) -> 20 values
    
    The tensor is only non-zero for valid (i,j,k) triples. For a given U index u = i*m+j,
    V index v = j'*p+k, W index w = k'*n+i', the product U[u,c]*V[v,c]*W[w,c] contributes
    to the tensor entry at position (u, v, w). This should equal 1 if there exist i,j,k such that
    u = i*m+j, v = j*p+k, w = k*n+i, and 0 otherwise.
    
    So we need: for each c, the tensor T_c = U[:,c] ⊗ V[:,c] ⊗ W[:,c] and the sum over c equals T.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 60)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def validate(U, V, W):
        """Check if decomposition is exact (within tolerance)."""
        n, m, p = DIMENSIONS
        # Build the target tensor
        # For each (i,j,k): target[i*m+j, j*p+k, k*n+i] = 1
        # We check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
        # And that for invalid combinations, the sum is 0.
        
        R = U.shape[1]
        max_err = 0.0
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(R):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        # Check that non-valid entries are 0 (spot check some)
        # The full tensor has 6*30*20 = 3600 entries, only 60 are valid
        # Let's check a few invalid ones
        for _ in range(100):
            u = random.randint(0, 5)
            v = random.randint(0, 29)
            w = random.randint(0, 19)
            # Check if this is a valid triple
            i = u // m
            j = u % m
            j2 = v // p
            k = v % p
            k2 = w // n
            i2 = w % n
            if not (i == i2 and j == j2 and k == k2):
                val = 0.0
                for c in range(R):
                    val += U[u, c] * V[v, c] * W[w, c]
                err = abs(val)
                if err > max_err:
                    max_err = err
        return max_err
    
    def try_strassen_like(rank_target):
        """Try to find a decomposition of given rank using iterative refinement."""
        # This is a heuristic approach - not guaranteed to find a solution
        # Use alternating least squares / gradient descent
        np.random.seed(42)
        U = np.random.randn(n*m, rank_target) * 0.1
        V = np.random.randn(m*p, rank_target) * 0.1
        W = np.random.randn(p*n, rank_target) * 0.1
        
        # Build list of valid tensor entries
        valid_entries = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_entries.append((i*m+j, j*p+k, k*n+i))
        
        for iteration in range(100):
            # Update U
            for c in range(rank_target):
                # For each row u of U, we want to minimize error
                # This is complex; use a simple gradient step
                pass
            # This approach is too slow/complex for the time limit
            break
        return None
    
    # Use the known Strassen-based construction for rank 55
    # Strategy: Use Strassen on 2x2 blocks, handle remaining columns with schoolbook
    # 
    # A is 2x3, B is 3x10.
    # Split A into A_left (2x2) and A_right (2x1)
    # Split B into B_top (2x10) and B_bottom (1x10)
    # 
    # C = A*B = [A_left B_top + A_right B_bottom]
    # 
    # Use Strassen for A_left * B_top (2x2 * 2x10 -> treat as 5 blocks of 2x2)
    # But B_top is 2x10, so we can split it into 5 blocks of 2x2
    # 
    # Actually, let's use a different approach:
    # Use the fact that rank(2x3 * 3x10) can be bounded using
    # known results. The tensor rank of 2x3*3x10 is at most 55 (from literature).
    # 
    # Let me construct a rank-55 decomposition:
    # 
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01, ..., b09], [b10, b11, ..., b19], [b20, b21, ..., b29]]
    #
    # C[0,k] = a00*b0k + a01*b1k + a02*b2k
    # C[1,k] = a10*b0k + a11*b1k + a12*b2k
    #
    # Use Strassen-like formulas on the 2x2 part (columns 0,1 of A and rows 0,1 of B):
    # For each pair of output columns (2k, 2k+1) for k=0..4:
    #   Let B_k = B[:, 2k:2k+2] (2 columns)
    #   C[:, 2k:2k+2] = A[:, 0:2] * B_k + A[:, 2:3] * B[2, 2k:2k+2]
    #
    # Use 7 Strassen products for the 2x2 * 2x2 part (for each k):
    # That's 7*5 = 35 products
    # Plus 2*5 = 10 products for the A[:,2] * B[2,:] part
    # Total: 45 products
    #
    # Wait, but the Strassen products involve additions/subtractions of A and B entries,
    # which means the U and V matrices will have entries that are sums of A and B entries.
    # The tensor decomposition still works because the linearity carries through.
    
    def strassen_based():
        """
        Construct a decomposition using Strassen-like formulas.
        
        For each block k (0..4), we have:
        - A_block = A[:, 0:2] (same for all k, it's 2x2)
        - B_block = B[0:2, 2k:2k+2] (2x2)
        
        C[:, 2k] and C[:, 2k+1] = A[:, 0:2] * B[0:2, 2k:2k+2] + A[:, 2:3] * B[2, 2k:2k+2]
        
        Use 7 Strassen products for the 2x2*2x2 part.
        Use 2 products for the 2x1 * 1x2 part.
        Total per block: 9 products, 5 blocks: 45 products.
        """
        # U: 6 rows (i=0,1; j=0,1,2), so U[i*3+j, c]
        # V: 30 rows (j=0,1,2; k=0..9), so V[j*10+k, c]
        # W: 20 rows (k=0..9; i=0,1), so W[k*2+i, c]
        
        rank = 45
        U = np.zeros((6, rank), dtype=float)
        V = np.zeros((30, rank), dtype=float)
        W = np.zeros((20, rank), dtype=float)
        
        col = 0
        for blk in range(5):
            # Block blk covers columns 2*blk and 2*blk+1
            k0 = 2 * blk
            k1 = 2 * blk + 1
            
            # Strassen products for 2x2 * 2x2:
            # M1 = (a00 + a11) * (b00 + b11)
            # M2 = (a10 + a11) * b00
            # M3 = a00 * (b01 - b11)
            # M4 = a11 * (b10 - b00)
            # M5 = (a00 + a01) * b11
            # M6 = (a10 - a00) * (b00 + b10)
            # M7 = (a01 - a11) * (b01 + b11)
            
            # A entries: a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
            # U indices: i*3+j -> a00: U[0], a01: U[1], a10: U[3], a11: U[4]
            
            # B entries for block blk:
            # b00=B[0,2*blk], b01=B[0,2*blk+1], b10=B[1,2*blk], b11=B[1,2*blk+1]
            # V indices: j*10+k -> b00: V[0*10+2*blk]=V[2*blk], b01: V[2*blk+1]
            #            b10: V[1*10+2*blk]=V[10+2*blk], b11: V[10+2*blk+1]
            
            # C entries:
            # C[0, 2*blk] = c00, C[0, 2*blk+1] = c01
            # C[1, 2*blk] = c10, C[1, 2*blk+1] = c11
            # W indices: k*2+i -> c00: W[2*blk*2+0]=W[4*blk], c01: W[4*blk+1]
            #            c10: W[4*blk+2], c11: W[4*blk+3]
            
            w_c00 = 4 * blk + 0
            w_c01 = 4 * blk + 1
            w_c10 = 4 * blk + 2
            w_c11 = 4 * blk + 3
            
            v_b00 = 2 * blk      # V[0*10 + 2*blk]
            v_b01 = 2 * blk + 1  # V[0*10 + 2*blk + 1]
            v_b10 = 10 + 2 * blk      # V[1*10 + 2*blk]
            v_b11 = 10 + 2 * blk + 1  # V[1*10 + 2*blk + 1]
            
            # M1 = (a00 + a11) * (b00 + b11)
            U[0, col] = 1   # a00
            U[4, col] = 1   # a11
            V[v_b00, col] = 1
            V[v_b11, col] = 1
            # Result: c00 += M1, c11 += M1
            W[w_c00, col] = 1
            W[w_c11, col] = 1
            col += 1
            
            # M2 = (a10 + a11) * b00
            U[3, col] = 1   # a10
            U[4, col] = 1   # a11
            V[v_b00, col] = 1
            # Result: c10 += M2, c11 -= M2
            W[w_c10, col] = 1
            W[w_c11, col] = -1
            col += 1
            
            # M3 = a00 * (b01 - b11)
            U[0, col] = 1   # a00
            V[v_b01, col] = 1
            V[v_b11, col] = -1
            # Result: c00 += M3, c01 += M3
            W[w_c00, col] = 1
            W[w_c01, col] = 1
            col += 1
            
            # M4 = a11 * (b10 - b00)
            U[4, col] = 1   # a11
            V[v_b10, col] = 1
            V[v_b00, col] = -1
            # Result: c10 += M4, c11 += M4
            W[w_c10, col] = 1
            W[w_c11, col] = 1
            col += 1
            
            # M5 = (a00 + a01) * b11
            U[0, col] = 1   # a00
            U[1, col] = 1   # a01
            V[v_b11, col] = 1
            # Result: c01 += M5, c11 -= M5
            W[w_c01, col] = 1
            W[w_c11, col] = -1
            col += 1
            
            # M6 = (a10 - a00) * (b00 + b10)
            U[3, col] = 1   # a10
            U[0, col] = -1  # -a00
            V[v_b00, col] = 1
            V[v_b10, col] = 1
            # Result: c10 += M6
            W[w_c10, col] = 1
            col += 1
            
            # M7 = (a01 - a11) * (b01 + b11)
            U[1, col] = 1   # a01
            U[4, col] = -1  # -a11
            V[v_b01, col] = 1
            V[v_b11, col] = 1
            # Result: c01 += M7
            W[w_c01, col] = 1
            col += 1
            
            # Now handle A[:, 2] * B[2, 2*blk:2*blk+2]
            # This is a 2x1 * 1x2 product, needs 2 products
            # A[0,2] * B[2, 2*blk] -> C[0, 2*blk]
            # A[1,2] * B[2, 2*blk] -> C[1, 2*blk]
            # A[0,2] * B[2, 2*blk+1] -> C[0, 2*blk+1]
            # A[1,2] * B[2, 2*blk+1] -> C[1, 2*blk+1]
            
            v_b20 = 20 + 2 * blk      # V[2*10 + 2*blk]
            v_b21 = 20 + 2 * blk + 1  # V[2*10 + 2*blk + 1]
            
            # P1 = A[0,2] * B[2, 2*blk]
            U[2, col] = 1  # A[0,2] -> i=0, j=2 -> index 0*3+2=2
            V[v_b20, col] = 1
            W[w_c00, col] = 1
            col += 1
            
            # P2 = A[1,2] * B[2, 2*blk]
            U[5, col] = 1  # A[1,2] -> i=1, j=2 -> index 1*3+2=5
            V[v_b20, col] = 1
            W[w_c10, col] = 1
            col += 1
            
            # P3 = A[0,2] * B[2, 2*blk+1]
            U[2, col] = 1
            V[v_b21, col] = 1
            W[w_c01, col] = 1
            col += 1
            
            # P4 = A[1,2] * B[2, 2*blk+1]
            U[5, col] = 1
            V[v_b21, col] = 1
            W[w_c11, col] = 1
            col += 1
        
        assert col == rank, f"Expected {rank}, got {col}"
        return U, V, W, rank
    
    # Build the Strassen-based decomposition
    U, V, W, rank = strassen_based()
    
    # Validate
    err = validate(U, V, W)
    if err < 1e-10:
        # Convert to list of lists for the expected format
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook if Strassen construction is wrong
    U, V, W, rank = schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
