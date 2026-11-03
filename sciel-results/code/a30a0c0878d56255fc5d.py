import time
import numpy as np

DIMENSIONS = [2, 4, 6]


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Build the full multiplication tensor
    # T[i, j, k] = delta(i,j,k) where i is row of A, j is col of A / row of B, k is col of B
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # Product: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 if valid, 0 otherwise
    
    # For efficiency, we can compute all products at once using einsum or loops
    # But the tensor is only 2*4*6 = 48 entries, so we can just loop
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
                # Check that off-diagonal terms are zero
                # Actually, the tensor for matrix multiplication is:
                # T[(i,j), (j,k), (k,i)] = 1 if the indices match, 0 otherwise
                # Wait, let me re-read the convention.
                # "The output factor follows the official trace convention W[k*n+i,r]"
                # So W index is k*n + i, where k is column of B (and output), i is row of A (and output)
                # U index is i*m + j, where i is row of A, j is column of A
                # V index is j*p + k, where j is row of B (column of A), k is column of B
                # The product sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # This should equal the (i,k) entry of A*B, which is sum_j A[i,j]*B[j,k]
                # So for fixed i,k, summing over j and r: sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
                # should equal C[i,k] = sum_j A[i,j] B[j,k]
                # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent
                # Actually, the multiplication tensor is defined as:
                # T[a, b, c] where a indexes A, b indexes B, c indexes C
                # T[a, b, c] = 1 if the indices match, i.e., if a=(i,j), b=(j,k), c=(k,i) for consistent i,j,k
                # Otherwise 0.
                # So for each triple (a, b, c), the value should be 1 if a=(i,j), b=(j,k), c=(k,i)
                # Let's verify: for fixed i, j, k, the triple is (i*m+j, j*p+k, k*n+i)
                # This should give 1. All other triples should give 0.
                # But we only check the "valid" triples here. We also need to check invalid ones.
                
    # Check invalid triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # This is a valid triple, should give 1
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    
    # Check a few invalid triples to make sure they're zero
    # For example, if we take i=0, j=0, k=0 for the first, but then use a different c
    # Actually, let's just check all triples where the indices don't match
    for a in range(n * m):
        i1 = a // m
        j1 = a % m
        for b in range(m * p):
            j2 = b // p
            k1 = b % p
            for c in range(p * n):
                k2 = c // n
                i2 = c % n
                if i1 == i2 and j1 == j2 and k1 == k2:
                    continue  # valid triple, already checked
                # This is an invalid triple, should give 0
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val) > 1e-10:
                    return False
    
    return True


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Rank 42 decomposition using 6 Strassen blocks
    rank = 42
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Block definitions: (a_rows, a_cols, b_rows, b_cols)
    blocks = [
        # Block 0: A[0:2, 0:2] * B[0:2, 0:2]
        ([0, 1], [0, 1], [0, 1], [0, 1]),
        # Block 1: A[0:2, 0:2] * B[0:2, 2:4]
        ([0, 1], [0, 1], [0, 1], [2, 3]),
        # Block 2: A[0:2, 0:2] * B[0:2, 4:6]
        ([0, 1], [0, 1], [0, 1], [4, 5]),
        # Block 3: A[0:2, 2:4] * B[2:4, 0:2]
        ([0, 1], [2, 3], [2, 3], [0, 1]),
        # Block 4: A[0:2, 2:4] * B[2:4, 2:4]
        ([0, 1], [2, 3], [2, 3], [2, 3]),
        # Block 5: A[0:2, 2:4] * B[2:4, 4:6]
        ([0, 1], [2, 3], [2, 3], [4, 5]),
    ]
    
    # Strassen 7 multiplications for 2x2x2
    # local_a: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    # local_b: 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
    # local_z: 0=Z[0,0], 1=Z[0,1], 2=Z[1,0], 3=Z[1,1]
    strassen = [
        # M1 = (A00+A11)(B00+B11) -> Z00 += M1, Z11 += M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10+A11)B00 -> Z10 += M2, Z11 -= M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A00(B11-B00) -> Z01 += M3, Z11 += M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A11(B00-B10) -> Z00 += M4, Z10 += M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (A00+A01)B11 -> Z00 -= M5, Z01 += M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (A10-A00)(B00+B01) -> Z10 += M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1)]),
        # M7 = (A01-A11)(B10+B11) -> Z00 += M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    col = 0
    for block in blocks:
        if time.time() > deadline:
            break
        
        a_rows, a_cols, b_rows, b_cols = block
        
        # Map local indices to global indices
        # U: i*m + j
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1],
        }
        
        # V: j*p + k
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1],
        }
        
        # W: k*n + i
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1],
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            for local_idx, coeff in u_coeffs:
                global_idx = a_map[local_idx]
                U[global_idx, col] += coeff
            
            for local_idx, coeff in v_coeffs:
                global_idx = b_map[local_idx]
                V[global_idx, col] += coeff
            
            for local_idx, coeff in w_coeffs:
                global_idx = w_map[local_idx]
                W[global_idx, col] += coeff
            
            col += 1
    
    # Verify correctness
    if not _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        raise ValueError("Invalid decomposition")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
