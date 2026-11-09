import numpy as np
import random
import time

DIMENSIONS = [3, 4, 4]

def _construct_322_rank_11():
    """
    Attempts to find a rank 11 decomposition for 3x2x2 matrix multiplication.
    Returns U, V, W matrices of shapes (6, 11), (4, 11), (6, 11) or None.
    """
    n, m, p = 3, 2, 2
    rank = 11
    
    # Target tensor T of shape (n*m, m*p, p*n) is not directly used, 
    # we verify via U, V, W.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # We search over a small finite field first to find a support structure.
    # Let's try F_3 or F_5.
    # The equations are:
    # For all i in [0, n), j in [0, m), k in [0, p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # For all other (i,j,k) combinations in the full box, the sum must be 0?
    # No, the tensor is only defined for the multiplication structure.
    # The condition is:
    # The tensor defined by U, V, W must equal the multiplication tensor.
    # The multiplication tensor T has T[i,j,k] = 1 if the indices correspond to a valid multiplication term?
    # Actually, the tensor T is sparse.
    # T[i*m+j, j*p+k, k*n+i] is not the right indexing for the tensor itself.
    # The tensor T is a 3-way tensor of size (n*m) x (m*p) x (p*n).
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.
    # Wait, for a fixed i,j,k, the term is U[i*m+j] V[j*p+k] W[k*n+i].
    # The product sum is sum_c U[a,c] V[b,c] W[c_idx,c].
    # We require that for all i,j,k:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # And for all (a,b,c) where (a,b,c) does NOT correspond to a valid (i,j,k) triple in a way that produces a 1?
    # Actually, the tensor is exactly the multiplication tensor.
    # The multiplication tensor has 1s at positions (i*m+j, j*p+k, k*n+i) for all i,j,k.
    # And 0s elsewhere.
    # So we have n*m*p = 12 equations equal to 1.
    # And (n*m * m*p * p*n - n*m*p) equations equal to 0.
    # 6*4*6 = 144 total entries.
    # 12 are 1, 132 are 0.
    
    # This is a system of polynomial equations.
    # We can try to solve it over a small field.
    
    fields = [3, 5, 7]
    for p_field in fields:
        # Random initialization
        for _ in range(1000):
            U = np.zeros((6, rank), dtype=int)
            V = np.zeros((4, rank), dtype=int)
            W = np.zeros((6, rank), dtype=int)
            
            # Random sparse initialization
            for c in range(rank):
                # Pick random non-zero entries
                U[np.random.randint(6), c] = np.random.randint(1, p_field)
                V[np.random.randint(4), c] = np.random.randint(1, p_field)
                W[np.random.randint(6), c] = np.random.randint(1, p_field)
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c_idx = k * n + i
                        # Compute sum
                        val = 0
                        for c in range(rank):
                            val = (val + U[a, c] * V[b, c] * W[c_idx, c]) % p_field
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
                
            # Check zeros
            for a in range(n * m):
                for b in range(m * p):
                    for c_idx in range(p * n):
                        # Check if (a,b,c_idx) corresponds to a valid (i,j,k)
                        # a = i*m + j => i = a//m, j = a%m
                        i = a // m
                        j = a % m
                        # b = j*p + k => j = b//p, k = b%p
                        j2 = b // p
                        k = b % p
                        # c_idx = k*n + i => k = c_idx//n, i = c_idx%n
                        k2 = c_idx // n
                        i2 = c_idx % n
                        
                        if not (j == j2 and k == k2 and i == i2):
                            # Should be 0
                            val = 0
                            for c in range(rank):
                                val = (val + U[a, c] * V[b, c] * W[c_idx, c]) % p_field
                            if val != 0:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find a better decomposition for 3x2x2
    # If found, use it to build 3x4x4
    # 3x4x4 can be split into 4 blocks of 3x2x2?
    # A (3x4) = [A1 A2] (3x2)
    # B (4x4) = [[B11 B12], [B21 B22]] (2x2 blocks)
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    # This requires 4 multiplications of type 3x2x2.
    # If we have a rank 11 decomposition for 3x2x2, we can use it 4 times.
    # Total rank 44.
    
    U_322, V_322, W_322 = _construct_322_rank_11()
    
    if U_322 is not None:
        # Construct 3x4x4
        # n=3, m=4, p=4
        # U shape (12, 44), V shape (16, 44), W shape (12, 44)
        rank_total = 4 * len(U_322[0]) # 4 * 11 = 44
        U = np.zeros((n * m, rank_total), dtype=int)
        V = np.zeros((m * p, rank_total), dtype=int)
        W = np.zeros((p * n, rank_total), dtype=int)
        
        # Map indices
        # Block 1: A1 (cols 0,1) * B11 (rows 0,1; cols 0,1) -> C1 (rows 0,1)
        # A1 is rows 0..2, cols 0..1 of A. Index i in 0..2, j in 0..1.
        # U index for A1: i*4 + j.
        # V index for B11: j*4 + k. j in 0..1, k in 0..1.
        # W index for C1: k*3 + i. k in 0..1, i in 0..2.
        
        # We have 4 blocks:
        # 1. A1 (j=0,1) * B11 (j=0,1; k=0,1) -> C1 (k=0,1)
        # 2. A1 (j=0,1) * B12 (j=0,1; k=2,3) -> C2 (k=2,3)
        # 3. A2 (j=2,3) * B21 (j=2,3; k=0,1) -> C1 (k=0,1)
        # 4. A2 (j=2,3) * B22 (j=2,3; k=2,3) -> C2 (k=2,3)
        
        # Wait, the decomposition of 3x2x2 assumes A is 3x2, B is 2x2.
        # In block 1: A_part is A[:, 0:2] (3x2). B_part is B[0:2, 0:2] (2x2).
        # The local indices for 3x2x2 are:
        # i_local = i (0..2)
        # j_local = j (0..1)
        # k_local = k (0..1)
        
        # U_322[i_local*2 + j_local, c_local]
        # V_322[j_local*2 + k_local, c_local]
        # W_322[k_local*3 + i_local, c_local]
        
        # Global indices:
        # U[i*4 + j, c_global]
        # V[j*4 + k, c_global]
        # W[k*3 + i, c_global]
        
        blocks = [
            (0, 0, 0), # A1, B11, C1 (j_offset 0, k_offset 0)
            (0, 0, 2), # A1, B12, C2 (j_offset 0, k_offset 2)
            (2, 0, 0), # A2, B21, C1 (j_offset 2, k_offset 0)
            (2, 2, 2), # A2, B22, C2 (j_offset 2, k_offset 2)
        ]
        
        for b_idx, (j_off, k_off, c_k_off) in enumerate(blocks):
            # c_k_off is the offset for k in the output W index?
            # W index is k*3 + i.
            # In block 1, k is 0,1. So k_global = k_local.
            # In block 2, k is 2,3. So k_global = k_local + 2.
            # In block 3, k is 0,1.
            # In block 4, k is 2,3.
            
            # j offset for A and B
            # A col j_global = j_local + j_off
            # B row j_global = j_local + j_off
            # B col k_global = k_local + k_off
            
            for c_local in range(len(U_322[0])):
                c_global = b_idx * len(U_322[0]) + c_local
                for i in range(3):
                    for j_local in range(2):
                        u_val = U_322[i * 2 + j_local, c_local]
                        if u_val != 0:
                            j_global = j_local + j_off
                            U[i * 4 + j_global, c_global] = u_val
                            
                    for j_local in range(2):
                        for k_local in range(2):
                            v_val = V_322[j_local * 2 + k_local, c_local]
                            if v_val != 0:
                                j_global = j_local + j_off
                                k_global = k_local + k_off
                                V[j_global * 4 + k_global, c_global] = v_val
                                
                    for k_local in range(2):
                        for i in range(3):
                            w_val = W_322[k_local * 3 + i, c_local]
                            if w_val != 0:
                                k_global = k_local + k_off
                                W[k_global * 3 + i, c_global] = w_val
                                
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

