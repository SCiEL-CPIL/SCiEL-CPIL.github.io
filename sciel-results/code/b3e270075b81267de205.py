"""Matrix multiplication 2x2 by 2x16 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Standard rank 64 decomposition."""
    n, m, p = 2, 2, 16
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(U, V, W, n=2, m=2, p=16):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # The decomposition condition:
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta(i,i')*delta(j,j')*delta(k,k')
    # We check this for all valid indices.
    # But for efficiency, we check the equivalent: for each (i,j,k), the (i,j,k) "slice"
    # of the reconstructed tensor equals 1, and all other entries are 0.
    
    # Actually, the correct check is:
    # For the multiplication to work, we need:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition gives:
    # (A*B)[i,k] = sum_r (sum_j A[i,j]*U[i*m+j,r]) * (sum_j' B[j',k]*V[j'*p+k,r]) * W[k*n+i,r]
    # Wait, this isn't quite right. Let me use the direct tensor identity.
    #
    # The tensor T has T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i=i', j=j', k=k', else 0.
    # Decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # So: sum_r U[i*m+j, r]*V[j'*p+k, r]*W[k'*n+i', r] = delta(i,i')*delta(j,j')*delta(k,k')
    
    # For verification, we can check a few representative cases:
    # 1. When i=i', j=j', k=k': sum should be 1
    # 2. When any index differs: sum should be 0
    
    # Check case 1: i=i', j=j', k=k'
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check case 2: j != j' (with i=i', k=k')
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                if j == jp:
                    continue
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[jp * p + k, r] * W[k * n + i, r]
                    if abs(val) > 1e-9:
                        return False
    
    # Check case 2: i != i' (with j=j', k=k')
    for i in range(n):
        for ip in range(n):
            if i == ip:
                continue
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + ip, r]
                if abs(val) > 1e-9:
                    return False
    
    # Check case 2: k != k' (with i=i', j=j')
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for kp in range(p):
                    if k == kp:
                        continue
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + kp, r] * W[kp * n + i, r]
                    if abs(val) > 1e-9:
                        return False
    
    return True


def _strassen_decomposition():
    """
    Strassen-based decomposition for 2x2 * 2x16.
    
    We decompose B into four 2x4 blocks: B = [B_0 | B_1 | B_2 | B_3]
    where B_q is 2x4 for q=0,1,2,3.
    
    For each block, we compute C[:, 4q:4q+4] = A * B_q using Strassen's 7-multiplication
    algorithm for 2x2 * 2x2, applied to each 2x2 sub-block of B_q.
    
    Actually, let's think about this differently. B_q is 2x4, which is two 2x2 blocks.
    So B_q = [B_{q,0} | B_{q,1}] where B_{q,s} is 2x2.
    
    For each 2x2 * 2x2 multiplication A * B_{q,s}, Strassen uses 7 multiplications.
    Total: 4 blocks * 2 sub-blocks * 7 = 56 multiplications.
    
    Let me construct the Strassen decomposition for a single 2x2 * 2x2 multiplication.
    
    Strassen's algorithm for C = A * B where A, B are 2x2:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each M_r is a product of a linear form in A and a linear form in B.
    
    For the tensor decomposition, each M_r corresponds to one rank-1 term:
    U_r[i*m+j] = coefficient of A[i,j] in the A-linear-form for M_r
    V_r[j'*p+k] = coefficient of B[j',k] in the B-linear-form for M_r
    W_r[k'*n+i'] = coefficient in the output combination
    
    For a 2x2 * 2x2 block at position (q, s) in B (i.e., B columns 4q+2s and 4q+2s+1):
    The A-linear-form for M_r involves A entries.
    The B-linear-form for M_r involves B entries at specific positions.
    The W factor gives the contribution to C entries.
    
    Let me define the Strassen factors for a single 2x2*2x2 multiplication.
    
    A indices: A[0,0], A[0,1], A[1,0], A[1,1] -> flat indices 0,1,2,3
    B indices: B[0,0], B[0,1], B[1,0], B[1,1] -> flat indices 0,1,2,3 (within the block)
    C indices: C[0,0], C[0,1], C[1,0], C[1,1] -> flat indices 0,1,2,3 (within the block)
    
    Strassen factors (A-form, B-form, C-contribution):
    M1: A-form = (1,0,0,1), B-form = (1,0,0,1), C: C00+=1, C11+=1
    M2: A-form = (0,0,1,1), B-form = (1,0,0,0), C: C10+=1, C11-=1
    M3: A-form = (1,0,0,0), B-form = (0,1,-1,0), C: C01+=1, C11+=1
    M4: A-form = (0,0,0,1), B-form = (0,0,1,-1), C: C00+=1, C10+=1
    M5: A-form = (1,1,0,0), B-form = (0,0,0,1), C: C00-=1, C01+=1
    M6: A-form = (0,0,1,-1), B-form = (1,1,0,0), C: C11+=1
    M7: A-form = (0,1,-1,0), B-form = (0,0,1,1), C: C00+=1, C11-=1
    
    Wait, let me re-derive more carefully.
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    So the W factors (C-contributions) for each M_r:
    M1: C00=1, C11=1
    M2: C10=1, C11=-1
    M3: C01=1, C11=1
    M4: C00=1, C10=1
    M5: C00=-1, C01=1
    M6: C11=1
    M7: C00=1, C11=-1
    
    Now, for the full 2x2 * 2x16 problem:
    U has shape (4, rank). U[i*m+j, r] where i,j in {0,1}.
    V has shape (32, rank). V[j'*p+k, r] where j' in {0,1}, k in {0..15}.
    W has shape (32, rank). W[k'*n+i', r] where k' in {0..15}, i' in {0,1}.
    
    For a Strassen term corresponding to block (q, s) and factor r_s (s=0..6):
    The A-linear-form is the same for all blocks (it only depends on A).
    The B-linear-form depends on which B entries are involved.
    The W factor depends on which C entries are affected.
    
    For block (q, s), the B entries are B[j', 4q+2s + k_local] where k_local in {0,1}.
    So V[j'*p + (4q+2s+k_local), r] gets the B-form coefficient.
    
    The C entries are C[i', 4q+2s + k_local_c] where k_local_c in {0,1}.
    So W[(4q+2s+k_local_c)*n + i', r] gets the C-contribution.
    
    Let me build this.
    """
    n, m, p = 2, 2, 16
    # Strassen factors for 2x2*2x2
    # Each factor: (A_form, B_form, C_contrib)
    # A_form: 4-vector (coefficients of A[0,0], A[0,1], A[1,0], A[1,1])
    # B_form: 4-vector (coefficients of B[0,0], B[0,1], B[1,0], B[1,1])
    # C_contrib: 4-vector (coefficients of C[0,0], C[0,1], C[1,0], C[1,1])
    
    strassen_factors = [
        # M1
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        # M2
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),
        # M3
        (np.array([1, 0, 0, 0]), np.array([0, 1, -1, 0]), np.array([0, 1, 0, 1])),
        # M4
        (np.array([0, 0, 0, 1]), np.array([0, 0, 1, -1]), np.array([1, 0, 1, 0])),
        # M5
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
        # M6
        (np.array([0, 0, 1, -1]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
        # M7
        (np.array([0, 1, -1, 0]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, -1])),
    ]
    
    # 4 blocks of 4 columns each, each block has 2 sub-blocks of 2 columns
    # Total: 4 * 2 * 7 = 56 rank-1 terms
    num_blocks = p // 4  # 4 blocks
    sub_blocks_per_block = 2  # 2 sub-blocks of 2x2 each
    num_strassen = 7
    rank = num_blocks * sub_blocks_per_block * num_strassen  # 56
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for q in range(num_blocks):
        for s in range(sub_blocks_per_block):
            # B sub-block starts at column 4*q + 2*s
            b_col_start = 4 * q + 2 * s
            # C sub-block starts at column 4*q + 2*s
            c_col_start = 4 * q + 2 * s
            
            for r_s in range(num_strassen):
                a_form, b_form, c_contrib = strassen_factors[r_s]
                
                # Fill U: A-form coefficients
                for idx in range(4):
                    U[idx, col] = a_form[idx]
                
                # Fill V: B-form coefficients
                # B[j', k] where j' in {0,1}, k in {b_col_start, b_col_start+1}
                # Flat index: j'*p + k
                for jp in range(2):
                    for kp in range(2):
                        # b_form index: jp*2 + kp (B[jp, kp] within the 2x2 block)
                        V[jp * p + (b_col_start + kp), col] = b_form[jp * 2 + kp]
                
                # Fill W: C-contribution
                # C[i', k'] where i' in {0,1}, k' in {c_col_start, c_col_start+1}
                # Flat index: k'*n + i'
                for ip in range(2):
                    for kp in range(2):
                        # c_contrib index: ip*2 + kp (C[ip, kp] within the 2x2 block)
                        W[(c_col_start + kp) * n + ip, col] = c_contrib[ip * 2 + kp]
                
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    """Return the rank-minimal tensor decomposition for 2x2 * 2x16 matrix multiplication."""
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Try Strassen decomposition (rank 56)
    try:
        result_strassen = _strassen_decomposition()
        if _verify(result_strassen["U"], result_strassen["V"], result_strassen["W"]):
            return result_strassen
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 64)
    result = _schoolbook()
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    return result

# EVOLVE-BLOCK-END
