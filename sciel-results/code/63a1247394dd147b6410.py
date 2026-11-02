import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _get_strassen_factors(n):
    """
    Returns U, V, W for n x n x n matrix multiplication using Strassen.
    U: (n*n, rank), V: (n*n, rank), W: (n*n, rank)
    Indices: U[i*n+j, r], V[j*n+k, r], W[k*n+i, r]
    """
    if n == 1:
        return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
    
    # Recursive step
    n_half = n // 2
    U_half, V_half, W_half = _get_strassen_factors(n_half)
    rank_half = U_half.shape[1]
    
    # There are 7 Strassen products
    # Each product involves a linear combination of blocks from A and B.
    # The "U" part for a specific Strassen factor (e.g. M1) is a linear combination of the 4 blocks of A.
    # The "V" part is a linear combination of the 4 blocks of B.
    # The "W" part is a linear combination of the 4 blocks of C.
    
    # We need to map the local 2x2 block structure to the global n x n structure.
    
    # Let's define the 7 Strassen formulas for 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00 + A11) (B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will iterate over the 7 factors (f in 0..6).
    # For each factor, we need to construct the U, V, W vectors for the n x n case.
    # The U vector for factor f is a concatenation of 4 blocks, each of size n_half * n_half.
    # Each block is a linear combination of the U_half vectors?
    # No, U_half is for the multiplication of the blocks themselves.
    
    # Let's clarify the structure.
    # The multiplication of two n_half x n_half matrices $X$ and $Y$ to produce $Z$ is decomposed into rank-1 terms using U_half, V_half, W_half.
    # $Z_{i,j} = \sum_r U_{i,j,r} X_{i',j'} ...$ No.
    # $Z = \sum_r u_r \otimes v_r \otimes w_r$ where $u_r$ is a vector of length $n_{half}^2$ representing the input to the "A" side, etc.
    # Actually, $Z_{p,q} = \sum_{i,j} X_{p,i} Y_{i,q}$.
    # The tensor decomposition gives $Z_{p,q} = \sum_r U[p,r] V[q,r] W[r,r]$? No.
    # Standard form: $T = \sum_r u_r \otimes v_r \otimes w_r$.
    # $u_r$ is a vector of size $n_{half}^2$ (flattened matrix).
    # $v_r$ is a vector of size $n_{half}^2$.
    # $w_r$ is a vector of size $n_{half}^2$.
    # The product is $\sum_r (u_r^T X) (v_r^T Y) w_r$? No.
    # The tensor contraction is:
    # $Z_{p,q} = \sum_r U_{p,q,r} V_{i,j,r} W_{k,l,r} X_{p,k} Y_{l,q}$?
    
    # Let's stick to the flat index definition provided in the task:
    # $T[a, b, c] = \sum_r U[a, r] V[b, r] W[c, r]$
    # $a = i \cdot n + j$ (row i, col j of A)
    # $b = j \cdot n + k$ (row j, col k of B)
    # $c = k \cdot n + i$ (row k, col i of C) -- Wait, C is $p \times n$ in task?
    # Task: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies C is indexed by $k, i$.
    # For $n \times n \times n$, $C_{k,i}$?
    # Usually $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # If $W$ is indexed by $k \cdot n + i$, then $W$ corresponds to $C_{k,i}$?
    # Let's check the validation code in the seed:
    # `c = k * n + i`
    # `s += U[a][r] * V[b][r] * W[c][r]`
    # `if abs(s - 1.0) > 1e-9: return False`
    # This checks if $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1$ for all $i,j,k$.
    # This corresponds to the tensor $T_{i,j,k} = \delta_{i,k}$? No.
    # Matrix multiplication: $(AB)_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # The tensor for matmul is $T = \sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{k,i}$?
    # If we contract $A$ with $U$ and $B$ with $V$, we get a vector of length $p \cdot n$.
    # Let $M_{k,i} = \sum_j A_{i,j} B_{j,k}$.
    # Then $M$ is the result.
    # The decomposition must satisfy:
    # $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1$ if the term $A_{i,j} B_{j,k}$ contributes to $M_{k,i}$?
    # Yes, $M_{k,i} = \sum_{j} A_{i,j} B_{j,k}$.
    # So the coefficient of $A_{i,j} B_{j,k}$ in $M_{k,i}$ is 1.
    # The coefficient in any other $M_{k', i'}$ is 0.
    # So the tensor is indeed defined by $T_{i,j,k} = 1$ if we view it as mapping inputs to outputs.
    # Specifically, the term $A_{i,j} B_{j,k}$ appears in $C_{k,i}$ (using the trace convention $W[k \cdot n + i]$).
    
    # So for the base case $n=1$:
    # $a=0, b=0, c=0$.
    # $U[0,0]=1, V[0,0]=1, W[0,0]=1$.
    # Sum = 1. Correct.
    
    # For the recursive step $n=2$ from $n=1$:
    # We have 4 blocks for A, 4 for B, 4 for C.
    # The decomposition is a sum of 7 terms.
    # Term $f$ (0..6) corresponds to a Strassen product $M_f$.
    # $M_f = A_{comb} B_{comb}$.
    # $A_{comb}$ is a linear combination of the 4 A-blocks.
    # $B_{comb}$ is a linear combination of the 4 B-blocks.
    # $M_f$ is a $n_{half} \times n_{half}$ matrix.
    # $M_f$ contributes to the 4 C-blocks with certain coefficients.
    
    # For each Strassen factor $f$, we need to generate $rank\_half$ rank-1 tensors for the multiplication of the $n_{half} \times n_{half}$ blocks.
    # Let $U_f, V_f, W_f$ be the $n_{half} \times n_{half} \times n_{half}$ decomposition (size $n_{half}^2 \times rank\_half$).
    # Wait, $U_f$ in the recursive call is for the "A" input of the block multiplication.
    # The "A" input for the block multiplication is $A_{comb}$.
    # $A_{comb} = \sum_{block\_idx} \alpha_{block\_idx} A_{block\_idx}$.
    # So the U-vector for the global factor corresponding to local factor $l$ in Strassen factor $f$ is:
    # $U_{global} = \sum_{block\_idx} \alpha_{block\_idx} (U_{half, l} \text{ placed in block\_idx})$.
    # Similarly for V and W.
    # But W is different.
    # $M_f = A_{comb} B_{comb}$.
    # $C = \sum_f \beta_{f, block\_idx} M_f$.
    # So $W_{global}$ for local factor $l$ in Strassen factor $f$ is:
    # $W_{global} = \sum_{block\_idx} \beta_{f, block\_idx} (W_{half, l} \text{ placed in block\_idx})$.
    
    # Let's define the coefficients for the 7 Strassen factors.
    # A-blocks: 00, 01, 10, 11 (indices 0,1,2,3 in 2x2 block grid)
    # B-blocks: 00, 01, 10, 11
    # C-blocks: 00, 01, 10, 11
    
    # M1: A=(00,11), B=(00,11)
    # M2: A=(10,11), B=(00)
    # M3: A=(00), B=(01,-11)
    # M4: A=(11), B=(10,-00)
    # M5: A=(00,01), B=(11)
    # M6: A=(10,-00), B=(00,01)
    # M7: A=(01,-11), B=(10,11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will build U, V, W for size n.
    
    U = np.zeros((n*n, 7 * rank_half))
    V = np.zeros((n*n, 7 * rank_half))
    W = np.zeros((n*n, 7 * rank_half))
    
    # Define the block coefficients
    # Each entry is a dict {block_idx: coeff}
    # Block indices: 0:00, 1:01, 2:10, 3:11
    
    A_coeffs = [
        {0: 1, 3: 1}, # M1
        {2: 1, 3: 1}, # M2
        {0: 1},       # M3
        {3: 1},       # M4
        {0: 1, 1: 1}, # M5
        {2: 1, 0: -1},# M6
        {1: 1, 3: -1} # M7
    ]
    
    B_coeffs = [
        {0: 1, 3: 1}, # M1
        {0: 1},       # M2
        {1: 1, 3: -1},# M3
        {2: 1, 0: -1},# M4
        {3: 1},       # M5
        {0: 1, 1: 1}, # M6
        {2: 1, 3: 1}  # M7
    ]
    
    C_coeffs = [
        {0: 1, 3: -1, 1: 1, 2: 1}, # M1 -> C00(1), C01(0), C10(0), C11(1)?
        # Wait, let's map C indices correctly.
        # C00 is block 0. C01 is block 1. C10 is block 2. C11 is block 3.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M1: C00 += 1, C11 += 1.
        # For M2: C10 += 1, C11 -= 1.
        # For M3: C01 += 1, C11 += 1.
        # For M4: C00 += 1, C10 += 1.
        # For M5: C00 -= 1, C01 += 1.
        # For M6: C11 += 1.
        # For M7: C00 += 1.
        
        # Let's re-verify Strassen:
        # C00 = M1 - M5 + M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: C00(+1), C11(+1)
        # M2: C10(+1), C11(-1)
        # M3: C01(+1), C11(+1)
        # M4: C00(+1), C10(+1)
        # M5: C00(-1), C01(+1)
        # M6: C11(+1)
        # M7: C00(+1)
        
        # Block indices for C: 0:00, 1:01, 2:10, 3:11
    ]
    
    C_coeffs = [
        {0: 1, 3: 1}, # M1
        {2: 1, 3: -1},# M2
        {1: 1, 3: 1}, # M3
        {0: 1, 2: 1}, # M4
        {0: -1, 1: 1},# M5
        {3: 1},       # M6
        {0: 1}        # M7
    ]
    
    # Now populate U, V, W
    for f in range(7):
        col_offset = f * rank_half
        for l in range(rank_half):
            col = col_offset + l
            
            # U part
            for b_idx, coeff in A_coeffs[f].items():
                # Place U_half[:, l] into the block b_idx of U[:, col]
                # Block b_idx corresponds to rows/cols in the n x n matrix.
                # Block 0: rows 0..n/2-1, cols 0..n/2-1
                # Block 1: rows 0..n/2-1, cols n/2..n-1
                # Block 2: rows n/2..n-1, cols 0..n/2-1
                # Block 3: rows n/2..n-1, cols n/2..n-1
                
                r_start = (b_idx // 2) * n_half
                c_start = (b_idx % 2) * n_half
                
                # Local indices in U_half are i*n_half + j
                # Global indices are (r_start + i)*n + (c_start + j)
                
                for i in range(n_half):
                    for j in range(n_half):
                        local_idx = i * n_half + j
                        global_idx = (r_start + i) * n + (c_start + j)
                        U[global_idx, col] += coeff * U_half[local_idx, l]
            
            # V part
            for b_idx, coeff in B_coeffs[f].items():
                r_start = (b_idx // 2) * n_half
                c_start = (b_idx % 2) * n_half
                
                for i in range(n_half):
                    for j in range(n_half):
                        local_idx = i * n_half + j
                        global_idx = (r_start + i) * n + (c_start + j)
                        V[global_idx, col] += coeff * V_half[local_idx, l]
            
            # W part
            # Note: W indices in the base case were $k \cdot n + i$.
            # In the recursive step, the "C" matrix is $n \times n$.
            # The blocks of C are also arranged in a 2x2 grid.
            # Block 0 (C00): rows 0..n/2-1, cols 0..n/2-1.
            # The local W indices are $k_{local} \cdot n_{half} + i_{local}$.
            # The global W indices are $k_{global} \cdot n + i_{global}$.
            # $k_{global} = r_start + k_{local}$
            # $i_{global} = c_start + i_{local}$
            
            for b_idx, coeff in C_coeffs[f].items():
                r_start = (b_idx // 2) * n_half
                c_start = (b_idx % 2) * n_half
                
                for k_local in range(n_half):
                    for i_local in range(n_half):
                        local_idx = k_local * n_half + i_local
                        global_idx = (r_start + k_local) * n + (c_start + i_local)
                        W[global_idx, col] += coeff * W_half[local_idx, l]
                        
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0]) if U else 0
    nm = n * m
    mp = m * p
    pn = p * n
    
    # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    # Check all 1 positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Check some 0 positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_alt = (k + 1) % p * n + i
                if c_alt != c:
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c_alt][r]
                    if abs(s) > 1e-9:
                        return False
    return True

def _construct_strassen_4x8():
    """
    Constructs a rank-98 decomposition for 4x4x8 matrix multiplication.
    """
    n, m, p = 4, 4, 8
    
    # Get 4x4x4 Strassen factors
    U4, V4, W4 = _get_strassen_factors(4)
    rank4 = U4.shape[1] # Should be 49
    
    U = np.zeros((n*m, 2*rank4))
    V = np.zeros((m*p, 2*rank4))
    W = np.zeros((p*n, 2*rank4))
    
    # Left half (columns 0-3 of B, rows 0-3 of C)
    # U is same for both halves (A is shared)
    U[:, 0:rank4] = U4
    
    # V for left half: rows j*4 + k for k in 0..3
    # V4 has rows j*4 + k for k in 0..3 (since n=4)
    # So V4 fits directly into the first 16 rows of V_left?
    # V is 32x98.
    # V[:, 0:rank4] should have V4 in rows 0..15.
    V[0:16, 0:rank4] = V4
    
    # W for left half: rows k*4 + i for k in 0..3
    # W4 has rows k*4 + i for k in 0..3.
    # So W4 fits directly into the first 16 rows of W_left?
    # W is 32x98.
    # W[:, 0:rank4] should have W4 in rows 0..15.
    W[0:16, 0:rank4] = W4
    
    # Right half (columns 4-7 of B, rows 4-7 of C)
    # U is same
    U[:, rank4:2*rank4] = U4
    
    # V for right half: rows j*4 + (k+4) for k in 0..3
    # These are rows 4, 8, 12, 16, 20, 24, 28, 32?
    # No, j is 0..3. k is 0..3.
    # Index b = j*4 + k.
    # For right half, k_global = k_local + 4.
    # So b = j*4 + (k_local + 4).
    # If j=0, k=0..3 -> b=4,5,6,7.
    # If j=1, k=0..3 -> b=8,9,10,11.
    # ...
    # So we need to map V4 rows (0..15) to these specific rows in V.
    
    # V4 row index: j_local * 4 + k_local.
    # V row index: j_global * 4 + k_global = j_local * 4 + (k_local + 4).
    
    for r in range(16):
        j_local = r // 4
        k_local = r % 4
        v_row_in_V4 = r
        v_row_in_V = j_local * 4 + (k_local + 4)
        V[v_row_in_V, rank4:2*rank4] = V4[v_row_in_V4, :]
        
    # W for right half: rows (k+4)*4 + i for k in 0..3
    # W4 row index: k_local * 4 + i_local.
    # W row index: k_global * 4 + i_global = (k_local + 4) * 4 + i_local.
    
    for r in range(16):
        k_local = r // 4
        i_local = r % 4
        w_row_in_W4 = r
        w_row_in_W = (k_local + 4) * 4 + i_local
        W[w_row_in_W, rank4:2*rank4] = W4[w_row_in_W4, :]
        
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try Strassen 4x8
    try:
        U, V, W = _construct_strassen_4x8()
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
        
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
