"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-88 tensor decomposition for 3x4 * 4x8 multiplication.
    Approach: Recursive Strassen-based decomposition.
    1. Split 3x4x8 into 2x4x8 (top 2 rows) and 1x4x8 (bottom row).
    2. Split 2x4x8 into two 2x4x4 blocks.
    3. Split each 2x4x4 into four 2x2x2 blocks.
    4. Use Strassen (rank 7) for each 2x2x2 block.
    5. Use schoolbook (rank 32) for the 1x4x8 block.
    Total Rank = 8 * 7 + 32 = 88.
    """
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_strassen_term(a_offsets, b_offsets, c_offsets, coeff_a, coeff_b, coeff_c):
        """
        Adds one term to the decomposition.
        a_offsets: (i, j) indices for A
        b_offsets: (j, k) indices for B
        c_offsets: (k, i) indices for C (W[k*n+i])
        coeff_a, coeff_b, coeff_c: coefficients for the rank-1 tensor
        """
        nonlocal current_rank
        idx = current_rank
        current_rank += 1
        
        # A index: i * m + j
        # U shape: (n*m, rank)
        # We need to add to U[i*m+j][idx]
        # But Strassen sums multiple A elements.
        # The function signature in the prompt implies U, V, W are matrices.
        # The decomposition is T = sum U[:,t] V[:,t]^T W[:,t]^T?
        # No, standard tensor decomposition:
        # T_{ijk} = sum_t U_{i, j, t} V_{j, k, t} W_{k, i, t} ?
        # The prompt says:
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        # This implies the tensor is flattened.
        # T_{(i,j), (j,k), (k,i)}?
        # Let's check the convention: "W[k*n+i,r]"
        # This suggests the third mode is indexed by k and i.
        # Usually matrix multiplication tensor is $T_{i,j,k} = A_{i,j} B_{j,k} C_{k,i}$?
        # No, $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
        # Tensor form: $T_{i,j,k} = A_{i,j} B_{j,k} C_{k,i}$ is for trace?
        # The prompt says: "output factor follows the official trace convention W[k*n+i,r]"
        # This is slightly ambiguous. Let's look at the schoolbook implementation in the seed.
        
        # Seed:
        # U[i*m + j][col] = 1
        # V[j*p + k][col] = 1
        # W[k*n + i][col] = 1
        # This corresponds to $T_{i,j,k} = A_{i,j} B_{j,k} C_{k,i}$?
        # If we sum over col:
        # sum_col U[i,j,col] V[j,k,col] W[k,i,col]
        # For schoolbook, term (i,j,k) has U=1, V=1, W=1.
        # Sum = 1.
        # This reconstructs the tensor $T_{i,j,k} = 1$ for all i,j,k?
        # No, the tensor for matrix multiplication is $T_{i,j,k}$ such that
        # $\langle A, T \cdot B \rangle$?
        # Standard definition: $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
        # This can be written as $C = T(A \otimes B)$.
        # The tensor $T$ has indices $i, j, k$.
        # $T_{i,j,k} = 1$ if it contributes to $C_{i,k}$ via $A_{i,j} B_{j,k}$.
        # In the seed:
        # U index: $i*m+j$.
        # V index: $j*p+k$.
        # W index: $k*n+i$.
        # This matches $A_{i,j}, B_{j,k}, C_{k,i}$?
        # Wait, $C$ is $n \times p$. $C_{i,k}$.
        # The W index is $k*n+i$. This is column-major for a $p \times n$ matrix?
        # Or row-major for a $n \times p$ matrix transposed?
        # If $W$ is $p \times n$, index $k*n+i$ is row $k$, col $i$.
        # So $W_{k,i}$.
        # The term is $A_{i,j} B_{j,k} W_{k,i}$.
        # This looks like the trace of $A B W$?
        # $\sum_{i,j,k} A_{i,j} B_{j,k} W_{k,i} = \text{Tr}(A B W)$.
        # Yes, this is the trace convention.
        
        # So for a rank-1 term with coefficients $u, v, w$ for specific indices:
        # We add $u$ to $U[i*m+j][idx]$
        # We add $v$ to $V[j*p+k][idx]$
        # We add $w$ to $W[k*n+i][idx]$
        
        pass # Placeholder for logic below

    # We need to implement Strassen for 2x2x2
    # A is 2x2, B is 2x2, C is 2x2.
    # Indices:
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1)
    
    # Strassen M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    def add_term(a_idx, b_idx, c_idx, coeff_a, coeff_b, coeff_c):
        nonlocal current_rank
        if current_rank >= rank:
            return
        idx = current_rank
        current_rank += 1
        # a_idx is (i, j)
        # b_idx is (j, k)
        # c_idx is (k, i)
        i, j = a_idx
        j2, k = b_idx
        k2, i2 = c_idx
        # Verify index consistency for the tensor structure
        # The tensor is fixed structure T_{i,j,k}.
        # A term is a rank-1 tensor $u_{i,j} v_{j,k} w_{k,i}$.
        # So we must have matching j in A and B, and matching k in B and C, and matching i in C and A?
        # No, the rank-1 tensor is a product of three vectors.
        # $T^{(t)}_{i,j,k} = U_{i,j,t} V_{j,k,t} W_{k,i,t}$.
        # So for a specific term, we specify the non-zero entries.
        # If $U_{i,j,t} = c_a$, $V_{j,k,t} = c_b$, $W_{k,i,t} = c_c$.
        # Then $T^{(t)}_{i,j,k} = c_a c_b c_c$.
        
        # In Strassen, we have linear combinations.
        # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
        # This is not a single rank-1 term $A_{ij} B_{jk} C_{ki}$.
        # It is a sum of 4 rank-1 terms?
        # $(A_{00}+A_{11})(B_{00}+B_{11}) = A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
        # But Strassen counts this as ONE multiplication.
        # How does this map to the tensor decomposition?
        # The tensor decomposition requires the final C to be correct.
        # $C_{ij} = \sum M_{...}$.
        # The intermediate $M$'s are linear combinations of products of linear combinations of A and B.
        # This is exactly a tensor decomposition!
        # Each $M$ is a "multiplication" node.
        # $M_1 = L_A(A) L_B(B)$.
        # $L_A(A) = A_{00} + A_{11}$.
        # $L_B(B) = B_{00} + B_{11}$.
        # Then $C$ is a linear combination of $M$'s.
        # $C_{00} = M_1 + M_4 - M_5 + M_7$.
        
        # So for each $M$, we define:
        # $U$ entries: coefficients of $A$ in $L_A$.
        # $V$ entries: coefficients of $B$ in $L_B$.
        # $W$ entries: coefficients of $M$ in $C$.
        
        # For $M_1$:
        # $L_A = A_{00} + A_{11}$.
        # $U[0,m+0] += 1$ (for A00)
        # $U[1,m+1] += 1$ (for A11)
        # $L_B = B_{00} + B_{11}$.
        # $V[0,p+0] += 1$ (for B00)
        # $V[1,p+1] += 1$ (for B11)
        # $C_{00} += M_1 \implies W[0,n+0] += 1$ (for C00, k=0, i=0)
        # $C_{11} += M_1 \implies W[1,n+1] += 1$ (for C11, k=1, i=1)
        
        # Wait, the W index is $k*n+i$.
        # For $C_{00}$, $k=0, i=0 \implies 0$.
        # For $C_{11}$, $k=1, i=1 \implies 1*n+1$.
        
        # So we need to add these contributions to the U, V, W matrices for the current rank index.

    def apply_strassen(offset_i, offset_j, offset_k):
        """
        Applies Strassen to a 2x2x2 block starting at A[offset_i:offset_i+2, offset_j:offset_j+2]
        and B[offset_j:offset_j+2, offset_k:offset_k+2], contributing to
        C[offset_k:offset_k+2, offset_i:offset_i+2].
        """
        nonlocal current_rank
        
        # Define the 7 multiplications
        # (coeffs for A, coeffs for B, coeffs for C)
        # A indices are relative to offset_i, offset_j
        # B indices are relative to offset_j, offset_k
        # C indices are relative to offset_k, offset_i
        
        strassen_terms = [
            # M1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2
            ([(1,0,1), (1,1,1)], [(0,0,1)], []), # C contribution handled separately?
            # No, better to list C contributions for each M.
            
            # Let's list (A_coeffs, B_coeffs, C_coeffs)
            # A_coeffs: list of (di, dj, coeff)
            # B_coeffs: list of (dj, dk, coeff)
            # C_coeffs: list of (dk, di, coeff)
            
            # M1: (A00+A11)(B00+B11)
            # C00 += M1, C11 += M1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            
            # M2: (A10+A11)B00
            # C10 += M2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1)]), # C10 is k=0, i=1
            
            # M3: A00(B01-B11)
            # C01 += M3
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1)]), # C01 is k=1, i=0
            
            # M4: A11(B10-B00)
            # C00 += M4, C10 += M4
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
            
            # M5: (A00+A01)B11
            # C00 -= M5, C01 += M5
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]),
            
            # M6: (A10-A00)(B00+B01)
            # C11 += M6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            
            # M7: (A01-A11)(B10+B11)
            # C00 += M7, C11 -= M7? No.
            # C11 = M1 - M2 + M3 + M6.
            # C00 = M1 + M4 - M5 + M7.
            # So M7 contributes to C00 (+1) and ... wait.
            # Standard Strassen:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            # So M7 only contributes to C00.
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            idx = current_rank
            current_rank += 1
            
            for di, dj, ca in a_coeffs:
                i = offset_i + di
                j = offset_j + dj
                U[i*m + j][idx] += ca
                
            for dj, dk, cb in b_coeffs:
                j = offset_j + dj
                k = offset_k + dk
                V[j*p + k][idx] += cb
                
            for dk, di, cc in c_coeffs:
                k = offset_k + dk
                i = offset_i + di
                W[k*n + i][idx] += cc

    # Apply Strassen to the 2x4x8 part
    # A rows 0,1. B cols 0..7. C cols 0..7.
    # Split B into left (0-3) and right (4-7).
    
    # Left part (2x4x4)
    # Split into 4 2x2x2 blocks
    # A cols 0,1 and 2,3
    # B cols 0,1 and 2,3
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    apply_strassen(0, 0, 0)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    apply_strassen(0, 0, 2)
    # Block 3: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    apply_strassen(0, 2, 0)
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    apply_strassen(0, 2, 2)
    
    # Right part (2x4x4)
    # Block 5: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    apply_strassen(0, 0, 4)
    # Block 6: A[0:2, 0:2] * B[0:2, 6:8] -> C[0:2, 6:8]
    apply_strassen(0, 0, 6)
    # Block 7: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    apply_strassen(0, 2, 4)
    # Block 8: A[0:2, 2:4] * B[2:4, 6:8] -> C[0:2, 6:8]
    apply_strassen(0, 2, 6)
    
    # Now the 1x4x8 part (Row 2 of A)
    # A[2, 0:4] * B[0:4, 0:8] -> C[0:8, 2]
    # Schoolbook: 32 terms
    for j in range(4):
        for k in range(8):
            idx = current_rank
            current_rank += 1
            # A[2, j]
            U[2*m + j][idx] = 1
            # B[j, k]
            V[j*p + k][idx] = 1
            # C[k, 2] -> W[k*n + 2]
            W[k*n + 2][idx] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
