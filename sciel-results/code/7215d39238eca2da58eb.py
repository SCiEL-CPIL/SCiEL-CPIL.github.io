"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def _to_frac(x):
    """Convert a float to a Fraction with small denominator."""
    if abs(x) < 1e-12:
        return Fraction(0, 1)
    x_float = float(x)
    # Try to find a simple fraction
    for den in range(1, 1001):
        num = x_float * den
        num_rounded = round(num)
        if abs(num - num_rounded) < 1e-10:
            return Fraction(int(num_rounded), den)
    # Fallback to high precision
    return Fraction(x_float).limit_denominator(10**8)

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We will build a decomposition using Strassen's algorithm for the 2x4x4 part
    # and standard multiplication for the 1x4x4 part.
    
    # Total rank:
    # 2x4x4: Can be done with 2 Strassen blocks (each 2x2x2) for the 2x2 parts?
    # No, 2x4x4 can be viewed as two 2x2x4 blocks? Or one 2x4x4.
    # Let's decompose 2x4x4 into four 2x2x2 blocks? No, that's not right.
    # A is 2x4, B is 4x4.
    # We can split A into A1 (2x2) and A2 (2x2) horizontally.
    # B into B1 (2x4) and B2 (2x4) vertically.
    # C = A1*B1 + A2*B2.
    # Each A_i * B_i is a 2x2 * 2x4 multiplication.
    # This can be done by treating it as two 2x2x2 multiplications?
    # A1 (2x2) * B1 (2x4) -> C1 (2x4).
    # We can split B1 into B1a (2x2) and B1b (2x2).
    # C1 = A1*B1a + A1*B1b.
    # Each A1*B1a is 2x2x2, rank 7 (Strassen).
    # So for A1*B1, we need 2 Strassen blocks -> rank 14.
    # Similarly for A2*B2, rank 14.
    # Total for 2x4x4 part: 14 + 14 = 28.
    
    # For the 1x4x4 part:
    # A is 1x4, B is 4x4.
    # C = A*B (1x4).
    # This is standard: C_k = sum_j A_j B_jk.
    # This requires 4 multiplications (or rank 4).
    # Can we do better? No, 1x4x4 tensor has rank 4.
    
    # Total rank = 28 + 4 = 32.
    
    rank = 32
    dim_U = n * m  # 12
    dim_V = m * p  # 16
    dim_W = p * n  # 12
    
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # Indices:
    # A rows: i1, i2 (local 0,1)
    # B cols: k1, k2 (local 0,1)
    # Shared index j: j1, j2 (local 0,1)
    # Global indices:
    # i_global, j_global, k_global
    def add_strassen_block(i_offset, j_offset, k_offset, col_start):
        # Strassen 2x2x2
        # A is 2x2, B is 2x2, C is 2x2
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map local indices to global.
        # A local (r, c) -> global U index: (i_offset + r)*m + (j_offset + c)
        # B local (r, c) -> global V index: (j_offset + r)*p + (k_offset + c)
        # C local (r, c) -> global W index: (k_offset + c)*n + (i_offset + r)
        
        # M1
        U[(i_offset+0)*m + (j_offset+0), col_start] = 1
        U[(i_offset+1)*m + (j_offset+1), col_start] = 1
        V[(j_offset+0)*p + (k_offset+0), col_start] = 1
        V[(j_offset+1)*p + (k_offset+1), col_start] = 1
        W[(k_offset+0)*n + (i_offset+0), col_start] = 1
        W[(k_offset+1)*n + (i_offset+1), col_start] = 1
        col_start += 1
        
        # M2
        U[(i_offset+1)*m + (j_offset+0), col_start] = 1
        U[(i_offset+1)*m + (j_offset+1), col_start] = 1
        V[(j_offset+0)*p + (k_offset+0), col_start] = 1
        W[(k_offset+0)*n + (i_offset+1), col_start] = 1
        W[(k_offset+1)*n + (i_offset+1), col_start] = 1
        col_start += 1
        
        # M3
        U[(i_offset+0)*m + (j_offset+0), col_start] = 1
        V[(j_offset+0)*p + (k_offset+1), col_start] = 1
        V[(j_offset+1)*p + (k_offset+1), col_start] = -1
        W[(k_offset+1)*n + (i_offset+0), col_start] = 1
        col_start += 1
        
        # M4
        U[(i_offset+1)*m + (j_offset+1), col_start] = 1
        V[(j_offset+1)*p + (k_offset+0), col_start] = 1
        V[(j_offset+0)*p + (k_offset+0), col_start] = -1
        W[(k_offset+0)*n + (i_offset+0), col_start] = 1
        W[(k_offset+1)*n + (i_offset+1), col_start] = 1
        col_start += 1
        
        # M5
        U[(i_offset+0)*m + (j_offset+0), col_start] = 1
        U[(i_offset+0)*m + (j_offset+1), col_start] = 1
        V[(j_offset+1)*p + (k_offset+1), col_start] = 1
        W[(k_offset+0)*n + (i_offset+0), col_start] = -1
        W[(k_offset+1)*n + (i_offset+0), col_start] = 1
        col_start += 1
        
        # M6
        U[(i_offset+1)*m + (j_offset+0), col_start] = 1
        U[(i_offset+0)*m + (j_offset+0), col_start] = -1
        V[(j_offset+0)*p + (k_offset+0), col_start] = 1
        V[(j_offset+0)*p + (k_offset+1), col_start] = 1
        W[(k_offset+1)*n + (i_offset+1), col_start] = 1
        col_start += 1
        
        # M7
        U[(i_offset+0)*m + (j_offset+1), col_start] = 1
        U[(i_offset+1)*m + (j_offset+1), col_start] = -1
        V[(j_offset+1)*p + (k_offset+0), col_start] = 1
        V[(j_offset+1)*p + (k_offset+1), col_start] = 1
        W[(k_offset+0)*n + (i_offset+0), col_start] = 1
        col_start += 1
        
        return col_start
    
    # Part 1: 2x4x4 (Rows 0,1 of A)
    # Split into two 2x2x4 blocks based on j (0,1) and (2,3)
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # This is 2x2x4. Split k into {0,1} and {2,3}.
    # Sub-block 1.1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen
    # Sub-block 1.2: i in {0,1}, j in {0,1}, k in {2,3} -> Strassen
    
    # Sub-block 1.1: i_offset=0, j_offset=0, k_offset=0
    col = add_strassen_block(0, 0, 0, col)
    
    # Sub-block 1.2: i_offset=0, j_offset=0, k_offset=2
    col = add_strassen_block(0, 0, 2, col)
    
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1,2,3}
    # Sub-block 2.1: i_offset=0, j_offset=2, k_offset=0
    col = add_strassen_block(0, 2, 0, col)
    
    # Sub-block 2.2: i_offset=0, j_offset=2, k_offset=2
    col = add_strassen_block(0, 2, 2, col)
    
    # Part 2: 1x4x4 (Row 2 of A)
    # Standard multiplication: C_k = sum_j A_2j B_jk
    # This is rank 4.
    # For each j in 0..3, k in 0..3? No, that's 16.
    # Wait, C is 1x4. C_k = sum_{j=0}^3 A_{2,j} B_{j,k}.
    # This is 4 dot products of length 4? No.
    # It's 4 outputs, each is a sum of 4 products.
    # The tensor is T[i,j,k] where i=2.
    # T[2,j,k] = 1.
    # We need sum_c U[2*m+j, c] V[j*p+k, c] W[k*n+2, c] = 1.
    # We can use 4 slices, one for each j?
    # If we set c=j, then U[2*m+j, j] = 1, V[j*p+k, j] = 1 for all k? No, V is fixed per slice.
    # Let's use 4 slices.
    # Slice j:
    # U[2*m+j, j] = 1
    # V[j*p+k, j] = 1 for all k? No, that would sum over k.
    # We want W[k*n+2, j] to pick out the k-th element.
    # If V[j*p+k, j] = 1 for all k, then the term is U * 1 * W.
    # Sum over j: sum_j U[2*m+j, j] W[k*n+2, j].
    # We want this to be 1 for the specific j? No, we want the sum to be 1 for ALL j,k?
    # No, the condition is T[2,j,k] = 1 for ALL j,k.
    # So for fixed j,k, sum_c ... = 1.
    # If we use 4 slices, one for each j:
    # Slice j: U[2*m+j, j] = 1. V[j*p+k, j] = 1 for all k. W[k*n+2, j] = 1 for all k?
    # Then for a specific j', k':
    # Sum over c=j: U[2*m+j', j] V[j'*p+k', j] W[k'*n+2, j]
    # = U[2*m+j', j'] * 1 * 1 = 1.
    # And for c != j', U[2*m+j', c] = 0.
    # So yes, this works.
    # But wait, V[j*p+k, j] = 1 for all k means the V vector for slice j has 1s in positions j*p, j*p+1, j*p+2, j*p+3.
    # And W[k*n+2, j] = 1 for all k means the W vector for slice j has 1s in positions 2, 6, 10 (since n=4, k=0,1,2,3 -> k*4+2).
    
    for j in range(4):
        U[2*m + j, col] = 1
        for k in range(4):
            V[j*p + k, col] = 1
            W[k*n + 2, col] = 1
        col += 1
    
    # Verify rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to list of Fractions
    U_list = [[_to_frac(U[i, c]) for c in range(rank)] for i in range(dim_U)]
    V_list = [[_to_frac(V[i, c]) for c in range(rank)] for i in range(dim_V)]
    W_list = [[_to_frac(W[i, c]) for c in range(rank)] for i in range(dim_W)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
