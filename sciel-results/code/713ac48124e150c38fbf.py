"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add an entry to the current tensor slice
    # We build the matrices column by column
    current_col = 0
    
    # Function to apply a standard rank-1 tensor addition
    # U[u] += u_coeff * x
    # V[v] += v_coeff * y
    # W[w] += w_coeff * z
    # In our context, coefficients are +/- 1.
    def add_term(u_idx, v_idx, w_idx, sign):
        nonlocal current_col
        U[u_idx, current_col] += sign
        V[v_idx, current_col] += sign
        W[w_idx, current_col] += sign

    # Helper to finalize the current column and move to next
    def next_col():
        nonlocal current_col
        current_col += 1

    # Define the Strassen block logic inline for a 2x2x2 sub-problem
    # i_base: row offset in A/C (0 or 1, but here always 0 since n=2)
    # j_base: col offset in A / row offset in B (0, 2)
    # k_base: col offset in B / col offset in C (0, 2, 4)
    
    def apply_strassen(i_b, j_b, k_b):
        # Local indices
        u00 = i_b * m + j_b
        u01 = i_b * m + (j_b + 1)
        u10 = (i_b + 1) * m + j_b
        u11 = (i_b + 1) * m + (j_b + 1)
        
        v00 = j_b * p + k_b
        v01 = j_b * p + (k_b + 1)
        v10 = (j_b + 1) * p + k_b
        v11 = (j_b + 1) * p + (k_b + 1)
        
        w00 = k_b * n + i_b
        w01 = (k_b + 1) * n + i_b
        w10 = k_b * n + (i_b + 1)
        w11 = (k_b + 1) * n + (i_b + 1)
        
        # M0: (A00+A11)(B00+B11) -> C00, C11
        add_term(u00, v00, w00, 1)
        add_term(u11, v00, w00, 0) # Placeholder, we need combined terms? 
        # Wait, the tensor decomposition is a sum of rank-1 tensors.
        # T = sum_k U_k \otimes V_k \otimes W_k
        # Each rank-1 term contributes 1 to U, 1 to V, 1 to W (times coefficients).
        # Strassen terms are of the form:
        # T_M0 = (A00+A11) \otimes (B00+B11) \otimes (C00 + C11) ?? No.
        # The standard Strassen decomposition expresses the multiplication map.
        # The tensor T_mul has entries T[i,j,k] = 1 if C_ik gets A_ij B_jk.
        # We need to decompose T into sum of rank-1 tensors.
        # Let's use the explicit rank-1 expansions for Strassen.
        
        # Actually, it's safer to just use the pre-computed indices from the seed logic 
        # but restructured. The seed logic was:
        # U[i*m+j][col] += 1, etc.
        
        # Let's re-verify the seed's Strassen implementation correctness first.
        # The seed used specific index combinations. I will replicate that exact math
        # but change the code structure significantly.
        
        pass # This helper is a stub for the new structure below

    # New Structure: Explicit loops over the block types
    
    # 1. Process Strassen blocks for j-pairs (0,1) and (2,3)
    # j_bases are 0 and 2
    # k_bases are 0, 2, 4
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            i_base = 0 # n=2, so only one i-pair
            
            # --- Strassen M0: (A00+A11)(B00+B11) ---
            # Contributes to C00 and C11?
            # Actually, let's look at the seed's `add_strassen_block`.
            # It had 7 distinct columns.
            
            # M0
            U[i_base*m+j_base, current_col] = 1
            U[(i_base+1)*m+(j_base+1), current_col] = 1
            V[j_base*p+k_base, current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = 1
            W[k_base*n+i_base, current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = 1
            next_col()
            
            # M1: (A10+A11)B00
            U[(i_base+1)*m+j_base, current_col] = 1
            U[(i_base+1)*m+(j_base+1), current_col] = 1
            V[j_base*p+k_base, current_col] = 1
            W[k_base*n+(i_base+1), current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = -1
            next_col()
            
            # M2: A00(B01-B11)
            U[i_base*m+j_base, current_col] = 1
            V[j_base*p+(k_base+1), current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = -1
            W[(k_base+1)*n+i_base, current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = 1
            next_col()
            
            # M3: A11(B10-B00)
            U[(i_base+1)*m+(j_base+1), current_col] = 1
            V[(j_base+1)*p+k_base, current_col] = 1
            V[j_base*p+k_base, current_col] = -1
            W[k_base*n+i_base, current_col] = 1
            W[k_base*n+(i_base+1), current_col] = 1
            next_col()
            
            # M4: (A00+A01)B11
            U[i_base*m+j_base, current_col] = 1
            U[i_base*m+(j_base+1), current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = 1
            W[k_base*n+i_base, current_col] = -1
            W[(k_base+1)*n+i_base, current_col] = 1
            next_col()
            
            # M5: (A10-A00)(B00+B01)
            U[(i_base+1)*m+j_base, current_col] = 1
            U[i_base*m+j_base, current_col] = -1
            V[j_base*p+k_base, current_col] = 1
            V[j_base*p+(k_base+1), current_col] = 1
            W[(k_base+1)*n+(i_base+1), current_col] = 1
            next_col()
            
            # M6: (A01-A11)(B10+B11)
            U[i_base*m+(j_base+1), current_col] = 1
            U[(i_base+1)*m+(j_base+1), current_col] = -1
            V[(j_base+1)*p+k_base, current_col] = 1
            V[(j_base+1)*p+(k_base+1), current_col] = 1
            W[k_base*n+i_base, current_col] = 1
            next_col()
            
    # 2. Process Naive blocks for j=4
    # j_base = 4
    # k_bases are 0, 2, 4
    # Block type: 2x1 * 1x2 -> 4 terms
    
    for k_base in [0, 2, 4]:
        i_base = 0
        j_idx = 4
        
        # Term 1: A[i0, j] * B[j, k0] -> C[i0, k0]
        U[i_base*m+j_idx, current_col] = 1
        V[j_idx*p+k_base, current_col] = 1
        W[k_base*n+i_base, current_col] = 1
        next_col()
        
        # Term 2: A[i0, j] * B[j, k1] -> C[i0, k1]
        U[i_base*m+j_idx, current_col] = 1
        V[j_idx*p+(k_base+1), current_col] = 1
        W[(k_base+1)*n+i_base, current_col] = 1
        next_col()
        
        # Term 3: A[i1, j] * B[j, k0] -> C[i1, k0]
        U[(i_base+1)*m+j_idx, current_col] = 1
        V[j_idx*p+k_base, current_col] = 1
        W[k_base*n+(i_base+1), current_col] = 1
        next_col()
        
        # Term 4: A[i1, j] * B[j, k1] -> C[i1, k1]
        U[(i_base+1)*m+j_idx, current_col] = 1
        V[j_idx*p+(k_base+1), current_col] = 1
        W[(k_base+1)*n+(i_base+1), current_col] = 1
        next_col()

    if current_col != rank:
        # Safety check, should be 54
        # 2 j-pairs * 3 k-pairs * 7 strassen = 42
        # 1 j-single * 3 k-pairs * 4 naive = 12
        # Total 54
        raise ValueError(f"Rank mismatch: expected {rank}, got {current_col}")

    # Convert to lists if required by interface, though numpy arrays are usually accepted
    # The interface says "factor matrices of shapes...". Numpy arrays are standard.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
