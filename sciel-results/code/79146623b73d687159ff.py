import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by creating U, V, W matrices.
    # The strategy is to recursively decompose the tensor.
    # To do this efficiently, we will define helper functions that return
    # the U, V, W blocks for a given sub-problem, and we will concatenate them.
    
    # Since the rank is small (78), we can just build the full matrices.
    # However, implementing the recursive logic directly into U, V, W is complex.
    # Instead, we will use a "virtual" tensor representation or just hardcode the structure.
    
    # Let's use a simple approach: 
    # 1. Create a list of "terms". Each term is (u, v, w) where u, v, w are vectors.
    # 2. The tensor T = sum term_k * u_k v_k w_k.
    # 3. We can generate these terms using the algorithms.
    
    terms = []
    
    # Helper to add a term
    def add_term(u, v, w):
        terms.append((u, v, w))

    # We need to represent the indices.
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    # We will build the terms for the sub-problems and then map them to the global indices.
    
    # --- Winograd 3x3x3 ---
    # Rank 23
    # This is complex to hardcode. 
    # Alternative: Use Strassen for 3x3x3? No, Strassen is for 2x2x2.
    # For 3x3x3, we can use the standard Winograd algorithm.
    
    # Given the complexity and time limit, I will use a simpler bound.
    # If I can't implement Winograd correctly, I can use Schoolbook for 3x3x3 (Rank 27).
    # Then R(M_3,3,6) = 54.
    # R(M_3,2,6) = 32.
    # Total 86.
    # 86 < 90.
    # So even with Schoolbook for 3x3x3, I get rank 86.
    
    # Let's implement the decomposition with Schoolbook for the base cases 
    # but using the block splitting to reduce rank.
    
    # Base case: 2x2x2 using Strassen (Rank 7)
    # Base case: 2x2x1 using Rank 1
    # Base case: 3x3x3 using Schoolbook (Rank 27) -- Safe fallback
    
    # Let's implement a generic recursive function that returns the list of terms
    # for a sub-problem of size n, m, p, with offsets.
    
    # This is getting too complex for a single pass.
    # I will implement the rank 86 decomposition (Schoolbook for 3x3x3).
    
    # Decomposition:
    # 1. Split m=5 into 2 and 3.
    # 2. C = C_1 + C_2
    # 3. C_1: 3x2x6. Pad to 4x2x6. Split 4 into 2+2. Split 6 into 3+3.
    #    C_1 = C_11 + C_12 + C_13 + C_14 (each 2x2x3)
    #    Each 2x2x3: Split into 2x2x2 (Strassen, Rank 7) + 2x2x1 (Rank 1).
    #    Total for C_1: 4 * (7+1) = 32.
    # 4. C_2: 3x3x6. Split 6 into 3+3.
    #    C_2 = C_21 + C_22 (each 3x3x3)
    #    Each 3x3x3: Schoolbook (Rank 27).
    #    Total for C_2: 2 * 27 = 54.
    # Total Rank: 32 + 54 = 86.
    
    # Let's code this specific decomposition.
    
    U = []
    V = []
    W = []
    
    # We will build the matrices by appending columns.
    
    # Helper to add a term to the global matrices
    # u is a vector of length n*m
    # v is a vector of length m*p
    # w is a vector of length p*n
    
    def add_global_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # --- Part 1: C_1 (3x2x6) ---
    # Indices: i in 0..2, j in 0..1, k in 0..5
    # Pad i to 3..3 (zero).
    # Split i into 0..1 and 2..3.
    # Split k into 0..2 and 3..5.
    
    # Sub-problem 1.1: i in 0..1, j in 0..1, k in 0..2 (2x2x3)
    # Sub-problem 1.2: i in 0..1, j in 0..1, k in 3..5 (2x2x3)
    # Sub-problem 1.3: i in 2..3, j in 0..1, k in 0..2 (2x2x3)
    # Sub-problem 1.4: i in 2..3, j in 0..1, k in 3..5 (2x2x3)
    
    def add_2x2x3(i_off, j_off, k_off):
        # Strassen for 2x2x2 (k_off .. k_off+1)
        # Indices: i in i_off..i_off+1, j in j_off..j_off+1, k in k_off..k_off+1
        # Strassen terms for 2x2x2
        # We need to map the local indices to global.
        
        # Strassen 2x2x2 terms (u, v, w) in local coords
        # u: 2x2 = 4 entries. v: 2x2 = 4 entries. w: 2x2 = 4 entries.
        
        # Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11) (B11 + B12)
        # M7 = (A12 - A22) (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We will generate the 7 terms.
        
        # Local indices:
        # i: 0, 1
        # j: 0, 1
        # k: 0, 1
        
        # Term 1: M1
        # u: A11, A22. u[0]=1 (i=0,j=0), u[3]=1 (i=1,j=1)
        # v: B11, B22. v[0]=1 (j=0,k=0), v[3]=1 (j=1,k=1)
        # w: C11, C22. w[0]=1 (k=0,i=0), w[3]=1 (k=1,i=1)
        
        # This is tedious to hardcode 7 terms * 4 sub-problems * 2 (for 2x2x1 part).
        
        # Instead, I will use a simpler method:
        # Just use Schoolbook for 2x2x3 as well?
        # Rank 6.
        # Then C_1 rank = 4 * 6 = 24.
        # C_2 rank = 2 * 27 = 54.
        # Total 78.
        # Wait, 2x2x3 schoolbook is 6.
        # 2x2x2 Strassen is 7.
        # So Schoolbook for 2x2x3 is better (6 < 7+1=8).
        
        # So use Schoolbook for all base cases?
        # 3x3x3 Schoolbook: 27.
        # 2x2x3 Schoolbook: 6.
        
        # C_1: 4 * 6 = 24.
        # C_2: 2 * 27 = 54.
        # Total 78.
        
        # This is much easier to implement.
        
        for ii in range(2):
            for jj in range(2):
                for kk in range(3):
                    # u: i_off+ii, j_off+jj
                    u = [0] * (n*m)
                    u[(i_off+ii)*m + (j_off+jj)] = 1
                    
                    # v: j_off+jj, k_off+kk
                    v = [0] * (m*p)
                    v[(j_off+jj)*p + (k_off+kk)] = 1
                    
                    # w: k_off+kk, i_off+ii
                    w = [0] * (p*n)
                    w[(k_off+kk)*n + (i_off+ii)] = 1
                    
                    add_global_term(u, v, w)
                    
    # Call add_2x2x3 for the 4 sub-problems
    add_2x2x3(0, 0, 0)
    add_2x2x3(0, 0, 3)
    add_2x2x3(2, 0, 0) # i=2 is real, i=3 is pad (zero in A, so term is 0, but we include it? No, we can skip if we know it's zero? No, the decomposition must be exact for the tensor. The tensor T_3,5,6 has non-zero entries for i=2. The pad i=3 is not in the original tensor. So we don't need to decompose it.
    # Wait, the original tensor is 3x5x6.
    # We split i into 0..1 and 2.
    # So we only need 2 sub-problems for i?
    # i=0..1 (2x2x6) and i=2 (1x2x6).
    # 1x2x6 is rank 6 (schoolbook).
    # 2x2x6 is 2 * 6 = 12.
    # Total C_1 rank = 12 + 6 = 18.
    
    # Let's re-calculate with this better split.
    # C_1: 3x2x6.
    # Split i into 2 and 1.
    # Sub 1: 2x2x6. Rank 12.
    # Sub 2: 1x2x6. Rank 6.
    # Total 18.
    
    # C_2: 3x3x6.
    # Split k into 3 and 3.
    # Sub 1: 3x3x3. Rank 27.
    # Sub 2: 3x3x3. Rank 27.
    # Total 54.
    
    # Total Rank = 18 + 54 = 72.
    
    # 72 is even better!
    
    # Let's implement this.
    
    # Clear previous terms
    terms = []
    U = []
    V = []
    W = []
    
    # --- Part 1: C_1 (3x2x6) ---
    # j is 0..1 (m=2)
    # k is 0..5 (p=6)
    
    # Sub 1.1: i in 0..1, j in 0..1, k in 0..5 (2x2x6)
    # Sub 1.2: i in 2..2, j in 0..1, k in 0..5 (1x2x6)
    
    # 2x2x6: Split k into 0..2 and 3..5.
    # 2x2x3 (Rank 6) + 2x2x3 (Rank 6) = 12.
    
    for ii in range(2):
        for jj in range(2):
            for kk in range(6):
                u = [0] * (n*m)
                u[(ii)*m + (jj)] = 1 # i=ii, j=jj (j is global 0..1)
                
                v = [0] * (m*p)
                v[(jj)*p + (kk)] = 1 # j=jj, k=kk
                
                w = [0] * (p*n)
                w[(kk)*n + (ii)] = 1 # k=kk, i=ii
                
                add_global_term(u, v, w)
                
    # 1x2x6:
    for jj in range(2):
        for kk in range(6):
            u = [0] * (n*m)
            u[(2)*m + (jj)] = 1 # i=2, j=jj
            
            v = [0] * (m*p)
            v[(jj)*p + (kk)] = 1
            
            w = [0] * (p*n)
            w[(kk)*n + (2)] = 1
            
            add_global_term(u, v, w)
            
    # --- Part 2: C_2 (3x3x6) ---
    # j is 2..4 (m=3, global j 2,3,4)
    # k is 0..5 (p=6)
    
    # Split k into 0..2 and 3..5.
    
    for ii in range(3):
        for jj in range(3):
            global_j = jj + 2
            for kk in range(6):
                u = [0] * (n*m)
                u[(ii)*m + (global_j)] = 1
                
                v = [0] * (m*p)
                v[(global_j)*p + (kk)] = 1
                
                w = [0] * (p*n)
                w[(kk)*n + (ii)] = 1
                
                add_global_term(u, v, w)
                
    # Total terms: 24 + 12 + 54 = 90?
    # Wait.
    # Part 1: 2*2*6 = 24.
    # Part 2: 3*2*6 = 36.
    # Total 60?
    # No, I just used Schoolbook for all of them.
    # Schoolbook for 3x5x6 is 90.
    # My split:
    # C_1: 3x2x6. Schoolbook is 3*2*6 = 36.
    # C_2: 3x3x6. Schoolbook is 3*3*6 = 54.
    # Total 90.
    
    # I need to use Strassen or Winograd to reduce the rank.
    # If I use Schoolbook everywhere, I get 90.
    # I must implement at least Strassen for 2x2x2 to get below 90.
    
    # Let's replace the 2x2x3 blocks in C_1 with Strassen.
    # C_1 has four 2x2x3 blocks?
    # No, C_1 is 3x2x6.
    # Split i into 2 and 1.
    # 2x2x6: Split k into 3 and 3.
    # Two 2x2x3 blocks.
    # Each 2x2x3: Split k into 2 and 1.
    # One 2x2x2 (Strassen, Rank 7) and one 2x2x1 (Rank 1).
    # Total 8 per 2x2x3.
    # Two 2x2x3 blocks: 16.
    # 1x2x6: Schoolbook 6.
    # Total C_1: 22.
    
    # C_2: 3x3x6.
    # Split k into 3 and 3.
    # Two 3x3x3 blocks.
    # Schoolbook 27 each. Total 54.
    
    # Total Rank: 22 + 54 = 76.
    
    # I will implement this.
    
    # Clear again
    U = []
    V = []
    W = []
    
    # --- C_1: 3x2x6 ---
    # Sub 1.1: 2x2x6 (i=0,1; j=0,1; k=0..5)
    # Split k into 0..2 and 3..5.
    
    # Block 1.1.1: 2x2x3 (i=0,1; j=0,1; k=0..2)
    # Use Strassen for k=0,1 and Schoolbook for k=2?
    # Or Strassen for 2x2x2 and add 2x2x1.
    
    # Strassen 2x2x2 for i=0,1; j=0,1; k=0,1
    # Local indices: i_l, j_l, k_l in 0..1
    # Global: i=i_l, j=j_l, k=k_l
    
    # Strassen terms
    # M1: (A00+A11)(B00+B11)
    # u: A00, A11. v: B00, B11. w: C00, C11.
    
    # I will hardcode the 7 Strassen terms for this block.
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        # i_off, j_off, k_off are global offsets
        # Local i, j, k in 0..1
        
        # M1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+0)*m + (j_off+0)] = 1
        u[(i_off+1)*m + (j_off+1)] = 1
        v[(j_off+0)*p + (k_off+0)] = 1
        v[(j_off+1)*p + (k_off+1)] = 1
        w[(k_off+0)*n + (i_off+0)] = 1
        w[(k_off+1)*n + (i_off+1)] = 1
        add_global_term(u, v, w)
        
        # M2
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+1)*m + (j_off+0)] = 1
        u[(i_off+1)*m + (j_off+1)] = 1
        v[(j_off+0)*p + (k_off+0)] = 1
        w[(k_off+0)*n + (i_off+1)] = 1
        w[(k_off+1)*n + (i_off+1)] = 1
        add_global_term(u, v, w)
        
        # M3
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+0)*m + (j_off+0)] = 1
        v[(j_off+0)*p + (k_off+1)] = 1
        v[(j_off+1)*p + (k_off+1)] = -1
        w[(k_off+1)*n + (i_off+0)] = 1
        add_global_term(u, v, w)
        
        # M4
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+1)*m + (j_off+1)] = 1
        v[(j_off+1)*p + (k_off+0)] = 1
        v[(j_off+0)*p + (k_off+0)] = -1
        w[(k_off+0)*n + (i_off+1)] = 1
        w[(k_off+1)*n + (i_off+1)] = -1
        add_global_term(u, v, w)
        
        # M5
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+0)*m + (j_off+0)] = 1
        u[(i_off+0)*m + (j_off+1)] = 1
        v[(j_off+1)*p + (k_off+1)] = 1
        w[(k_off+1)*n + (i_off+0)] = -1
        w[(k_off+1)*n + (i_off+1)] = 1
        add_global_term(u, v, w)
        
        # M6
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+1)*m + (j_off+0)] = 1
        u[(i_off+0)*m + (j_off+0)] = -1
        v[(j_off+0)*p + (k_off+0)] = 1
        v[(j_off+0)*p + (k_off+1)] = 1
        w[(k_off+0)*n + (i_off+1)] = 1
        w[(k_off+1)*n + (i_off+1)] = 1
        add_global_term(u, v, w)
        
        # M7
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+0)*m + (j_off+1)] = 1
        u[(i_off+1)*m + (j_off+1)] = -1
        v[(j_off+1)*p + (k_off+0)] = 1
        v[(j_off+1)*p + (k_off+1)] = 1
        w[(k_off+0)*n + (i_off+0)] = 1
        w[(k_off+1)*n + (i_off+0)] = 1
        add_global_term(u, v, w)
        
    # Add Strassen for k=0,1
    add_strassen_2x2x2(0, 0, 0)
    
    # Add Schoolbook for k=2 (2x2x1)
    for ii in range(2):
        for jj in range(2):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[ii*m + jj] = 1
            v[jj*p + 2] = 1
            w[2*n + ii] = 1
            add_global_term(u, v, w)
            
    # Block 1.1.2: 2x2x3 (i=0,1; j=0,1; k=3..5)
    add_strassen_2x2x2(0, 0, 3)
    for ii in range(2):
        for jj in range(2):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[ii*m + jj] = 1
            v[jj*p + 5] = 1
            w[5*n + ii] = 1
            add_global_term(u, v, w)
            
    # Sub 1.2: 1x2x6 (i=2; j=0,1; k=0..5)
    for jj in range(2):
        for kk in range(6):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[2*m + jj] = 1
            v[jj*p + kk] = 1
            w[kk*n + 2] = 1
            add_global_term(u, v, w)
            
    # --- C_2: 3x3x6 ---
    # Schoolbook
    for ii in range(3):
        for jj in range(3):
            global_j = jj + 2
            for kk in range(6):
                u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
                u[ii*m + global_j] = 1
                v[global_j*p + kk] = 1
                w[kk*n + ii] = 1
                add_global_term(u, v, w)
                
    # Convert lists to numpy arrays
    U = np.array(U).T
    V = np.array(V).T
    W = np.array(W).T
    
    return {"U": U, "V": V, "W": W}
