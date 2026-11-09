import time
import sys

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44 from Laser + Strassen
    rank = 44
    
    # Initialize matrices
    # U: (n*m, rank)
    U = [[0] * rank for _ in range(n * m)]
    # V: (m*p, rank)
    V = [[0] * rank for _ in range(m * p)]
    # W: (p*n, rank)
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # i, j, k are global indices
    # ui, vj, wk are coefficients (usually 1, -1, etc)
    def add_term(i, j, k, ui, vj, wk):
        nonlocal col
        if col >= rank:
            return
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] += ui
        V[v_idx][col] += vj
        W[w_idx][col] += wk
        col += 1

    # Helper to add a 1x2x2 block
    # i0: start row of A
    # j0: start col of A / row of B
    # k0: start col of B
    # A is 1x2, B is 2x2
    def add_1x2x2(i0, j0, k0):
        # u1 = A[i0, j0], u2 = A[i0, j0+1]
        # v11 = B[j0, k0], v12 = B[j0, k0+1]
        # v21 = B[j0+1, k0], v22 = B[j0+1, k0+1]
        # w1 = C[i0, k0], w2 = C[i0, k0+1]
        
        # T1: u1 * v11 -> w1
        add_term(i0, j0, k0, 1, 1, 1)
        # T2: u2 * v21 -> w1
        add_term(i0, j0+1, k0, 1, 1, 1)
        # T3: u1 * v12 -> w2
        add_term(i0, j0, k0+1, 1, 1, 1)
        # T4: u2 * v22 -> w2
        add_term(i0, j0+1, k0+1, 1, 1, 1)

    # Helper to add a 2x2x2 block using Strassen
    # i0: start row of A
    # j0: start col of A / row of B
    # k0: start col of B
    def add_2x2x2_strassen(i0, j0, k0):
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # C indices: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
        
        # S1 = (A00 + A11) * (B00 + B11)
        # A00=(i0,j0), A11=(i0+1,j0+1)
        # B00=(j0,k0), B11=(j0+1,k0+1)
        # C00 += S1, C11 += S1
        # We need to distribute the product of sums into rank-1 terms?
        # No, the tensor decomposition of Strassen consists of 7 terms.
        # Each term is a product of a linear combination of A entries and a linear combination of B entries.
        # The linear combinations are fixed.
        # We must add the contributions to U, V, W for each of the 7 terms.
        
        # Term 1: (A00 + A11) * (B00 + B11) -> C00 + C11
        # U: A00 += 1, A11 += 1
        # V: B00 += 1, B11 += 1
        # W: C00 += 1, C11 += 1
        add_term(i0, j0, k0, 1, 1, 1) # A00*B00 -> C00
        add_term(i0, j0, k0+1, 1, 1, 0) # A00*B11 -> C00? No.
        
        # Wait, the "add_term" function adds a single rank-1 tensor $u_i v_j w_k$.
        # Strassen's $S_1$ is $(A_{00} + A_{11})(B_{00} + B_{11})$.
        # This expands to $A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
        # This is 4 rank-1 terms.
        # But Strassen replaces the 8 schoolbook terms with 7 "composite" terms.
        # In the context of tensor rank, a "composite" term like $(A_{00} + A_{11})(B_{00} + B_{11})$ is NOT a single rank-1 tensor.
        # A rank-1 tensor is of the form $u \otimes v \otimes w$ where $u, v, w$ are vectors.
        # The term $(A_{00} + A_{11}) \otimes (B_{00} + B_{11}) \otimes (C_{00} + C_{11})$ is NOT rank 1.
        # It is a sum of 4 rank-1 tensors.
        
        # Ah, I made a conceptual error.
        # The rank of the $2 \times 2 \times 2$ tensor is 7.
        # This means it can be decomposed into 7 rank-1 tensors.
        # The Strassen algorithm provides 7 multiplications $S_1, \dots, S_7$.
        # Each $S_k$ is a product of two linear forms.
        # $S_k = L_k(A) M_k(B)$.
        # The contribution to C is $L_k(A) M_k(B) R_k(C)$.
        # This IS a rank-1 tensor!
        # $L_k$ is a vector in the dual space of A (or A itself if we view it as input).
        # $M_k$ is a vector in the dual space of B.
        # $R_k$ is a vector in the dual space of C.
        # So yes, each Strassen step corresponds to exactly ONE rank-1 term in the decomposition.
        
        # So, for Term 1:
        # $L_1(A) = A_{00} + A_{11}$
        # $M_1(B) = B_{00} + B_{11}$
        # $R_1(C) = C_{00} + C_{11}$
        
        # We need to add:
        # U[A00] += 1, U[A11] += 1
        # V[B00] += 1, V[B11] += 1
        # W[C00] += 1, W[C11] += 1
        
        # My `add_term` function only handles single indices.
        # I need a function `add_rank1` that takes lists of indices and coefficients.
        
        pass # Redefine logic below

    # Redefine helper to handle linear combinations
    def add_rank1(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for idx, coeff in zip(u_indices, u_coeffs):
            if coeff != 0:
                U[idx][col] += coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            if coeff != 0:
                V[idx][col] += coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            if coeff != 0:
                W[idx][col] += coeff
        col += 1

    # Re-implement add_1x2x2 with add_rank1
    def add_1x2x2(i0, j0, k0):
        # u1=A[i0,j0], u2=A[i0,j0+1]
        # v11=B[j0,k0], v12=B[j0,k0+1], v21=B[j0+1,k0], v22=B[j0+1,k0+1]
        # w1=C[i0,k0], w2=C[i0,k0+1]
        
        # T1: A[i0,j0] * B[j0,k0] -> C[i0,k0]
        add_rank1([i0*m+j0], [1], [j0*p+k0], [1], [k0*n+i0], [1])
        # T2: A[i0,j0+1] * B[j0+1,k0] -> C[i0,k0]
        add_rank1([i0*m+j0+1], [1], [(j0+1)*p+k0], [1], [k0*n+i0], [1])
        # T3: A[i0,j0] * B[j0,k0+1] -> C[i0,k0+1]
        add_rank1([i0*m+j0], [1], [j0*p+k0+1], [1], [(k0+1)*n+i0], [1])
        # T4: A[i0,j0+1] * B[j0+1,k0+1] -> C[i0,k0+1]
        add_rank1([i0*m+j0+1], [1], [(j0+1)*p+k0+1], [1], [(k0+1)*n+i0], [1])

    # Re-implement add_2x2x2_strassen
    def add_2x2x2_strassen(i0, j0, k0):
        # Indices
        A00 = i0*m + j0
        A01 = i0*m + j0 + 1
        A10 = (i0+1)*m + j0
        A11 = (i0+1)*m + j0 + 1
        
        B00 = j0*p + k0
        B01 = j0*p + k0 + 1
        B10 = (j0+1)*p + k0
        B11 = (j0+1)*p + k0 + 1
        
        C00 = k0*n + i0
        C01 = (k0+1)*n + i0
        C10 = k0*n + (i0+1)
        C11 = (k0+1)*n + (i0+1)
        
        # S1 = (A00 + A11) * (B00 + B11) -> C00 + C11
        add_rank1([A00, A11], [1, 1], [B00, B11], [1, 1], [C00, C11], [1, 1])
        
        # S2 = (A10 + A11) * B00 -> C10
        add_rank1([A10, A11], [1, 1], [B00], [1], [C10], [1])
        
        # S3 = A00 * (B01 - B11) -> C01
        add_rank1([A00], [1], [B01, B11], [1, -1], [C01], [1])
        
        # S4 = A11 * (B10 - B00) -> C10
        add_rank1([A11], [1], [B10, B00], [1, -1], [C10], [1])
        
        # S5 = (A00 + A01) * B11 -> C01
        add_rank1([A00, A01], [1, 1], [B11], [1], [C01], [1])
        
        # S6 = (A10 - A00) * (B00 + B01) -> C11
        add_rank1([A10, A00], [1, -1], [B00, B01], [1, 1], [C11], [1])
        
        # S7 = (A01 - A11) * (B10 + B11) -> C00
        add_rank1([A01, A11], [1, -1], [B10, B11], [1, 1], [C00], [1])

    # Block 1: A11 (1x2) * B11 (2x2) -> C11 (1x2)
    # i0=0, j0=0, k0=0
    add_1x2x2(0, 0, 0)
    
    # Block 2: A12 (1x2) * B21 (2x2) -> C12 (1x2)
    # i0=0, j0=2, k0=0
    add_1x2x2(0, 2, 0)
    
    # Block 3: A11 (1x2) * B12 (2x2) -> C12 (1x2)
    # i0=0, j0=0, k0=2
    add_1x2x2(0, 0, 2)
    
    # Block 4: A12 (1x2) * B22 (2x2) -> C12 (1x2)
    # i0=0, j0=2, k0=2
    add_1x2x2(0, 2, 2)
    
    # Block 5: A21 (2x2) * B11 (2x2) -> C21 (2x2)
    # i0=1, j0=0, k0=0
    add_2x2x2_strassen(1, 0, 0)
    
    # Block 6: A22 (2x2) * B21 (2x2) -> C21 (2x2)
    # i0=1, j0=2, k0=0
    add_2x2x2_strassen(1, 2, 0)
    
    # Block 7: A21 (2x2) * B12 (2x2) -> C22 (2x2)
    # i0=1, j0=0, k0=2
    add_2x2x2_strassen(1, 0, 2)
    
    # Block 8: A22 (2x2) * B22 (2x2) -> C22 (2x2)
    # i0=1, j0=2, k0=2
    add_2x2x2_strassen(1, 2, 2)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
