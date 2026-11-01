"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Strassen's algorithm uses rank 7
    rank = 7
    
    # Initialize U, V, W
    # U shape: (n*m, rank) = (4, 7)
    # V shape: (m*p, rank) = (4, 7)
    # W shape: (p*n, rank) = (4, 7)
    # Indices:
    # U row: i*m + j (i in 0..n-1, j in 0..m-1)
    # V row: j*p + k (j in 0..m-1, k in 0..p-1)
    # W row: k*n + i (k in 0..p-1, i in 0..n-1)
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Strassen's 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Output C:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Define the coefficients for each of the 7 columns (rank dimension)
    # col 0: M1
    # col 1: M2
    # col 2: M3
    # col 3: M4
    # col 4: M5
    # col 5: M6
    # col 6: M7
    
    # Helper to set U, V, W for a specific product term
    # U[i, j] coefficient for A_ij
    # V[j, k] coefficient for B_jk
    # W[k, i] coefficient for C_ki (note: task says W[k*n+i, r])
    
    # Product 1: (A00 + A11)(B00 + B11)
    # A coeffs: A00:1, A11:1
    # B coeffs: B00:1, B11:1
    # C coeffs: C00:1, C11:1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Product 2: (A10 + A11)B00
    # A coeffs: A10:1, A11:1
    # B coeffs: B00:1
    # C coeffs: C10:1, C11:-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = 1; W[3, 1] = -1
    
    # Product 3: A00(B01 - B11)
    # A coeffs: A00:1
    # B coeffs: B01:1, B11:-1
    # C coeffs: C01:1, C11:1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Product 4: A11(B10 - B00)
    # A coeffs: A11:1
    # B coeffs: B10:1, B00:-1
    # C coeffs: C00:1, C10:1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Product 5: (A00 + A01)B11
    # A coeffs: A00:1, A01:1
    # B coeffs: B11:1
    # C coeffs: C00:-1, C01:1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Product 6: (A10 - A00)(B00 + B01)
    # A coeffs: A10:1, A00:-1
    # B coeffs: B00:1, B01:1
    # C coeffs: C11:1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Product 7: (A01 - A11)(B10 + B11)
    # A coeffs: A01:1, A11:-1
    # B coeffs: B10:1, B11:1
    # C coeffs: C00:1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    # Verification
    # Check if sum_c U[:,c] outer V[:,c] outer W[:,c] reconstructs the tensor
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For 2x2x2, we can check all 8 entries of the multiplication tensor
    # T[i,j,k] should be 1 if i==k (for standard matrix mult A*B -> C, C_ik = sum_j A_ij B_jk)
    # Wait, the tensor for matrix multiplication T_{ijk} = delta_{ik} is NOT correct.
    # The tensor is T_{ijk} such that C_{ik} = sum_j A_{ij} B_{jk}.
    # So the structure tensor T has T[i, j, k] = 1 if and only if the output C[k, i] depends on A[i, j] and B[j, k]?
    # Let's check the convention:
    # U[i*m+j, c]
    # V[j*p+k, c]
    # W[k*n+i, c]
    # Term: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This contributes to the coefficient of the basis element associated with (i, j, k) in the tensor product space.
    # The matrix multiplication tensor T is defined such that T[i, j, k] = 1 if the operation is C[k, i] += A[i, j] * B[j, k]?
    # No, standard index notation: C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor component T_{ijk} is 1 if the indices match the pattern A_{ij} B_{jk} -> C_{ik}.
    # In the decomposition: sum_c U_{ij, c} V_{jk, c} W_{ki, c}
    # This sums to the coefficient of the monomial A_{ij} B_{jk} in the expression for C_{ki}.
    # For matrix multiplication, C_{ki} = sum_j A_{ij} B_{jk}.
    # So the coefficient of A_{ij} B_{jk} in C_{ki} is 1 if i==k? No.
    # C_{ki} is the element at row k, col i.
    # C_{ki} = sum_j A_{ij} B_{jk}.
    # So for a fixed i, j, k, the term A_{ij} B_{jk} appears in C_{ki} with coefficient 1.
    # Therefore, the reconstructed tensor T'_{ijk} = sum_c U_{ij, c} V_{jk, c} W_{ki, c} should equal 1 for ALL i, j, k.
    # Wait, let's re-read carefully.
    # C is p x n. A is n x m. B is m x p.
    # C_{k, i} = sum_j A_{i, j} B_{j, k}.
    # The tensor T is in V_A^* \otimes V_B^* \otimes V_C.
    # T = sum_{i,j,k} e^i_i \otimes e^j_j \otimes e^k_k ? No.
    # T = sum_{i,j,k} A_{ij} B_{jk} C_{ki}^* ?
    # The decomposition is T = sum_c u_c \otimes v_c \otimes w_c.
    # <T, A \otimes B \otimes C> = sum_c <u_c, A> <v_c, B> <w_c, C>.
    # <u_c, A> = sum_{i,j} U_{ij, c} A_{ij}.
    # <v_c, B> = sum_{j,k} V_{jk, c} B_{jk}.
    # <w_c, C> = sum_{k,i} W_{ki, c} C_{ki}.
    # So the scalar is sum_{i,j,k} (sum_c U_{ij,c} V_{jk,c} W_{ki,c}) A_{ij} B_{jk} C_{ki}.
    # We want this to equal sum_{i,j,k} A_{ij} B_{jk} C_{ki} (since C_{ki} = sum_j A_{ij} B_{jk}).
    # Thus, we require sum_c U_{ij,c} V_{jk,c} W_{ki,c} = 1 for ALL i, j, k.
    
    # Let's verify this condition.
    is_valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if val != 1:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break
            
    if not is_valid:
        # Fallback to schoolbook if Strassen fails (should not happen)
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        rank = n*m*p # Update rank for return? No, rank is determined by shape.
        
    # Convert to lists of lists as required by interface?
    # "factor matrices of shapes ... with integer/half-integer entries"
    # Usually numpy arrays are accepted if they are convertible, but let's stick to lists to be safe if the evaluator is strict about types, or return numpy arrays if allowed.
    # The prompt says "Python standard library, NumPy and SciPy are available".
    # The seed program returns lists. I will return lists to match the seed's type signature closely, or numpy if it's more standard.
    # The seed returns `{'U': U, 'V': V, 'W': W}` where U is a list of lists.
    # I will return numpy arrays as they are standard in these tasks, but if I must match the seed exactly, I should convert.
    # However, "score is the rank". The rank is inferred from the shape.
    # Let's return numpy arrays for efficiency and clarity, as NumPy is available.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
