import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # i_off: row offset in A (0)
        # j_off: col offset in A (0)
        # k_off: col offset in B (0)
        
        # Local indices for A: 00, 01, 10, 11 -> global i, j
        # A indices: (i_off+0, j_off+0), (i_off+0, j_off+1), (i_off+1, j_off+0), (i_off+1, j_off+1)
        # B indices: (j_off+0, k_off+0), (j_off+0, k_off+1), (j_off+1, k_off+0), (j_off+1, k_off+1)
        # C indices: (i_off+0, k_off+0), (i_off+0, k_off+1), (i_off+1, k_off+0), (i_off+1, k_off+1)
        
        # Strassen M1-M7 definitions
        # U (A)
        # M1: A00+A11
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        # V (B)
        # M1: B00+B11
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        # W (C)
        # C00 += M1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        # C11 += M1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M2: A10+A11, B00
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        # C10 += M2
        W[(k_off+0)*n + (i_off+1)][col] = 1
        # C11 += M2
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M3: A00, B01-B11
        U[(i_off+0)*m + (j_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = -1
        # C01 += M3
        W[(k_off+1)*n + (i_off+0)][col] = 1
        # C11 -= M3
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # M4: A11, B10-B00
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = -1
        # C00 -= M4
        W[(k_off+0)*n + (i_off+0)][col] = -1
        # C10 += M4
        W[(k_off+0)*n + (i_off+1)][col] = 1
        col += 1
        
        # M5: A00+A01, B11
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        # C01 += M5
        W[(k_off+1)*n + (i_off+0)][col] = 1
        # C11 -= M5
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # M6: A10-A00, B00+B01
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+0)][col] = -1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = 1
        # C00 += M6
        W[(k_off+0)*n + (i_off+0)][col] = 1
        # C10 -= M6
        W[(k_off+0)*n + (i_off+1)][col] = -1
        col += 1
        
        # M7: A01-A11, B10+B11
        U[(i_off+0)*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        # C00 -= M7
        W[(k_off+0)*n + (i_off+0)][col] = -1
        # C01 += M7
        W[(k_off+1)*n + (i_off+0)][col] = 1
        col += 1

    # Helper to add schoolbook block
    def add_schoolbook(i_off, j_off, k_off, i_len, j_len, k_len):
        nonlocal col
        for i in range(i_len):
            for j in range(j_len):
                for k in range(k_len):
                    U[(i_off+i)*m + (j_off+j)][col] = 1
                    V[(j_off+j)*p + (k_off+k)][col] = 1
                    W[(k_off+k)*n + (i_off+i)][col] = 1
                    col += 1

    # Block 1: A1 (2x2) * B11 (2x2) -> C11 (2x2)
    # i in 0..1, j in 0..1, k in 0..1
    add_strassen(0, 0, 0)
    
    # Block 2: A1 (2x2) * B12 (2x2) -> C12 (2x2)
    # i in 0..1, j in 0..1, k in 2..3
    add_strassen(0, 0, 2)
    
    # Block 3: A1 (2x2) * B13 (2x1) -> C13 (2x1)
    # i in 0..1, j in 0..1, k in 4..4
    add_schoolbook(0, 0, 4, 2, 2, 1)
    
    # Block 4: A2 (2x1) * B2 (1x5) -> C2 (2x5)
    # i in 0..1, j in 2..2, k in 0..4
    add_schoolbook(0, 2, 0, 2, 1, 5)

    return {"U": U, "V": V, "W": W}

# Verification logic (optional, but good for debugging)
if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Test with random matrices
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    C_calc = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0
            for r in range(U.shape[1]):
                u_val = 0
                v_val = 0
                # U[i*m+j, r]
                # V[j*p+k, r]
                # W[k*n+i, r]
                # Sum over j
                # Actually, the decomposition is:
                # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])?
                # No, the tensor decomposition is:
                # C[i,k] = sum_r sum_j U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] ?
                # This would imply C is a sum of rank 1 tensors of A and B?
                # No, A and B are inputs.
                # The evaluator likely does:
                # C_calc[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]?
                # This doesn't make sense dimensionally if U and V are just coefficients.
                
                # Let's assume the standard tensor evaluation:
                # The tensor T has components T_{u,v,w}.
                # C_{i,k} = sum_{u,v,w} T_{u,v,w} A_u B_v ? No.
                
                # The correct interpretation:
                # The product is C = A B.
                # The decomposition provides a way to compute C using R multiplications.
                # Each multiplication r computes M_r = (linear comb of A) * (linear comb of B).
                # Then C is a linear comb of M_r.
                # So:
                # M_r = (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
                # C[i,k] = sum_r W[k*n+i, r] M_r
                # Wait, U depends on i, V depends on k?
                # U index is i*m+j. So U depends on i.
                # V index is j*p+k. So V depends on k.
                # This means M_r depends on i and k?
                # That would mean we compute a different product for each i,k?
                # That is not how bilinear complexity works.
                
                # In bilinear complexity, the multiplications are shared.
                # The standard form is:
                # C_{i,k} = sum_r alpha_{i,k}^r (sum_j beta_{i,j}^r A_{i,j}) (sum_l gamma_{k,l}^r B_{j,l})?
                # No, the indices in U, V, W in the prompt are:
                # U: (n*m, R)
                # V: (m*p, R)
                # W: (p*n, R)
                
                # This specific index mapping (U[i*m+j], V[j*p+k], W[k*n+i])
                # implies that the term for a specific (i,j,k) is U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
                # If we sum over r, j, and (implicitly) the structure is fixed.
                # Actually, if we fix i and k, and sum over j and r:
                # C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # This looks like C[i,k] is computed from A and B?
                # Where are A and B?
                # The evaluator must be substituting A and B.
                # Likely:
                # C_calc[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
                # This would mean U and V are not just coefficients, but define the linear combinations.
                # However, U[i*m+j, r] is a single number.
                # If U[i*m+j, r] is non-zero, it means A[i,j] is included.
                # So sum_j U[i*m+j, r] A[i,j] is a linear combination of row i of A.
                # sum_j V[j*p+k, r] B[j,k] is a linear combination of column k of B.
                # Then we multiply them and weight by W.
                # This corresponds to the decomposition:
                # C[i,k] = sum_r W_{ik}^r (u_r^T A_i) (v_r^T B_k)
                # This is a valid form of bilinear decomposition.
                
                u_comb = 0
                for j in range(m):
                    u_comb += U[i*m+j, r] * A[i,j]
                v_comb = 0
                for j in range(m):
                    v_comb += V[j*p+k, r] * B[j,k]
                s += W[k*n+i, r] * u_comb * v_comb
            C_calc[i,k] = s
            
    if not np.allclose(C_calc, C_true):
        print("Verification failed")
    else:
        print("Verification passed")
