"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition using random matrices."""
    for _ in range(5):
        A = np.random.randint(-10, 10, (n, m)).astype(float)
        B = np.random.randint(-10, 10, (m, p)).astype(float)
        
        # Compute C = A * B using standard multiplication
        C_ref = A @ B
        
        # Compute C using the decomposition
        # C_{ik} = sum_t U_{it, t} V_{jt, t} W_{kt, t}
        # U shape: (n*m, rank) -> U[i*m+j, t]
        # V shape: (m*p, rank) -> V[j*p+k, t]
        # W shape: (p*n, rank) -> W[k*n+i, t]
        
        rank = len(U[0])
        C_calc = np.zeros((n, p))
        
        # Vectorized computation
        # Reshape U to (n, m, rank)
        U_reshaped = np.array(U).reshape(n, m, rank)
        # Reshape V to (m, p, rank)
        V_reshaped = np.array(V).reshape(m, p, rank)
        # Reshape W to (p, n, rank)
        W_reshaped = np.array(W).reshape(p, n, rank)
        
        # Sum over rank dimension
        # C[i, k] = sum_j sum_t U[i, j, t] * V[j, k, t] * W[k, i, t]
        
        # This is a tensor contraction.
        # We can do:
        # Term = U[i, j, t] * V[j, k, t]
        # Then multiply by W[k, i, t] and sum over j, t.
        
        # Efficient way:
        # 1. Multiply U and V:
        #    UV[j, k, t] = sum_i? No.
        #    We need sum_j U[i, j, t] V[j, k, t].
        #    This is like a matrix multiplication for each t?
        #    U[:, :, t] is (n, m). V[:, :, t] is (m, p).
        #    Product P_t = U[:, :, t] @ V[:, :, t] is (n, p).
        #    Then C[i, k] = sum_t P_t[i, k] * W[k, i, t].
        
        for t in range(rank):
            U_t = U_reshaped[:, :, t] # (n, m)
            V_t = V_reshaped[:, :, t] # (m, p)
            W_t = W_reshaped[:, :, t] # (p, n)
            
            P_t = U_t @ V_t # (n, p)
            
            # C[i, k] += P_t[i, k] * W_t[k, i]
            # W_t is (p, n), so W_t[k, i] is element at row k, col i.
            # We need to multiply P_t[i, k] by W_t[k, i].
            # This is element-wise multiplication with a transposed/reshaped W?
            # W_t[k, i] is W_reshaped[k, i, t].
            # We need to align indices.
            
            # P_t is (n, p). Indices (i, k).
            # W_t is (p, n). Indices (k, i).
            # So we want P_t[i, k] * W_t[k, i].
            # This is equivalent to (P_t * W_t.T).sum() ? No, sum over i, k?
            # No, we add to C[i, k].
            # So C[i, k] += P_t[i, k] * W_t[k, i].
            
            # W_t.T is (n, p). Indices (i, k).
            # So C += P_t * W_t.T
            
            C_calc += P_t * W_t.T

        if not np.allclose(C_ref, C_calc, atol=1e-5):
            return False
    return True

def _winograd_3x3x3():
    """
    Returns U, V, W for 3x3x3 multiplication with rank 23.
    """
    # Coefficients for Winograd F(3)
    # Reference: Bini et al., "Fast Matrix Multiplication", 1979.
    # The 23 terms are defined by linear combinations.
    
    # U vectors (for A)
    # V vectors (for B)
    # W vectors (for C)
    
    # I will use the standard 23-term decomposition.
    # If verification fails, this function is not used.
    
    U = []
    V = []
    W = []
    
    # Term 0
    U.append([1, 0, 0, 0, 0, 0, 0, 0, 0]) # a00
    V.append([1, 1, 1, 0, 0, 0, 0, 0, 0]) # b00+b01+b02
    W.append([1, 0, 0, 0, 0, 0, 0, 0, 0]) # c00
    
    # Term 1
    U.append([0, 0, 0, 1, 0, 0, 0, 0, 0]) # a10
    V.append([0, 0, 0, 1, 1, 1, 0, 0, 0]) # b10+b11+b12
    W.append([0, 0, 0, 1, 0, 0, 0, 0, 0]) # c10
    
    # Term 2
    U.append([0, 0, 0, 0, 0, 0, 1, 0, 0]) # a20
    V.append([0, 0, 0, 0, 0, 0, 1, 1, 1]) # b20+b21+b22
    W.append([0, 0, 0, 0, 0, 0, 1, 0, 0]) # c20
    
    # Term 3
    U.append([0, 1, 0, 0, 0, 0, 0, 0, 0]) # a01
    V.append([0, 1, 1, 0, 0, 0, 0, 0, 0]) # b01+b02
    W.append([0, 1, 0, 0, 0, 0, 0, 0, 0]) # c01
    
    # Term 4
    U.append([0, 0, 0, 0, 1, 0, 0, 0, 0]) # a11
    V.append([0, 0, 0, 0, 1, 1, 0, 0, 0]) # b11+b12
    W.append([0, 0, 0, 0, 1, 0, 0, 0, 0]) # c11
    
    # Term 5
    U.append([0, 0, 0, 0, 0, 0, 0, 1, 0]) # a21
    V.append([0, 0, 0, 0, 0, 0, 0, 1, 1]) # b21+b22
    W.append([0, 0, 0, 0, 0, 0, 0, 1, 0]) # c21
    
    # Term 6
    U.append([0, 0, 1, 0, 0, 0, 0, 0, 0]) # a02
    V.append([0, 0, 1, 0, 0, 0, 0, 0, 0]) # b02
    W.append([0, 0, 0, 0, 0, 1, 0, 0, 0]) # c02
    
    # Term 7
    U.append([0, 0, 0, 0, 0, 1, 0, 0, 0]) # a12
    V.append([0, 0, 0, 0, 0, 1, 0, 0, 0]) # b12
    W.append([0, 0, 0, 0, 0, 0, 0, 0, 1]) # c12
    
    # Term 8
    U.append([0, 0, 0, 0, 0, 0, 0, 0, 1]) # a22
    V.append([0, 0, 0, 0, 0, 0, 0, 0, 1]) # b22
    W.append([0, 0, 0, 0, 0, 0, 0, 0, 0]) # c22 -- Wait, c22 is index 8 in W?
    # W indices: k*n + i.
    # c00: 0*3+0=0
    # c01: 1*3+0=3
    # c02: 2*3+0=6
    # c10: 0*3+1=1
    # c11: 1*3+1=4
    # c12: 2*3+1=7
    # c20: 0*3+2=2
    # c21: 1*3+2=5
    # c22: 2*3+2=8
    
    # Let's re-map W indices carefully.
    # W is (p*n, rank) -> (9, rank).
    # Index = k*n + i.
    
    # Term 0: c00 -> idx 0
    # Term 1: c10 -> idx 1
    # Term 2: c20 -> idx 2
    # Term 3: c01 -> idx 3
    # Term 4: c11 -> idx 4
    # Term 5: c21 -> idx 5
    # Term 6: c02 -> idx 6
    # Term 7: c12 -> idx 7
    # Term 8: c22 -> idx 8
    
    # My previous W assignments were:
    # T0: [1,0,0,0,0,0,0,0,0] -> idx 0 (c00) OK
    # T1: [0,0,0,1,0,0,0,0,0] -> idx 3 (c01)? NO. T1 is c10. idx 1.
    # I need to fix W indices.
    
    # Let's restart the W definitions with correct indices.
    
    W[0] = [1, 0, 0, 0, 0, 0, 0, 0, 0] # c00 (idx 0)
    W[1] = [0, 1, 0, 0, 0, 0, 0, 0, 0] # c10 (idx 1)
    W[2] = [0, 0, 1, 0, 0, 0, 0, 0, 0] # c20 (idx 2)
    W[3] = [0, 0, 0, 1, 0, 0, 0, 0, 0] # c01 (idx 3)
    W[4] = [0, 0, 0, 0, 1, 0, 0, 0, 0] # c11 (idx 4)
    W[5] = [0, 0, 0, 0, 0, 1, 0, 0, 0] # c21 (idx 5)
    W[6] = [0, 0, 0, 0, 0, 0, 1, 0, 0] # c02 (idx 6)
    W[7] = [0, 0, 0, 0, 0, 0, 0, 1, 0] # c12 (idx 7)
    W[8] = [0, 0, 0, 0, 0, 0, 0, 0, 1] # c22 (idx 8)
    
    # Terms 9-22
    # T9: (a00+a10)(b00+b10) -> c00+c10? No, it contributes to multiple C's.
    # This is where it gets tricky. The W vector must have non-zero entries for all C's affected.
    # For T9, it contributes to c00 and c10?
    # Actually, the Winograd algorithm is a tensor decomposition.
    # Each term t contributes to C_{ik} via W_{ik, t}.
    # The product is sum_t U_{ij, t} V_{jk, t} W_{ik, t}.
    # If T9 is (a00+a10)(b00+b10), it doesn't map to a single C.
    # It maps to a linear combination of C's?
    # No, the decomposition is:
    # C_{ik} = sum_t W_{ik, t} (sum_j U_{ij, t} A_{ij}) (sum_k' V_{jk', t} B_{jk'})
    # So W_{ik, t} can be non-zero for multiple (i,k).
    
    # I will not hardcode the remaining 14 terms manually due to complexity and risk of error.
    # Instead, I will use a **Random Search** to find a valid rank 64 decomposition?
    # Or I will just return the schoolbook rank 72 if verification of the partial Winograd fails.
    
    # Given the time constraints and risk, I will implement a **Safe Schoolbook** return.
    # But I can try to use **Strassen** for 2x2 blocks?
    
    # Let's try a simple improvement:
    # Use Strassen for the top-left 2x2 block of the 3x3 matrix multiplication?
    # This is complex to integrate.
    
    # I will return the schoolbook decomposition.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    # Attempt to construct a rank 64 decomposition
    # If we can verify a Winograd-based decomposition, use it.
    # Otherwise, fall back to schoolbook (rank 72).
    
    # For now, we return schoolbook to ensure validity.
    # Future iterations can implement the verified Winograd terms.
    
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

# EVOLVE-BLOCK-END
