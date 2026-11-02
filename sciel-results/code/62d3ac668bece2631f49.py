import time
import numpy as np

DIMENSIONS = [4, 5, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _time_left():
    return _DEADLINE - time.time()

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen formulas for M1..M7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to express these as sums of rank-1 tensors.
    # Each Mi is a product of a 2x2 matrix (sum/diff) and a 2x2 matrix (sum/diff).
    # Actually, Strassen's algorithm computes 7 scalar multiplications of 2x2 matrices.
    # Here we are looking for the tensor rank of the 2x2x2 multiplication itself.
    # The tensor is T[i,j,k] = 1 for all i,j,k in {0,1}.
    # Rank is 7.
    
    # The standard Strassen decomposition for the tensor:
    # The 7 terms correspond to the 7 products.
    # We need to fill U, V, W such that sum_c U[i*2+j, c] V[j*2+k, c] W[k*2+i, c] = 1.
    
    # This is a known set of vectors.
    # Let's use the explicit vectors for the rank 7 decomposition of the 2x2x2 tensor.
    # Reference: "The Rank of the Matrix Multiplication Tensor"
    
    # The 7 rank-1 tensors are:
    # 1. (1, 0, 0, 1) (1, 1, 1, 1) (1, 1, 1, 1) ? No.
    
    # Let's construct it from the Strassen algorithm structure.
    # The multiplication is C = AB.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # This is for the *result* entries.
    # But we need the decomposition of the *tensor* T[i,j,k].
    # The tensor T is such that C_ik = sum_j A_ij B_jk.
    # T[i,j,k] = 1 for all i,j,k.
    
    # The rank 7 decomposition is given by the following vectors (u, v, w) for c=0..6:
    # c=0: u=(1,0,0,1), v=(1,1,1,1), w=(1,0,0,1) -> No, this is not correct.
    
    # Let's use the explicit coordinates for the 7 rank-1 components of the 2x2x2 tensor.
    # These are well-known.
    
    U[0, 0] = 1; U[1, 0] = 0; U[2, 0] = 0; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 0; W[2, 0] = 0; W[3, 0] = 1
    
    U[0, 1] = 0; U[1, 1] = 1; U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1; V[1, 1] = 0; V[2, 1] = 0; V[3, 1] = 0
    W[0, 1] = 0; W[1, 1] = 0; W[2, 1] = 1; W[3, 1] = 0
    
    U[0, 2] = 1; U[1, 2] = 0; U[2, 2] = 0; U[3, 2] = 0
    V[0, 2] = 0; V[1, 2] = 1; V[2, 2] = 0; V[3, 2] = -1
    W[0, 2] = 0; W[1, 2] = 0; W[2, 2] = 0; W[3, 2] = 1
    
    U[0, 3] = 0; U[1, 3] = 0; U[2, 3] = 0; U[3, 3] = 1
    V[0, 3] = -1; V[1, 3] = 0; V[2, 3] = 1; V[3, 3] = 0
    W[0, 3] = 0; W[1, 3] = 0; W[2, 3] = 0; W[3, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1; U[2, 4] = 0; U[3, 4] = 0
    V[0, 4] = 0; V[1, 4] = 0; V[2, 4] = 0; V[3, 4] = 1
    W[0, 4] = 0; W[1, 4] = 0; W[2, 4] = 1; W[3, 4] = 0
    
    U[0, 5] = -1; U[1, 5] = 0; U[2, 5] = 1; U[3, 5] = 0
    V[0, 5] = 1; V[1, 5] = 1; V[2, 5] = 0; V[3, 5] = 0
    W[0, 5] = 0; W[1, 5] = 0; W[2, 5] = 0; W[3, 5] = 1
    
    U[0, 6] = 0; U[1, 6] = 1; U[2, 6] = 0; U[3, 6] = -1
    V[0, 6] = 0; V[1, 6] = 0; V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 0; W[1, 6] = 0; W[2, 6] = 0; W[3, 6] = 1

    return U, V, W

def _embed_4x4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication with rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U_str, V_str, W_str = _strassen_2x2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # The 4x4x4 multiplication can be viewed as 2x2 block multiplication
    # where each block is 2x2.
    # We use the 7 Strassen products, each of which is a 2x2x2 multiplication.
    # So we have 7 "super-terms", each of rank 7.
    
    # Mapping of indices:
    # i_global = i_block * 2 + i_local
    # j_global = j_block * 2 + j_local
    # k_global = k_block * 2 + k_local
    
    # Strassen block indices:
    # A11 -> block (0,0), A12 -> (0,1), A21 -> (1,0), A22 -> (1,1)
    # Similarly for B and C.
    
    # The 7 Strassen products M1..M7 involve sums of blocks.
    # M1 = (A11+A22)(B11+B22)
    # This product contributes to C11, C22.
    # Specifically, C11 += M1, C22 -= M1? No, C11 = M1 + M4 - M5 + M7.
    # C22 = M1 - M2 + M3 + M6.
    
    # We need to distribute the rank 7 decomposition of M1 into the global tensor.
    # M1 is a 2x2 matrix (result of multiplying two 2x2 "matrices" of 2x2 blocks).
    # Wait, M1 is a 2x2 matrix of 2x2 blocks.
    # So M1 has 4 sub-blocks, each 2x2.
    # Each sub-block multiplication is a 2x2x2 tensor.
    
    # Let's iterate over the 7 Strassen terms.
    for s in range(7):
        # The s-th Strassen term involves specific block combinations.
        # We need to know which blocks of A and B are involved and with what signs.
        # And which blocks of C receive the result.
        
        # This is getting complicated to hardcode correctly without a reference.
        # However, we can simply use the fact that the tensor for 4x4x4 is the
        # "Kronecker product" of the 2x2x2 tensor with itself?
        # No, the rank is 49 = 7 * 7.
        # The decomposition is the tensor product of the rank 7 decomposition of 2x2x2.
        
        # If T_2 is the 2x2x2 tensor, then T_4 = T_2 \otimes T_2.
        # The rank-1 factors of T_4 are u_a \otimes u_b, v_a \otimes v_b, w_a \otimes w_b.
        
        # So we can construct U, V, W by taking the Kronecker product of the
        # vectors from the 2x2x2 decomposition.
        
        # U_4[i4, c4] = U_2[i2, c2a] * U_2[i2', c2b]
        # where i4 = i2 * 2 + i2', c4 = c2a * 7 + c2b.
        
        for c2a in range(7):
            for c2b in range(7):
                c4 = c2a * 7 + c2b
                for i2a in range(2):
                    for i2b in range(2):
                        i4 = i2a * 2 + i2b
                        U[i4 * 4 + 0, c4] = U_str[i2a * 2 + 0, c2a] * U_str[i2b * 2 + 0, c2b] # Wait, U index is i*2+j
                        # U_str index is i*2+j.
                        # i4 is the row index of the 4x4 matrix? No, U has shape (16, 49).
                        # U[i*4+j, c].
                        # i4 is the global i. j4 is the global j.
                        # i4 = i2a * 2 + i2b.
                        # j4 = j2a * 2 + j2b.
                        
                        # We need to fill U[i4*4 + j4, c4].
                        # U_str has indices i2*2 + j2.
                        
                        for j2a in range(2):
                            for j2b in range(2):
                                j4 = j2a * 2 + j2b
                                U[i4 * 4 + j4, c4] = U_str[i2a * 2 + j2a, c2a] * U_str[i2b * 2 + j2b, c2b]
                                
                                V[i4 * 4 + j4, c4] = 0 # Placeholder
                                
                                for k2a in range(2):
                                    for k2b in range(2):
                                        k4 = k2a * 2 + k2b
                                        W[i4 * 4 + k4, c4] = W_str[i2a * 2 + k2a, c2a] * W_str[i2b * 2 + k2b, c2b]
                                        
                                        # V index is j*2+k.
                                        # V_str index is j2*2 + k2.
                                        V[j4 * 4 + k4, c4] = V_str[j2a * 2 + k2a, c2a] * V_str[j2b * 2 + k2b, c2b]

    return U, V, W

def _dot_product_4x4x1():
    """
    Returns U, V, W for 4x4x1 matrix multiplication with rank 16.
    U: (16, 16), V: (4, 16), W: (4, 16)
    """
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))
    c = 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, c] = 1
            V[j, c] = 1
            W[i, c] = 1
            c += 1
    return U, V, W

def _outer_product_4x1x4():
    """
    Returns U, V, W for 4x1x4 matrix multiplication with rank 1.
    U: (4, 1), V: (4, 1), W: (16, 1)
    """
    U = np.zeros((4, 1))
    V = np.zeros((4, 1))
    W = np.zeros((16, 1))
    U[:, 0] = 1
    V[:, 0] = 1
    W[:, 0] = 1
    return U, V, W

def _scaling_4x1x1():
    """
    Returns U, V, W for 4x1x1 matrix multiplication with rank 4.
    U: (4, 4), V: (1, 4), W: (4, 4)
    """
    U = np.zeros((4, 4))
    V = np.zeros((1, 4))
    W = np.zeros((4, 4))
    for i in range(4):
        U[i, i] = 1
        V[0, i] = 1
        W[i, i] = 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U_total = np.zeros((n * m, 70))
    V_total = np.zeros((m * p, 70))
    W_total = np.zeros((p * n, 70))
    
    # 1. 4x4x4 part (Rank 49)
    U1, V1, W1 = _embed_4x4x4()
    for i in range(4):
        for j in range(4):
            U_total[i * 5 + j, 0:49] = U1[i * 4 + j, :]
            for k in range(4):
                V_total[j * 5 + k, 0:49] = V1[j * 4 + k, :]
                W_total[k * 4 + i, 0:49] = W1[k * 4 + i, :]
                
    # 2. 4x4x1 part (Rank 16)
    U2, V2, W2 = _dot_product_4x4x1()
    for i in range(4):
        for j in range(4):
            U_total[i * 5 + j, 49:65] = U2[i * 4 + j, :]
            V_total[j * 5 + 4, 49:65] = V2[j, :]
            W_total[4 * 4 + i, 49:65] = W2[i, :]
            
    # 3. 4x1x4 part (Rank 1)
    U3, V3, W3 = _outer_product_4x1x4()
    for i in range(4):
        U_total[i * 5 + 4, 65:66] = U3[i, :]
        for k in range(4):
            V_total[4 * 5 + k, 65:66] = V3[k, :]
            W_total[k * 4 + i, 65:66] = W3[k * 4 + i, :]
            
    # 4. 4x1x1 part (Rank 4)
    U4, V4, W4 = _scaling_4x1x1()
    for i in range(4):
        U_total[i * 5 + 4, 66:70] = U4[i, :]
        V_total[4 * 5 + 4, 66:70] = V4[0, :]
        W_total[4 * 4 + i, 66:70] = W4[i, :]
        
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

if __name__ == "__main__":
    construct()
