"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication
    using two levels of Strassen's algorithm.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Strassen's algorithm for 2x2 matrix multiplication (rank 7)
    # U2: shape (2*2, 7) = (4, 7)
    # V2: shape (2*2, 7) = (4, 7)
    # W2: shape (2*2, 7) = (4, 7)
    
    # Define the 7 rank-1 tensors for 2x2 Strassen
    # Indexing: U[i*m + j], V[j*p + k], W[k*n + i]
    # For 2x2: i,j,k in {0,1}
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    # U1: A00=1, A11=1 -> U[0]=1, U[3]=1
    # V1: B00=1, B11=1 -> V[0]=1, V[3]=1
    # W1: C00 += M1, C11 += M1 -> W[0]=1, W[3]=1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # U2: A10=1, A11=1 -> U[2]=1, U[3]=1
    # V2: B00=1 -> V[0]=1
    # W2: C10 += M2, C11 -= M2 -> W[1]=1, W[3]=-1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # M3 = A10(B10-B00)
    # U3: A10=1 -> U[2]=1
    # V3: B10=1, B00=-1 -> V[2]=1, V[0]=-1
    # W3: C10 += M3, C11 += M3 -> W[1]=1, W[3]=1
    U2[2, 2] = 1
    V2[2, 2] = 1; V2[0, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4 = A00(B11-B10)
    # U4: A00=1 -> U[0]=1
    # V4: B11=1, B10=-1 -> V[3]=1, V[2]=-1
    # W4: C00 += M4, C10 += M4 -> W[0]=1, W[1]=1
    U2[0, 3] = 1
    V2[3, 3] = 1; V2[2, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M5 = (A00+A01)B10
    # U5: A00=1, A01=1 -> U[0]=1, U[1]=1
    # V5: B10=1 -> V[2]=1
    # W5: C00 -= M5, C10 += M5 -> W[0]=-1, W[1]=1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[2, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (A10-A00)B01
    # U6: A10=1, A00=-1 -> U[2]=1, U[0]=-1
    # V6: B01=1 -> V[1]=1
    # W6: C01 += M6, C11 += M6 -> W[2]=1, W[3]=1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[1, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1
    
    # M7 = (A01-A11)B11
    # U7: A01=1, A11=-1 -> U[1]=1, U[3]=-1
    # V7: B11=1 -> V[3]=1
    # W7: C00 += M7, C01 += M7 -> W[0]=1, W[2]=1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[3, 6] = 1
    W2[0, 6] = 1; W2[2, 6] = 1
    
    # Now construct the 4x4 decomposition using Kronecker products
    # Rank = 7 * 7 = 49
    rank = 49
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    # For each pair of Strassen factors (a, b) where a, b in {0..6}
    # The corresponding 4x4 factor is the Kronecker product
    for a in range(7):
        for b in range(7):
            col = a * 7 + b
            # U4 = U2[:, a] ⊗ U2[:, b]
            U4[:, col] = np.kron(U2[:, a], U2[:, b])
            V4[:, col] = np.kron(V2[:, a], V2[:, b])
            W4[:, col] = np.kron(W2[:, a], W2[:, b])
    
    # Convert to lists
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
