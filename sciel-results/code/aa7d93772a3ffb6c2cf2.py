import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Target: M_{4,4,6}
    # Strategy: Kronecker Product M_{2,2,2} x M_{2,2,3}
    # Rank M_{2,2,2} = 7 (Strassen)
    # Rank M_{2,2,3} = 11 (Strassen 7 + Schoolbook 4)
    # Total Rank = 77
    
    # 1. Construct M_{2,2,2} (Strassen)
    # U1: (4, 7), V1: (4, 7), W1: (4, 7)
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # Indices: 00, 01, 10, 11
    # Strassen Terms:
    # 0: (A00+A11)(B00+B11) -> C00, C11
    U1[0,0]=1; U1[3,0]=1
    V1[0,0]=1; V1[3,0]=1
    W1[0,0]=1; W1[3,0]=1
    
    # 1: (A10+A11)B00 -> C10, C11
    U1[2,1]=1; U1[3,1]=1
    V1[0,1]=1
    W1[1,1]=1; W1[3,1]=-1
    
    # 2: A00(B01-B11) -> C01, C11
    U1[0,2]=1
    V1[1,2]=1; V1[3,2]=-1
    W1[2,2]=1; W1[3,2]=1
    
    # 3: A11(B10-B00) -> C00, C10
    U1[3,3]=1
    V1[2,3]=1; V1[0,3]=-1
    W1[0,3]=1; W1[1,3]=1
    
    # 4: (A00+A01)B11 -> C00, C01
    U1[0,4]=1; U1[1,4]=1
    V1[3,4]=1
    W1[0,4]=-1; W1[2,4]=1
    
    # 5: (A10-A00)(B00+B01) -> C11
    U1[2,5]=1; U1[0,5]=-1
    V1[0,5]=1; V1[1,5]=1
    W1[3,5]=1
    
    # 6: (A01-A11)(B10+B11) -> C00
    U1[1,6]=1; U1[3,6]=-1
    V1[2,6]=1; V1[3,6]=1
    W1[0,6]=1

    # 2. Construct M_{2,2,3}
    # U2: (4, 11), V2: (6, 11), W2: (6, 11)
    U2 = np.zeros((4, 11))
    V2 = np.zeros((6, 11))
    W2 = np.zeros((6, 11))
    
    # Part 1: Strassen for cols 0,1 of B (which are cols 0,1 of the 2x3 block)
    # Note: In M_{2,2,3}, B is 2x3. Cols 0,1 form a 2x2 matrix.
    # We use the same Strassen terms as U1, V1, W1 but mapped to the 2x3 indices.
    # B indices: 00, 10 (Col 0); 01, 11 (Col 1); 02, 12 (Col 2)
    # V2 indices: 0:00, 1:01, 2:10, 3:11, 4:02, 5:12 ? 
    # Let's stick to the previous program's mapping for consistency:
    # V2 rows: 0:00, 1:01, 2:10, 3:11, 4:02, 5:12
    # Wait, the previous program used:
    # V2[0] = V1[0] (00)
    # V2[1] = V1[1] (01)
    # V2[3] = V1[2] (10)
    # V2[4] = V1[3] (11)
    # This mapping is confusing. Let's use a standard row-major mapping for B (2x3).
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=3, B[1,1]=4, B[1,2]=5.
    
    # Strassen terms for B cols 0,1 (indices 0,1,3,4)
    for c in range(7):
        U2[:, c] = U1[:, c]
        
        # Map V1 (2x2) to V2 (2x3)
        # V1 indices: 0:00, 1:01, 2:10, 3:11
        # V2 indices: 0:00, 1:01, 2:10, 3:11, 4:02, 5:12
        # So V2[0]=V1[0], V2[1]=V1[1], V2[2]=V1[2], V2[3]=V1[3]
        V2[0, c] = V1[0, c]
        V2[1, c] = V1[1, c]
        V2[2, c] = V1[2, c]
        V2[3, c] = V1[3, c]
        V2[4, c] = 0
        V2[5, c] = 0
        
        # Map W1 (2x2) to W2 (2x3)
        # W1 indices: 0:00, 1:01, 2:10, 3:11 (k*n+i, k=0,1; i=0,1)
        # W2 indices: 0:00, 1:01, 2:10, 3:11, 4:02, 5:12 (k=0,1,2; i=0,1)
        # So W2[0]=W1[0], W2[1]=W1[1], W2[2]=W1[2], W2[3]=W1[3]
        W2[0, c] = W1[0, c]
        W2[1, c] = W1[1, c]
        W2[2, c] = W1[2, c]
        W2[3, c] = W1[3, c]
        W2[4, c] = 0
        W2[5, c] = 0

    # Part 2: Schoolbook for col 2 of B (indices 2, 5)
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # Term 7: A00 B02 -> C02
    U2[0, 7] = 1
    V2[2, 7] = 1  # B[0,2] is index 2
    W2[4, 7] = 1  # C[0,2] is index 4
    
    # Term 8: A01 B12 -> C02
    U2[1, 8] = 1
    V2[5, 8] = 1  # B[1,2] is index 5
    W2[4, 8] = 1
    
    # Term 9: A10 B02 -> C12
    U2[2, 9] = 1
    V2[2, 9] = 1
    W2[5, 9] = 1  # C[1,2] is index 5
    
    # Term 10: A11 B12 -> C12
    U2[3, 10] = 1
    V2[5, 10] = 1
    W2[5, 10] = 1

    # 3. Kronecker Product
    # U_full = kron(U1, U2) ?
    # Dimensions:
    # U1: (4, 7), U2: (4, 11) -> U_full: (16, 77)
    # V1: (4, 7), V2: (6, 11) -> V_full: (24, 77)
    # W1: (4, 7), W2: (6, 11) -> W_full: (24, 77)
    
    # The Kronecker product of matrices A (m1, k1) and B (m2, k2) is (m1*m2, k1*k2).
    # np.kron(U1, U2) gives shape (4*4, 7*11) = (16, 77).
    # This matches the required shapes.
    
    U_full = np.kron(U1, U2)
    V_full = np.kron(V1, V2)
    W_full = np.kron(W1, W2)
    
    # Convert to list of lists as required by interface
    U = U_full.tolist()
    V = V_full.tolist()
    W = W_full.tolist()
    
    return {"U": U, "V": V, "W": W}
