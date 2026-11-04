import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build a decomposition using block Strassen for 2x2 blocks
    # and schoolbook for the rest.
    
    # List of terms: (U_vec, V_vec, W_vec)
    # U_vec is length n*m = 9
    # V_vec is length m*p = 21
    # W_vec is length p*n = 21
    
    terms = []
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        # Adds a Strassen decomposition for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]
        # Indices in U: (i, j) -> i*m + j
        # Indices in V: (j, k) -> j*p + k
        # Indices in W: (k, i) -> k*n + i
        
        # The 7 linear forms on A (local 2x2)
        # A00, A01, A10, A11
        # L1: A00 + A11
        # L2: A10 + A11
        # L3: A00
        # L4: A11
        # L5: A00 + A01
        # L6: A10 - A00
        # L7: A01 - A11
        
        # The 7 linear forms on B (local 2x2)
        # B00, B01, B10, B11
        # M1: B00 + B11
        # M2: B00
        # M3: B10 - B11
        # M4: B01 - B00
        # M5: B11
        # M6: B00 + B01
        # M7: B10 + B11
        
        # Reconstruction coefficients for C (local 2x2)
        # C00 = M1 - M4 + M6 - M7
        # C01 = M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 - M6
        
        # Term 1
        U1 = np.zeros(n*m); U1[(i_off)*m + (j_off)] = 1; U1[(i_off+1)*m + (j_off+1)] = 1
        V1 = np.zeros(m*p); V1[(j_off)*p + (k_off)] = 1; V1[(j_off+1)*p + (k_off+1)] = 1
        W1 = np.zeros(p*n); W1[(k_off)*n + i_off] = 1; W1[(k_off)*n + (i_off+1)] = 1; W1[(k_off+1)*n + (i_off)] = -1; W1[(k_off+1)*n + (i_off+1)] = 1
        terms.append((U1, V1, W1))
        
        # Term 2
        U2 = np.zeros(n*m); U2[(i_off+1)*m + (j_off)] = 1; U2[(i_off+1)*m + (j_off+1)] = 1
        V2 = np.zeros(m*p); V2[(j_off)*p + (k_off)] = 1
        W2 = np.zeros(p*n); W2[(k_off+1)*n + (i_off+1)] = 1
        terms.append((U2, V2, W2))
        
        # Term 3
        U3 = np.zeros(n*m); U3[(i_off)*m + (j_off)] = 1
        V3 = np.zeros(m*p); V3[(j_off+1)*p + (k_off)] = 1; V3[(j_off+1)*p + (k_off+1)] = -1
        W3 = np.zeros(p*n); W3[(k_off+1)*n + (i_off+1)] = -1
        terms.append((U3, V3, W3))
        
        # Term 4
        U4 = np.zeros(n*m); U4[(i_off+1)*m + (j_off+1)] = 1
        V4 = np.zeros(m*p); V4[(j_off)*p + (k_off+1)] = 1; V4[(j_off)*p + (k_off)] = -1
        W4 = np.zeros(p*n); W4[(k_off)*n + i_off] = -1; W4[(k_off+1)*n + (i_off+1)] = 1
        terms.append((U4, V4, W4))
        
        # Term 5
        U5 = np.zeros(n*m); U5[(i_off)*m + (j_off)] = 1; U5[(i_off)*m + (j_off+1)] = 1
        V5 = np.zeros(m*p); V5[(j_off+1)*p + (k_off+1)] = 1
        W5 = np.zeros(p*n); W5[(k_off+1)*n + i_off] = 1; W5[(k_off+1)*n + (i_off+1)] = 1
        terms.append((U5, V5, W5))
        
        # Term 6
        U6 = np.zeros(n*m); U6[(i_off+1)*m + (j_off)] = 1; U6[(i_off)*m + (j_off)] = -1
        V6 = np.zeros(m*p); V6[(j_off)*p + (k_off)] = 1; V6[(j_off)*p + (k_off+1)] = 1
        W6 = np.zeros(p*n); W6[(k_off)*n + i_off] = 1; W6[(k_off+1)*n + (i_off+1)] = -1
        terms.append((U6, V6, W6))
        
        # Term 7
        U7 = np.zeros(n*m); U7[(i_off)*m + (j_off+1)] = 1; U7[(i_off+1)*m + (j_off+1)] = -1
        V7 = np.zeros(m*p); V7[(j_off+1)*p + (k_off)] = 1; V7[(j_off+1)*p + (k_off+1)] = 1
        W7 = np.zeros(p*n); W7[(k_off+1)*n + i_off] = 1; W7[(k_off+1)*n + (i_off+1)] = -1
        terms.append((U7, V7, W7))

    def add_schoolbook(i, j, k):
        U = np.zeros(n*m); U[i*m + j] = 1
        V = np.zeros(m*p); V[j*p + k] = 1
        W = np.zeros(p*n); W[k*n + i] = 1
        terms.append((U, V, W))

    # Block decomposition:
    # i: 0,1,2
    # j: 0,1,2
    # k: 0..6
    
    # Block 1: i in 0,1; j in 0,1; k in 0,1
    add_strassen_2x2x2(0, 0, 0)
    # Block 2: i in 0,1; j in 0,1; k in 2,3
    add_strassen_2x2x2(0, 0, 2)
    # Block 3: i in 0,1; j in 0,1; k in 4,5
    add_strassen_2x2x2(0, 0, 4)
    # Block 4: i in 0,1; j in 0,1; k in 6
    # 2x2x1: Rank 4
    add_schoolbook(0, 0, 6)
    add_schoolbook(0, 1, 6)
    add_schoolbook(1, 0, 6)
    add_schoolbook(1, 1, 6)
    
    # Block 5: i in 2; j in 0,1; k in 0,1
    # 1x2x2: Rank 4
    add_schoolbook(2, 0, 0)
    add_schoolbook(2, 0, 1)
    add_schoolbook(2, 1, 0)
    add_schoolbook(2, 1, 1)
    
    # Block 6: i in 2; j in 0,1; k in 2,3
    add_schoolbook(2, 0, 2)
    add_schoolbook(2, 0, 3)
    add_schoolbook(2, 1, 2)
    add_schoolbook(2, 1, 3)
    
    # Block 7: i in 2; j in 0,1; k in 4,5
    add_schoolbook(2, 0, 4)
    add_schoolbook(2, 0, 5)
    add_schoolbook(2, 1, 4)
    add_schoolbook(2, 1, 5)
    
    # Block 8: i in 2; j in 0,1; k in 6
    add_schoolbook(2, 0, 6)
    add_schoolbook(2, 1, 6)
    
    # Block 9: i in 0,1,2; j in 2; k in 0..6
    # 3x1x7: Rank 21
    for i in range(3):
        for k in range(7):
            add_schoolbook(i, 2, k)
            
    rank = len(terms)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
