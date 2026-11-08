import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target rank is 46
    R = 46
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    # Helper to add a sub-decomposition
    # i_off, j_off, k_off are global offsets
    # sub_U, sub_V, sub_W are local factor matrices
    # rank_start is the starting index in the global rank dimension
    def add_sub(i_off, j_off, k_off, sub_U, sub_V, sub_W, rank_start):
        n_sub = sub_U.shape[0]
        m_sub = sub_V.shape[0]
        p_sub = sub_W.shape[0]
        r_sub = sub_U.shape[1]
        
        for i in range(n_sub):
            for r in range(r_sub):
                U[i_off + i, rank_start + r] = sub_U[i, r]
        
        for j in range(m_sub):
            for r in range(r_sub):
                V[j_off + j, rank_start + r] = sub_V[j, r]
        
        for k in range(p_sub):
            for r in range(r_sub):
                W[k_off + k, rank_start + r] = sub_W[k, r]

    # --- Define Sub-Decompositions ---
    
    # 1. M_2_2_2 (Rank 7) - Strassen
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    U_222 = np.zeros((2, 7))
    V_222 = np.zeros((2, 7))
    W_222 = np.zeros((2, 7))
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    U_222[0, 0] = 1; U_222[1, 0] = 1
    V_222[0, 0] = 1; V_222[1, 0] = 1
    W_222[0, 0] = 1; W_222[1, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10
    U_222[1, 1] = 1
    V_222[0, 1] = 1
    W_222[0, 1] = 1
    
    # M3 = A00(B01-B11) -> C01
    U_222[0, 2] = 1
    V_222[0, 2] = 1; V_222[1, 2] = -1
    W_222[1, 2] = 1
    
    # M4 = A11(B10-B00) -> C10
    U_222[1, 3] = 1
    V_222[0, 3] = -1; V_222[1, 3] = 1
    W_222[0, 3] = 1
    
    # M5 = (A00+A01)B11 -> C01
    U_222[0, 4] = 1; U_222[1, 4] = 1 # Wait, A00+A01 is row 0?
    # A00 is i=0, j=0. A01 is i=0, j=1.
    # So U[0,4]=1. U[1,4]=0?
    # M5 = (A00+A01)B11.
    # Linear form in A: A00+A01. i=0.
    # U[0,4]=1. U[1,4]=0.
    U_222[0, 4] = 1
    U_222[1, 4] = 0
    V_222[1, 4] = 1
    W_222[1, 4] = 1
    
    # M6 = (A10-A11)(B01+B10) -> C11
    U_222[1, 5] = 1
    V_222[0, 5] = 1; V_222[1, 5] = -1
    W_222[1, 5] = 1
    
    # M7 = (B11-B01)(A00+A10) -> C00
    # (A00+A10) is column 0? No, A00 is i=0,j=0. A10 is i=1,j=0.
    # Linear form in A: A00+A10. j=0.
    # U[0,6]=1, U[1,6]=1.
    # Linear form in B: B11-B01. i=1? No, B11 is j=1,k=1. B01 is j=0,k=1.
    # V[1,6]=1, V[0,6]=-1.
    # W[0,6]=1.
    U_222[0, 6] = 1; U_222[1, 6] = 1
    V_222[0, 6] = -1; V_222[1, 6] = 1
    W_222[0, 6] = 1
    
    # 2. M_2_2_1 (Rank 4)
    # Local: i in {0,1}, j in {0,1}, k in {0}
    U_221 = np.zeros((2, 4))
    V_221 = np.zeros((2, 4))
    W_221 = np.zeros((1, 4))
    
    # A00 B00 -> C0
    U_221[0, 0] = 1; V_221[0, 0] = 1; W_221[0, 0] = 1
    # A01 B10 -> C0
    U_221[0, 1] = 1; V_221[1, 1] = 1; W_221[0, 1] = 1
    # A10 B00 -> C1
    U_221[1, 2] = 1; V_221[0, 2] = 1; W_221[0, 2] = 1
    # A11 B10 -> C1
    U_221[1, 3] = 1; V_221[1, 3] = 1; W_221[0, 3] = 1
    
    # 3. M_2_1_2 (Rank 4)
    # Local: i in {0,1}, j in {0}, k in {0,1}
    U_212 = np.zeros((2, 4))
    V_212 = np.zeros((1, 4))
    W_212 = np.zeros((2, 4))
    
    # A00 B00 -> C00
    U_212[0, 0] = 1; V_212[0, 0] = 1; W_212[0, 0] = 1
    # A00 B01 -> C01
    U_212[0, 1] = 1; V_212[0, 1] = 1; W_212[1, 1] = 1
    # A10 B00 -> C10
    U_212[1, 2] = 1; V_212[0, 2] = 1; W_212[0, 2] = 1
    # A10 B01 -> C11
    U_212[1, 3] = 1; V_212[0, 3] = 1; W_212[1, 3] = 1
    
    # 4. M_2_1_1 (Rank 2)
    # Local: i in {0,1}, j in {0}, k in {0}
    U_211 = np.zeros((2, 2))
    V_211 = np.zeros((1, 2))
    W_211 = np.zeros((1, 2))
    
    # A00 B00 -> C0
    U_211[0, 0] = 1; V_211[0, 0] = 1; W_211[0, 0] = 1
    # A10 B00 -> C1
    U_211[1, 1] = 1; V_211[0, 1] = 1; W_211[0, 1] = 1
    
    # 5. M_2_1_3 (Rank 6)
    # Local: i in {0,1}, j in {0}, k in {0,1,2}
    U_213 = np.zeros((2, 6))
    V_213 = np.zeros((1, 6))
    W_213 = np.zeros((3, 6))
    
    # A00 B00 -> C0
    U_213[0, 0] = 1; V_213[0, 0] = 1; W_213[0, 0] = 1
    # A00 B01 -> C1
    U_213[0, 1] = 1; V_213[0, 1] = 1; W_213[1, 1] = 1
    # A00 B02 -> C2
    U_213[0, 2] = 1; V_213[0, 2] = 1; W_213[2, 2] = 1
    # A10 B00 -> C0
    U_213[1, 3] = 1; V_213[0, 3] = 1; W_213[0, 3] = 1
    # A10 B01 -> C1
    U_213[1, 4] = 1; V_213[0, 4] = 1; W_213[1, 4] = 1
    # A10 B02 -> C2
    U_213[1, 5] = 1; V_213[0, 5] = 1; W_213[2, 5] = 1
    
    # --- Assemble Global Decomposition ---
    
    # Block 0,0: i in {0,1}, j in {0,1}, k in {0,1}
    # Use M_2_2_2 (Rank 7)
    add_sub(0, 0, 0, U_222, V_222, W_222, 0)
    
    # Block 0,1: i in {0,1}, j in {0,1}, k in {2,3,4}
    # Use M_2_2_3 (Rank 11)
    # M_2_2_3 is M_2_2_2 (for k=0,1) + M_2_2_1 (for k=2)
    # Local k indices: 0,1,2 map to global 2,3,4
    # We need to construct M_2_2_3 locally
    U_223 = np.hstack([U_222, U_221])
    V_223 = np.hstack([V_222, V_221])
    # W_223: W_222 is (2,7), W_221 is (1,4).
    # We need W_223 to be (3, 11).
    # W_222 corresponds to local k=0,1.
    # W_221 corresponds to local k=2.
    W_223 = np.zeros((3, 11))
    W_223[0:2, 0:7] = W_222
    W_223[2, 7:11] = W_221
    
    add_sub(0, 0, 2, U_223, V_223, W_223, 7)
    
    # Block 1,0: i in {0,1}, j in {2,3,4}, k in {0,1}
    # Use M_2_3_2 (Rank 11)
    # M_2_3_2 is M_2_2_2 (for j=0,1) + M_2_1_2 (for j=2)
    # Local j indices: 0,1,2 map to global 2,3,4
    U_232 = np.hstack([U_222, U_212])
    # V_232: V_222 is (2,7), V_212 is (1,4).
    # V_222 corresponds to local j=0,1.
    # V_212 corresponds to local j=2.
    V_232 = np.zeros((3, 11))
    V_232[0:2, 0:7] = V_222
    V_232[2, 7:11] = V_212
    W_232 = np.hstack([W_222, W_212])
    
    add_sub(0, 2, 0, U_232, V_232, W_232, 18)
    
    # Block 1,1: i in {0,1}, j in {2,3,4}, k in {2,3,4}
    # Use M_2_3_3 (Rank 17)
    # M_2_3_3 is M_2_2_3 (for j=0,1) + M_2_1_3 (for j=2)
    # Local j indices: 0,1,2 map to global 2,3,4
    # Local k indices: 0,1,2 map to global 2,3,4
    U_233 = np.hstack([U_223, U_213])
    # V_233: V_223 is (3,11), V_213 is (1,6).
    # V_223 corresponds to local j=0,1,2? No, V_223 is for M_2_2_3 which has j=0,1.
    # Wait, M_2_3_3 split j into 2+1.
    # j=0,1 use M_2_2_3. j=2 uses M_2_1_3.
    # So V_233 should have rows for j=0,1,2.
    # V_223 has 3 rows? No, V_223 was constructed for M_2_2_3.
    # M_2_2_3 has j=0,1. So V_223 should have 2 rows.
    # Let's re-verify V_223 construction.
    # V_222 is (2,7). V_221 is (2,4).
    # V_223 = hstack(V_222, V_221) -> (2, 11).
    # My previous construction of V_223 was wrong.
    
    # Correct V_223:
    V_223_correct = np.hstack([V_222, V_221]) # (2, 11)
    
    # Correct V_233:
    # Rows: j=0,1 from V_223_correct. Row j=2 from V_213.
    V_233 = np.zeros((3, 17))
    V_233[0:2, 0:11] = V_223_correct
    V_233[2, 11:17] = V_213
    
    # W_233:
    # W_223 is (3, 11). W_213 is (3, 6).
    W_233 = np.hstack([W_223, W_213]) # (3, 17)
    
    add_sub(0, 2, 2, U_233, V_233, W_233, 29)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

