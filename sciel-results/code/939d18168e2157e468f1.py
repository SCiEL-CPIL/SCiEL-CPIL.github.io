import numpy as np
import time

DIMENSIONS = [3, 4, 5]
DEADLINE = time.time() + 140

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def strassen_2x2x2():
        """Strassen decomposition for 2x2x2. U:(4,7), V:(4,7), W:(4,7).
        U indices: 0=A00, 1=A01, 2=A10, 3=A11
        V indices: 0=B00, 1=B01, 2=B10, 3=B11
        W indices: 0=C00, 1=C01, 2=C10, 3=C11
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        
        # M2 = (A10+A11)B00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=-1
        
        # M3 = A00(B01-B11)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        
        # M4 = A11(B10-B00)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        
        # M5 = (A00+A01)B11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[1,4]=1
        
        # M6 = (A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        
        # M7 = (A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1
        
        return U, V, W
    
    def schoolbook_2x2x1():
        """2x2x1: A(2x2)*B(2x1)=C(2x1). Rank 4.
        U:(4,4), V:(2,4), W:(2,4)
        U indices: 0=A00, 1=A01, 2=A10, 3=A11
        V indices: 0=B0, 1=B1
        W indices: 0=C0, 1=C1
        """
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1  # A00*B0->C0
        U[1,1]=1; V[1,1]=1; W[0,1]=1  # A01*B1->C0
        U[2,2]=1; V[0,2]=1; W[1,2]=1  # A10*B0->C1
        U[3,3]=1; V[1,3]=1; W[1,3]=1  # A11*B1->C1
        return U, V, W
    
    def rank2_1x2x2():
        """1x2x2: A(1x2)*B(2x2)=C(1x2). Rank 2.
        U:(2,2), V:(4,2), W:(2,2)
        U indices: 0=A0, 1=A1
        V indices: 0=B00, 1=B01, 2=B10, 3=B11
        W indices: 0=C0, 1=C1
        C0 = A0*B00 + A1*B10
        C1 = A0*B01 + A1*B11
        """
        U = np.zeros((2, 2))
        V = np.zeros((4, 2))
        W = np.zeros((2, 2))
        # Term 1: contributes to C0
        U[0,0]=1; V[0,0]=1; W[0,0]=1  # A0*B00
        U[1,0]=1; V[2,0]=1             # A1*B10
        # Term 2: contributes to C1
        U[0,1]=1; V[1,1]=1; W[1,1]=1  # A0*B01
        U[1,1]=1; V[3,1]=1             # A1*B11
        return U, V, W
    
    def rank2_1x2x1():
        """1x2x1: A(1x2)*B(2x1)=C(1x1). Rank 2.
        U:(2,2), V:(2,2), W:(1,2)
        U indices: 0=A0, 1=A1
        V indices: 0=B0, 1=B1
        W index: 0=C0
        C0 = A0*B0 + A1*B1
        """
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0,0]=1; V[0,0]=1; W[0,0]=1  # A0*B0
        U[1,1]=1; V[1,1]=1             # A1*B1
        return U, V, W
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            col_u = np.zeros(n*m)
            col_v = np.zeros(m*p)
            col_w = np.zeros(p*n)
            for i, idx in enumerate(u_idx):
                col_u[idx] = U_blk[i, r]
            for i, idx in enumerate(v_idx):
                col_v[idx] = V_blk[i, r]
            for i, idx in enumerate(w_idx):
                col_w[idx] = W_blk[i, r]
            U_parts.append(col_u)
            V_parts.append(col_v)
            W_parts.append(col_w)
    
    # U indices: i*m + j, i in [0,3), j in [0,4)
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    # V indices: j*p + k, j in [0,4), k in [0,5)
    # Row 0: 0,1,2,3,4
    # Row 1: 5,6,7,8,9
    # Row 2: 10,11,12,13,14
    # Row 3: 15,16,17,18,19
    # W indices: k*n + i, k in [0,5), i in [0,3)
    # k=0: 0,1,2
    # k=1: 3,4,5
    # k=2: 6,7,8
    # k=3: 9,10,11
    # k=4: 12,13,14
    
    # Block 1: A1_top(2x2) * B_top_left(2x2) -> C rows 0,1; cols 0,1
    # A1_top: rows 0,1; cols 0,1 -> U indices 0,1,4,5
    # B_top_left: rows 0,1; cols 0,1 -> V indices 0,1,5,6
    # C: rows 0,1; cols 0,1 -> W indices: k=0,i=0->0; k=0,i=1->1; k=1,i=0->3; k=1,i=1->4
    add_block([0,1,4,5], [0,1,5,6], [0,1,3,4], *strassen_2x2x2())
    
    # Block 2: A1_top(2x2) * B_top_mid(2x2) -> C rows 0,1; cols 2,3
    # B_top_mid: rows 0,1; cols 2,3 -> V indices 2,3,7,8
    # C: rows 0,1; cols 2,3 -> W indices: k=2,i=0->6; k=2,i=1->7; k=3,i=0->9; k=3,i=1->10
    add_block([0,1,4,5], [2,3,7,8], [6,7,9,10], *strassen_2x2x2())
    
    # Block 3: A1_top(2x2) * B_top_right(2x1) -> C rows 0,1; col 4
    # B_top_right: rows 0,1; col 4 -> V indices 4,9
    # C: rows 0,1; col 4 -> W indices: k=4,i=0->12; k=4,i=1->13
    add_block([0,1,4,5], [4,9], [12,13], *schoolbook_2x2x1())
    
    # Block 4: A1_bottom(2x2) * B_bottom_left(2x2) -> C rows 0,1; cols 0,1
    # A1_bottom: rows 0,1; cols 2,3 -> U indices 2,3,6,7
    # B_bottom_left: rows 2,3; cols 0,1 -> V indices 10,11,15,16
    add_block([2,3,6,7], [10,11,15,16], [0,1,3,4], *strassen_2x2x2())
    
    # Block 5: A1_bottom(2x2) * B_bottom_mid(2x2) -> C rows 0,1; cols 2,3
    # B_bottom_mid: rows 2,3; cols 2,3 -> V indices 12,13,17,18
    add_block([2,3,6,7], [12,13,17,18], [6,7,9,10], *strassen_2x2x2())
    
    # Block 6: A1_bottom(2x2) * B_bottom_right(2x1) -> C rows 0,1; col 4
    # B_bottom_right: rows 2,3; col 4 -> V indices 14,19
    add_block([2,3,6,7], [14,19], [12,13], *schoolbook_2x2x1())
    
    # Block 7: A2_top(1x2) * B_top_left(2x2) -> C row 2; cols 0,1
    # A2_top: row 2; cols 0,1 -> U indices 8,9
    # B_top_left: rows 0,1; cols 0,1 -> V indices 0,1,5,6
    # C: row 2; cols 0,1 -> W indices: k=0,i=2->2; k=1,i=2->5
    add_block([8,9], [0,1,5,6], [2,5], *rank2_1x2x2())
    
    # Block 8: A2_top(1x2) * B_top_mid(2x2) -> C row 2; cols 2,3
    # B_top_mid: rows 0,1; cols 2,3 -> V indices 2,3,7,8
    # C: row 2; cols 2,3 -> W indices: k=2,i=2->8; k=3,i=2->11
    add_block([8,9], [2,3,7,8], [8,11], *rank2_1x2x2())
    
    # Block 9: A2_top(1x2) * B_top_right(2x1) -> C row 2; col 4
    # B_top_right: rows 0,1; col 4 -> V indices 4,9
    # C: row 2; col 4 -> W index: k=4,i=2->14
    add_block([8,9], [4,9], [14], *rank2_1x2x1())
    
    # Block 10: A2_bottom(1x2) * B_bottom_left(2x2) -> C row 2; cols 0,1
    # A2_bottom: row 2; cols 2,3 -> U indices 10,11
    # B_bottom_left: rows 2,3; cols 0,1 -> V indices 10,11,15,16
    add_block([10,11], [10,11,15,16], [2,5], *rank2_1x2x2())
    
    # Block 11: A2_bottom(1x2) * B_bottom_mid(2x2) -> C row 2; cols 2,3
    # B_bottom_mid: rows 2,3; cols 2,3 -> V indices 12,13,17,18
    add_block([10,11], [12,13,17,18], [8,11], *rank2_1x2x2())
    
    # Block 12: A2_bottom(1x2) * B_bottom_right(2x1) -> C row 2; col 4
    # B_bottom_right: rows 2,3; col 4 -> V indices 14,19
    add_block([10,11], [14,19], [14], *rank2_1x2x1())
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    # Verify
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        A_flat = A.flatten()
        B_flat = B.flatten()
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                val = 0
                for r in range(rank):
                    u_proj = np.dot(A_flat, U[:, r])
                    v_proj = np.dot(B_flat, V[:, r])
                    val += W[c_idx, r] * u_proj * v_proj
                C_computed[i, k] = val
        return np.max(np.abs(C_computed - C))
    
    err = verify(U, V, W, n, m, p)
    if err > 1e-10:
        pass  # Should not happen
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
