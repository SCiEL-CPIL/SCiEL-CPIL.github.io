import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            U = np.zeros(n * m)
            V = np.zeros(m * p)
            W = np.zeros(p * n)
            
            for i, idx in enumerate(u_idx):
                U[idx] = U_blk[i, r]
            for i, idx in enumerate(v_idx):
                V[idx] = V_blk[i, r]
            for i, idx in enumerate(w_idx):
                W[idx] = W_blk[i, r]
                
            U_parts.append(U)
            V_parts.append(V)
            W_parts.append(W)

    # Strassen 2x2x2: rank 7
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Schoolbook 2x2x1: rank 4
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Schoolbook 1x2x2: rank 4
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Schoolbook 1x2x1: rank 2
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # U indices: A is 3x4, flat index i*m+j where i=0..2, j=0..3
    # V indices: B is 4x5, flat index j*p+k where j=0..3, k=0..4
    # W indices: C is 5x3, flat index k*n+i where k=0..4, i=0..2

    # === 2x4x5 block (A rows 0-1) ===
    # Split B columns into 2+2+1
    
    # Cols 0-1: 2x4x2, split B rows 0:4 as 2+2
    # Block: A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2]
    u_idx = [0, 1, 4, 5]    # A[0,0],A[0,1],A[1,0],A[1,1]
    v_idx = [0, 1, 5, 6]    # B[0,0],B[0,1],B[1,0],B[1,1]
    w_idx = [0, 3, 1, 4]    # C[0,0],C[1,0],C[0,1],C[1,1]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    # Block: A[0:2, 2:4] x B[2:4, 0:2] -> C[0:2, 0:2]
    u_idx = [2, 3, 6, 7]    # A[0,2],A[0,3],A[1,2],A[1,3]
    v_idx = [10, 11, 15, 16]  # B[2,0],B[2,1],B[3,0],B[3,1]
    w_idx = [0, 3, 1, 4]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    # Cols 2-3: 2x4x2, split B rows 0:4 as 2+2
    # Block: A[0:2, 0:2] x B[0:2, 2:4] -> C[2:4, 0:2]
    u_idx = [0, 1, 4, 5]
    v_idx = [2, 3, 7, 8]    # B[0,2],B[0,3],B[1,2],B[1,3]
    w_idx = [6, 9, 7, 10]   # C[2,0],C[3,0],C[2,1],C[3,1]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    # Block: A[0:2, 2:4] x B[2:4, 2:4] -> C[2:4, 0:2]
    u_idx = [2, 3, 6, 7]
    v_idx = [12, 13, 17, 18]  # B[2,2],B[2,3],B[3,2],B[3,3]
    w_idx = [6, 9, 7, 10]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    # Col 4: 2x4x1, split B rows 0:4 as 2+2
    # Block: A[0:2, 0:2] x B[0:2, 4] -> C[4, 0:2]
    u_idx = [0, 1, 4, 5]
    v_idx = [4, 9]          # B[0,4],B[1,4]
    w_idx = [12, 13]        # C[4,0],C[4,1]
    add_block(u_idx, v_idx, w_idx, *schoolbook_2x2x1())
    
    # Block: A[0:2, 2:4] x B[2:4, 4] -> C[4, 0:2]
    u_idx = [2, 3, 6, 7]
    v_idx = [14, 19]        # B[2,4],B[3,4]
    w_idx = [12, 13]
    add_block(u_idx, v_idx, w_idx, *schoolbook_2x2x1())
    
    # === 1x4x5 block (A row 2) ===
    # Split B rows 0:4 as 2+2
    
    # B rows 0:2: 1x2x5, split B cols 0:5 as 2+2+1
    # Block: A[2, 0:2] x B[0:2, 0:2] -> C[0:2, 2]
    u_idx = [8, 9]          # A[2,0],A[2,1]
    v_idx = [0, 1, 5, 6]    # B[0,0],B[0,1],B[1,0],B[1,1]
    w_idx = [2, 5]          # C[0,2],C[1,2]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    # Block: A[2, 0:2] x B[0:2, 2:4] -> C[2:4, 2]
    u_idx = [8, 9]
    v_idx = [2, 3, 7, 8]    # B[0,2],B[0,3],B[1,2],B[1,3]
    w_idx = [8, 11]         # C[2,2],C[3,2]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    # Block: A[2, 0:2] x B[0:2, 4] -> C[4, 2]
    u_idx = [8, 9]
    v_idx = [4, 9]          # B[0,4],B[1,4]
    w_idx = [14]            # C[4,2]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x1())
    
    # B rows 2:4: 1x2x5, split B cols 0:5 as 2+2+1
    # Block: A[2, 2:4] x B[2:4, 0:2] -> C[0:2, 2]
    u_idx = [10, 11]        # A[2,2],A[2,3]
    v_idx = [10, 11, 15, 16]  # B[2,0],B[2,1],B[3,0],B[3,1]
    w_idx = [2, 5]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    # Block: A[2, 2:4] x B[2:4, 2:4] -> C[2:4, 2]
    u_idx = [10, 11]
    v_idx = [12, 13, 17, 18]  # B[2,2],B[2,3],B[3,2],B[3,3]
    w_idx = [8, 11]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    # Block: A[2, 2:4] x B[2:4, 4] -> C[4, 2]
    u_idx = [10, 11]
    v_idx = [14, 19]        # B[2,4],B[3,4]
    w_idx = [14]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x1())
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    rank = U.shape[1]
    
    # Verify correctness with random test
    if time.time() - start_time < 100:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Compute via tensor decomposition
        C_computed = np.zeros((p, n))
        for r in range(rank):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            
            # u_vec is flat (n*m), reshape to (n, m)
            # v_vec is flat (m*p), reshape to (m, p)
            # w_vec is flat (p*n), reshape to (p, n)
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            
            # Contribution: sum over r of (A^T u_r) * (v_r^T B) * w_r^T
            # Actually: C = sum_r (A * u_r) * (v_r * B) * w_r^T
            # Where u_r is (n,m), v_r is (m,p), w_r is (p,n)
            
            # The tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * ... 
            # Wait, let me think about this more carefully.
            
            # The multiplication tensor T[i,j,k] = delta (meaning C[i,k] = sum_j A[i,j] B[j,k])
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]  (using flat indices)
            # So C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Let me compute differently:
            # For each r, compute: (A @ u_r_col) where u_r is extracted from U
            # Actually the standard way:
            # C = sum_r (A * U_r) * (V_r * B) * W_r^T
            # where U_r is (n,m), V_r is (m,p), W_r is (p,n)
            # A * U_r means element-wise? No...
            
            # Let me just verify by direct tensor contraction
            pass
        
        # Simpler verification: check that the tensor is correct
        # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And T[i,j,k] = 1 if this contributes to C[i,k] via A[i,j]*B[j,k]
        # Actually T[i,j,k] = delta_{contribution} means the tensor for matrix mult is:
        # T[i,j,k] = 1 for all i,j,k (it's the identity tensor for the multiplication)
        # No wait, the multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
        # because C[i,k] = sum_j A[i,j] B[j,k]
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # Let's verify this
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    max_err = max(max_err, abs(val - 1))
        
        if max_err > 1e-10:
            # Something wrong, fall back to schoolbook
            rank_school = n * m * p
            U_school = [[0] * rank_school for _ in range(n * m)]
            V_school = [[0] * rank_school for _ in range(m * p)]
            W_school = [[0] * rank_school for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i * m + j][col] = 1
                        V_school[j * p + k][col] = 1
                        W_school[k * n + i][col] = 1
                        col += 1
            return {"U": U_school, "V": V_school, "W": W_school}
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
