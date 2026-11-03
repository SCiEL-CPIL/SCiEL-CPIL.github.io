import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 115
    
    # Build Strassen rank-7 decomposition for 2x2x2
    # Strassen: C = A*B where A,B,C are 2x2
    # Uses 7 products instead of 8
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For the full 3x5x6 tensor, we'll use a block decomposition
    # Split: rows [2,1], inner cols [2,2,1], outer cols [2,2,2]
    
    # For each block combination, we need a decomposition
    # Block (ri, mi, pi) where ri=2 or 1, mi=2 or 1, pi=2 or 1
    
    # Let's build the decomposition by enumerating all blocks
    
    # First, let's create the Strassen factors for 2x2x2
    # U_strassen[i,j,c], V_strassen[j,k,c], W_strassen[k,i,c] for i,j,k in {0,1}, c in 0..6
    
    def make_strassen_222():
        """Return U, V, W for 2x2x2 Strassen decomposition, each shape (4, 7)"""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Index mapping: i*2+j for U, j*2+k for V, k*2+i for W
        
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        # A indices: A11=(0,0)=idx0, A22=(1,1)=idx3
        # B indices: B11=(0,0)=idx0, B22=(1,1)=idx3
        U[0, 0] = 1; U[3, 0] = 1  # A11, A22
        V[0, 0] = 1; V[3, 0] = 1  # B11, B22
        W[0, 0] = 1  # C11 (k=0,i=0 -> idx 0)
        W[3, 0] = 1  # C22 (k=1,i=1 -> idx 3)
        
        # M2 = (A21+A22)B11 -> C21 += M2
        # A21=(1,0)=idx2, A22=(1,1)=idx3, B11=(0,0)=idx0
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C21 (k=1,i=0 -> idx 2)
        
        # M3 = A11(B21-B22) -> C12 += M3, C22 += M3
        # A11=(0,0)=idx0, B21=(1,0)=idx2, B22=(1,1)=idx3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # C12 (k=0,i=1 -> idx 1)
        W[3, 2] = 1  # C22
        
        # M4 = A22(B12-B11) -> C11 += M4, C21 += M4
        # A22=(1,1)=idx3, B12=(0,1)=idx1, B11=(0,0)=idx0
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1  # C11
        W[2, 3] = 1  # C21
        
        # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
        # A11=(0,0)=idx0, A12=(0,1)=idx1, B22=(1,1)=idx3
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1  # C11
        W[1, 4] = 1   # C12
        
        # M6 = (A21-A11)(B11+B12) -> C22 += M6
        # A21=(1,0)=idx2, A11=(0,0)=idx0, B11=(0,0)=idx0, B12=(0,1)=idx1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C22
        
        # M7 = (A12-A22)(B21+B22) -> C11 += M7
        # A12=(0,1)=idx1, A22=(1,1)=idx3, B21=(1,0)=idx2, B22=(1,1)=idx3
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # C11
        
        return U, V, W
    
    def make_schoolbook_222():
        """Return U, V, W for 2x2x2 schoolbook decomposition, each shape (4, 8)"""
        U = np.zeros((4, 8))
        V = np.zeros((4, 8))
        W = np.zeros((4, 8))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j, col] = 1
                    V[j*2+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return U, V, W
    
    # For 1x2x2: A is 1x2, B is 2x2, C is 1x2
    # Schoolbook: rank = 1*2*2 = 4
    def make_schoolbook_122():
        """U shape (2,4), V shape (4,4), W shape (2,4) for 1x2x2"""
        U = np.zeros((2, 4))  # 1*2 = 2
        V = np.zeros((4, 4))  # 2*2 = 4
        W = np.zeros((2, 4))  # 2*1 = 2
        col = 0
        for i in range(1):
            for j in range(2):
                for k in range(2):
                    U[i*2+j, col] = 1
                    V[j*2+k, col] = 1
                    W[k*1+i, col] = 1
                    col += 1
        return U, V, W
    
    # For 2x1x2: A is 2x1, B is 1x2, C is 2x2
    def make_schoolbook_212():
        """U shape (2,4), V shape (2,4), W shape (4,4) for 2x1x2"""
        U = np.zeros((2, 4))  # 2*1 = 2
        V = np.zeros((2, 4))  # 1*2 = 2
        W = np.zeros((4, 4))  # 2*2 = 4
        col = 0
        for i in range(2):
            for j in range(1):
                for k in range(2):
                    U[i*1+j, col] = 1
                    V[j*2+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return U, V, W
    
    # For 2x2x1: A is 2x2, B is 2x1, C is 2x1
    def make_schoolbook_221():
        """U shape (4,4), V shape (2,4), W shape (2,4) for 2x2x1"""
        U = np.zeros((4, 4))  # 2*2 = 4
        V = np.zeros((2, 4))  # 2*1 = 2
        W = np.zeros((2, 4))  # 1*2 = 2
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    U[i*2+j, col] = 1
                    V[j*1+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return U, V, W
    
    # For 1x1x2: rank 2
    def make_schoolbook_112():
        U = np.zeros((1, 2))
        V = np.zeros((2, 2))
        W = np.zeros((2, 2))
        for k in range(2):
            U[0, k] = 1
            V[k, k] = 1
            W[k, k] = 1
        return U, V, W
    
    # For 1x2x1: rank 2
    def make_schoolbook_121():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        for j in range(2):
            U[j, j] = 1
            V[j, j] = 1
            W[0, j] = 1
        return U, V, W
    
    # For 2x1x1: rank 2
    def make_schoolbook_211():
        U = np.zeros((2, 2))
        V = np.zeros((1, 2))
        W = np.zeros((2, 2))
        for i in range(2):
            U[i, i] = 1
            V[0, i] = 1
            W[i, i] = 1
        return U, V, W
    
    # For 1x1x1: rank 1
    def make_schoolbook_111():
        U = np.ones((1, 1))
        V = np.ones((1, 1))
        W = np.ones((1, 1))
        return U, V, W
    
    # Now build the full decomposition
    # Row blocks: [2, 1] (sizes)
    # Inner col blocks: [2, 2, 1]
    # Outer col blocks: [2, 2, 2]
    
    row_blocks = [2, 1]
    inner_blocks = [2, 2, 1]
    outer_blocks = [2, 2, 2]
    
    # Pre-compute block decompositions
    block_decomp = {}
    block_decomp[(2,2,2)] = make_strassen_222()
    block_decomp[(2,2,1)] = make_schoolbook_221()
    block_decomp[(2,1,2)] = make_schoolbook_212()
    block_decomp[(2,1,1)] = make_schoolbook_211()
    block_decomp[(1,2,2)] = make_schoolbook_122()
    block_decomp[(1,2,1)] = make_schoolbook_121()
    block_decomp[(1,1,2)] = make_schoolbook_112()
    block_decomp[(1,1,1)] = make_schoolbook_111()
    
    # Build full U, V, W
    total_rank = 0
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Row offsets
    row_offsets = [0]
    for s in row_blocks[:-1]:
        row_offsets.append(row_offsets[-1] + s)
    
    # Inner col offsets  
    inner_offsets = [0]
    for s in inner_blocks[:-1]:
        inner_offsets.append(inner_offsets[-1] + s)
    
    # Outer col offsets
    outer_offsets = [0]
    for s in outer_blocks[:-1]:
        outer_offsets.append(outer_offsets[-1] + s)
    
    for ri, rs in enumerate(row_blocks):
        for mi, ms in enumerate(inner_blocks):
            for pi, ps in enumerate(outer_blocks):
                key = (rs, ms, ps)
                U_b, V_b, W_b = block_decomp[key]
                rank_b = U_b.shape[1]
                total_rank += rank_b
                
                # Create full-size slices
                U_full = np.zeros((n*m, rank_b))
                V_full = np.zeros((m*p, rank_b))
                W_full = np.zeros((p*n, rank_b))
                
                # Fill U_full for this block
                for local_i in range(rs):
                    for local_j in range(ms):
                        global_i = row_offsets[ri] + local_i
                        global_j = inner_offsets[mi] + local_j
                        U_full[global_i*m + global_j, :] = U_b[local_i*ms + local_j, :]
                
                # Fill V_full for this block
                for local_j in range(ms):
                    for local_k in range(ps):
                        global_j = inner_offsets[mi] + local_j
                        global_k = outer_offsets[pi] + local_k
                        V_full[global_j*p + global_k, :] = V_b[local_j*ps + local_k, :]
                
                # Fill W_full for this block
                for local_k in range(ps):
                    for local_i in range(rs):
                        global_k = outer_offsets[pi] + local_k
                        global_i = row_offsets[ri] + local_i
                        W_full[global_k*n + global_i, :] = W_b[local_k*rs + local_i, :]
                
                U_parts.append(U_full)
                V_parts.append(V_full)
                W_parts.append(W_full)
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    # Verify
    if verify_decomp(n, m, p, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

def verify_decomp(n, m, p, U, V, W):
    rank = U.shape[1]
    np.random.seed(12345)
    A = np.random.randint(-10, 10, (n, m))
    B = np.random.randint(-10, 10, (m, p))
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n).T
        
        X = np.sum(A * u_c, axis=1)
        Y = np.sum(B * v_c, axis=0)
        C_computed += w_c * np.outer(X, Y)
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
