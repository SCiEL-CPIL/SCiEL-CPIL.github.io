import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 130
    
    def find_2x2x2_rank7():
        """Find a rank-7 decomposition of the 2x2x2 all-ones tensor."""
        # Use Strassen-style rank-7 construction for 2x2x2 all-ones tensor
        # The 2x2x2 multiplication tensor has rank 7 (Strassen)
        # For all-ones tensor, we need U[i,r]*V[j,r]*W[k,r] = 1 for all i,j,k
        # This is equivalent to finding rank-7 decomposition of the all-ones 2x2x2 tensor
        # 
        # Simple approach: use the fact that for all-ones, we can use
        # specific rational/half-integer values
        #
        # Actually, let's use gradient descent with better initialization
        np.random.seed(42)
        U = np.random.randn(2, 7) * 0.5
        V = np.random.randn(2, 7) * 0.5
        W = np.random.randn(2, 7) * 0.5
        
        lr = 0.05
        for it in range(20000):
            T = np.zeros((2,2,2))
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        T[i,j,k] = np.sum(U[i,:]*V[j,:]*W[k,:])
            
            residual = T - 1.0
            loss = np.sum(residual**2)
            if loss < 1e-14:
                break
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        dU[i,:] += 2*residual[i,j,k]*V[j,:]*W[k,:]
                        dV[j,:] += 2*residual[i,j,k]*U[i,:]*W[k,:]
                        dW[k,:] += 2*residual[i,j,k]*U[i,:]*V[j,:]
            
            U -= lr*dU
            V -= lr*dV
            W -= lr*dW
        
        return U, V, W
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] for r in range(rank))
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    U2, V2, W2 = find_2x2x2_rank7()
    
    # Partition j into {0,1,2} and {3,4}
    # Partition k into {0,1,2} and {3,4,5}
    
    # Blocks:
    # 1. j∈{0,1,2}, k∈{0,1,2}: 2x3x3, rank 10
    # 2. j∈{0,1,2}, k∈{3,4,5}: 2x3x3, rank 10
    # 3. j∈{3,4}, k∈{0,1,2}: 2x2x3, rank 8
    # 4. j∈{3,4}, k∈{3,4,5}: 2x2x3, rank 8
    
    rank = 10 + 10 + 8 + 8  # 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_2x3x3_block(j_offset, k_offset, U2, V2, W2, col):
        """Add a 2x3x3 block starting at j_offset, k_offset. Returns new col."""
        # 2x2x2 sub-block: j∈{0,1}, k∈{0,1}
        for r in range(7):
            for i in range(2):
                for jp in range(2):
                    U[i*m + (j_offset+jp), col] = U2[i, r]
            for jp in range(2):
                for kp in range(2):
                    V[(j_offset+jp)*p + (k_offset+kp), col] = V2[jp, r]
            for kp in range(2):
                for i in range(2):
                    W[(k_offset+kp)*n + i, col] = W2[kp, r]
            col += 1
        
        # 2x1x2 sub-block: j=2, k∈{0,1}
        for i in range(2):
            U[i*m + (j_offset+2), col] = 1.0
        for kp in range(2):
            V[(j_offset+2)*p + (k_offset+kp), col] = 1.0
        for kp in range(2):
            for i in range(2):
                W[(k_offset+kp)*n + i, col] = 1.0
        col += 1
        
        # 2x2x1 sub-block: j∈{0,1}, k=2
        for i in range(2):
            for jp in range(2):
                U[i*m + (j_offset+jp), col] = 1.0
        for jp in range(2):
            V[(j_offset+jp)*p + (k_offset+2), col] = 1.0
        for i in range(2):
            W[(k_offset+2)*n + i, col] = 1.0
        col += 1
        
        # 2x1x1 sub-block: j=2, k=2
        for i in range(2):
            U[i*m + (j_offset+2), col] = 1.0
        V[(j_offset+2)*p + (k_offset+2), col] = 1.0
        for i in range(2):
            W[(k_offset+2)*n + i, col] = 1.0
        col += 1
        
        return col
    
    def add_2x2x3_block(j_offset, k_offset, U2, V2, W2, col):
        """Add a 2x2x3 block starting at j_offset, k_offset. Returns new col."""
        # 2x2x2 sub-block: j∈{0,1}, k∈{0,1}
        for r in range(7):
            for i in range(2):
                for jp in range(2):
                    U[i*m + (j_offset+jp), col] = U2[i, r]
            for jp in range(2):
                for kp in range(2):
                    V[(j_offset+jp)*p + (k_offset+kp), col] = V2[jp, r]
            for kp in range(2):
                for i in range(2):
                    W[(k_offset+kp)*n + i, col] = W2[kp, r]
            col += 1
        
        # 2x2x1 sub-block: j∈{0,1}, k=2
        for i in range(2):
            for jp in range(2):
                U[i*m + (j_offset+jp), col] = 1.0
        for jp in range(2):
            V[(j_offset+jp)*p + (k_offset+2), col] = 1.0
        for i in range(2):
            W[(k_offset+2)*n + i, col] = 1.0
        col += 1
        
        return col
    
    # Block 1: j∈{0,1,2}, k∈{0,1,2}
    col = add_2x3x3_block(0, 0, U2, V2, W2, 0)
    
    # Block 2: j∈{0,1,2}, k∈{3,4,5}
    col = add_2x3x3_block(0, 3, U2, V2, W2, col)
    
    # Block 3: j∈{3,4}, k∈{0,1,2}
    col = add_2x2x3_block(3, 0, U2, V2, W2, col)
    
    # Block 4: j∈{3,4}, k∈{3,4,5}
    col = add_2x2x3_block(3, 3, U2, V2, W2, col)
    
    # Verify
    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}
    
    return {"U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]}
