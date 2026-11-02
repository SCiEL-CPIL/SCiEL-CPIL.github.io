import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = 2, 3, 9
    # We decompose into 3 blocks of 2x3x3
    # Rank of full tensor = 3 * Rank(2x3x3)
    
    # We try to find the minimal rank for 2x3x3
    # Schoolbook rank is 18.
    # We try R_block = 12, 14, 16, 18
    
    def solve_block_233(R, time_budget):
        """
        Tries to find U, V, W of shapes (6,R), (9,R), (6,R)
        such that sum_r U[i*3+j, r] V[j*3+k, r] W[k*2+i, r] = 1 for all i,j,k.
        Returns (U, V, W) or None.
        """
        np.random.seed(int(time.time() * 1000) % 10000)
        
        U = np.random.randn(6, R) * 0.5
        V = np.random.randn(9, R) * 0.5
        W = np.random.randn(6, R) * 0.5
        
        # Precompute indices
        indices = []
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    indices.append((i, j, k))
                    
        lr = 0.1
        prev_loss = np.inf
        
        for it in range(10000):
            if time.time() > start_time + time_budget:
                return None
                
            # Compute T and Error
            T = np.zeros((2,3,3))
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        T[i,j,k] = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*2+i, :])
                        
            E = T - 1.0
            loss = np.sum(E**2)
            
            if loss < 1e-8:
                return U, V, W
                
            if loss > prev_loss * 1.1:
                # Increase learning rate or reset? Just break and try again with different seed
                return None
            prev_loss = loss
            
            # Compute Gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        e = E[i,j,k]
                        dU[i*3+j, :] += e * V[j*3+k, :] * W[k*2+i, :]
                        dV[j*3+k, :] += e * U[i*3+j, :] * W[k*2+i, :]
                        dW[k*2+i, :] += e * U[i*3+j, :] * V[j*3+k, :]
                        
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Normalize to prevent divergence
            # This might help
            # U /= np.linalg.norm(U) + 1e-6
            # V /= np.linalg.norm(V) + 1e-6
            # W /= np.linalg.norm(W) + 1e-6
            
        return None

    U_block = None
    V_block = None
    W_block = None
    R_block = 18 # Default to schoolbook
    
    # Try lower ranks
    for R in [12, 14, 16]:
        if time.time() > deadline - 10:
            break
        res = solve_block_233(R, 5) # 5 seconds per attempt
        if res is not None:
            U_block, V_block, W_block = res
            R_block = R
            break
            
    # If not found, use schoolbook R_block = 18
    if U_block is None:
        R_block = 18
        U_block = np.zeros((6, R_block))
        V_block = np.zeros((9, R_block))
        W_block = np.zeros((6, R_block))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U_block[i*3+j, col] = 1
                    V_block[j*3+k, col] = 1
                    W_block[k*2+i, col] = 1
                    col += 1
                    
    # Construct full 2x3x9 decomposition
    R_total = 3 * R_block
    U = np.zeros((6, R_total))
    V = np.zeros((27, R_total))
    W = np.zeros((18, R_total))
    
    for b in range(3):
        base_r = b * R_block
        for c in range(R_block):
            c_full = base_r + c
            U[:, c_full] = U_block[:, c]
            for j in range(3):
                for k in range(3):
                    V[j*9 + 3*b + k, c_full] = V_block[j*3 + k, c]
                    W[(3*b + k)*2 + i, c_full] = W_block[k*2 + i, c] # Note: i is not in loop, W depends on i?
                    
    # Wait, W index is (k*n + i).
    # In the block, W_block[k*2 + i, c].
    # In the full, W_full[(3b+k)*2 + i, c_full].
    # The loop above for W is wrong because i is not iterated in the inner loop structure properly.
    
    # Let's rewrite the expansion carefully.
    U = np.zeros((6, R_total))
    V = np.zeros((27, R_total))
    W = np.zeros((18, R_total))
    
    for b in range(3):
        for c in range(R_block):
            c_full = b * R_block + c
            # U is same for all blocks
            U[:, c_full] = U_block[:, c]
            
            # V and W depend on the block b
            for j in range(3):
                for k in range(3):
                    # V index: j*p + k -> j*9 + (3b+k)
                    V[j*9 + 3*b + k, c_full] = V_block[j*3 + k, c]
                    
            for i in range(2):
                for k in range(3):
                    # W index: k*n + i -> (3b+k)*2 + i
                    W[(3*b + k)*2 + i, c_full] = W_block[k*2 + i, c]
                    
    # Verify
    valid = True
    for i in range(2):
        for j in range(3):
            for k in range(9):
                val = np.sum(U[i*3+j, :] * V[j*9+k, :] * W[k*2+i, :])
                if abs(val - 1.0) > 1e-5:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to full schoolbook 54
        R_total = 54
        U = np.zeros((6, R_total))
        V = np.zeros((27, R_total))
        W = np.zeros((18, R_total))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(9):
                    U[i*3+j, col] = 1
                    V[j*9+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
