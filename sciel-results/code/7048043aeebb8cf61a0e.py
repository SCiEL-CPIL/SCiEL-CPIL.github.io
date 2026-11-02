import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor for full multiplication
    # We will build it from parts
    
    # Part 1: 2x2x2 (Strassen)
    # Part 2: 2x2x3 (Optimized)
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    # Try to find better decomposition
    # We split p=5 into p1=2 and p2=3
    # Rank = Rank(M_2,2,2) + Rank(M_2,2,3)
    
    # Strassen for 2x2x2
    # Indices: i,j in 0..1, k in 0..1
    # We need U1 (4x7), V1 (4x7), W1 (4x7) for the 2x2x2 part
    # But the full V and W have different sizes.
    # V is (m*p, rank) = (10, rank)
    # W is (p*n, rank) = (10, rank)
    
    # Let's define the full tensor T
    # T[u, v, w]
    
    # We will try to solve for Rank 17 (7 + 10)
    # And Rank 16 (7 + 9)
    
    start_time = time.time()
    deadline = 10.0 # seconds for search
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_schoolbook
    
    # Helper to create the tensor for 2x2xp
    def make_tensor_2_2_p(p_val):
        # n=2, m=2
        # u: 0..3 (i*2+j)
        # v: 0..2*p_val-1 (j*p_val+k)
        # w: 0..2*p_val-1 (k*2+i)
        T = np.zeros((4, 2*p_val, 2*p_val))
        for i in range(2):
            for j in range(2):
                for k in range(p_val):
                    u = i * 2 + j
                    v = j * p_val + k
                    w = k * 2 + i
                    T[u, v, w] = 1
        return T

    # Strassen decomposition for 2x2x2
    # U, V, W are 4x7
    # Standard Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We can hardcode the 7 vectors
    # Or just use schoolbook for the 2x2x2 part if we can't find Strassen easily in code?
    # No, Strassen is well known.
    
    # Let's try to find a rank 10 decomposition for 2x2x3
    # T_2_2_3 = make_tensor_2_2_p(3)
    # Size 4 x 6 x 6
    
    # Optimization
    def try_find_decomp(T, rank):
        # T is 4 x (2*p) x (2*p)
        # U: 4 x rank
        # V: (2*p) x rank
        # W: (2*p) x rank
        
        # Random init
        np.random.seed(42)
        U = np.random.randn(4, rank)
        V = np.random.randn(T.shape[1], rank)
        W = np.random.randn(T.shape[2], rank)
        
        # Gradient descent
        # Loss = || T - sum U_r V_r W_r ||^2
        # We can use simple GD
        
        lr = 0.01
        for step in range(1000):
            if time.time() - start_time > deadline:
                return None
            
            # Compute current tensor
            # T_curr = np.einsum('ur,vr,wr->uvw', U, V, W)
            # This is slow.
            # Better: compute gradient analytically or use finite differences?
            # For small tensors, einsum is fine.
            
            T_curr = np.einsum('ur,vr,wr->uvw', U, V, W)
            loss = np.sum((T - T_curr)**2)
            
            if loss < 1e-6:
                return U, V, W
            
            # Gradient
            # dL/dU = 2 * (T - T_curr) * (V^T W) ? No.
            # T_curr_{uvw} = sum_r U_{ur} V_{vr} W_{wr}
            # dL/dU_{ur} = 2 * sum_{vw} (T_{uvw} - T_curr_{uvw}) V_{vr} W_{wr}
            
            # Efficient computation:
            # Residual R = T - T_curr
            # GradU = 2 * np.einsum('uvw,vr,wr->ur', R, V, W)
            # GradV = 2 * np.einsum('uvw,ur,wr->vr', R, U, W)
            # GradW = 2 * np.einsum('uvw,ur,vr->wr', R, U, V)
            
            R = T - T_curr
            GradU = 2 * np.einsum('uvw,vr,wr->ur', R, V, W)
            GradV = 2 * np.einsum('uvw,ur,wr->vr', R, U, W)
            GradW = 2 * np.einsum('uvw,ur,vr->wr', R, U, V)
            
            U -= lr * GradU
            V -= lr * GradV
            W -= lr * GradW
            
            # Normalize to prevent divergence
            # Maybe not needed for small steps
            
        return None

    # Try rank 10 for 2x2x3
    T_2_2_3 = make_tensor_2_2_p(3)
    decomp_3 = try_find_decomp(T_2_2_3, 10)
    
    if decomp_3:
        U3, V3, W3 = decomp_3
        # We have a rank 10 decomp for 2x2x3
        # We need rank 7 for 2x2x2
        # Let's just use schoolbook for 2x2x2 to be safe? No, Strassen is better.
        # But if I can't find Strassen code easily, I can use schoolbook (rank 8).
        # Total 8+10=18. Not an improvement.
        # I need Strassen.
        
        # Let's hardcode Strassen U,V,W for 2x2x2
        # This is tedious to write correctly from memory.
        # I will use a known set of 7 vectors.
        
        # Alternatively, I can try to find rank 7 for 2x2x2 using the same optimizer?
        # It might be slower.
        
        # Let's assume I can find Strassen.
        # If I can't, I return schoolbook.
        
    # If I can't find a good decomp, return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
