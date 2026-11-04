"""Matrix multiplication 3x3x4 rank minimization."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 4]

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _verify_exact(U, V, W, n, m, p):
    """
    Exact verification using Python integers to avoid floating point issues.
    """
    rank = len(U[0])
    
    T = [[[0 for _ in range(p)] for _ in range(m)] for _ in range(n)]
    
    for r in range(rank):
        u_col = [U[i * m + j][r] for i in range(n) for j in range(m)]
        v_col = [V[j * p + k][r] for j in range(m) for k in range(p)]
        w_col = [W[k * n + i][r] for k in range(p) for i in range(n)]
        
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0:
                        continue
                    w_val = w_col[k * n + i]
                    T[i][j][k] += u_val * v_val * w_val
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i][j][k] != 1:
                    return False
    return True

def _sa_search(rank, n, m, p, deadline, max_restarts=100):
    """
    Simulated Annealing search for integer decomposition.
    Uses integer arithmetic to avoid floating point issues.
    """
    # Shapes: U[n, m, rank], V[m, p, rank], W[p, n, rank]
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize random integers in a small range
        U = np.random.randint(-3, 4, size=(n, m, rank), dtype=np.int64)
        V = np.random.randint(-3, 4, size=(m, p, rank), dtype=np.int64)
        W = np.random.randint(-3, 4, size=(p, n, rank), dtype=np.int64)
        
        # Compute initial T (integer)
        T = np.zeros((n, m, p), dtype=np.int64)
        for r in range(rank):
            u_r = U[:, :, r]
            v_r = V[:, :, r]
            w_r = W[:, :, r]
            for i in range(n):
                for j in range(m):
                    u_val = u_r[i, j]
                    if u_val == 0: continue
                    contrib = u_val * (v_r[j, :] * w_r[:, i])
                    T[i, j, :] += contrib
                    
        # Energy: sum of squared differences from target (1 if in support, 0 else)
        # Target T_target[i,j,k] is 1 for all i,j,k? No, matrix multiplication tensor has 1s at (i,j,k) 
        # where the term A[i,j]*B[j,k] contributes to C[i,k]. 
        # The tensor T_mmm has entries T[i,j,k] = 1.
        # Wait, the decomposition is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
        # For matrix multiplication, the tensor is 1 for all i,j,k? 
        # Yes, the trilinear form is sum_{i,j,k} A[i,j] B[j,k] C[k,i]. 
        # The coefficient of A[i,j]B[j,k]C[k,i] is 1.
        # So the target tensor is all 1s? 
        # Let's check the schoolbook.
        # U[i*m+j][col]=1, V[j*p+k][col]=1, W[k*n+i][col]=1.
        # Sum over col: U[i,j,r]V[j,k,r]W[k,i,r]. 
        # Only one term is non-zero for each (i,j,k) in schoolbook. It is 1.
        # So yes, target is 1 for all i,j,k.
        
        target = np.ones((n, m, p), dtype=np.int64)
        energy = np.sum((T - target)**2)
        
        current_energy = energy
        temp = 10.0
        cooling = 0.95
        
        steps = 2000
        for step in range(steps):
            if time.time() > deadline:
                return None
            
            # Pick a random move
            r = random.randint(0, rank - 1)
            mat = random.randint(0, 2) # 0:U, 1:V, 2:W
            
            if mat == 0:
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                old_val = U[i, j, r]
                delta = random.choice([-1, 1])
                new_val = old_val + delta
                if abs(new_val) > 10:
                    continue
                change_vec = delta * (V[j, :, r] * W[:, i, r])
                old_T_slice = T[i, j, :]
                new_T_slice = old_T_slice + change_vec
                
                old_e_slice = np.sum((old_T_slice - target[i, j, :])**2)
                new_e_slice = np.sum((new_T_slice - target[i, j, :])**2)
                delta_e = new_e_slice - old_e_slice
                
                if delta_e < 0 or (delta_e > 0 and random.random() < np.exp(-delta_e / temp)):
                    U[i, j, r] = new_val
                    T[i, j, :] = new_T_slice
                    current_energy += delta_e
            elif mat == 1:
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                old_val = V[j, k, r]
                delta = random.choice([-1, 1])
                new_val = old_val + delta
                if abs(new_val) > 10:
                    continue
                change_vec = delta * (U[:, j, r] * W[k, :, r])
                old_T_slice = T[:, j, k]
                new_T_slice = old_T_slice + change_vec
                
                old_e_slice = np.sum((old_T_slice - target[:, j, k])**2)
                new_e_slice = np.sum((new_T_slice - target[:, j, k])**2)
                delta_e = new_e_slice - old_e_slice
                
                if delta_e < 0 or (delta_e > 0 and random.random() < np.exp(-delta_e / temp)):
                    V[j, k, r] = new_val
                    T[:, j, k] = new_T_slice
                    current_energy += delta_e
            else:
                k = random.randint(0, p - 1)
                i = random.randint(0, n - 1)
                old_val = W[k, i, r]
                delta = random.choice([-1, 1])
                new_val = old_val + delta
                if abs(new_val) > 10:
                    continue
                change_vec = delta * (U[i, :, r] * V[:, k, r])
                old_T_slice = T[i, :, k]
                new_T_slice = old_T_slice + change_vec
                
                old_e_slice = np.sum((old_T_slice - target[i, :, k])**2)
                new_e_slice = np.sum((new_T_slice - target[i, :, k])**2)
                delta_e = new_e_slice - old_e_slice
                
                if delta_e < 0 or (delta_e > 0 and random.random() < np.exp(-delta_e / temp)):
                    W[k, i, r] = new_val
                    T[i, :, k] = new_T_slice
                    current_energy += delta_e
            
            temp *= cooling
            
            if current_energy == 0:
                # Found solution!
                U_out = [[0]*rank for _ in range(n*m)]
                V_out = [[0]*rank for _ in range(m*p)]
                W_out = [[0]*rank for _ in range(p*n)]
                
                for i in range(n):
                    for j in range(m):
                        for r in range(rank):
                            U_out[i*m+j][r] = int(U[i, j, r])
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            V_out[j*p+k][r] = int(V[j, k, r])
                for k in range(p):
                    for i in range(n):
                        for r in range(rank):
                            W_out[k*n+i][r] = int(W[k, i, r])
                
                # Verify
                if _verify_exact(U_out, V_out, W_out, n, m, p):
                    return U_out, V_out, W_out
                else:
                    pass
                    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    
    U_sb, V_sb, W_sb = _schoolbook()
    
    # Try SA for lower ranks
    # Start from a rank slightly lower than schoolbook to find improvements
    # Schoolbook is 36.
    # We try to find a valid decomposition with rank < 36.
    # If SA fails (returns None), we fall back to schoolbook.
    
    # We iterate ranks in increasing order of likelihood? Or decreasing rank (harder)?
    # Lower rank is better score. So we try high rank first (easier to find) to ensure we get *some* improvement, 
    # then try lower? 
    # Actually, if we find rank 35, we can stop. If we want rank 30, we must search for it.
    # Searching for rank 30 is harder than 35.
    # Let's try 35 first, then 34, etc.
    
    for target_rank in [35, 34, 33, 32, 31, 30, 29, 28, 27]:
        if time.time() > deadline - 10:
            break
            
        result = _sa_search(target_rank, n, m, p, deadline - 5)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    return {"U": U_sb, "V": V_sb, "W": W_sb}
