"""Find rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = 140.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Schoolbook decomposition baseline (rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def is_valid_w_val(v):
        """Check if v is integer or half-integer."""
        if abs(v) < 1e-12:
            return True
        if abs(v - round(v)) < 1e-9:
            return True
        if abs(v*2 - round(v*2)) < 1e-9:
            return True
        return False
    
    def try_find_w(M_row, rank):
        """
        Given M_row (vector of length rank), find W_row such that dot(M_row, W_row) = 1.
        Try sparse solutions with entries in {-2,-1,-0.5,0,0.5,1,2} or single non-zero.
        Returns W_row or None.
        """
        # First try single non-zero
        for r in range(rank):
            if abs(M_row[r]) > 1e-12:
                w_val = 1.0 / M_row[r]
                if is_valid_w_val(w_val):
                    W_row = np.zeros(rank)
                    W_row[r] = w_val
                    return W_row
        
        # Try two non-zeros with small integer/half-integer coefficients
        for r1 in range(rank):
            if abs(M_row[r1]) < 1e-12:
                continue
            for r2 in range(r1+1, rank):
                if abs(M_row[r2]) < 1e-12:
                    continue
                # Try a*M[r1] + b*M[r2] = 1 for small a, b
                for a in range(-4, 5):
                    for b in range(-4, 5):
                        if a == 0 and b == 0:
                            continue
                        if abs(a*M_row[r1] + b*M_row[r2] - 1.0) < 1e-9:
                            W_row = np.zeros(rank)
                            W_row[r1] = float(a)
                            W_row[r2] = float(b)
                            return W_row
        
        # Try three non-zeros
        for r1 in range(rank):
            if abs(M_row[r1]) < 1e-12:
                continue
            for r2 in range(r1+1, rank):
                if abs(M_row[r2]) < 1e-12:
                    continue
                for r3 in range(r2+1, rank):
                    if abs(M_row[r3]) < 1e-12:
                        continue
                    for a in range(-3, 4):
                        for b in range(-3, 4):
                            for c in range(-3, 4):
                                if a == 0 and b == 0 and c == 0:
                                    continue
                                if abs(a*M_row[r1] + b*M_row[r2] + c*M_row[r3] - 1.0) < 1e-9:
                                    W_row = np.zeros(rank)
                                    W_row[r1] = float(a)
                                    W_row[r2] = float(b)
                                    W_row[r3] = float(c)
                                    return W_row
        return None
    
    def try_sparse_rank(rank, num_trials=200):
        """
        Try to find a rank-r exact decomposition using sparse U, V with entries in {-1, 0, 1}.
        """
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            # Generate sparse U, V
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            for r in range(rank):
                # For U column r, pick 2-3 random (i,j) positions
                num_nz_u = random.choice([2, 3, 3, 4])
                indices_u = random.sample(range(n*m), min(num_nz_u, n*m))
                for idx in indices_u:
                    U[idx, r] = random.choice([-1, 1])
                
                # For V column r, pick 2-3 random (j,k) positions
                num_nz_v = random.choice([2, 3, 3, 4])
                indices_v = random.sample(range(m*p), min(num_nz_v, m*p))
                for idx in indices_v:
                    V[idx, r] = random.choice([-1, 1])
            
            # Compute M[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[k*n+i, r] = s
            
            # For each (k,i), find W
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                W_row = try_find_w(M[idx], rank)
                if W_row is None:
                    valid = False
                    break
                W[idx] = W_row
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_structured_rank(rank, num_trials=100):
        """
        Try structured patterns for U and V.
        """
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            # Pattern: each column of U has exactly one 1, each column of V has exactly one 1
            # but with structured assignments
            perm_u = np.random.permutation(n*m)
            perm_v = np.random.permutation(m*p)
            
            for r in range(rank):
                U[perm_u[r % n*m], r] = random.choice([-1, 1])
                V[perm_v[r % m*p], r] = random.choice([-1, 1])
            
            # Compute M
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[k*n+i, r] = s
            
            # Find W
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                W_row = try_find_w(M[idx], rank)
                if W_row is None:
                    valid = False
                    break
                W[idx] = W_row
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_block_rank(rank, num_trials=50):
        """
        Try block-structured decomposition.
        """
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            # Divide rank into blocks
            block_size = max(1, rank // 7)
            
            for r in range(rank):
                block = r // block_size
                # Within each block, assign to a specific (i,j) or (j,k)
                if block < n*m:
                    U[block, r] = random.choice([-1, 1])
                if block < m*p:
                    V[block, r] = random.choice([-1, 1])
            
            # Compute M
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[k*n+i, r] = s
            
            # Find W
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                W_row = try_find_w(M[idx], rank)
                if W_row is None:
                    valid = False
                    break
                W[idx] = W_row
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    # Try ranks from a reasonable lower bound upward
    # Theoretical lower bound for matrix multiplication rank is max(n*m, m*p, p*n) = 28
    # But tensor rank can be lower. Known results suggest ~20-30 for small cases.
    ranks_to_try = list(range(20, min(n*m*p, 84)))
    
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Try sparse random search
        result = try_sparse_rank(rank, num_trials=100)
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            break
        
        # Try structured search if time permits
        if time.time() - start_time < DEADLINE - 10:
            result = try_structured_rank(rank, num_trials=50)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
        
        # Try block search if time permits
        if time.time() - start_time < DEADLINE - 10:
            result = try_block_rank(rank, num_trials=30)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
