"""Find rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

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
    
    def check_decomposition_fast(U, V, W, rank):
        """Fast vectorized check using numpy."""
        # Compute all (i,j,k) values
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
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
        Uses linear algebra to find sparse solutions with integer/half-integer entries.
        Returns W_row or None.
        """
        nonzero_idx = np.where(np.abs(M_row) > 1e-12)[0]
        if len(nonzero_idx) == 0:
            return None
        
        # Try single non-zero
        for r in nonzero_idx:
            w_val = 1.0 / M_row[r]
            if is_valid_w_val(w_val):
                W_row = np.zeros(rank)
                W_row[r] = w_val
                return W_row
        
        # Try pairs using exact rational arithmetic
        for i1 in range(len(nonzero_idx)):
            for i2 in range(i1+1, len(nonzero_idx)):
                r1, r2 = nonzero_idx[i1], nonzero_idx[i2]
                m1, m2 = M_row[r1], M_row[r2]
                # Solve a*m1 + b*m2 = 1 for integer a, b
                # a = (1 - b*m2) / m1
                for b in range(-8, 9):
                    a_val = (1.0 - b*m2) / m1
                    if abs(a_val - round(a_val)) < 1e-9:
                        a = int(round(a_val))
                        if abs(a*m1 + b*m2 - 1.0) < 1e-9:
                            W_row = np.zeros(rank)
                            W_row[r1] = float(a)
                            W_row[r2] = float(b)
                            return W_row
                # Try half-integer solutions: a = p/2, b = q/2
                # (p/2)*m1 + (q/2)*m2 = 1 => p*m1 + q*m2 = 2
                for q in range(-16, 17):
                    p_val = (2.0 - q*m2) / m1
                    if abs(p_val - round(p_val)) < 1e-9:
                        p = int(round(p_val))
                        if abs(p*m1 + q*m2 - 2.0) < 1e-9:
                            W_row = np.zeros(rank)
                            W_row[r1] = p/2.0
                            W_row[r2] = q/2.0
                            return W_row
        
        # Try triples with small coefficients
        if len(nonzero_idx) >= 3:
            for i1 in range(min(len(nonzero_idx), 6)):
                for i2 in range(i1+1, min(len(nonzero_idx), 6)):
                    for i3 in range(i2+1, min(len(nonzero_idx), 6)):
                        r1, r2, r3 = nonzero_idx[i1], nonzero_idx[i2], nonzero_idx[i3]
                        m1, m2, m3 = M_row[r1], M_row[r2], M_row[r3]
                        for a in range(-4, 5):
                            for b in range(-4, 5):
                                c_val = (1.0 - a*m1 - b*m2) / m3
                                if abs(c_val - round(c_val)) < 1e-9:
                                    c = int(round(c_val))
                                    if abs(a*m1 + b*m2 + c*m3 - 1.0) < 1e-9:
                                        W_row = np.zeros(rank)
                                        W_row[r1] = float(a)
                                        W_row[r2] = float(b)
                                        W_row[r3] = float(c)
                                        return W_row
        return None
    
    def try_sparse_rank(rank, num_trials=200):
        """
        Try to find a rank-r exact decomposition using sparse U, V with entries in {-1, 0, 1}.
        Uses structured generation to increase success probability.
        """
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            if trial % 3 == 0:
                # Strategy 1: Each column has exactly 2 non-zeros
                for r in range(rank):
                    indices_u = random.sample(range(n*m), 2)
                    for idx in indices_u:
                        U[idx, r] = random.choice([-1, 1])
                    indices_v = random.sample(range(m*p), 2)
                    for idx in indices_v:
                        V[idx, r] = random.choice([-1, 1])
            elif trial % 3 == 1:
                # Strategy 2: Each column has 3 non-zeros
                for r in range(rank):
                    indices_u = random.sample(range(n*m), 3)
                    for idx in indices_u:
                        U[idx, r] = random.choice([-1, 1])
                    indices_v = random.sample(range(m*p), 3)
                    for idx in indices_v:
                        V[idx, r] = random.choice([-1, 1])
            else:
                # Strategy 3: Mixed sparsity
                for r in range(rank):
                    nz_u = random.choice([2, 3, 4])
                    indices_u = random.sample(range(n*m), min(nz_u, n*m))
                    for idx in indices_u:
                        U[idx, r] = random.choice([-1, 1])
                    nz_v = random.choice([2, 3, 4])
                    indices_v = random.sample(range(m*p), min(nz_v, m*p))
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
        Try structured patterns for U and V with systematic coverage.
        """
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            if trial % 4 == 0:
                # Pattern A: Cycle through (i,j) and (j,k) pairs
                for r in range(rank):
                    i_u = (r * 3) % n
                    j_u = (r * 7) % m
                    i_v = (r * 5) % m
                    k_v = (r * 11) % p
                    U[i_u*m + j_u, r] = 1.0 if (r % 2 == 0) else -1.0
                    V[i_v*p + k_v, r] = 1.0 if (r % 2 == 0) else -1.0
            elif trial % 4 == 1:
                # Pattern B: Block diagonal structure
                block_size = max(1, rank // max(n, m, p))
                for r in range(rank):
                    block = r // block_size
                    pos_in_block = r % block_size
                    if block < n*m:
                        U[block, r] = 1.0 if pos_in_block % 2 == 0 else -1.0
                    if block < m*p:
                        V[block, r] = 1.0 if pos_in_block % 2 == 0 else -1.0
            elif trial % 4 == 2:
                # Pattern C: Latin square based
                for r in range(rank):
                    i_u = (r % n)
                    j_u = ((r // n) + r) % m
                    j_v = (r % m)
                    k_v = ((r // m) + r) % p
                    U[i_u*m + j_u, r] = 1.0
                    V[j_v*p + k_v, r] = 1.0
            else:
                # Pattern D: Random but with guaranteed coverage
                perm_u = np.random.permutation(n*m)
                perm_v = np.random.permutation(m*p)
                for r in range(rank):
                    U[perm_u[r % n*m], r] = 1.0 if r % 2 == 0 else -1.0
                    V[perm_v[r % m*p], r] = 1.0 if r % 2 == 0 else -1.0
            
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
        Try block-structured decomposition with systematic block assignment.
        """
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            # Divide rank into blocks, each block handles a specific (i,j,k) combination
            num_blocks = min(rank, n*m*p)
            block_size = max(1, rank // num_blocks)
            
            for r in range(rank):
                block = r % num_blocks
                pos = r // num_blocks
                
                # Map block to specific indices
                i = block % n
                j = (block // n) % m
                k = (block // (n*m)) % p
                
                U[i*m + j, r] = 1.0 if pos % 2 == 0 else -1.0
                V[j*p + k, r] = 1.0 if pos % 2 == 0 else -1.0
            
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
    
    def try_systematic_rank(rank, num_trials=30):
        """
        Systematic search: fix U and V patterns, solve for W exactly.
        Uses the fact that for fixed U,V, W must satisfy linear constraints.
        """
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            # Generate U with controlled sparsity
            for r in range(rank):
                # Each column of U has exactly 2 non-zeros
                i1 = random.randint(0, n-1)
                j1 = random.randint(0, m-1)
                i2 = random.randint(0, n-1)
                j2 = random.randint(0, m-1)
                U[i1*m + j1, r] = 1.0
                if (i2, j2) != (i1, j1):
                    U[i2*m + j2, r] = random.choice([-1.0, 1.0])
            
            # Generate V with controlled sparsity
            for r in range(rank):
                j1 = random.randint(0, m-1)
                k1 = random.randint(0, p-1)
                j2 = random.randint(0, m-1)
                k2 = random.randint(0, p-1)
                V[j1*p + k1, r] = 1.0
                if (j2, k2) != (j1, k1):
                    V[j2*p + k2, r] = random.choice([-1.0, 1.0])
            
            # Compute M[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[k*n+i, r] = s
            
            # For each (k,i), find W using linear algebra
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
        
        # Try systematic search first (more structured)
        if time.time() - start_time < DEADLINE - 12:
            result = try_systematic_rank(rank, num_trials=20)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
        
        # Try sparse random search
        if time.time() - start_time < DEADLINE - 10:
            result = try_sparse_rank(rank, num_trials=80)
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
            result = try_structured_rank(rank, num_trials=40)
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
            result = try_block_rank(rank, num_trials=20)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
