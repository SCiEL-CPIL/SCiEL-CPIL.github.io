import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Generate all possible 2x2x2 blocks
    i_pairs = [(i, i+1) for i in range(n-1)]
    j_pairs = [(j, j+1) for j in range(m-1)]
    k_pairs = [(k, k+1) for k in range(p-1)]
    
    all_blocks = []
    for ip in i_pairs:
        for jp in j_pairs:
            for kp in k_pairs:
                triples = set()
                for i in ip:
                    for j in jp:
                        for k in kp:
                            triples.add((i, j, k))
                all_blocks.append((ip, jp, kp, triples))
    
    num_blocks = len(all_blocks)
    
    # Greedy + backtracking search for max disjoint blocks
    def is_disjoint(triple_set_a, triple_set_b):
        return len(triple_set_a & triple_set_b) == 0
    
    def greedy_search(order):
        selected = []
        selected_triples = set()
        for idx in order:
            ip, jp, kp, triples = all_blocks[idx]
            if is_disjoint(selected_triples, triples):
                selected.append(idx)
                selected_triples |= triples
        return selected
    
    import random
    rng = random.Random(42)
    
    best_greedy = []
    for trial in range(500):
        order = list(range(num_blocks))
        rng.shuffle(order)
        result = greedy_search(order)
        if len(result) > len(best_greedy):
            best_greedy = result
        if time.time() > deadline - 10:
            break
    
    # Define Strassen terms for 2x2x2
    # u, v, w are length 4 vectors
    # u[i*m+j], v[j*p+k], w[k*n+i]
    # Local indices: i,j,k in {0,1}
    # u[2*i+j], v[2*j+k], w[2*i+k]
    
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,1,1,1]),  # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),  # M2
        ([1,0,0,0], [0,1,0,-1], [1,1,0,0]),  # M3
        ([0,0,0,1], [0,0,1,-1], [0,0,1,1]),  # M4
        ([1,1,0,0], [0,0,0,1], [0,1,0,0]),  # M5
        ([0,-1,1,0], [1,1,0,0], [0,0,1,1]),  # M6
        ([0,0,-1,1], [0,0,1,1], [0,0,1,1]),  # M7
    ]
    
    # Verify Strassen terms
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for u, v, w in strassen_terms:
                        total += u[2*i+j] * v[2*j+k] * w[2*i+k]
                    if total != 1:
                        return False
        return True
    
    use_strassen = verify_strassen()
    
    if not use_strassen:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build decomposition
    rank = n*m*p - len(best_greedy)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for block_idx in best_greedy:
        ip, jp, kp, triples = all_blocks[block_idx]
        i1, i2 = ip
        j1, j2 = jp
        k1, k2 = kp
        
        for u, v, w in strassen_terms:
            for di in range(2):
                for dj in range(2):
                    if u[2*di+dj] != 0:
                        gi = (i1, i2)[di]
                        gj = (j1, j2)[dj]
                        U[gi*m+gj][col] += u[2*di+dj]
            
            for dj in range(2):
                for dk in range(2):
                    if v[2*dj+dk] != 0:
                        gj = (j1, j2)[dj]
                        gk = (k1, k2)[dk]
                        V[gj*p+gk][col] += v[2*dj+dk]
            
            for di in range(2):
                for dk in range(2):
                    if w[2*di+dk] != 0:
                        gi = (i1, i2)[di]
                        gk = (k1, k2)[dk]
                        W[gk*n+gi][col] += w[2*di+dk]
            
            col += 1
    
    covered = set()
    for block_idx in best_greedy:
        ip, jp, kp, triples = all_blocks[block_idx]
        covered |= triples
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in covered:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    assert col == rank, f"Column mismatch: {col} vs {rank}"
    
    # Final verification with random matrices
    # If it fails, return schoolbook
    if time.time() < deadline - 5:
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_vec = np.array([U[i*m+j][r] for i in range(n) for j in range(m)]).reshape(n, m)
            v_vec = np.array([V[j*p+k][r] for j in range(m) for k in range(p)]).reshape(m, p)
            w_vec = np.array([W[k*n+i][r] for k in range(p) for i in range(n)]).reshape(p, n)
            # C += u_vec * v_vec * w_vec (element-wise? No, this is tensor contraction)
            # T[i,j,k] = sum U V W. C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
            # Actually, the decomposition is T = sum U_r V_r W_r.
            # C = sum_r (U_r A) * (B V_r^T) W_r^T? No.
            # C[i,k] = sum_j sum_r U_r[i,j] V_r[j,k] W_r[k,i] A[i,j] B[j,k]
            # This is not a simple matrix product.
            # Let's just check the tensor entries.
            pass
        
        # Simpler check: Reconstruct T and check against identity-like tensor?
        # T[i,j,k] should be 1 for all i,j,k? No, T is the multiplication tensor.
        # T[i,j,k] = 1.
        # sum_r U_r[i,j] V_r[j,k] W_r[k,i] = 1 for all i,j,k.
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
