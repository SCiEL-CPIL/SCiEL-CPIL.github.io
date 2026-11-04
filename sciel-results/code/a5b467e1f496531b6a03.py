"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 145  # leave 5s margin
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p  # 54
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    def verify_decomposition_full(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.sum(U[a,:] * V[b,:] * W[c,:])
                    if abs(val - 1) > tol:
                        return False
        # Check all zero entries
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (j == j2 and k == k2 and i == i2):
                        val = np.sum(U[a,:] * V[b,:] * W[c,:])
                        if abs(val) > tol:
                            return False
        return True
    
    def compute_loss(U, V, W):
        """Compute the total squared error over all tensor entries."""
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.sum(U[a,:] * V[b,:] * W[c,:])
                    loss += (val - 1)**2
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (j == j2 and k == k2 and i == i2):
                        val = np.sum(U[a,:] * V[b,:] * W[c,:])
                        loss += val**2
        return loss
    
    # Precompute valid triples and zero triples
    valid_triples = []
    zero_triples = []
    for a in range(dim_u):
        for b in range(dim_v):
            for c in range(dim_w):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if j == j2 and k == k2 and i == i2:
                    valid_triples.append((a, b, c))
                else:
                    zero_triples.append((a, b, c))
    
    def alternating_minimization(U, V, W, max_iters=300, tol=1e-12):
        """Perform alternating least squares to minimize the decomposition error."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        prev_loss = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Solve for U given V and W
            A_rows = []
            b_vec = []
            for (a, b, c) in valid_triples:
                row = np.zeros(dim_u * rank)
                for d in range(rank):
                    row[a*rank + d] = V[b, d] * W[c, d]
                A_rows.append(row)
                b_vec.append(1.0)
            for (a, b, c) in zero_triples:
                row = np.zeros(dim_u * rank)
                for d in range(rank):
                    row[a*rank + d] = V[b, d] * W[c, d]
                A_rows.append(row)
                b_vec.append(0.0)
            
            A = np.array(A_rows)
            b = np.array(b_vec)
            
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U = U_flat.reshape(dim_u, rank)
            except:
                pass
            
            # Solve for V given U and W
            A_rows = []
            b_vec = []
            for (a, b, c) in valid_triples:
                row = np.zeros(dim_v * rank)
                for d in range(rank):
                    row[b*rank + d] = U[a, d] * W[c, d]
                A_rows.append(row)
                b_vec.append(1.0)
            for (a, b, c) in zero_triples:
                row = np.zeros(dim_v * rank)
                for d in range(rank):
                    row[b*rank + d] = U[a, d] * W[c, d]
                A_rows.append(row)
                b_vec.append(0.0)
            
            A = np.array(A_rows)
            b = np.array(b_vec)
            
            try:
                V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V = V_flat.reshape(dim_v, rank)
            except:
                pass
            
            # Solve for W given U and V
            A_rows = []
            b_vec = []
            for (a, b, c) in valid_triples:
                row = np.zeros(dim_w * rank)
                for d in range(rank):
                    row[c*rank + d] = U[a, d] * V[b, d]
                A_rows.append(row)
                b_vec.append(1.0)
            for (a, b, c) in zero_triples:
                row = np.zeros(dim_w * rank)
                for d in range(rank):
                    row[c*rank + d] = U[a, d] * V[b, d]
                A_rows.append(row)
                b_vec.append(0.0)
            
            A = np.array(A_rows)
            b = np.array(b_vec)
            
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(dim_w, rank)
            except:
                pass
            
            loss = compute_loss(U, V, W)
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
        
        return U, V, W, prev_loss
    
    # Try to exploit cyclic structure: group columns of B (3x9) mod 3
    # This gives 3 polynomials of degree <3, multiplication in Z[x]/(x^3-1)
    # Try ranks from low to high
    for rank in range(18, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        
        found = False
        num_trials = 20 if rank < 40 else 10
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.RandomState(trial * 1000 + rank)
            U = rng.randn(dim_u, rank) * 0.1
            V = rng.randn(dim_v, rank) * 0.1
            W = rng.randn(dim_w, rank) * 0.1
            
            U, V, W, loss = alternating_minimization(U, V, W, max_iters=150, tol=1e-14)
            
            if loss < 1e-10:
                if verify_decomposition_full(U, V, W, tol=1e-10):
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if verify_decomposition_full(U_r, V_r, W_r, tol=1e-10):
                        best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = rank
                        found = True
                    else:
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = rank
                        found = True
                    break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
