import time
import random
import math

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115
    
    # Target values for the tensor T[i, j, k]
    # T[i, j, k] = 1 if i==j and k==i? No, matrix multiplication C = A * B
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor is T[i, j, k] which is 1 if the term A[i, j] B[j, k] contributes to C[i, k].
    # Wait, the tensor decomposition is for the multiplication map.
    # The tensor T has indices i, j, k where i is row of A, j is col of A/row of B, k is col of B.
    # T[i, j, k] = 1 if this term exists in the product. It always exists.
    # So T[i, j, k] = 1 for all i, j, k.
    # The decomposition is T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c] ?
    # No, the standard decomposition is T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c] is for 3-tensor.
    # But matrix multiplication is a 3-tensor T_{i,j,k} where T_{i,j,k} = 1.
    # The factors are U (n x R), V (m x R), W (p x R)?
    # The task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # And the formula is: sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1.
    # This matches the standard matrix multiplication tensor.
    
    def get_sse(rank, U, V, W):
        sse = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    diff = s - 1.0
                    sse += diff * diff
        return sse

    def verify(rank, U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best = make_schoolbook()
    best_rank = n * m * p
    
    vals = [0, 0.5, -0.5, 1, -1]
    
    def local_search_sse(rank, U, V, W, seed, max_steps=5000):
        rng = random.Random(seed)
        U_ = [row[:] for row in U]
        V_ = [row[:] for row in V]
        W_ = [row[:] for row in W]
        
        cur_sse = get_sse(rank, U_, V_, W_)
        if cur_sse < 1e-9:
            return {"U": U_, "V": V_, "W": W_}
        
        best_sse = cur_sse
        no_improve = 0
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if cur_sse < 1e-9:
                break
                
            # Pick a random factor and entry
            choice = rng.random()
            if choice < 0.34:
                idx = rng.randrange(n*m)
                c = rng.randrange(rank)
                old = U_[idx][c]
                best_val = old
                best_move_sse = cur_sse
                for v in vals:
                    if v == old: continue
                    U_[idx][c] = v
                    new_sse = get_sse(rank, U_, V_, W_)
                    if new_sse < best_move_sse:
                        best_move_sse = new_sse
                        best_val = v
                U_[idx][c] = best_val
                if best_val != old:
                    cur_sse = best_move_sse
                    if cur_sse < best_sse:
                        best_sse = cur_sse
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            elif choice < 0.67:
                idx = rng.randrange(m*p)
                c = rng.randrange(rank)
                old = V_[idx][c]
                best_val = old
                best_move_sse = cur_sse
                for v in vals:
                    if v == old: continue
                    V_[idx][c] = v
                    new_sse = get_sse(rank, U_, V_, W_)
                    if new_sse < best_move_sse:
                        best_move_sse = new_sse
                        best_val = v
                V_[idx][c] = best_val
                if best_val != old:
                    cur_sse = best_move_sse
                    if cur_sse < best_sse:
                        best_sse = cur_sse
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                idx = rng.randrange(p*n)
                c = rng.randrange(rank)
                old = W_[idx][c]
                best_val = old
                best_move_sse = cur_sse
                for v in vals:
                    if v == old: continue
                    W_[idx][c] = v
                    new_sse = get_sse(rank, U_, V_, W_)
                    if new_sse < best_move_sse:
                        best_move_sse = new_sse
                        best_val = v
                W_[idx][c] = best_val
                if best_val != old:
                    cur_sse = best_move_sse
                    if cur_sse < best_sse:
                        best_sse = cur_sse
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            if no_improve > 200:
                # Try a random jump to escape local minimum
                if rng.random() < 0.5:
                    # Perturb a few random entries
                    for _ in range(3):
                        f = rng.randrange(3)
                        if f == 0:
                            idx = rng.randrange(n*m)
                            c = rng.randrange(rank)
                            U_[idx][c] = rng.choice(vals)
                        elif f == 1:
                            idx = rng.randrange(m*p)
                            c = rng.randrange(rank)
                            V_[idx][c] = rng.choice(vals)
                        else:
                            idx = rng.randrange(p*n)
                            c = rng.randrange(rank)
                            W_[idx][c] = rng.choice(vals)
                    cur_sse = get_sse(rank, U_, V_, W_)
                    no_improve = 0
            
        if cur_sse < 1e-9:
            return {"U": U_, "V": V_, "W": W_}
        return None
    
    # Try ranks from 16 upward
    for rank in range(16, min(n*m*p, 40)):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            break
        
        num_restarts = 10
        found = None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            seed = rank * 1000 + restart
            rng = random.Random(seed)
            
            # Initialize with sparse random values
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # Bias towards non-zero values to start with some structure
            for idx in range(n*m):
                for c in range(rank):
                    if rng.random() < 0.3:
                        U[idx][c] = rng.choice(vals)
            for idx in range(m*p):
                for c in range(rank):
                    if rng.random() < 0.3:
                        V[idx][c] = rng.choice(vals)
            for idx in range(p*n):
                for c in range(rank):
                    if rng.random() < 0.3:
                        W[idx][c] = rng.choice(vals)
            
            result = local_search_sse(rank, U, V, W, seed + 7777, max_steps=2000)
            if result:
                found = result
                break
        
        if found:
            best = found
            best_rank = rank
            break
            
    return best
