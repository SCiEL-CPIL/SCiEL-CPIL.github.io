import time
import random

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115
    
    def verify(rank, U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1) > 1e-9:
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
    
    def get_error_count(rank, U, V, W):
        count = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        count += 1
        return count
    
    def local_search(rank, U, V, W, seed, max_steps=200):
        rng = random.Random(seed)
        U_ = [row[:] for row in U]
        V_ = [row[:] for row in V]
        W_ = [row[:] for row in W]
        
        cur_err = get_error_count(rank, U_, V_, W_)
        if cur_err == 0:
            return {"U": U_, "V": V_, "W": W_}
        
        best_err = cur_err
        no_improve = 0
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if cur_err == 0:
                break
            
            # Choose a random factor and entry to flip
            choice = rng.random()
            if choice < 0.34:
                idx = rng.randrange(n*m)
                c = rng.randrange(rank)
                old = U_[idx][c]
                # Try changing to another value
                if old == 0:
                    U_[idx][c] = rng.choice([-1, 1, -0.5, 0.5])
                elif old == 1:
                    U_[idx][c] = rng.choice([0, -1, 0.5, -0.5])
                elif old == -1:
                    U_[idx][c] = rng.choice([0, 1, 0.5, -0.5])
                elif old == 0.5:
                    U_[idx][c] = rng.choice([0, 1, -1, -0.5])
                elif old == -0.5:
                    U_[idx][c] = rng.choice([0, 1, -1, 0.5])
                new_err = get_error_count(rank, U_, V_, W_)
                if new_err < best_err:
                    best_err = new_err
                    cur_err = new_err
                    no_improve = 0
                else:
                    U_[idx][c] = old
                    no_improve += 1
            elif choice < 0.67:
                idx = rng.randrange(m*p)
                c = rng.randrange(rank)
                old = V_[idx][c]
                if old == 0:
                    V_[idx][c] = rng.choice([-1, 1, -0.5, 0.5])
                elif old == 1:
                    V_[idx][c] = rng.choice([0, -1, 0.5, -0.5])
                elif old == -1:
                    V_[idx][c] = rng.choice([0, 1, 0.5, -0.5])
                elif old == 0.5:
                    V_[idx][c] = rng.choice([0, 1, -1, -0.5])
                elif old == -0.5:
                    V_[idx][c] = rng.choice([0, 1, -1, 0.5])
                new_err = get_error_count(rank, U_, V_, W_)
                if new_err < best_err:
                    best_err = new_err
                    cur_err = new_err
                    no_improve = 0
                else:
                    V_[idx][c] = old
                    no_improve += 1
            else:
                idx = rng.randrange(p*n)
                c = rng.randrange(rank)
                old = W_[idx][c]
                if old == 0:
                    W_[idx][c] = rng.choice([-1, 1, -0.5, 0.5])
                elif old == 1:
                    W_[idx][c] = rng.choice([0, -1, 0.5, -0.5])
                elif old == -1:
                    W_[idx][c] = rng.choice([0, 1, 0.5, -0.5])
                elif old == 0.5:
                    W_[idx][c] = rng.choice([0, 1, -1, -0.5])
                elif old == -0.5:
                    W_[idx][c] = rng.choice([0, 1, -1, 0.5])
                new_err = get_error_count(rank, U_, V_, W_)
                if new_err < best_err:
                    best_err = new_err
                    cur_err = new_err
                    no_improve = 0
                else:
                    W_[idx][c] = old
                    no_improve += 1
            
            if no_improve > 50:
                break
        
        if cur_err == 0:
            return {"U": U_, "V": V_, "W": W_}
        return None
    
    # Try ranks from 50 upward
    for rank in range(50, min(n*m*p, 66)):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            break
        
        num_restarts = 15 if rank <= 55 else 8
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
            
            vals = [0, 0, 0, 0, 1, -1, 0.5, -0.5]
            for idx in range(n*m):
                for c in range(rank):
                    if rng.random() < 0.4:
                        U[idx][c] = rng.choice(vals)
            for idx in range(m*p):
                for c in range(rank):
                    if rng.random() < 0.4:
                        V[idx][c] = rng.choice(vals)
            for idx in range(p*n):
                for c in range(rank):
                    if rng.random() < 0.4:
                        W[idx][c] = rng.choice(vals)
            
            result = local_search(rank, U, V, W, seed + 7777, max_steps=100)
            if result:
                found = result
                break
        
        if found:
            best = found
            best_rank = rank
            break
    
    return best
