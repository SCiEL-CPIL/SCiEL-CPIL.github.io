import time
import random
import math

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
    
    def get_smooth_error(rank, U, V, W):
        """Sum of squared errors - smooth surrogate for discrete error count"""
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    total += (s - 1.0) ** 2
        return total
    
    def local_search_smooth(rank, U, V, W, seed, max_steps=300):
        """Local search using smooth error as objective, then verify with exact count"""
        rng = random.Random(seed)
        U_ = [row[:] for row in U]
        V_ = [row[:] for row in V]
        W_ = [row[:] for row in W]
        
        cur_smooth = get_smooth_error(rank, U_, V_, W_)
        cur_err = get_error_count(rank, U_, V_, W_)
        if cur_err == 0:
            return {"U": U_, "V": V_, "W": W_}
        
        best_smooth = cur_smooth
        best_err = cur_err
        no_improve = 0
        
        # Candidate values for each entry
        candidate_vals = [0, 1, -1, 0.5, -0.5, 2, -2, 0.25, -0.25]
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if cur_err == 0:
                break
            
            # Choose a random factor and entry to modify
            choice = rng.random()
            if choice < 0.34:
                idx = rng.randrange(n*m)
                c = rng.randrange(rank)
                old = U_[idx][c]
                # Try a few candidate values
                best_candidate = None
                best_candidate_smooth = cur_smooth
                for val in candidate_vals:
                    if val == old:
                        continue
                    U_[idx][c] = val
                    new_smooth = get_smooth_error(rank, U_, V_, W_)
                    if new_smooth < best_candidate_smooth:
                        best_candidate_smooth = new_smooth
                        best_candidate = val
                if best_candidate is not None:
                    U_[idx][c] = best_candidate
                    new_err = get_error_count(rank, U_, V_, W_)
                    if new_err < best_err:
                        best_err = new_err
                        cur_err = new_err
                        cur_smooth = best_candidate_smooth
                        no_improve = 0
                    else:
                        U_[idx][c] = old
                        no_improve += 1
                else:
                    no_improve += 1
            elif choice < 0.67:
                idx = rng.randrange(m*p)
                c = rng.randrange(rank)
                old = V_[idx][c]
                best_candidate = None
                best_candidate_smooth = cur_smooth
                for val in candidate_vals:
                    if val == old:
                        continue
                    V_[idx][c] = val
                    new_smooth = get_smooth_error(rank, U_, V_, W_)
                    if new_smooth < best_candidate_smooth:
                        best_candidate_smooth = new_smooth
                        best_candidate = val
                if best_candidate is not None:
                    V_[idx][c] = best_candidate
                    new_err = get_error_count(rank, U_, V_, W_)
                    if new_err < best_err:
                        best_err = new_err
                        cur_err = new_err
                        cur_smooth = best_candidate_smooth
                        no_improve = 0
                    else:
                        V_[idx][c] = old
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                idx = rng.randrange(p*n)
                c = rng.randrange(rank)
                old = W_[idx][c]
                best_candidate = None
                best_candidate_smooth = cur_smooth
                for val in candidate_vals:
                    if val == old:
                        continue
                    W_[idx][c] = val
                    new_smooth = get_smooth_error(rank, U_, V_, W_)
                    if new_smooth < best_candidate_smooth:
                        best_candidate_smooth = new_smooth
                        best_candidate = val
                if best_candidate is not None:
                    W_[idx][c] = best_candidate
                    new_err = get_error_count(rank, U_, V_, W_)
                    if new_err < best_err:
                        best_err = new_err
                        cur_err = new_err
                        cur_smooth = best_candidate_smooth
                        no_improve = 0
                    else:
                        W_[idx][c] = old
                        no_improve += 1
                else:
                    no_improve += 1
            
            if no_improve > 80:
                break
        
        if cur_err == 0:
            return {"U": U_, "V": V_, "W": W_}
        return None
    
    def structured_init(rank, rng):
        """Initialize with structure based on matrix multiplication patterns"""
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        # Assign random but structured values
        for c in range(rank):
            # Each column c gets a pattern
            # For U: row i*m+j, assign value based on pattern
            for i in range(n):
                for j in range(m):
                    if rng.random() < 0.5:
                        U[i*m+j][c] = rng.choice([1, -1, 0.5, -0.5])
            for j in range(m):
                for k in range(p):
                    if rng.random() < 0.5:
                        V[j*p+k][c] = rng.choice([1, -1, 0.5, -0.5])
            for k in range(p):
                for i in range(n):
                    if rng.random() < 0.5:
                        W[k*n+i][c] = rng.choice([1, -1, 0.5, -0.5])
        
        return U, V, W
    
    # Try ranks from 50 upward
    for rank in range(50, min(n*m*p, 66)):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            break
        
        num_restarts = 20 if rank <= 55 else 12
        found = None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            seed = rank * 1000 + restart
            rng = random.Random(seed)
            
            # Initialize with structured random values
            U, V, W = structured_init(rank, rng)
            
            result = local_search_smooth(rank, U, V, W, seed + 7777, max_steps=150)
            if result:
                found = result
                break
        
        if found:
            best = found
            best_rank = rank
            break
    
    return best
