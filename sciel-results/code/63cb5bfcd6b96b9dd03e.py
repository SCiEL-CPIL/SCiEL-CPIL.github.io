import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 5.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _validate(U, V, W, n=3, m=3, p=5):
    rank = len(U[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # U: (n*m, rank) -> reshape to (n, m, rank)
    # V: (m*p, rank) -> reshape to (m, p, rank)
    # W: (p*n, rank) -> reshape to (p, n, rank)
    # Note: W index in code is k*n+i. So k is first, i is second.
    # Reshape W to (p, n, rank)
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # We need to check:
    # sum_r W[k, i, r] * U[i, j, r] * V[j', k, r] == delta(j, j')
    # for all i, k, j, j'
    
    # Let's compute the tensor T[i, k, j, j'] = sum_r W[k, i, r] U[i, j, r] V[j', k, r]
    # Indices:
    # i: 0..n-1
    # k: 0..p-1
    # j: 0..m-1
    # j': 0..m-1
    
    # Using einsum:
    # W is (p, n, r) -> W[k, i, r]
    # U is (n, m, r) -> U[i, j, r]
    # V is (m, p, r) -> V[j', k, r]
    
    # T[i, k, j, j'] = sum_r W[k, i, r] U[i, j, r] V[j', k, r]
    # Einsum: 'kir,ijr,jkr->ikjj' ? No.
    # W: k, i, r
    # U: i, j, r
    # V: j', k, r (let's call j' jj)
    
    # T[i, k, j, jj] = sum_r W[k, i, r] * U[i, j, r] * V[jj, k, r]
    # einsum('kir,ijr,jkr->ikjj', W_reshaped, U_reshaped, V_reshaped)
    
    try:
        T = np.einsum('kir,ijr,jkr->ikjj', W_reshaped, U_reshaped, V_reshaped, optimize=True)
    except:
        return False
        
    # Check if T[i, k, j, j'] == delta(j, j')
    # For each i, k, the matrix T[i, k, :, :] should be Identity(m)
    
    for i in range(n):
        for k in range(p):
            if not np.allclose(T[i, k, :, :], np.eye(m), atol=1e-9):
                return False
    return True

def _validate_half_integer(U, V, W, n=3, m=3, p=5):
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return _validate(U, V, W, n, m, p)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = 3, 3, 5
    
    # Schoolbook fallback
    U = [[0]*45 for _ in range(n*m)]
    V = [[0]*45 for _ in range(m*p)]
    W = [[0]*45 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    schoolbook = {"U": U, "V": V, "W": W}
    
    best_rank = 45
    best_sol = schoolbook
    
    # Try to find a better rank using random search + local improvement
    rng = random.Random(42)
    
    # We will try ranks 30-44
    for r in range(30, 45):
        if _time_remaining() < 10.0:
            break
            
        found = False
        for attempt in range(50):
            if _time_remaining() < 10.0:
                break
                
            U = [[0]*r for _ in range(n*m)]
            V = [[0]*r for _ in range(m*p)]
            W = [[0]*r for _ in range(p*n)]
            
            # Random initialization
            for i in range(n*m):
                for rr in range(r):
                    if rng.random() < 0.5:
                        U[i][rr] = rng.choice([-1, 1])
            for j in range(m*p):
                for rr in range(r):
                    if rng.random() < 0.5:
                        V[j][rr] = rng.choice([-1, 1])
            for k in range(p*n):
                for rr in range(r):
                    if rng.random() < 0.5:
                        W[k][rr] = rng.choice([-1, 1])
                        
            # Local search
            for it in range(100):
                if _time_remaining() < 10.0:
                    break
                    
                # Compute errors using NumPy for speed
                U_np = np.array(U)
                V_np = np.array(V)
                W_np = np.array(W)
                
                U_reshaped = U_np.reshape(n, m, r)
                V_reshaped = V_np.reshape(m, p, r)
                W_reshaped = W_np.reshape(p, n, r)
                
                try:
                    T = np.einsum('kir,ijr,jkr->ikjj', W_reshaped, U_reshaped, V_reshaped, optimize=True)
                except:
                    break
                    
                # Error is sum of squared differences from Identity
                errors = 0.0
                for i in range(n):
                    for k in range(p):
                        errors += np.sum((T[i, k, :, :] - np.eye(m))**2)
                        
                if errors < 1e-9:
                    if _validate_half_integer(U, V, W, n, m, p):
                        best_rank = r
                        best_sol = {"U": U, "V": V, "W": W}
                        found = True
                        break
                    
                # Pick a random error and try to fix it
                if not found:
                    # We need to pick a random (i, k, j, j') that has error
                    # And modify a random entry in U, V, W
                    # This is complex to map back to single entry changes effectively
                    # We'll just do random perturbation
                    
                    mat_choice = rng.randint(0, 2)
                    if mat_choice == 0:
                        idx_u = rng.randint(0, n*m-1)
                        idx_r = rng.randint(0, r-1)
                        old_val = U[idx_u][idx_r]
                        new_val = old_val + rng.choice([-1, 1])
                        U[idx_u][idx_r] = new_val
                    elif mat_choice == 1:
                        idx_v = rng.randint(0, m*p-1)
                        idx_r = rng.randint(0, r-1)
                        old_val = V[idx_v][idx_r]
                        new_val = old_val + rng.choice([-1, 1])
                        V[idx_v][idx_r] = new_val
                    else:
                        idx_w = rng.randint(0, p*n-1)
                        idx_r = rng.randint(0, r-1)
                        old_val = W[idx_w][idx_r]
                        new_val = old_val + rng.choice([-1, 1])
                        W[idx_w][idx_r] = new_val
                        
                    # Check if it improved (using a quick check on a subset or full)
                    # For speed, we might skip the full re-evaluation every step
                    # But for correctness, we need to know if we improved
                    # Let's just keep the change if it doesn't increase error significantly
                    # Or simply accept random walks
                    
        if found:
            break
            
    return best_sol
