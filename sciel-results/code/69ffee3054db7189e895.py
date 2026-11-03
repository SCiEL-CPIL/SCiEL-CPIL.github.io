import time
import numpy as np
import random

DIMENSIONS = [2, 3, 5]
START_TIME = time.time()
DEADLINE = START_TIME + 140.0

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    """
    n, m, p = 2, 3, 5
    
    # Target tensor T[i, k, i', j', j'', k'] = 1 if i'=i, j'=j'', k'=k else 0
    # We will verify against this.
    
    def error(U, V, W):
        """
        U: (n*m, R)
        V: (m*p, R)
        W: (p*n, R)
        """
        R = U.shape[1]
        # Reshape to 3D
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R) # W[k, i, r]
        
        # Compute E[i, k, i', j', j'', k'] = sum_r W3[k, i, r] * U3[i', j', r] * V3[j'', k', r]
        # Note: W3 is indexed by [k, i, r]
        # U3 is [i', j', r]
        # V3 is [j'', k', r]
        
        # Using einsum
        # 'kir,i\'jr,j''k\'r->iki\'j\'j''k\''
        # Let's use labels a,b,c,d,e,f,g
        # W: k, i, r -> a, b, r
        # U: i', j', r -> c, d, r
        # V: j'', k', r -> e, f, r
        # Result: a, b, c, d, e, f
        try:
            E = np.einsum('arb,cdr,esr->abcde f'.replace(' ', ''), W3, U3, V3)
            # Wait, labels:
            # W3: k, i, r. Let's say indices 0, 1, 2.
            # U3: i', j', r. Indices 0, 1, 2.
            # V3: j'', k', r. Indices 0, 1, 2.
            # Result: k, i, i', j', j'', k'.
            
            # k -> 0
            # i -> 1
            # i' -> 2
            # j' -> 3
            # j'' -> 4
            # k' -> 5
            
            # W3[k, i, r] -> 'kir'
            # U3[i', j', r] -> 'i\'jr'
            # V3[j'', k', r] -> 'j''k\'r'
            
            # String: 'kir,i\'jr,j''k\'r->iki\'j\'j''k\''
            # This is getting messy with quotes.
            # Let's use integers or simple letters.
            # W: 0, 1, 2
            # U: 3, 4, 2
            # V: 5, 6, 2
            # Result: 0, 1, 3, 4, 5, 6
            
            E = np.einsum('a b r, c d r, e f r -> a b c d e f', W3, U3, V3)
        except Exception:
            # Fallback to loop if einsum string is tricky
            E = np.zeros((n, p, n, m, m, p))
            for r in range(R):
                # W3[:, :, r] is (p, n) -> k, i
                # U3[:, :, r] is (n, m) -> i', j'
                # V3[:, :, r] is (m, p) -> j'', k'
                # Outer product
                # E[i, k, i', j', j'', k'] += W3[k, i, r] * U3[i', j', r] * V3[j'', k', r]
                # This is a tensor product of 3 matrices.
                # We can do it in steps.
                # T1[i, k, i', j'] = W3[k, i, r] * U3[i', j', r]
                # T2[i, k, j'', k'] = ... no, V is separate.
                
                # Let's just do it with broadcasting
                # W3_r: (p, n) -> k, i
                # U3_r: (n, m) -> i', j'
                # V3_r: (m, p) -> j'', k'
                
                # We want to add to E
                # E shape: (p, n, n, m, m, p)
                # W3_r[:, :, None, None, None, None] * U3_r[None, None, :, :, None, None] * V3_r[None, None, None, None, :, :]
                # Dimensions:
                # W: k, i
                # U: i', j'
                # V: j'', k'
                
                term = W3[:, :, r][:, :, None, None, None, None] * \
                       U3[:, :, r][None, None, :, :, None, None] * \
                       V3[:, :, r][None, None, None, None, :, :]
                E += term
            
        # Compute target T
        # T[i, k, i', j', j'', k'] = 1 if i'=i, j'=j'', k'=k
        # Note: i is from W (1st dim of result in my einsum 'a'), k is 2nd dim 'b'
        # Wait, my einsum result is 'a b c d e f' which is k, i, i', j', j'', k'
        # So a=k, b=i, c=i', d=j', e=j'', f=k'
        
        # Target condition: c == b, d == e, f == a
        T = np.zeros_like(E)
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    T[k, i, i, j, j, k] = 1.0
        
        diff = E - T
        return np.sum(diff * diff)

    # Search for rank R
    best_rank = 30
    best_solution = None
    
    for R in [27, 26, 25]:
        if time.time() > DEADLINE - 20:
            break
            
        found = False
        # Try a few random restarts
        for restart in range(5):
            if time.time() > DEADLINE - 20:
                break
                
            # Initialize with random integers
            U = np.random.randint(-1, 2, (n*m, R))
            V = np.random.randint(-1, 2, (m*p, R))
            W = np.random.randint(-1, 2, (p*n, R))
            
            # Local search
            current_error = error(U, V, W)
            
            # If error is already 0, we are done
            if current_error < 1e-9:
                found = True
                best_rank = R
                best_solution = (U, V, W)
                break
                
            # Iterate
            for it in range(1000):
                if time.time() > DEADLINE - 20:
                    break
                    
                # Pick a random variable
                var_type = random.choice(['U', 'V', 'W'])
                if var_type == 'U':
                    idx = random.randint(0, n*m - 1)
                    r = random.randint(0, R - 1)
                    old_val = U[idx, r]
                    new_val = random.choice([-1, 0, 1, 2])
                    if new_val == old_val:
                        continue
                    U[idx, r] = new_val
                    new_error = error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                    else:
                        U[idx, r] = old_val
                elif var_type == 'V':
                    idx = random.randint(0, m*p - 1)
                    r = random.randint(0, R - 1)
                    old_val = V[idx, r]
                    new_val = random.choice([-1, 0, 1, 2])
                    if new_val == old_val:
                        continue
                    V[idx, r] = new_val
                    new_error = error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                    else:
                        V[idx, r] = old_val
                else:
                    idx = random.randint(0, p*n - 1)
                    r = random.randint(0, R - 1)
                    old_val = W[idx, r]
                    new_val = random.choice([-1, 0, 1, 2])
                    if new_val == old_val:
                        continue
                    W[idx, r] = new_val
                    new_error = error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                    else:
                        W[idx, r] = old_val
                        
                if current_error < 1e-9:
                    found = True
                    best_rank = R
                    best_solution = (U, V, W)
                    break
                    
            if found:
                break

    if not found:
        # Fallback to rank 28 (current best)
        # Reuse the logic from the current program or just return rank 30 if 28 is too complex to re-implement quickly
        # For safety, return rank 30 schoolbook if 28 is not implemented here
        # But we should try to return the best we found, even if not 0 error? No, must be exact.
        # If no exact solution found, return rank 30.
        R = 30
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        best_solution = (U, V, W)
        best_rank = 30

    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
