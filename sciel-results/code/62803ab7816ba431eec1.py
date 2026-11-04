import numpy as np
import time

def construct():
    n, m, p = 2, 5, 5
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    deadline = start_time + 135.0
    
    # Schoolbook
    U_school = np.zeros((nm, 50), dtype=float)
    V_school = np.zeros((mp, 50), dtype=float)
    W_school = np.zeros((pn, 50), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1

    def verify(U, V, W):
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    error += abs(s - 1.0)
        return error < 1e-6

    # Try to find a lower rank solution
    # We will try to reduce the rank from 50 downwards
    # Using a local search strategy: try to merge pairs of columns
    
    U = U_school.copy()
    V = V_school.copy()
    W = W_school.copy()
    current_rank = 50
    
    # We'll try to perform merges
    # Each merge reduces rank by 1
    # We'll attempt this for a certain number of iterations
    
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        if current_rank <= 45: # Stop if we reach a good rank
            break
            
        merged = False
        # Try random pairs
        for _ in range(50):
            if time.time() > deadline - 5:
                break
                
            c1, c2 = np.random.choice(current_rank, 2, replace=False)
            
            u1, v1, w1 = U[:, c1], V[:, c1], W[:, c1]
            u2, v2, w2 = U[:, c2], V[:, c2], W[:, c2]
            
            # Fit u'v'w' to u1v1w1 + u2v2w2
            u_prime = (u1 + u2) / 2.0
            v_prime = (v1 + v2) / 2.0
            w_prime = (w1 + w2) / 2.0
            
            lr = 0.1
            best_err = float('inf')
            best_u, best_v, best_w = u_prime.copy(), v_prime.copy(), w_prime.copy()
            
            for step in range(50):
                if time.time() > deadline - 2:
                    break
                    
                grad_u = np.zeros_like(u_prime)
                grad_v = np.zeros_like(v_prime)
                grad_w = np.zeros_like(w_prime)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            t_val = u_prime[i*m+j] * v_prime[j*p+k] * w_prime[k*n+i]
                            t_target = u1[i*m+j]*v1[j*p+k]*w1[k*n+i] + u2[i*m+j]*v2[j*p+k]*w2[k*n+i]
                            diff = t_val - t_target
                            err += diff*diff
                            
                            grad_u[i*m+j] += 2 * diff * v_prime[j*p+k] * w_prime[k*n+i]
                            grad_v[j*p+k] += 2 * diff * u_prime[i*m+j] * w_prime[k*n+i]
                            grad_w[k*n+i] += 2 * diff * u_prime[i*m+j] * v_prime[j*p+k]
                
                u_prime -= lr * grad_u
                v_prime -= lr * grad_v
                w_prime -= lr * grad_w
                
                if err < best_err:
                    best_err = err
                    best_u, best_v, best_w = u_prime.copy(), v_prime.copy(), w_prime.copy()
                    
                if err < 1e-8:
                    break
                    
            # Check if the merge is valid with integer/half-integer snapping
            u_snap = np.round(best_u * 2) / 2
            v_snap = np.round(best_v * 2) / 2
            w_snap = np.round(best_w * 2) / 2
            
            # Verify the snap
            err_snap = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t_val = u_snap[i*m+j] * v_snap[j*p+k] * w_snap[k*n+i]
                        t_target = u1[i*m+j]*v1[j*p+k]*w1[k*n+i] + u2[i*m+j]*v2[j*p+k]*w2[k*n+i]
                        err_snap += (t_val - t_target)**2
                        
            if err_snap < 1e-4:
                # Perform merge
                U[:, c1] = u_snap
                V[:, c1] = v_snap
                W[:, c1] = w_snap
                
                # Remove c2
                if c2 < current_rank - 1:
                    U[:, c2] = U[:, c2+1:current_rank]
                    V[:, c2] = V[:, c2+1:current_rank]
                    W[:, c2] = W[:, c2+1:current_rank]
                    
                current_rank -= 1
                merged = True
                break
                
        if not merged:
            # If no merge found in 50 tries, break
            break

    if verify(U[:, :current_rank], V[:, :current_rank], W[:, :current_rank]):
        return {
            'U': U[:, :current_rank].tolist(),
            'V': V[:, :current_rank].tolist(),
            'W': W[:, :current_rank].tolist()
        }
    else:
        return {
            'U': U_school.tolist(),
            'V': V_school.tolist(),
            'W': W_school.tolist()
        }
