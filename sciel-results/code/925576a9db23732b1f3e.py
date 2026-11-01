import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Strategy: Use structured initialization based on tensor decomposition theory
    # For 2x3x3, try ranks from 12 upward with smart initialization
    
    def try_refinement(U, V, W, T, rank, max_iter=200):
        """Local refinement: try to fix small errors by adjusting entries"""
        for it in range(max_iter):
            R = np.einsum('as,bs,cs->abc', U, V, W)
            if np.allclose(R, T, atol=1e-9):
                return True
            diff = R - T
            sq = np.sum(diff**2)
            if sq < 1e-12:
                return True
            # Find the worst entry
            flat_diff = diff.flatten()
            idx = np.argmax(np.abs(flat_diff))
            a, b, c = np.unravel_index(idx, diff.shape)
            # Try adjusting one entry in each matrix at column s
            # The contribution to T[a,b,c] from rank s is U[a,s]*V[b,s]*W[c,s]
            # We want to change this by -diff[a,b,c]
            # Try small adjustments
            for s in range(rank):
                u_val = U[a, s]
                v_val = V[b, s]
                w_val = W[c, s]
                prod = u_val * v_val * w_val
                if abs(prod) > 1e-9:
                    # Try adjusting U[a,s]
                    # New contribution: (u_val + delta)*v_val*w_val
                    # We want (u_val + delta)*v_val*w_val = prod - diff[a,b,c]
                    # delta = -diff[a,b,c] / (v_val * w_val)
                    delta_u = -diff[a, b, c] / (v_val * w_val)
                    new_u = u_val + delta_u
                    if abs(new_u * 2) == round(abs(new_u * 2)):  # half-integer check
                        U[a, s] = new_u
                        R2 = np.einsum('as,bs,cs->abc', U, V, W)
                        if np.sum((R2 - T)**2) < sq:
                            sq = np.sum((R2 - T)**2)
                            if np.allclose(R2, T, atol=1e-9):
                                return True
                        else:
                            U[a, s] = u_val  # revert
                    # Try adjusting V[b,s]
                    if abs(u_val * w_val) > 1e-9:
                        delta_v = -diff[a, b, c] / (u_val * w_val)
                        new_v = v_val + delta_v
                        if abs(new_v * 2) == round(abs(new_v * 2)):
                            V[b, s] = new_v
                            R2 = np.einsum('as,bs,cs->abc', U, V, W)
                            if np.sum((R2 - T)**2) < sq:
                                sq = np.sum((R2 - T)**2)
                                if np.allclose(R2, T, atol=1e-9):
                                    return True
                            else:
                                V[b, s] = v_val  # revert
                    # Try adjusting W[c,s]
                    if abs(u_val * v_val) > 1e-9:
                        delta_w = -diff[a, b, c] / (u_val * v_val)
                        new_w = w_val + delta_w
                        if abs(new_w * 2) == round(abs(new_w * 2)):
                            W[c, s] = new_w
                            R2 = np.einsum('as,bs,cs->abc', U, V, W)
                            if np.sum((R2 - T)**2) < sq:
                                sq = np.sum((R2 - T)**2)
                                if np.allclose(R2, T, atol=1e-9):
                                    return True
                            else:
                                W[c, s] = w_val  # revert
        return False
    
    def structured_init(rank):
        """Create structured initialization based on tensor structure"""
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # For 2x3x3: A[i,j], B[j,k], C[k,i]
        # Structured approach: use rank components that correspond to
        # (i, j, k) combinations but with shared structure
        
        # Initialize with schoolbook-like structure but compressed
        # Use the fact that for each (j,k) pair, we need to handle both i values
        col = 0
        for j in range(m):
            for k in range(p):
                if col >= rank:
                    break
                # For this (j,k), we need C[k,0] += A[0,j]*B[j,k] and C[k,1] += A[1,j]*B[j,k]
                # Use two rank components if possible
                if col < rank:
                    U[0 * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + 0, col] = 1.0
                    col += 1
                if col < rank:
                    U[1 * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + 1, col] = 1.0
                    col += 1
        
        # Fill remaining with random half-integers
        for s in range(col, rank):
            U[:, s] = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=nm)
            V[:, s] = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=mp)
            W[:, s] = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=pn)
        
        return U, V, W
    
    # Try ranks from 12 to 17
    for rank in [12, 13, 14, 15, 16, 17]:
        if time.time() > deadline - 15:
            break
        
        max_trials = 3000
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Structured initialization
            U, V, W = structured_init(rank)
            
            # Compute R and check
            R = np.einsum('as,bs,cs->abc', U, V, W)
            if np.allclose(R, T, atol=1e-9):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # If close, try refinement
            diff = R - T
            sq = np.sum(diff**2)
            if sq < 10.0:
                if try_refinement(U, V, W, T, rank, max_iter=50):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Random perturbation for diversity
            if trial % 10 == 0:
                for _ in range(5):
                    mat_idx = np.random.randint(3)
                    if mat_idx == 0:
                        r, c = np.random.randint(nm), np.random.randint(rank)
                        U[r, c] += np.random.choice([-0.5, 0, 0.5])
                    elif mat_idx == 1:
                        r, c = np.random.randint(mp), np.random.randint(rank)
                        V[r, c] += np.random.choice([-0.5, 0, 0.5])
                    else:
                        r, c = np.random.randint(pn), np.random.randint(rank)
                        W[r, c] += np.random.choice([-0.5, 0, 0.5])
    
    # Fallback: Schoolbook decomposition of rank n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
