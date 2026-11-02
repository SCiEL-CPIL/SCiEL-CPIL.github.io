import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Pre-compute consistent triples
    # x = i*m + j, y = j*p + k, z = k*n + i
    # Check consistency: 
    # i = x // m, j = x % m
    # j' = y // p, k = y % p
    # k' = z // n, i' = z % n
    # Consistent if j == j' and k == k' and i == i'
    
    triples = []
    consistent_flags = []
    for x in range(nm):
        for y in range(mp):
            for z in range(pn):
                i_x, j_x = divmod(x, m)
                j_y, k_y = divmod(y, p)
                k_z, i_z = divmod(z, n)
                if j_x == j_y and k_y == k_z and i_z == i_x:
                    triples.append((x, y, z))
                    consistent_flags.append(1)
                else:
                    triples.append((x, y, z))
                    consistent_flags.append(0)
    
    triples = np.array(triples)
    flags = np.array(consistent_flags)
    num_triples = len(triples)
    
    # Try ranks from 41 down to 35
    for R in range(41, 34, -1):
        start_time = time.time()
        
        # Initialize U, V, W
        # Use schoolbook for first min(R, 42) terms, rest random
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        
        # Fill with schoolbook terms
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < R:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
                    else:
                        break
                if col >= R: break
            if col >= R: break
        
        # Add random noise to remaining columns or all to escape local minima
        if R < 42:
             # We removed some terms. Let's randomize all to allow restructuring
             U = np.random.randn(nm, R) * 0.1
             V = np.random.randn(mp, R) * 0.1
             W = np.random.randn(pn, R) * 0.1
             # Scale up to get closer to 1
             U /= np.linalg.norm(U)
             V /= np.linalg.norm(V)
             W /= np.linalg.norm(W)
        
        def residual(params):
            # params: flat array of U, V, W
            U_flat = params[:nm * R]
            V_flat = params[nm * R : nm * R + mp * R]
            W_flat = params[nm * R + mp * R :]
            
            U_mat = U_flat.reshape(nm, R)
            V_mat = V_flat.reshape(mp, R)
            W_mat = W_flat.reshape(pn, R)
            
            # Compute sum_r U[x,r] V[y,r] W[z,r]
            # This is a tensor contraction
            # Reshape to (nm, R) x (mp, R) -> (nm, mp, R)
            # Then multiply with W
            # Efficient way: 
            # M_xy = U[:, :, None] * V[None, :, :] # (nm, mp, R)
            # Res = M_xy @ W.T # (nm, mp, pn)
            # This might be memory heavy: 6 * 21 * 14 * 41 ~ 72k floats. OK.
            
            M = U_mat[:, :, None] * V_mat[None, :, :]
            # M is (nm, mp, R)
            # We need sum over R: M[x, y, r] * W[z, r]
            # This is (nm, mp, pn)
            # Res = np.einsum('xyr,zr->xyp', M, W_mat)
            
            Res = np.einsum('xyr,zr->xyp', M, W_mat)
            
            # Reshape to (num_triples,)
            Res_flat = Res[triples[:, 0], triples[:, 1], triples[:, 2]]
            
            # Target: 1 if consistent, 0 otherwise
            target = flags
            
            return Res_flat - target
        
        # Initial guess
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Optimize
        try:
            result = least_squares(residual, x0, method='lm', max_nfev=10000)
            if result.cost < 1e-8:
                # Check if solution is close to half-integers
                U_sol = result.x[:nm * R].reshape(nm, R)
                V_sol = result.x[nm * R : nm * R + mp * R].reshape(mp, R)
                W_sol = result.x[nm * R + mp * R :].reshape(pn, R)
                
                # Round to half-integers
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_sol * 2) / 2
                
                # Verify exactness with rounded values
                M_round = U_round[:, :, None] * V_round[None, :, :]
                Res_round = np.einsum('xyr,zr->xyp', M_round, W_round)
                Res_flat_round = Res_round[triples[:, 0], triples[:, 1], triples[:, 2]]
                
                if np.allclose(Res_flat_round, flags, atol=1e-9):
                    # Convert to lists
                    U_list = U_round.tolist()
                    V_list = V_round.tolist()
                    W_list = W_round.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception as e:
            pass
        
        # If time is running out, break
        if time.time() - start_time > 10:
            continue
            
    # Fallback to schoolbook
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
