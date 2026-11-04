import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, np_ = n*m, m*p, p*n
    
    # Precompute T
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    T_mat = T.reshape(nm, mp*np_)
    T_mat2 = np.transpose(T, (1,0,2)).reshape(mp, nm*np_)
    T_mat3 = np.transpose(T, (2,0,1)).reshape(np_, nm*mp)
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((np_, rank_school))
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, t] = 1
                V_school[j*p+k, t] = 1
                W_school[k*n+i, t] = 1
                t += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    def to_half_integer(matrix):
        return np.round(matrix * 2) / 2
    
    def check_exact(U, V, W):
        T_approx = np.zeros((nm, mp, np_))
        for t in range(U.shape[1]):
            T_approx += np.outer(np.outer(U[:, t], V[:, t]), W[:, t])
        return np.linalg.norm(T_approx - T, ord='fro') < 1e-10
    
    # Try ranks from 85 to 99 (start higher to save time)
    for r in range(85, 100):
        if time.time() > deadline:
            break
        
        # Try 2 random initializations
        for trial in range(2):
            if time.time() > deadline:
                break
            
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(np_, r)
            
            # Alternating least squares with fewer iterations
            for iteration in range(20):
                if time.time() > deadline:
                    break
                
                # Update U
                VW = np.zeros((mp*np_, r))
                for t in range(r):
                    VW[:, t] = np.outer(V[:, t], W[:, t]).flatten()
                try:
                    U = np.linalg.lstsq(VW, T_mat.T, rcond=None)[0].T
                except:
                    pass
                
                # Update V
                UW = np.zeros((nm*np_, r))
                for t in range(r):
                    UW[:, t] = np.outer(U[:, t], W[:, t]).flatten()
                try:
                    V = np.linalg.lstsq(UW, T_mat2.T, rcond=None)[0].T
                except:
                    pass
                
                # Update W
                UV = np.zeros((nm*mp, r))
                for t in range(r):
                    UV[:, t] = np.outer(U[:, t], V[:, t]).flatten()
                try:
                    W = np.linalg.lstsq(UV, T_mat3.T, rcond=None)[0].T
                except:
                    pass
            
            # Round to half-integers
            U_r = to_half_integer(U)
            V_r = to_half_integer(V)
            W_r = to_half_integer(W)
            
            if check_exact(U_r, V_r, W_r):
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if best_rank < rank_school:
            break
    
    # Convert to lists
    def to_list(matrix):
        result = []
        for row in matrix:
            new_row = []
            for val in row:
                if abs(val) < 1e-10:
                    new_row.append(0)
                elif abs(val - round(val)) < 1e-10:
                    new_row.append(int(round(val)))
                elif abs(val*2 - round(val*2)) < 1e-10:
                    new_row.append(round(val*2)/2)
                else:
                    new_row.append(float(val))
            result.append(new_row)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
