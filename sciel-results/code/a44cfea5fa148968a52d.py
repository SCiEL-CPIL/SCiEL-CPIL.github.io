import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    def check_deadline():
        return time.time() > deadline
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    def is_valid(U, V, W, r):
        """Check if the decomposition is valid (all entries integer/half-integer, exact decomposition)."""
        # Check entries
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        # Check decomposition
        for i in range(n):
            for i2 in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k in range(p):
                            for k2 in range(p):
                                s = 0.0
                                for c in range(r):
                                    s += U[i*m+j][c] * V[j2*p+k][c] * W[k2*n+i2][c]
                                target = 1.0 if (i==i2 and j==j2 and k==k2) else 0.0
                                if abs(s - target) > 1e-6:
                                    return False
        return True
    
    # Try to find a better decomposition
    for target_rank in range(28, 60):
        if check_deadline():
            break
        
        for trial in range(3):
            if check_deadline():
                break
            
            np.random.seed(42 + trial * 1000 + target_rank)
            
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            for iteration in range(20):
                if check_deadline():
                    break
                
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        A_rows = []
                        b_vals = []
                        for j2 in range(m):
                            for k2 in range(p):
                                for i2 in range(n):
                                    vec = V[j2*p+k2, :target_rank] * W[k2*n+i2, :target_rank]
                                    A_rows.append(vec)
                                    target = 1.0 if (j2==j and k2==k and i2==i) else 0.0
                                    b_vals.append(target)
                        A = np.array(A_rows)
                        b = np.array(b_vals)
                        try:
                            U[i*m+j, :target_rank] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        A_rows = []
                        b_vals = []
                        for i in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    vec = U[i*m+j2, :target_rank] * W[k2*n+i, :target_rank]
                                    A_rows.append(vec)
                                    target = 1.0 if (j2==j and k2==k) else 0.0
                                    b_vals.append(target)
                        A = np.array(A_rows)
                        b = np.array(b_vals)
                        try:
                            V[j*p+k, :target_rank] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Solve for W
                for k in range(p):
                    for i in range(n):
                        A_rows = []
                        b_vals = []
                        for i2 in range(n):
                            for j in range(m):
                                for k2 in range(p):
                                    vec = U[i2*m+j, :target_rank] * V[j*p+k2, :target_rank]
                                    A_rows.append(vec)
                                    target = 1.0 if (i2==i and k2==k) else 0.0
                                    b_vals.append(target)
                        A = np.array(A_rows)
                        b = np.array(b_vals)
                        try:
                            W[k*n+i, :target_rank] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
            
            # Round to integers/half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Convert to lists
            U_list = [[float(U_r[i, c]) for c in range(target_rank)] for i in range(n*m)]
            V_list = [[float(V_r[i, c]) for c in range(target_rank)] for i in range(m*p)]
            W_list = [[float(W_r[i, c]) for c in range(target_rank)] for i in range(p*n)]
            
            if is_valid(U_list, V_list, W_list, target_rank) and target_rank < best_rank:
                best_U, best_V, best_W, best_rank = U_list, V_list, W_list, target_rank
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
