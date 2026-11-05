import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    
    def check_validity(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def generate_strassen(n, m, p):
        if n == 2 and m == 2 and p == 2:
            rank = 7
            U = [[0]*rank for _ in range(4)]
            V = [[0]*rank for _ in range(4)]
            W = [[0]*rank for _ in range(4)]
            
            # M1 = (A00+A11)(B00+B11)
            U[0][0] += 1; U[3][0] += 1
            V[0][0] += 1; V[3][0] += 1
            W[0][0] += 1; W[3][0] += 1
            
            # M2 = (A10+A11)B00
            U[2][1] += 1; U[3][1] += 1
            V[0][1] += 1
            W[1][1] += 1; W[3][1] += -1
            
            # M3 = A00(B01-B11)
            U[0][2] += 1
            V[1][2] += 1; V[3][2] += -1
            W[2][2] += 1; W[3][2] += 1
            
            # M4 = A11(B10-B00)
            U[3][3] += 1
            V[2][3] += 1; V[0][3] += -1
            W[0][3] += 1; W[1][3] += 1
            
            # M5 = (A00+A01)B11
            U[0][4] += 1; U[1][4] += 1
            V[3][4] += 1
            W[0][4] += -1; W[2][4] += 1
            
            # M6 = (A10-A00)(B00+B01)
            U[2][5] += 1; U[0][5] += -1
            V[0][5] += 1; V[1][5] += 1
            W[3][5] += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[1][6] += 1; U[3][6] += -1
            V[2][6] += 1; V[3][6] += 1
            W[0][6] += 1
            
            return U, V, W, rank
        
        if m > 2:
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1, r1 = generate_strassen(n, m1, p)
            U2, V2, W2, r2 = generate_strassen(n, m2, p)
            rank = r1 + r2
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            for i in range(n):
                for j in range(m1):
                    U[i*m+j][0:r1] = U1[i*m1+j]
                for j in range(m2):
                    U[i*m+m1+j][r1:r1+r2] = U2[i*m2+j]
            for j in range(m1):
                for k in range(p):
                    V[j*p+k][0:r1] = V1[j*p+k]
            for j in range(m2):
                for k in range(p):
                    V[(m1+j)*p+k][r1:r1+r2] = V2[j*p+k]
            for k in range(p):
                for i in range(n):
                    W[k*n+i][0:r1] = W1[k*n+i]
                    W[k*n+i][r1:r1+r2] = W2[k*n+i]
            return U, V, W, rank
        elif p > 2:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1, r1 = generate_strassen(n, m, p1)
            U2, V2, W2, r2 = generate_strassen(n, m, p2)
            rank = r1 + r2
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            for i in range(n):
                for j in range(m):
                    U[i*m+j][0:r1] = U1[i*m+j]
                    U[i*m+j][r1:r1+r2] = U2[i*m+j]
            for j in range(m):
                for k in range(p1):
                    V[j*p+k][0:r1] = V1[j*p1+k]
                for k in range(p2):
                    V[j*p+p1+k][r1:r1+r2] = V2[j*p2+k]
            for k in range(p1):
                for i in range(n):
                    W[k*n+i][0:r1] = W1[k*n+i]
            for k in range(p2):
                for i in range(n):
                    W[(p1+k)*n+i][r1:r1+r2] = W2[k*n+i]
            return U, V, W, rank
        elif n > 2:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1, r1 = generate_strassen(n1, m, p)
            U2, V2, W2, r2 = generate_strassen(n2, m, p)
            rank = r1 + r2
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            for i in range(n1):
                for j in range(m):
                    U[i*m+j][0:r1] = U1[i*m+j]
            for i in range(n2):
                for j in range(m):
                    U[(n1+i)*m+j][r1:r1+r2] = U2[i*m+j]
            for j in range(m):
                for k in range(p):
                    V[j*p+k][0:r1] = V1[j*p+k]
                    V[j*p+k][r1:r1+r2] = V2[j*p+k]
            for k in range(p):
                for i in range(n1):
                    W[k*n+i][0:r1] = W1[k*n+i]
                for i in range(n2):
                    W[k*n+n1+i][r1:r1+r2] = W2[k*n+i]
            return U, V, W, rank
        else:
            raise ValueError("Base case not reached")
    
    U, V, W, rank = generate_strassen(n, m, p)
    
    # Verify
    if not check_validity(U, V, W, n, m, p):
        # Fallback: schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    # Now attempt rank reduction via BCR-style search
    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def check_validity_np(U_np, V_np, W_np, n, m, p):
        rank = U_np.shape[1]
        # Tensor product check: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Build the tensor T[i,j,k,r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We want to find a subset of rank indices that still satisfies the constraint
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Strategy: try to remove rank components one at a time
    # For each rank index r, check if removing it (setting to 0) still gives valid decomposition
    # If not, keep it. This is a greedy approach.
    
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = rank
    
    # Track which rank indices are active
    active = [True] * rank
    
    # Try removing each rank index
    for r in range(rank):
        if _time_left() < 10:
            break
        if not active[r]:
            continue
        
        # Try removing rank index r
        old_U_r = current_U[:, r].copy()
        old_V_r = current_V[:, r].copy()
        old_W_r = current_W[:, r].copy()
        
        current_U[:, r] = 0
        current_V[:, r] = 0
        current_W[:, r] = 0
        
        if check_validity_np(current_U, current_V, current_W, n, m, p):
            active[r] = False
            current_rank -= 1
            # Keep it removed
        else:
            # Restore
            current_U[:, r] = old_U_r
            current_V[:, r] = old_V_r
            current_W[:, r] = old_W_r
    
    # Now try to find a minimal subset by checking all active indices
    # Try removing each active index again (in case removal order matters)
    for r in range(rank):
        if _time_left() < 10:
            break
        if not active[r]:
            continue
        
        old_U_r = current_U[:, r].copy()
        old_V_r = current_V[:, r].copy()
        old_W_r = current_W[:, r].copy()
        
        current_U[:, r] = 0
        current_V[:, r] = 0
        current_W[:, r] = 0
        
        if check_validity_np(current_U, current_V, current_W, n, m, p):
            active[r] = False
            current_rank -= 1
        else:
            current_U[:, r] = old_U_r
            current_V[:, r] = old_V_r
            current_W[:, r] = old_W_r
    
    # Extract the reduced matrices
    if current_rank < best_rank:
        active_indices = [r for r in range(rank) if active[r]]
        U_reduced = current_U[:, active_indices]
        V_reduced = current_V[:, active_indices]
        W_reduced = current_W[:, active_indices]
        
        # Convert back to lists
        U_list = U_reduced.tolist()
        V_list = V_reduced.tolist()
        W_list = W_reduced.tolist()
        
        # Verify the reduced version
        if check_validity(U_list, V_list, W_list, n, m, p):
            best_rank = current_rank
            best_U, best_V, best_W = U_list, V_list, W_list
    
    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
