import time
import numpy as np
from scipy.optimize import least_squares
import random

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Default rank 55 solution
    def get_rank_55():
        rank = 55
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for pair in range(5):
            k0 = pair * 2
            k1 = pair * 2 + 1
            
            # Strassen for (2,2,2)
            # M1
            U[0][col] = 1; U[4][col] = 1
            V[k0][col] = 1; V[10+k1][col] = 1
            W[2*k0][col] = 1; W[2*k1+1][col] = 1
            col += 1
            # M2
            U[3][col] = 1; U[4][col] = 1
            V[k0][col] = 1
            W[2*k0+1][col] = 1; W[2*k1+1][col] = -1
            col += 1
            # M3
            U[0][col] = 1
            V[k1][col] = 1; V[10+k1][col] = -1
            W[2*k1][col] = 1; W[2*k1+1][col] = 1
            col += 1
            # M4
            U[4][col] = 1
            V[10+k0][col] = 1; V[k0][col] = -1
            W[2*k0][col] = 1; W[2*k0+1][col] = 1
            col += 1
            # M5
            U[0][col] = 1; U[1][col] = 1
            V[10+k1][col] = 1
            W[2*k0][col] = -1; W[2*k1][col] = 1
            col += 1
            # M6
            U[4][col] = 1; U[0][col] = -1
            V[k0][col] = 1; V[k1][col] = 1
            W[2*k1+1][col] = 1
            col += 1
            # M7
            U[1][col] = 1; U[4][col] = -1
            V[10+k0][col] = 1; V[10+k1][col] = 1
            W[2*k0][col] = 1; W[2*k1+1][col] = -1
            col += 1
        
        # Schoolbook for j=2
        for i in range(2):
            for k in range(10):
                U[i*3+2][col] = 1
                V[20+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
        
        return {"U": U, "V": V, "W": W}

    result_55 = get_rank_55()
    
    # Try to find rank 54 solution
    # Split m=3 into 2 and 1.
    # Part 1: M(2,1,10) -> Rank 20
    # Part 2: M(2,2,10) -> Split p=10 into 3,3,3,1
    # M(2,2,3) rank 10?
    # If we find rank 10 for 2x2x3, total rank = 20 + 3*10 + 4 = 54.
    
    def find_2x2x3_rank_10():
        # Returns U, V, W for 2x2x3 if found, else None
        # U: 4x10, V: 6x10, W: 6x10
        # Target tensor T[i,j,k] = delta(i,j) * delta(j,k) is wrong.
        # T is the multiplication tensor.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor T[i,j,k] is 1 if i=j and j=k? No.
        # The tensor T is defined by T_{i,j,k} such that C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
        # In the decomposition sum_r U_{i,j}^r V_{j,k}^r W_{i,k}^r = T_{i,j,k}
        # Wait, the standard form is C_{i,k} = sum_r (sum_j U_{i,j}^r A_{i,j}) (sum_l V_{j,l}^r B_{j,l}) W_{i,k}^r
        # This corresponds to tensor T_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{i,k}^r
        # For matrix multiplication, T_{i,j,k} is 1 if the indices correspond to the product term?
        # Actually, the tensor of matrix multiplication M_{n,m,p} has entries T_{i,j,k} = 1 if ...?
        # No, the tensor is the linear map.
        # Let's define the target values for the 2x2x3 case.
        # Indices: i in {0,1}, j in {0,1}, k in {0,1,2}
        # U indices: (i,j) -> 4 rows
        # V indices: (j,k) -> 6 rows (j in 0,1; k in 0,1,2)
        # W indices: (k,i) -> 6 rows (k in 0,1,2; i in 0,1) -- Wait, W is p*n = 3*2 = 6.
        # Convention: W[k*n+i, r]
        
        def residuals(x):
            U = x[:40].reshape(4, 10)
            V = x[40:100].reshape(6, 10)
            W = x[100:160].reshape(6, 10)
            
            res = []
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        # Target T[i,j,k]
                        # The term A[i,j] * B[j,k] contributes to C[i,k].
                        # In the decomposition, the coefficient of A[i,j] B[j,k] in C[i,k] must be 1.
                        # Sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 if (i,j,k) is a valid product term?
                        # Yes, for every i,j,k, the term A[i,j]B[j,k] appears in C[i,k].
                        # So T[i,j,k] = 1 for all i,j,k?
                        # No, that's not right.
                        # C[i,k] = sum_j A[i,j] B[j,k].
                        # The decomposition is C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_l V[j,l,r] B[j,l]) W[k,i,r]
                        # Expanding: sum_r sum_j sum_l U[i,j,r] V[j,l,r] W[k,i,r] A[i,j] B[j,l]
                        # We need the coefficient of A[i,j] B[j,k] to be 1 in C[i,k].
                        # So sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
                        # And for A[i,j] B[j,l] with l != k, the coefficient in C[i,k] must be 0.
                        # So sum_r U[i,j,r] V[j,l,r] W[k,i,r] = 0 if l != k.
                        
                        val = np.sum(U[i, j] * V[j, k] * W[k, i])
                        res.append(val - 1.0)
                        
                        for l in range(3):
                            if l != k:
                                val_zero = np.sum(U[i, j] * V[j, l] * W[k, i])
                                res.append(val_zero)
            return res

        # Initial guess: Random
        best_sol = None
        best_cost = float('inf')
        
        for attempt in range(10):
            if time.time() > DEADLINE - 10:
                break
            x0 = np.random.normal(0, 1, 160)
            try:
                sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol
            except:
                pass
        
        if best_sol and best_sol.cost < 1e-8:
            x = best_sol.x
            U = x[:40].reshape(4, 10)
            V = x[40:100].reshape(6, 10)
            W = x[100:160].reshape(6, 10)
            
            # Check if we can round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify exactness
            if np.allclose(residuals(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])), 0, atol=1e-9):
                # Convert to lists
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                return U_list, V_list, W_list
        
        return None

    # Attempt to build rank 54
    if time.time() < DEADLINE - 20:
        try:
            sol_223 = find_2x2x3_rank_10()
            if sol_223:
                U_223, V_223, W_223 = sol_223
                # Build the full 54 rank solution
                rank = 54
                U = [[0] * rank for _ in range(6)]
                V = [[0] * rank for _ in range(30)]
                W = [[0] * rank for _ in range(20)]
                
                col = 0
                # Part 1: M(2,1,10) using j=0,1? No, we split m=3 into 2 and 1.
                # Let's say j=0,1 are the 2x2 part, j=2 is the 1 part.
                # M(2,2,10) part:
                # Split p=10 into 3,3,3,1.
                # k=0,1,2 -> Block 1
                # k=3,4,5 -> Block 2
                # k=6,7,8 -> Block 3
                # k=9 -> Block 4 (1x1)
                
                # Block 1: k=0,1,2
                for r in range(10):
                    for i in range(2):
                        for j in range(2):
                            U[i*3+j][col] = U_223[i*2+j][r]
                            V[j*10+k][col] = V_223[j*3+k][r] # V index j*10+k? No, V is 2x3 block, so j*3+k.
                            # Wait, V in full matrix is 3x10. Here we use j=0,1.
                            # V_223 is 2x3.
                            # Full V index: row j (0 or 1), col k (0,1,2).
                            V[j*10+k][col] = V_223[j*3+k][r]
                            W[k*2+i][col] = W_223[k*2+i][r]
                    col += 1
                
                # Block 2: k=3,4,5
                for r in range(10):
                    for i in range(2):
                        for j in range(2):
                            U[i*3+j][col] = U_223[i*2+j][r]
                            V[j*10+(3+k)][col] = V_223[j*3+k][r]
                            W[(3+k)*2+i][col] = W_223[k*2+i][r]
                    col += 1
                    
                # Block 3: k=6,7,8
                for r in range(10):
                    for i in range(2):
                        for j in range(2):
                            U[i*3+j][col] = U_223[i*2+j][r]
                            V[j*10+(6+k)][col] = V_223[j*3+k][r]
                            W[(6+k)*2+i][col] = W_223[k*2+i][r]
                    col += 1

                # Block 4: k=9 (1x1)
                # M(2,2,1) rank 4
                # A is 2x2, B is 2x1. C is 2x1.
                # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
                # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
                # 4 terms.
                U[i*3+0][col] = 1; V[0*10+9][col] = 1; W[9*2+0][col] = 1; col += 1
                U[i*3+1][col] = 1; V[1*10+9][col] = 1; W[9*2+0][col] = 1; col += 1
                U[i*3+0][col] = 1; V[0*10+9][col] = 1; W[9*2+1][col] = 1; col += 1
                U[i*3+1][col] = 1; V[1*10+9][col] = 1; W[9*2+1][col] = 1; col += 1
                
                # Part 2: M(2,1,10) for j=2
                # Rank 20
                for i in range(2):
                    for k in range(10):
                        U[i*3+2][col] = 1
                        V[2*10+k][col] = 1
                        W[k*2+i][col] = 1
                        col += 1
                        
                assert col == 54, f"Rank 54 construction failed: {col}"
                
                # Verify
                if verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
                    
        except Exception as e:
            print(f"Rank 54 attempt failed: {e}")
            
    return result_55

def verify(U, V, W, n, m, p, max_checks=5):
    rank = len(U[0])
    random.seed(42)
    for _ in range(max_checks):
        A = [[random.uniform(-2, 2) for _ in range(m)] for _ in range(n)]
        B = [[random.uniform(-2, 2) for _ in range(p)] for _ in range(m)]
        C = [[0.0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C[i][k] += A[i][j] * B[j][k]
        Cc = [[0.0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(rank):
                    ua = sum(U[i*m+j][r] * A[i][j] for j in range(m))
                    vb = sum(V[j*p+k][r] * B[j][k] for j in range(m))
                    acc += W[k*n+i][r] * ua * vb
                Cc[i][k] = acc
        for i in range(n):
            for k in range(p):
                if abs(C[i][k] - Cc[i][k]) > 1e-9:
                    return False
    return True
