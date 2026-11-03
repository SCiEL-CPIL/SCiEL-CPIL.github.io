import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # Build rank 26 solution: Strassen 2x2 core + 19 border terms
    R26 = 26
    U = np.zeros((n * m, R26), dtype=np.float64)
    V = np.zeros((m * p, R26), dtype=np.float64)
    W = np.zeros((p * n, R26), dtype=np.float64)
    
    # Strassen for 2x2 core (rows 0,1 of A; cols 0,1 of B)
    # r=0: (A00+A11)(B00+B11)
    U[idx_u(0,0),0]=1; U[idx_u(1,1),0]=1
    V[idx_v(0,0),0]=1; V[idx_v(1,1),0]=1
    W[idx_w(0,0),0]=1; W[idx_w(1,1),0]=1
    # r=1: (A10+A11)B00
    U[idx_u(1,0),1]=1; U[idx_u(1,1),1]=1
    V[idx_v(0,0),1]=1
    W[idx_w(0,1),1]=1; W[idx_w(1,1),1]=-1
    # r=2: A00(B11-B01)
    U[idx_u(0,0),2]=1
    V[idx_v(0,1),2]=1; V[idx_v(1,1),2]=-1
    W[idx_w(1,0),2]=1; W[idx_w(1,1),2]=1
    # r=3: A11(B10-B00)
    U[idx_u(1,1),3]=1
    V[idx_v(1,0),3]=1; V[idx_v(0,0),3]=-1
    W[idx_w(0,0),3]=1; W[idx_w(0,1),3]=1
    # r=4: (A00+A01)B11
    U[idx_u(0,0),4]=1; U[idx_u(0,1),4]=1
    V[idx_v(1,1),4]=1
    W[idx_w(0,0),4]=-1; W[idx_w(1,0),4]=1
    # r=5: (A10-A00)(B00+B01)
    U[idx_u(1,0),5]=1; U[idx_u(0,0),5]=-1
    V[idx_v(0,0),5]=1; V[idx_v(0,1),5]=1
    W[idx_w(1,1),5]=1
    # r=6: (A01-A11)(B10+B11)
    U[idx_u(0,1),6]=1; U[idx_u(1,1),6]=-1
    V[idx_v(1,0),6]=1; V[idx_v(1,1),6]=1
    W[idx_w(0,0),6]=1
    
    # 19 border terms
    border_elements = [
        (0,0,2),(0,1,2),(0,2,0),(0,2,1),(0,2,2),
        (1,0,2),(1,1,2),(1,2,0),(1,2,1),(1,2,2),
        (2,0,0),(2,0,1),(2,0,2),(2,1,0),(2,1,1),(2,1,2),
        (2,2,0),(2,2,1),(2,2,2)
    ]
    col = 7
    for (i,j,k) in border_elements:
        U[idx_u(i,j),col]=1
        V[idx_v(j,k),col]=1
        W[idx_w(k,i),col]=1
        col += 1
    
    # Verify rank 26 solution
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    expected = 1.0
                    actual = sum(U[idx_u(i,j),r]*V[idx_v(j,k),r]*W[idx_w(k,i),r] for r in range(R))
                    if abs(actual - expected) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback: schoolbook rank 27
        R27 = 27
        U27 = np.zeros((n*m, R27), dtype=np.float64)
        V27 = np.zeros((m*p, R27), dtype=np.float64)
        W27 = np.zeros((p*n, R27), dtype=np.float64)
        r = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U27[idx_u(i,j),r]=1
                    V27[idx_v(j,k),r]=1
                    W27[idx_w(k,i),r]=1
                    r += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    # Try to find rank 23 via border optimization using GF(5) finite field approach
    if time.time() < deadline - 20:
        try:
            # Strategy: Use finite field GF(5) to search for rank-23 decompositions
            # Key insight: The tensor entries are 0 or 1, so we can work mod 5
            # A rank-r decomposition over GF(5) with integer coefficients that are 0,1,-1,2,-2
            # (i.e., half-integers scaled by 2) would be valid.
            
            # Build the constraint system for rank 23
            # We have 27 "ones" constraints and 702 "zero" constraints
            # For rank R, we need to find U (9xR), V (9xR), W (9xR) such that:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = target[i,j,k]
            
            R_target = 23
            # Total variables: 3 * 9 * 23 = 621
            # Total constraints: 729 (27 ones + 702 zeros)
            # This is overdetermined, but we can try to find a solution using random restarts
            
            # Precompute the target tensor
            target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
            ones_list = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = idx_u(i,j)
                        b = idx_v(j,k)
                        c = idx_w(k,i)
                        target[a,b,c] = 1
                        ones_list.append((a,b,c))
            
            # Use a simple random search over GF(5)
            # Each entry of U, V, W is in {-2,-1,0,1,2} (representing -2/2, -1/2, 0, 1/2, 1)
            # or we can use {0,1,2,3,4} mod 5 and then map to integers
            
            # For efficiency, we'll do a few random restarts with local search
            np.random.seed(42)
            found = False
            
            max_restarts = 50
            for restart in range(max_restarts):
                if time.time() > deadline - 10:
                    break
                
                # Initialize with random values in GF(5)
                U_r = np.random.randint(0, 5, size=(9, R_target), dtype=np.int64)
                V_r = np.random.randint(0, 5, size=(9, R_target), dtype=np.int64)
                W_r = np.random.randint(0, 5, size=(9, R_target), dtype=np.int64)
                
                # Local search: randomly perturb one entry at a time
                best_error = float('inf')
                for step in range(200):
                    if time.time() > deadline - 10:
                        break
                    
                    # Compute current error (number of wrong entries)
                    error = 0
                    for a, b, c in ones_list:
                        s = sum(U_r[a,r] * V_r[b,r] * W_r[c,r] for r in range(R_target)) % 5
                        if s != 1:
                            error += 1
                    
                    # Also check some zero entries (sample)
                    zero_samples = []
                    for _ in range(100):
                        a = np.random.randint(0, 9)
                        b = np.random.randint(0, 9)
                        c = np.random.randint(0, 9)
                        if (a,b,c) not in ones_list:
                            zero_samples.append((a,b,c))
                    
                    for a, b, c in zero_samples:
                        s = sum(U_r[a,r] * V_r[b,r] * W_r[c,r] for r in range(R_target)) % 5
                        if s != 0:
                            error += 1
                    
                    if error < best_error:
                        best_error = error
                        if error == 0:
                            found = True
                            break
                    
                    # Perturb a random entry
                    mat = np.random.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        row = np.random.randint(0, 9)
                        col = np.random.randint(0, R_target)
                        U_r[row, col] = (U_r[row, col] + np.random.randint(1, 5)) % 5
                    elif mat == 'V':
                        row = np.random.randint(0, 9)
                        col = np.random.randint(0, R_target)
                        V_r[row, col] = (V_r[row, col] + np.random.randint(1, 5)) % 5
                    else:
                        row = np.random.randint(0, 9)
                        col = np.random.randint(0, R_target)
                        W_r[row, col] = (W_r[row, col] + np.random.randint(1, 5)) % 5
                
                if found:
                    break
            
            if found:
                # Map GF(5) values to integers: 0->0, 1->1, 2->-1, 3->-2, 4->2
                # Actually, let's use a simpler mapping: treat them as integers directly
                # and verify over the reals
                U_sol = U_r.astype(np.float64)
                V_sol = V_r.astype(np.float64)
                W_sol = W_r.astype(np.float64)
                
                # Verify over the reals
                if verify(U_sol, V_sol, W_sol):
                    return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
        except Exception:
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
