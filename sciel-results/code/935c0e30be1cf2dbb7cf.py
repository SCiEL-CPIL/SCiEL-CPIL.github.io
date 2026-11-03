import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Rank-55 solution using Strassen for 2x2 blocks
    def build_rank55():
        rank = 55
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        c = 0
        # Part A: j=0
        for i in range(n):
            for k in range(p):
                U[i*m+0, c] = 1
                V[0*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
        
        # Part B: A1*B1 + A2*B2 with Strassen for 2x2 blocks
        for b in range(5):
            k0 = 2*b
            k1 = 2*b+1
            
            u11, u12 = 0*m+1, 0*m+2
            u21, u22 = 1*m+1, 1*m+2
            
            v11, v12 = 1*p+k0, 1*p+k1
            v21, v22 = 2*p+k0, 2*p+k1
            
            w00, w01 = k0*n+0, k1*n+0
            w10, w11 = k0*n+1, k1*n+1
            
            # Strassen for 2x2 * 2x2
            U[u11,c]+=1; U[u22,c]+=1; V[v11,c]+=1; V[v22,c]+=1; W[w00,c]+=1; W[w11,c]+=1; c+=1
            U[u21,c]+=1; U[u22,c]+=1; V[v11,c]+=1; W[w10,c]+=1; W[w11,c]+=-1; c+=1
            U[u11,c]+=1; V[v12,c]+=1; V[v22,c]+=-1; W[w01,c]+=1; W[w11,c]+=1; c+=1
            U[u22,c]+=1; V[v21,c]+=1; V[v11,c]+=-1; W[w00,c]+=1; W[w10,c]+=1; c+=1
            U[u11,c]+=1; U[u12,c]+=1; V[v22,c]+=1; W[w00,c]+=-1; W[w01,c]+=1; c+=1
            U[u21,c]+=1; U[u11,c]+=-1; V[v11,c]+=1; V[v12,c]+=1; W[w11,c]+=1; c+=1
            U[u12,c]+=1; U[u22,c]+=-1; V[v21,c]+=1; V[v22,c]+=1; W[w00,c]+=1; c+=1
        
        return U, V, W, rank
    
    U55, V55, W55, rank55 = build_rank55()
    
    # Verify rank-55 solution
    def verify(U, V, W, T, atol=1e-6):
        r = U.shape[1]
        for a in range(U.shape[0]):
            for b in range(V.shape[0]):
                for cc in range(W.shape[0]):
                    val = np.sum(U[a,:r] * V[b,:r] * W[cc,:r])
                    if abs(val - T[a,b,cc]) > atol:
                        return False
        return True
    
    if not verify(U55, V55, W55, T):
        # Fallback to schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try to find rank-10 decomposition for M_{2,2,3}
    # M_{2,2,3}: A is 2x2, B is 2x3, C is 3x2
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # i in 0..1, j in 0..1, k in 0..2
    
    T_223 = np.zeros((4, 6, 6))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                a_idx = i*2 + j
                b_idx = j*3 + k
                c_idx = k*2 + i
                T_223[a_idx, b_idx, c_idx] = 1.0
    
    def verify_223(U, V, W, r, atol=1e-6):
        for a in range(4):
            for b in range(6):
                for cc in range(6):
                    val = np.sum(U[a,:r] * V[b,:r] * W[cc,:r])
                    if abs(val - T_223[a,b,cc]) > atol:
                        return False
        return True
    
    def find_m223_rank10(time_budget):
        """Search for rank-10 decomposition of M_{2,2,3}"""
        best_result = None
        best_score = float('inf')
        
        # Start with rank-11 solution (Strassen for k=0,1 + trivial for k=2)
        U_223 = np.zeros((4, 11))
        V_223 = np.zeros((6, 11))
        W_223 = np.zeros((6, 11))
        
        c = 0
        u00, u01, u10, u11 = 0, 1, 2, 3
        v00, v01, v10, v11 = 0, 1, 2, 3
        w00, w01, w10, w11 = 0, 1, 2, 3
        
        # Strassen part for k=0,1
        U_223[u00,c]+=1; U_223[u11,c]+=1; V_223[v00,c]+=1; V_223[v11,c]+=1; W_223[w00,c]+=1; W_223[w11,c]+=1; c+=1
        U_223[u10,c]+=1; U_223[u11,c]+=1; V_223[v00,c]+=1; W_223[w10,c]+=1; W_223[w11,c]+=-1; c+=1
        U_223[u00,c]+=1; V_223[v01,c]+=1; V_223[v11,c]+=-1; W_223[w01,c]+=1; W_223[w11,c]+=1; c+=1
        U_223[u11,c]+=1; V_223[v10,c]+=1; V_223[v00,c]+=-1; W_223[w00,c]+=1; W_223[w10,c]+=1; c+=1
        U_223[u00,c]+=1; U_223[u01,c]+=1; V_223[v11,c]+=1; W_223[w00,c]+=-1; W_223[w01,c]+=1; c+=1
        U_223[u10,c]+=1; U_223[u00,c]+=-1; V_223[v00,c]+=1; V_223[v01,c]+=1; W_223[w11,c]+=1; c+=1
        U_223[u01,c]+=1; U_223[u11,c]+=-1; V_223[v10,c]+=1; V_223[v11,c]+=1; W_223[w00,c]+=1; c+=1
        
        # Trivial part for k=2
        U_223[u00,c]=1; V_223[v00+3,c]=1; W_223[w00+3,c]=1; c+=1
        U_223[u01,c]=1; V_223[v10+3,c]=1; W_223[w00+3,c]=1; c+=1
        U_223[u10,c]=1; V_223[v00+3,c]=1; W_223[w10+3,c]=1; c+=1
        U_223[u11,c]=1; V_223[v10+3,c]=1; W_223[w10+3,c]=1; c+=1
        
        if not verify_223(U_223, V_223, W_223, 11):
            return None
        
        # Try to reduce to rank 10 by finding a linear combination that cancels one column
        # We'll try random perturbations and local search
        np.random.seed(42)
        
        for attempt in range(200):
            if elapsed() > start_time + time_budget:
                break
            
            # Start from rank-11 and try to find a null vector in the column space
            # that allows us to merge two columns
            U_curr = U_223.copy()
            V_curr = V_223.copy()
            W_curr = W_223.copy()
            
            # Try to find a linear combination of columns that is zero
            # This is equivalent to finding a rank-10 decomposition
            
            # Use randomized approach: try to find coefficients alpha, beta such that
            # alpha*col_i + beta*col_j can be absorbed
            for i in range(10):
                for j in range(i+1, 11):
                    if elapsed() > start_time + time_budget:
                        break
                    
                    # Try to find a combination that zeros out the difference
                    # We'll use a simple local search
                    U_test = U_curr.copy()
                    V_test = V_curr.copy()
                    W_test = W_curr.copy()
                    
                    # Try to eliminate column j by expressing it in terms of others
                    # This is complex, so we'll try a different approach
                    # Use gradient descent to minimize the tensor error for rank 10
                    
                    U10 = np.random.randn(4, 10) * 0.1 + U_curr[:, :10]
                    V10 = np.random.randn(6, 10) * 0.1 + V_curr[:, :10]
                    W10 = np.random.randn(6, 10) * 0.1 + W_curr[:, :10]
                    
                    # Local optimization
                    lr = 0.01
                    for step in range(100):
                        if elapsed() > start_time + time_budget:
                            break
                        
                        # Compute gradient
                        U_grad = np.zeros_like(U10)
                        V_grad = np.zeros_like(V10)
                        W_grad = np.zeros_like(W10)
                        
                        for a in range(4):
                            for b in range(6):
                                for cc in range(6):
                                    target = T_223[a, b, cc]
                                    current = np.sum(U10[a, :] * V10[b, :] * W10[cc, :])
                                    err = current - target
                                    
                                    for r in range(10):
                                        U_grad[a, r] += err * V10[b, r] * W10[cc, r]
                                        V_grad[b, r] += err * U10[a, r] * W10[cc, r]
                                        W_grad[cc, r] += err * U10[a, r] * V10[b, r]
                        
                        U10 -= lr * U_grad
                        V10 -= lr * V_grad
                        W10 -= lr * W_grad
                    
                    if verify_223(U10, V10, W10, 10):
                        return U10, V10, W10
        
        return None
    
    # Search for rank-10 M_{2,2,3}
    m223_result = None
    time_for_search = min(30, max(5, deadline - elapsed() - 5))
    if time_for_search > 0:
        m223_result = find_m223_rank10(time_for_search)
    
    if m223_result is not None:
        U_223, V_223, W_223 = m223_result
        # Build rank-54 solution
        # M_{2,2,10} = 3 * M_{2,2,3} + 1 * M_{2,2,1}
        # M_{2,2,1} has rank 4
        # Total rank = 3*10 + 4 = 34 for the 2x2 part
        # Plus 20 for j=0 part = 54
        
        rank_54 = 54
        U_54 = np.zeros((n*m, rank_54))
        V_54 = np.zeros((m*p, rank_54))
        W_54 = np.zeros((p*n, rank_54))
        
        c = 0
        # Part A: j=0 (same as before)
        for i in range(n):
            for k in range(p):
                U_54[i*m+0, c] = 1
                V_54[0*p+k, c] = 1
                W_54[k*n+i, c] = 1
                c += 1
        
        # Part B: Use M_{2,2,3} decomposition for 3 blocks + M_{2,2,1} for 1 block
        # M_{2,2,3} covers k=0,1,2 with rank 10
        # M_{2,2,1} covers k=3 with rank 4
        # We need to tile this for 5 blocks of 2 columns each
        
        # For each group of 3 k-values, use M_{2,2,3}
        # k=0,1,2: use M_{2,2,3} (rank 10)
        # k=3,4,5: use M_{2,2,3} (rank 10)
        # k=6,7,8: use M_{2,2,3} (rank 10)
        # k=9: use M_{2,2,1} (rank 4)
        
        # This gives 3*10 + 4 = 34 for the 2x2 part
        
        # Actually, let's reconsider. We have 10 k-values.
        # We can do 3 blocks of M_{2,2,3} (covering k=0-2, 3-5, 6-8) and 1 block of M_{2,2,1} (k=9)
        # That's 3*10 + 4 = 34
        
        # M_{2,2,3} decomposition: U is 4x10, V is 6x10, W is 6x10
        # For block starting at k_offset:
        #   A indices: i*2+j for i in 0..1, j in 1..2 (U[1], U[2], U[4], U[5])
        #   B indices: j*3+k for j in 1..2, k in 0..2 (V[10+k_offset], V[10+k_offset+1], V[10+k_offset+2], V[20+k_offset], V[20+k_offset+1], V[20+k_offset+2])
        #   C indices: k*2+i for k in 0..2, i in 0..1 (W[2*k_offset+i], W[2*k_offset+1+i], ..., W[2*k_offset+5])
        
        for block in range(3):
            k_offset = block * 3
            for r in range(10):
                # U part
                for a in range(4):
                    i, j = divmod(a, 2)
                    if j == 0:
                        continue  # j=0 is handled in Part A
                    U_54[i*m+j, c] = U_223[a, r]
                
                # V part
                for b in range(6):
                    j, k = divmod(b, 3)
                    if j == 0:
                        continue  # j=0 is handled in Part A
                    V_54[j*p+k_offset+k, c] = V_223[b, r]
                
                # W part
                for cc in range(6):
                    k, i = divmod(cc, 2)
                    W_54[(k_offset+k)*n+i, c] = W_223[cc, r]
                
                c += 1
        
        # M_{2,2,1} for k=9
        # A indices: i*2+j for i in 0..1, j in 1..2
        # B indices: j*1+0 for j in 1..2 (i.e., j=1,k=9 and j=2,k=9)
        # C indices: 0*2+i for i in 0..1 (i.e., k=9,i=0 and k=9,i=1)
        
        # Trivial decomposition for M_{2,2,1}
        # (i,j,k) = (0,1,9), (0,2,9), (1,1,9), (1,2,9)
        for i in range(2):
            for j in range(1, 3):
                U_54[i*m+j, c] = 1
                V_54[j*p+9, c] = 1
                W_54[9*n+i, c] = 1
                c += 1
        
        if verify(U_54, V_54, W_54, T):
            U_list = [[float(x) for x in row] for row in U_54]
            V_list = [[float(x) for x in row] for row in V_54]
            W_list = [[float(x) for x in row] for row in W_54]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to rank-55
    U_list = [[float(x) for x in row] for row in U55]
    V_list = [[float(x) for x in row] for row in V55]
    W_list = [[float(x) for x in row] for row in W55]
    return {"U": U_list, "V": V_list, "W": W_list}
