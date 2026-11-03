import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Returns {'U', 'V', 'W'} where:
      U has shape (n*m, rank) = (9, rank)
      V has shape (m*p, rank) = (18, rank)
      W has shape (p*n, rank) = (18, rank)
    The decomposition satisfies:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Fallback: Schoolbook decomposition
    rank_sb = n * m * p  # 54
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try to find a better decomposition using Strassen-like techniques
    # Strategy: Decompose the 3x3 * 3x6 multiplication into 3 blocks of 3x3 * 3x2
    # For each 3x3 * 3x2 block, use a Strassen-based decomposition

    try:
        # Strassen decomposition for 2x2 * 2x2 -> 2x2 (rank 7)
        # Returns list of (U_dict, V_dict, W_dict) where each dict maps indices to coefficients
        def strassen_2x2():
            terms = []
            
            # M1 = (A00+A11)(B00+B11)
            terms.append((
                {(0,0):1, (1,1):1},  # U: A indices
                {(0,0):1, (1,1):1},  # V: B indices
                {(0,0):1, (0,1):1}   # W: C indices (k,i)
            ))
            
            # M2 = (A10+A11)B00
            terms.append((
                {(1,0):1, (1,1):1},
                {(0,0):1},
                {(0,1):1}
            ))
            
            # M3 = A00(B01-B11)
            terms.append((
                {(0,0):1},
                {(0,1):1, (1,1):-1},
                {(1,0):1, (1,1):-1}
            ))
            
            # M4 = A11(B10-B00)
            terms.append((
                {(1,1):1},
                {(1,0):1, (0,0):-1},
                {(0,0):-1, (1,0):1}
            ))
            
            # M5 = (A00+A01)B11
            terms.append((
                {(0,0):1, (0,1):1},
                {(1,1):1},
                {(1,1):1}
            ))
            
            # M6 = (A10-A00)(B00+B01)
            terms.append((
                {(1,0):1, (0,0):-1},
                {(0,0):1, (0,1):1},
                {(0,0):1}
            ))
            
            # M7 = (A01-A11)(B10+B11)
            terms.append((
                {(0,1):1, (1,1):-1},
                {(1,0):1, (1,1):1},
                {(0,0):1, (1,1):1}
            ))
            
            return terms
        
        strassen_terms = strassen_2x2()
        
        all_U = []
        all_V = []
        all_W = []
        
        # For each of the 3 blocks (k in {0,1}, {2,3}, {4,5})
        for b in range(3):
            k_offset = 2 * b
            
            # Part 1: 2x2 * 2x2 subproblem (rows 0,1 of A, rows 0,1 of B, cols 0,1 of B)
            # This covers C[0,0], C[0,1], C[1,0], C[1,1] contributions from A[0:2,0:2]*B[0:2,k_offset:k_offset+2]
            for u_dict, v_dict, w_dict in strassen_terms:
                U_term = [0] * (n * m)
                V_term = [0] * (m * p)
                W_term = [0] * (p * n)
                
                for (i, j), coeff in u_dict.items():
                    U_term[i * m + j] = coeff
                for (j, k), coeff in v_dict.items():
                    V_term[j * p + (k_offset + k)] = coeff
                for (k, i), coeff in w_dict.items():
                    W_term[(k_offset + k) * n + i] = coeff
                    
                all_U.append(U_term)
                all_V.append(V_term)
                all_W.append(W_term)
            
            # Part 2: Handle terms involving A[2,*] and B[0:2,:]
            # C[2,k] += A[2,j]*B[j,k] for j in {0,1}
            for j in range(2):
                for k in range(2):
                    U_term = [0] * (n * m)
                    V_term = [0] * (m * p)
                    W_term = [0] * (p * n)
                    
                    U_term[2 * m + j] = 1
                    V_term[j * p + (k_offset + k)] = 1
                    W_term[(k_offset + k) * n + 2] = 1
                    
                    all_U.append(U_term)
                    all_V.append(V_term)
                    all_W.append(W_term)
            
            # Part 3: Handle terms involving A[0:2,2] and B[2,:]
            # C[i,k] += A[i,2]*B[2,k] for i in {0,1}
            for i in range(2):
                for k in range(2):
                    U_term = [0] * (n * m)
                    V_term = [0] * (m * p)
                    W_term = [0] * (p * n)
                    
                    U_term[i * m + 2] = 1
                    V_term[2 * p + (k_offset + k)] = 1
                    W_term[(k_offset + k) * n + i] = 1
                    
                    all_U.append(U_term)
                    all_V.append(V_term)
                    all_W.append(W_term)
            
            # Part 4: Handle A[2,2]*B[2,:]
            # C[2,k] += A[2,2]*B[2,k]
            for k in range(2):
                U_term = [0] * (n * m)
                V_term = [0] * (m * p)
                W_term = [0] * (p * n)
                
                U_term[2 * m + 2] = 1
                V_term[2 * p + (k_offset + k)] = 1
                W_term[(k_offset + k) * n + 2] = 1
                
                all_U.append(U_term)
                all_V.append(V_term)
                all_W.append(W_term)
        
        rank = len(all_U)
        
        # Verify correctness
        if verify(all_U, all_V, all_W, n, m, p):
            if rank < rank_sb:
                return {"U": all_U, "V": all_V, "W": all_W}
                
    except Exception:
        pass

    # If time allows, try to further optimize using algebraic techniques
    if time.time() < deadline - 10:
        try:
            # Try to find a lower rank decomposition using random initialization and optimization
            # This is a heuristic approach that may or may not improve on the Strassen-based solution
            
            def random_decomposition(rank, n, m, p):
                """Generate a random rank-r decomposition."""
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                for c in range(rank):
                    # Random non-zero entries
                    u_idx = np.random.choice(n * m, size=3, replace=False)
                    v_idx = np.random.choice(m * p, size=3, replace=False)
                    w_idx = np.random.choice(p * n, size=3, replace=False)
                    
                    U[u_idx, c] = np.random.randn(3)
                    V[v_idx, c] = np.random.randn(3)
                    W[w_idx, c] = np.random.randn(3)
                
                return U, V, W
            
            def check_decomposition(U, V, W, n, m, p, tol=1e-6):
                """Check if the decomposition is valid."""
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(U.shape[1]):
                                s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            if abs(s - 1.0) > tol:
                                return False
                return True
            
            # Try ranks from 40 to 53
            for test_rank in range(40, 54):
                if time.time() > deadline - 5:
                    break
                
                for attempt in range(100):
                    if time.time() > deadline - 5:
                        break
                    
                    U, V, W = random_decomposition(test_rank, n, m, p)
                    
                    if check_decomposition(U, V, W, n, m, p):
                        # Convert to list format
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        
                        if verify(U_list, V_list, W_list, n, m, p):
                            return {"U": U_list, "V": V_list, "W": W_list}
                
                if time.time() > deadline - 10:
                    break
        
        except Exception:
            pass

    return {"U": U_sb, "V": V_sb, "W": W_sb}
