import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    """
    Finds a low-rank exact tensor decomposition for 3x3x6 matrix multiplication.
    
    Strategy:
    1. Generate the schoolbook rank-54 solution as a safe fallback.
    2. Construct a rank-45 decomposition by treating 3x3x6 as three parallel 3x3x2 multiplications.
       - Use a known rank-15 decomposition for 3x3x2 (derived from Strassen-like formulas or specific small matrix multiplication schemes).
       - Total rank = 3 * 15 = 45.
    3. Verify the decomposition exactly using integer arithmetic (or high-precision floats).
    4. If verification fails, return the schoolbook rank-54 solution.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = 3, 3, 6

    def generate_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = generate_schoolbook()
    best_rank = n * m * p

    def check_half_integer(arr, tol=1e-8):
        doubled = arr * 2.0
        rounded = np.round(doubled)
        if np.max(np.abs(doubled - rounded)) < tol:
            return rounded / 2.0
        return None

    def verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        """
        Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{i,i'} delta_{j,j'} delta_{k,k'}
        U: shape (n*m, R)
        V: shape (m*p, R)
        W: shape (p*n, R)
        """
        R = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n):
            for ip in range(n):
                for k in range(p):
                    for kp in range(p):
                        for j in range(m):
                            for jp in range(m):
                                expected = 1.0 if (i == ip and k == kp and j == jp) else 0.0
                                actual = 0.0
                                for r in range(R):
                                    actual += U_np[i * m + j, r] * V_np[jp * p + k, r] * W_np[kp * n + ip, r]
                                if abs(actual - expected) > tol:
                                    return False
        return True

    def build_3x3x2_rank15():
        """
        Construct a rank-15 decomposition for 3x3x2 matrix multiplication.
        U: 9x15, V: 6x15, W: 6x15
        
        We use a specific known decomposition. Since finding one from scratch is hard,
        we will use a structured approach. 
        One common way to get low rank for small matrices is to use bilinear forms.
        
        Let's try to construct one using the following logic:
        We want T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
        
        Actually, the tensor is T_{i,j,k} = A_{i,j} * B_{j,k}.
        The output is C_{i,k}.
        So we need: sum_j A_{i,j} B_{j,k} = C_{i,k}.
        
        Wait, the tensor for matrix multiplication M_{n,m,p} has entries T_{i,j,k} which are 1 if we are multiplying A_{i,j} and B_{j,k} to contribute to C_{i,k}.
        The decomposition is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = delta_{i,i'} delta_{j,j'} delta_{k,k'} ? No.
        
        The standard definition:
        T_{i,j,k} = 1 if (i,j,k) is a valid index tuple? No.
        The tensor T is defined such that C = A*B is given by C_{i,k} = sum_j A_{i,j} B_{j,k}.
        In tensor form, T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
        So T_{i,j,k} = 1 if the indices align? 
        Actually, the tensor T is of shape (n*m) x (m*p) x (p*n).
        T[(i*m+j), (j'*p+k), (k'*n+i')] = 1 if i=i', j=j', k=k'? 
        No, the multiplication is:
        C_{i,k} = sum_j A_{i,j} B_{j,k}.
        So the term A_{i,j} B_{j,k} contributes to C_{i,k}.
        Thus, the tensor entry T_{i*m+j, j*p+k, k*n+i} = 1.
        All other entries are 0.
        
        We need to find U, V, W such that:
        sum_r U[i*m+j, r] V[j'*p+k, r] W[k'*n+i', r] = T[i*m+j, j'*p+k, k'*n+i']
        
        This means:
        sum_r U[i,j,r] V[j',k,r] W[k',i',r] = 1 if i=i', j=j', k=k' else 0.
        
        Let's use a known rank-15 decomposition for 3x3x2.
        I will hardcode a specific solution if I can find one, or generate one via search if time permits.
        Given the time limit, hardcoding a verified solution is safer.
        
        However, I don't have a hardcoded rank-15 solution for 3x3x2 readily available in memory.
        I will attempt a quick search for a rank-15 solution for 3x3x2.
        If it fails, I will fall back to rank-16 or rank-18 or schoolbook.
        Actually, the rank of 3x3x2 is known to be 15? 
        Let's check: rank(3,3,2) <= rank(3,3,3) = 23? No, rank(3,3,3) is at least 19.
        rank(3,3,2) is at most 3*3*2 = 18.
        Is it 15? 
        A paper by Bläser et al. might have it.
        
        Let's try to search for a rank-15 solution for 3x3x2 quickly.
        """
        n2, m2, p2 = 3, 3, 2
        R = 15
        
        # Random search for rank-15 decomposition
        # We fix U and V to random integer values in {-1, 0, 1} and solve for W.
        # Then check if W is half-integer.
        
        # To increase success rate, we can try different random seeds.
        # But we must be fast.
        
        # Let's try a few random attempts.
        for attempt in range(100):
            if time.time() > deadline - 10:
                return None
                
            np.random.seed(attempt)
            U_r = np.random.randint(-1, 2, size=(n2 * m2, R))
            V_r = np.random.randint(-1, 2, size=(m2 * p2, R))
            
            W_r = np.zeros((p2 * n2, R))
            valid = True
            
            # Solve for W
            for i in range(n2):
                for k in range(p2):
                    # Equation: sum_r U[i,j,r] V[j',k,r] W[k,i,r] = delta_{j,j'}
                    # M[j, j', :] = U[i,j,:] * V[j',k,:]
                    # rhs[j, j'] = delta_{j,j'}
                    M = np.zeros((m2 * m2, R))
                    for j in range(m2):
                        for jp in range(m2):
                            M[j * m2 + jp, :] = U_r[i * m2 + j, :] * V_r[jp * p2 + k, :]
                    
                    rhs = np.zeros(m2 * m2)
                    for j in range(m2):
                        rhs[j * m2 + j] = 1.0
                    
                    try:
                        w_vec, residuals, rank_m, s = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        valid = False
                        break
                    
                    w_check = check_half_integer(w_vec)
                    if w_check is None:
                        valid = False
                        break
                    else:
                        W_r[k * n2 + i, :] = w_check
            
            if valid:
                # Verify
                if verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n2, m2, p2):
                    return (U_r.tolist(), V_r.tolist(), W_r.tolist())
        
        return None

    # Try to build the 3x3x2 rank-15 decomposition
    decomp = build_3x3x2_rank15()
    
    if decomp is not None:
        U_core, V_core, W_core = decomp
        R_core = len(U_core[0])
        total_rank = 3 * R_core
        
        if total_rank < best_rank:
            # Extend to 3x3x6
            U_tot = [[0] * total_rank for _ in range(n * m)]
            V_tot = [[0] * total_rank for _ in range(m * p)]
            W_tot = [[0] * total_rank for _ in range(p * n)]
            
            for block in range(3):
                offset = block * R_core
                for i in range(3):
                    for j in range(3):
                        for r in range(R_core):
                            U_tot[i * 3 + j][offset + r] = U_core[i * 3 + j][r]
                
                for j in range(3):
                    for k in range(2):
                        for r in range(R_core):
                            V_tot[j * 6 + block * 2 + k][offset + r] = V_core[j * 2 + k][r]
                
                for i in range(3):
                    for k in range(2):
                        for r in range(R_core):
                            W_tot[(block * 2 + k) * 3 + i][offset + r] = W_core[k * 3 + i][r]
            
            # Verify the full 3x3x6 decomposition
            if verify_decomposition(U_tot, V_tot, W_tot, n, m, p):
                best_result = {"U": U_tot, "V": V_tot, "W": W_tot}
                best_rank = total_rank

    return best_result
