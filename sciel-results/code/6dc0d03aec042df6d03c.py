import time
import numpy as np
import random
from math import gcd
from functools import reduce
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook fallback
    def get_schoolbook():
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

    def validate(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m+j]
                    v_val = V_arr[j*p+k]
                    w_val = W_arr[k*n+i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def solve_diophantine(coeffs, target=1):
        """
        Find integer x such that dot(coeffs, x) = target.
        Returns None if gcd(coeffs) does not divide target.
        """
        g = reduce(gcd, [abs(c) for c in coeffs if c != 0], 0)
        if g == 0:
            return None
        if target % g != 0:
            return None
        
        best_x = None
        for r1 in range(len(coeffs)):
            if coeffs[r1] == 0: continue
            for r2 in range(r1 + 1, len(coeffs)):
                if coeffs[r2] == 0: continue
                g = gcd(abs(coeffs[r1]), abs(coeffs[r2]))
                if target % g == 0:
                    def egcd(a, b):
                        if b == 0:
                            return (a, 1, 0)
                        else:
                            g, x, y = egcd(b, a % b)
                            return (g, y, x - (a // b) * y)
                    
                    g, x, y = egcd(abs(coeffs[r1]), abs(coeffs[r2]))
                    scale = target // g
                    a = x * scale
                    if coeffs[r1] < 0: a = -a
                    b = y * scale
                    if coeffs[r2] < 0: b = -b
                    
                    x_vec = [0] * len(coeffs)
                    x_vec[r1] = a
                    x_vec[r2] = b
                    if sum(x_vec[i] * coeffs[i] for i in range(len(coeffs))) == target:
                        return x_vec
        return None

    def try_structured_search(target_rank, max_attempts=50000):
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            U = np.random.randint(-2, 3, size=(n*m, target_rank))
            V = np.random.randint(-2, 3, size=(m*p, target_rank))
            
            C = np.zeros((n, p, target_rank), dtype=int)
            for i in range(n):
                for k in range(p):
                    for r in range(target_rank):
                        s = 0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        C[i, k, r] = s
            
            valid = True
            W = np.zeros((p*n, target_rank), dtype=float)
            
            for i in range(n):
                for k in range(p):
                    coeffs = C[i, k, :]
                    non_zero = [abs(c) for c in coeffs if c != 0]
                    if not non_zero:
                        valid = False
                        break
                    g = reduce(gcd, non_zero)
                    if g not in (1, 2):
                        valid = False
                        break
                    
                    target = 1
                    if g == 2:
                        target = 2
                        scale_factor = 2.0
                    else:
                        scale_factor = 1.0
                        
                    w_vec = solve_diophantine(coeffs, target)
                    if w_vec is None:
                        valid = False
                        break
                    
                    for r in range(target_rank):
                        W[k*n+i, r] = w_vec[r] / scale_factor
                if not valid:
                    break
            
            if valid:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if validate(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    def try_strassen_based_rank27():
        """
        Try to construct a rank-27 decomposition using a structured ansatz
        based on Strassen's 2x2x2 algorithm extended to 2x3x5.
        
        The idea: Use 7 Strassen-like terms for the 2x2x2 core (rows 0-1, cols 0-1),
        then add 20 more terms to handle the remaining dimensions.
        
        Actually, let's think more carefully. We need to multiply:
        A (2x3) * B (3x5) = C (2x5)
        
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        For rank r, we need:
        C[i,k] = sum_{r=1}^{rank} U[i,j... wait, no.
        
        The tensor decomposition: T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}
        
        Wait, let me re-read the convention. W[k*n+i, r].
        
        So: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        This must equal sum_j A[i,j] * B[j,k] for all i,j,k.
        
        So the tensor T_{i,j,k} = delta (it's the multiplication tensor).
        T_{i,j,k} = 1 if we're computing C[i,k] using A[i,j] and B[j,k], but actually
        the multiplication tensor has T_{i,j,k} = 1 for the "correct" indices.
        
        Actually, the matrix multiplication tensor T satisfies:
        T_{i,j,k} = 1 if i in [0,n), j in [0,m), k in [0,p)
        and the product C[i,k] = sum_j T_{i,j,k} * A[i,j] * B[j,k]
        
        Wait no. The standard definition: The tensor for matrix multiplication of (n x m) * (m x p)
        is a 3-tensor of size (n*m) x (m*p) x (p*n) where:
        T_{(i,j), (j',k), (k',i')} = 1 if j = j' and i = i' and k = k'
        
        So T_{(i,j), (j',k), (k',i')} = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
        
        And a rank-r decomposition gives:
        T = sum_{r=1}^{rank} U_r * V_r * W_r
        where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
        
        So: delta_{j,j'} * delta_{i,i'} * delta_{k,k'} = sum_r U_r[i*m+j] * V_r[j'*p+k] * W_r[k'*n+i']
        
        Setting j'=j, k'=k, i'=i:
        1 = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
        
        For other combinations, the LHS is 0, so:
        0 = sum_r U_r[i*m+j] * V_r[j'*p+k] * W_r[k'*n+i']
        whenever (j,j',i,i',k,k') don't all match appropriately.
        
        This is a very constrained system. Let me try a different approach.
        
        Let's use the fact that we can write:
        sum_j A[i,j] B[j,k] = sum_r (sum_j U_r[i*m+j] A[i,j]) * (sum_j V_r[j*p+k] B[j,k]) * W_r[k*n+i]
        
        Wait, that's not right either. The decomposition is:
        C[i,k] = sum_r U_r[i*m+j]... no, j is summed over.
        
        Let me re-derive. We have:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        And we want: C[i,k] = sum_r f_r(A) * g_r(B) * h_r(C_position)
        where f_r, g_r are linear forms, and h_r is a scalar depending on (i,k).
        
        Actually: C[i,k] = sum_r (sum_j U_r[i*m+j] A[i,j]) * (sum_k' V_r[j'*p+k] B[j',k]) ... this doesn't work cleanly.
        
        Let me just use the direct condition:
        For all i in [0,n), j in [0,m), k in [0,p):
        1 = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
        
        And for "wrong" combinations:
        If we fix i, j, k and vary j' != j, or i' != i, or k' != k, we need:
        0 = sum_r U_r[i*m+j] * V_r[j'*p+k] * W_r[k'*n+i']
        
        This is a system of bilinear equations in the unknowns U, V, W.
        
        Let's try a specific structured ansatz for rank 27.
        
        Approach: Start with the schoolbook 30-term decomposition and try to merge 3 terms.
        
        Actually, let's try a simpler approach: use random search with a better initialization.
        """
        if time.time() > deadline:
            return None
        
        # Try rank 27 with a structured approach
        target_rank = 27
        
        # Initialize with schoolbook-like structure and perturb
        best_result = None
        
        for attempt in range(10000):
            if time.time() > deadline - 5:
                break
            
            # Create U, V with a mix of structured and random entries
            U = np.zeros((n*m, target_rank), dtype=float)
            V = np.zeros((m*p, target_rank), dtype=float)
            
            # First 7 columns: Strassen-like for the 2x2x2 subproblem (i=0,1; j=0,1; k=0,1)
            # Strassen for 2x2: 7 multiplications
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            
            # For our 2x3x5 problem, let's use a similar idea
            # Columns 0-6: handle the "core" 2x2x2 part
            # Columns 7-26: handle the remaining parts
            
            # Let's just use random initialization with small integers
            U = np.random.randint(-1, 2, size=(n*m, target_rank))
            V = np.random.randint(-1, 2, size=(m*p, target_rank))
            
            # Compute C[i,k,r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            C = np.zeros((n, p, target_rank), dtype=int)
            for i in range(n):
                for k in range(p):
                    for r in range(target_rank):
                        s = 0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        C[i, k, r] = s
            
            # For each (i,k), we need: sum_r C[i,k,r] * W[k*n+i, r] = 1
            # And we need W entries to be integers or half-integers
            
            valid = True
            W = np.zeros((p*n, target_rank), dtype=float)
            
            for i in range(n):
                for k in range(p):
                    coeffs = C[i, k, :]
                    non_zero = [abs(c) for c in coeffs if c != 0]
                    if not non_zero:
                        valid = False
                        break
                    g = reduce(gcd, non_zero)
                    if g not in (1, 2):
                        valid = False
                        break
                    
                    target = 1
                    if g == 2:
                        target = 2
                        scale_factor = 2.0
                    else:
                        scale_factor = 1.0
                        
                    w_vec = solve_diophantine(coeffs, target)
                    if w_vec is None:
                        valid = False
                        break
                    
                    for r in range(target_rank):
                        W[k*n+i, r] = w_vec[r] / scale_factor
                if not valid:
                    break
            
            if valid:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if validate(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    def try_rank27_linear_algebra():
        """
        Try to find a rank-27 decomposition using exact linear algebra.
        
        Key insight: If we fix U and V, then W is determined by linear equations.
        Specifically, for each (i,k):
        sum_r C[i,k,r] * W[k*n+i, r] = 1
        
        where C[i,k,r] = sum_j U[i*m+j, r] * V[j*p+k, r]
        
        So if we fix U and V, W is uniquely determined (if the system is consistent).
        
        The constraint is that W must have integer or half-integer entries.
        
        Let's try: fix U to be a fixed structured matrix, then search over V.
        """
        if time.time() > deadline:
            return None
        
        target_rank = 27
        
        # Try several structured U matrices
        U_structures = []
        
        # Structure 1: Identity-like
        U1 = np.zeros((n*m, target_rank))
        for idx in range(n*m):
            if idx < target_rank:
                U1[idx, idx] = 1
        U_structures.append(U1)
        
        # Structure 2: Block diagonal
        U2 = np.zeros((n*m, target_rank))
        for i in range(n):
            for j in range(m):
                col = i * m + j
                if col < target_rank:
                    U2[i*m+j, col] = 1
        U_structures.append(U2)
        
        # For each U structure, try random V and solve for W
        for U_struct in U_structures:
            for attempt in range(5000):
                if time.time() > deadline - 5:
                    return None
                
                V = np.random.randint(-1, 2, size=(m*p, target_rank))
                
                # Compute C[i,k,r] = sum_j U[i*m+j, r] * V[j*p+k, r]
                C = np.zeros((n, p, target_rank), dtype=int)
                for i in range(n):
                    for k in range(p):
                        for r in range(target_rank):
                            s = 0
                            for j in range(m):
                                s += int(U_struct[i*m+j, r]) * int(V[j*p+k, r])
                            C[i, k, r] = s
                
                valid = True
                W = np.zeros((p*n, target_rank), dtype=float)
                
                for i in range(n):
                    for k in range(p):
                        coeffs = C[i, k, :]
                        non_zero = [abs(c) for c in coeffs if c != 0]
                        if not non_zero:
                            valid = False
                            break
                        g = reduce(gcd, non_zero)
                        if g not in (1, 2):
                            valid = False
                            break
                        
                        target = 1
                        if g == 2:
                            target = 2
                            scale_factor = 2.0
                        else:
                            scale_factor = 1.0
                            
                        w_vec = solve_diophantine(coeffs, target)
                        if w_vec is None:
                            valid = False
                            break
                        
                        for r in range(target_rank):
                            W[k*n+i, r] = w_vec[r] / scale_factor
                    if not valid:
                        break
                
                if valid:
                    U_list = U_struct.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    if validate(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # First try the structured rank-27 approaches
    result = try_rank27_linear_algebra()
    if result is not None:
        return result
    
    result = try_strassen_based_rank27()
    if result is not None:
        return result

    # Try ranks from lower bound upwards with random search
    for rank in range(15, 30):
        if time.time() > deadline:
            break
        result = try_structured_search(rank, max_attempts=10000)
        if result is not None:
            return result

    # Fallback to schoolbook
    return get_schoolbook()
