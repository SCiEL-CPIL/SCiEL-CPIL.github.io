import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def _check_deadline(start_time, deadline=145.0):
    """Check if we've exceeded the deadline."""
    return (time.time() - start_time) > deadline

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    Starts with rank-28 Strassen solution, then attempts to reduce rank.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Known rank-28 solution
    best = _strassen_28()
    
    # Try to find lower rank solutions
    for target_rank in [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16]:
        if _check_deadline(start_time):
            break
        result = _search_lower_rank(target_rank, start_time)
        if result is not None:
            best = result
            break
    
    return best

def _strassen_28():
    """Returns rank-28 solution using Strassen per 2x2 block."""
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    A_forms = [
        {0: 1, 3: 1},       # M1: a+d
        {2: 1, 3: 1},       # M2: c+d
        {0: 1},             # M3: a
        {3: 1},             # M4: d
        {0: 1, 1: 1},       # M5: a+b
        {2: 1, 0: -1},      # M6: c-a
        {1: 1, 3: -1}       # M7: b-d
    ]
    
    B_forms = [
        {0: 1, 3: 1},       # M1: x0+y1
        {0: 1},             # M2: x0
        {1: 1, 3: -1},      # M3: x1-y1
        {2: 1, 0: -1},      # M4: y0-x0
        {3: 1},             # M5: y1
        {0: 1, 1: 1},       # M6: x0+x1
        {2: 1, 3: 1}        # M7: y0+y1
    ]
    
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # M1, M4, M5, M7
        {2: 1, 4: 1},              # M3, M5
        {1: 1, 3: 1},              # M2, M4
        {0: 1, 1: -1, 2: 1, 5: 1}  # M1, M2, M3, M6
    ]

    for block in range(4):
        k0 = block * 2
        
        for s in range(7):
            r = block * 7 + s
            
            for idx, val in A_forms[s].items():
                i = idx // 2
                j = idx % 2
                U[i * m + j][r] = val
            
            for idx, val in B_forms[s].items():
                j = idx // 2
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global][r] = val
            
            for c_idx, coeffs in enumerate(C_coeffs):
                if s in coeffs:
                    coeff = coeffs[s]
                    i = c_idx // 2
                    k_local = c_idx % 2
                    k_global = k0 + k_local
                    W[k_global * n + i][r] = coeff

    return {"U": U, "V": V, "W": W}

def _search_lower_rank(target_rank, start_time):
    """
    Search for a rank-target_rank decomposition.
    Uses randomized approach with verification.
    Returns None if not found within time budget.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor T[a,b,c] where:
    # a = i*m + j (A index)
    # b = j*p + k (B index)  
    # c = k*n + i (C index)
    # T[a,b,c] = 1 if indices are consistent, else 0
    
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Try random initialization with integer/half-integer entries
    np.random.seed(42)
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if _check_deadline(start_time):
            break
        
        # Initialize with random integer/half-integer values
        U = np.random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(nm, target_rank))
        V = np.random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(mp, target_rank))
        W = np.random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], size=(pn, target_rank))
        
        # Compute decomposed tensor
        T_decomposed = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check if close to target
        if np.allclose(T_decomposed, T_target, atol=1e-9):
            # Convert to list of lists with integer/half-integer entries
            U_list = [[float(val) for val in row] for row in U]
            V_list = [[float(val) for val in row] for row in V]
            W_list = [[float(val) for val in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

# EVOLVE-BLOCK-END
