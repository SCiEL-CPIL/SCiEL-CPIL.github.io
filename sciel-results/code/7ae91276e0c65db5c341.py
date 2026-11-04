import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (U, V, W) where U, V, W are 4-element lists.
    U[i*2+j], V[j*2+k], W[k*2+i].
    The condition is: sum_c U_c[i*2+j] * V_c[j*2+k] * W_c[k*2+i] = 1 for all i,j,k.
    """
    # Strassen's 7 products for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0] = M1 - M4 + M5 - M7
    # C[1] = M3 + M5
    # C[2] = M2 + M4
    # C[3] = M1 - M2 + M6 - M7
    
    terms = []
    
    # M1: A: a+d -> U=[1,0,0,1]; B: e+h -> V=[1,0,0,1]; C: +1 to C[0], +1 to C[3] -> W=[1,0,0,1]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2: A: c+d -> U=[0,0,1,1]; B: e -> V=[1,0,0,0]; C: +1 to C[2] -> W=[0,0,1,0]
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    
    # M3: A: a -> U=[1,0,0,0]; B: f-h -> V=[0,1,0,-1]; C: +1 to C[1] -> W=[0,1,0,0]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    
    # M4: A: d -> U=[0,0,0,1]; B: g-e -> V=[0,0,1,-1]; C: -1 to C[0], +1 to C[2] -> W=[-1,0,1,0]
    terms.append(([0,0,0,1], [0,0,1,-1], [-1,0,1,0]))
    
    # M5: A: a+b -> U=[1,1,0,0]; B: h -> V=[0,0,0,1]; C: +1 to C[0], +1 to C[1] -> W=[1,1,0,0]
    terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    
    # M6: A: c-a -> U=[-1,0,1,0]; B: e+f -> V=[1,1,0,0]; C: +1 to C[3] -> W=[0,0,0,1]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7: A: b-d -> U=[0,1,0,-1]; B: g+h -> V=[0,0,1,1]; C: -1 to C[0], -1 to C[3] -> W=[-1,0,0,-1]
    terms.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,-1]))
    
    return terms

def _kron_4x4_terms():
    """
    Generate 49 terms for 4x4x4 matrix multiplication using Kronecker product of Strassen.
    Each term is (U, V, W) where U, V, W are 16-element lists.
    """
    base_terms = _strassen_2x2_terms()
    terms_4x4 = []
    
    for t1 in base_terms:
        for t2 in base_terms:
            U1, V1, W1 = t1
            U2, V2, W2 = t2
            
            U = [0]*16
            V = [0]*16
            W = [0]*16
            
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            idx = (i1*2+i2)*4 + (j1*2+j2)
                            U[idx] = U1[i1*2+j1] * U2[i2*2+j2]
                            V[idx] = V1[i1*2+j1] * V2[i2*2+j2]
                            W[idx] = W1[i1*2+j1] * W2[i2*2+j2]
            
            terms_4x4.append((U, V, W))
    
    return terms_4x4

def _verify_decomposition(U, V, W, r, n=4, m=4, p=7):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _peel_terms(U, V, W, n=4, m=4, p=7, deadline=None):
    """
    Try to peel linearly dependent terms from the decomposition.
    Returns a reduced (U, V, W) if successful.
    """
    r = len(U[0])
    
    # We want to find a minimal subset of columns that still gives the correct tensor.
    # This is a combinatorial problem, so we use a greedy approach.
    
    # First, check if any single term is redundant (i.e., removing it still gives the correct tensor)
    # This is expensive, so we do it carefully.
    
    # For each term c, check if removing it still gives the correct tensor.
    # If so, remove it and repeat.
    
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_r = r
    
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for c in range(current_r):
            if deadline is not None and time.time() > deadline:
                break
            
            # Temporarily remove term c
            temp_U = [row[:c] + row[c+1:] for row in current_U]
            temp_V = [row[:c] + row[c+1:] for row in current_V]
            temp_W = [row[:c] + row[c+1:] for row in current_W]
            
            if _verify_decomposition(temp_U, temp_V, temp_W, current_r - 1, n, m, p):
                # Term c is redundant
                current_U = temp_U
                current_V = temp_V
                current_W = temp_W
                current_r -= 1
                improved = True
                break
    
    return current_U, current_V, current_W, current_r

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Step 1: Build 4x4x4 decomposition from Kronecker product of Strassen
    kron_terms = _kron_4x4_terms()
    r_444 = len(kron_terms)  # 49
    
    U_444 = [[0]*r_444 for _ in range(16)]
    V_444 = [[0]*r_444 for _ in range(16)]
    W_444 = [[0]*r_444 for _ in range(16)]
    
    for c, (U, V, W) in enumerate(kron_terms):
        for idx in range(16):
            U_444[idx][c] = U[idx]
            V_444[idx][c] = V[idx]
            W_444[idx][c] = W[idx]
    
    # Step 2: Build 4x4x3 decomposition using schoolbook
    r_443 = 4*4*3  # 48
    U_443 = [[0]*r_443 for _ in range(16)]
    V_443 = [[0]*r_443 for _ in range(28)]  # 4*7=28
    W_443 = [[0]*r_443 for _ in range(28)]  # 7*4=28
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k_rel in range(3):
                k_abs = k_rel + 4  # columns 4,5,6
                U_443[i*4+j][col] = 1
                V_443[j*7+k_abs][col] = 1
                W_443[k_abs*4+i][col] = 1
                col += 1
    
    # Step 3: Combine
    total_rank = r_444 + r_443  # 49 + 48 = 97
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    # Embed 4x4x4 part
    for c in range(r_444):
        for idx in range(16):
            U[idx][c] = U_444[idx][c]
            V[idx][c] = V_444[idx][c]
            W[idx][c] = W_444[idx][c]
    
    # Add 4x4x3 part
    for c in range(r_443):
        for idx in range(16):
            U[idx][r_444+c] = U_443[idx][c]
        for idx in range(28):
            V[idx][r_444+c] = V_443[idx][c]
            W[idx][r_444+c] = W_443[idx][c]
    
    # Step 4: Verify
    if _verify_decomposition(U, V, W, total_rank):
        # Step 5: Try to peel terms
        U, V, W, reduced_rank = _peel_terms(U, V, W, n, m, p, deadline)
        return {"U": U, "V": V, "W": W}
    
    # Fallback: full schoolbook
    rank = 4 * 4 * 7
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(7):
                U[i*4+j][col] = 1
                V[j*7+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
