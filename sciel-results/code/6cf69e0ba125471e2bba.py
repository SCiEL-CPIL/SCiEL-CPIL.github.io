"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140

    def elapsed():
        return time.time() - start_time

    def validate(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p  # 72
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

    best_U, best_V, best_W, best_rank = schoolbook()

    # Strategy: Try to find a valid decomposition with rank < 72
    # using a structured approach based on Laderman for 3x3x3 blocks
    
    # First, let's get a CORRECT Laderman by trying known formulations
    
    def try_laderman_formulation():
        """Try several known Laderman formulations and return the first valid one."""
        
        # Formulation 1: Based on standard Laderman
        # The 23 terms are defined as follows (0-indexed):
        
        formulations = []
        
        # Standard Laderman algorithm terms
        # Each term is (u_entries, v_entries, w_entries)
        # u_entries: list of (idx, coeff) where idx = i*3+j for A[i,j]
        # v_entries: list of (idx, coeff) where idx = j*3+k for B[j,k]  
        # w_entries: list of (idx, coeff) where idx = k*3+i for C[k,i]
        
        # Term 0
        formulations.append((
            [(0,1),(4,1),(8,1)],   # A[0,0], A[1,1], A[2,2]
            [(0,1),(4,1),(8,1)],   # B[0,0], B[1,1], B[2,2]
            [(0,1),(4,1),(8,1)]    # C[0,0], C[1,1], C[2,2]
        ))
        # Term 1
        formulations.append((
            [(1,1),(5,1),(6,1)],   # A[0,1], A[1,2], A[2,0]
            [(1,1),(5,1),(6,1)],   # B[0,1], B[1,2], B[2,0]
            [(1,1),(5,1),(6,1)]    # C[0,1], C[1,2], C[2,0]
        ))
        # Term 2
        formulations.append((
            [(2,1),(3,1),(7,1)],   # A[0,2], A[1,0], A[2,1]
            [(2,1),(3,1),(7,1)],   # B[0,2], B[1,0], B[2,1]
            [(2,1),(3,1),(7,1)]    # C[0,2], C[1,0], C[2,1]
        ))
        
        # Terms 3-8: row-pair terms
        # (A[i,a]+A[i,b])*(B[j,a]+B[j,b]) -> C[i,a]-C[i,b]
        formulations.append((
            [(0,1),(1,1)],   # A[0,0]+A[0,1]
            [(3,1),(4,1)],   # B[1,0]+B[1,1]
            [(0,1),(1,-1)]   # C[0,0]-C[0,1]
        ))
        formulations.append((
            [(3,1),(4,1)],   # A[1,0]+A[1,1]
            [(0,1),(1,1)],   # B[0,0]+B[0,1]
            [(3,1),(4,-1)]   # C[1,0]-C[1,1]
        ))
        formulations.append((
            [(6,1),(7,1)],   # A[2,0]+A[2,1]
            [(6,1),(7,1)],   # B[2,0]+B[2,1]
            [(6,1),(7,-1)]   # C[2,0]-C[2,1]
        ))
        formulations.append((
            [(0,1),(2,1)],   # A[0,0]+A[0,2]
            [(6,1),(8,1)],   # B[2,0]+B[2,2]
            [(0,1),(2,-1)]   # C[0,0]-C[0,2]
        ))
        formulations.append((
            [(3,1),(5,1)],   # A[1,0]+A[1,2]
            [(6,1),(8,1)],   # B[2,0]+B[2,2]
            [(3,1),(5,-1)]   # C[1,0]-C[1,2]
        ))
        formulations.append((
            [(6,1),(8,1)],   # A[2,0]+A[2,2]
            [(0,1),(2,1)],   # B[0,0]+B[0,2]
            [(6,1),(8,-1)]   # C[2,0]-C[2,2]
        ))
        
        # Terms 9-14: more row-pair terms
        formulations.append((
            [(1,1),(2,1)],   # A[0,1]+A[0,2]
            [(4,1),(5,1)],   # B[1,1]+B[1,2]
            [(1,1),(2,-1)]   # C[0,1]-C[0,2]
        ))
        formulations.append((
            [(4,1),(5,1)],   # A[1,1]+A[1,2]
            [(1,1),(2,1)],   # B[0,1]+B[0,2]
            [(4,1),(5,-1)]   # C[1,1]-C[1,2]
        ))
        formulations.append((
            [(7,1),(8,1)],   # A[2,1]+A[2,2]
            [(4,1),(5,1)],   # B[1,1]+B[1,2]
            [(7,1),(8,-1)]   # C[2,1]-C[2,2]
        ))
        
        # Terms 15-20: column-pair terms
        # (A[i,a]+A[j,a])*(B[i,a]+B[j,a]) -> C[i,a]-C[j,a]
        formulations.append((
            [(0,1),(3,1)],   # A[0,0]+A[1,0]
            [(0,1),(3,1)],   # B[0,0]+B[1,0]
            [(0,1),(3,-1)]   # C[0,0]-C[1,0]
        ))
        formulations.append((
            [(3,1),(6,1)],   # A[1,0]+A[2,0]
            [(3,1),(6,1)],   # B[1,0]+B[2,0]
            [(3,1),(6,-1)]   # C[1,0]-C[2,0]
        ))
        formulations.append((
            [(6,1),(0,1)],   # A[2,0]+A[0,0]
            [(6,1),(0,1)],   # B[2,0]+B[0,0]
            [(6,1),(0,-1)]   # C[2,0]-C[0,0]
        ))
        formulations.append((
            [(1,1),(4,1)],   # A[0,1]+A[1,1]
            [(1,1),(4,1)],   # B[0,1]+B[1,1]
            [(1,1),(4,-1)]   # C[0,1]-C[1,1]
        ))
        formulations.append((
            [(4,1),(7,1)],   # A[1,1]+A[2,1]
            [(4,1),(7,1)],   # B[1,1]+B[2,1]
            [(4,1),(7,-1)]   # C[1,1]-C[2,1]
        ))
        formulations.append((
            [(7,1),(1,1)],   # A[2,1]+A[0,1]
            [(7,1),(1,1)],   # B[2,1]+B[0,1]
            [(7,1),(1,-1)]   # C[2,1]-C[0,1]
        ))
        
        # Terms 21-22: remaining
        formulations.append((
            [(2,1),(5,1)],   # A[0,2]+A[1,2]
            [(2,1),(5,1)],   # B[0,2]+B[1,2]
            [(2,1),(5,-1)]   # C[0,2]-C[1,2]
        ))
        formulations.append((
            [(5,1),(8,1)],   # A[1,2]+A[2,2]
            [(5,1),(8,1)],   # B[1,2]+B[2,2]
            [(5,1),(8,-1)]   # C[1,2]-C[2,2]
        ))
        
        # That's 23 terms. Let's verify.
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))
        
        for t, (u_entries, v_entries, w_entries) in enumerate(formulations):
            for idx, coeff in u_entries:
                U[idx, t] = coeff
            for idx, coeff in v_entries:
                V[idx, t] = coeff
            for idx, coeff in w_entries:
                W[idx, t] = coeff
        
        # Verify
        valid = True
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = np.dot(U[i*3+j], V[j*3+k] * W[k*3+i])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U, V, W
        return None
    
    U_l, V_l, W_l = try_laderman_formulation()
    
    if U_l is not None:
        # Build 3x3 * 3x8 using Laderman for two 3x3 blocks + schoolbook for 3x2
        rank = 23 + 23 + 18  # 64
        U_big = np.zeros((9, rank))
        V_big = np.zeros((24, rank))
        W_big = np.zeros((24, rank))
        
        # Block 1: columns 0-2
        U_big[:, 0:23] = U_l
        for j in range(3):
            for k in range(3):
                V_big[j*3+k, 0:23] = V_l[j*3+k]
                W_big[k*3+j, 0:23] = W_l[k*3+j]
        
        # Block 2: columns 3-5
        U_big[:, 23:46] = U_l
        for j in range(3):
            for k in range(3):
                V_big[j*3+(k+3), 23:46] = V_l[j*3+k]
                W_big[(k+3)*3+j, 23:46] = W_l[k*3+j]
        
        # Block 3: columns 6-7, schoolbook
        col_offset = 46
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U_big[i*3+j, col_offset] = 1
                    V_big[j*3+(k+6), col_offset] = 1
                    W_big[(k+6)*3+i, col_offset] = 1
                    col_offset += 1
        
        # Verify full decomposition
        valid = True
        for i in range(3):
            for j in range(3):
                for k in range(8):
                    val = np.dot(U_big[i*3+j], V_big[j*3+k] * W_big[k*3+i])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_U = U_big.tolist()
            best_V = V_big.tolist()
            best_W = W_big.tolist()
            best_rank = rank
    else:
        # If Laderman failed, try a simpler improvement
        # Use schoolbook but see if we can merge any terms
        # For now, just return schoolbook
        pass
    
    if elapsed() > deadline - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

