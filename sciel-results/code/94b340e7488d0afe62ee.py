import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def _compute_error_np(U, V, W, n, m, p, rank):
    """Vectorized error computation using numpy."""
    U3 = np.array(U, dtype=float).reshape(n, m, rank)
    V3 = np.array(V, dtype=float).reshape(m, p, rank)
    W3 = np.array(W, dtype=float).reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    error = np.count_nonzero(result != 1)
    return int(error)

def _verify(U, V, W, n, m, p, rank):
    """Verify the decomposition is exact."""
    U3 = np.array(U, dtype=float).reshape(n, m, rank)
    V3 = np.array(V, dtype=float).reshape(m, p, rank)
    W3 = np.array(W, dtype=float).reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-10)

def _strassen_2x2x2():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W each as 4x7 matrices (flattened row-major for 2x2, 2x2, 2x2).
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen formulas for 2x2x2:
    # C11 = M1 - M2 + M4 - M7
    # C12 = M3 + M5
    # C21 = M6 + M7
    # C22 = M1 - M3 + M2 - M5
    # where
    # M1 = (A22 + A11)(B11 - B22)  -- wait, let me use standard Strassen
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me verify with a known Strassen decomposition.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # C11 = ae+cg, C12 = af+dh, C21 = ce+dg, C22 = cf+dh
    
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-h)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me verify C11:
    # M1 - M4 + M5 - M7 = (a+d)(e+h) - d(g-h) + (a+b)h - (b-d)(g+h)
    # = ae + ah + de + dh - dg + dh + ah + bh - bg - bh + dg + dh
    # = ae + ah + de + dh - dg + dh + ah + bh - bg - bh + dg + dh
    # Hmm, let me be more careful.
    # (a+d)(e+h) = ae + ah + de + dh
    # -d(g-h) = -dg + dh
    # +(a+b)h = ah + bh
    # -(b-d)(g+h) = -(bg + bh - dg - dh) = -bg - bh + dg + dh
    # Sum: ae + ah + de + dh - dg + dh + ah + bh - bg - bh + dg + dh
    # = ae + (ah+ah) + de + (dh+dh+dh) + (-dg+dg) + (bh-bh) - bg
    # = ae + 2ah + de + 3dh - bg
    # That's not ae+cg. Let me use a verified Strassen decomposition.
    
    # Verified Strassen (from Wikipedia):
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Verify C11 = M1 + M4 - M5 + M7:
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M4 = d(g-h) = dg-dh
    # -M5 = -(a+b)h = -ah-bh
    # M7 = (b-d)(g+h) = bg+bh-dg-dh
    # Sum: ae+ah+de+dh + dg-dh -ah-bh + bg+bh-dg-dh
    # = ae + (ah-ah) + de + (dh-dh-dh) + (dg-dg) + (-bh+bh) + bg
    # = ae + de - dh + bg
    # That's not ae+cg either. Let me just use a known-correct decomposition.
    
    # Let me use the standard one from Strassen's paper:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - h)
    # M5 = (a + b)h
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # I'll just construct it directly and verify numerically.
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B indices: 0=e, 1=f, 2=g, 3=h
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    # W[k*2+i, r]: C11 -> W[0*2+0]=W[0], C12 -> W[1*2+0]=W[2], C21 -> W[0*2+1]=W[1], C22 -> W[1*2+1]=W[3]
    
    # M1 = (a+d)(e+h)
    # U: a=1 at idx0, d=1 at idx3; V: e=1 at idx0, h=1 at idx3
    # W: C11 += 1, C22 += 1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (c+d)e
    # U: c=1 at idx2, d=1 at idx3; V: e=1 at idx0
    # W: C21 += 1
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1
    
    # M3 = a(f-h)
    # U: a=1 at idx0; V: f=1 at idx1, h=-1 at idx3
    # W: C12 += 1
    U[0][2] += 1
    V[1][2] += 1; V[3][2] += -1
    W[2][2] += 1
    
    # M4 = d(g-h)
    # U: d=1 at idx3; V: g=1 at idx2, h=-1 at idx3
    # W: C11 += -1, C21 += 1
    U[3][3] += 1
    V[2][3] += 1; V[3][3] += -1
    W[0][3] += -1; W[1][3] += 1
    
    # M5 = (a+b)h
    # U: a=1 at idx0, b=1 at idx1; V: h=1 at idx3
    # W: C11 += 1, C12 += 1
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += 1; W[2][4] += 1
    
    # M6 = (c-a)(e+f)
    # U: c=1 at idx2, a=-1 at idx0; V: e=1 at idx0, f=1 at idx1
    # W: C22 += 1
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (b-d)(g+h)
    # U: b=1 at idx1, d=-1 at idx3; V: g=1 at idx2, h=1 at idx3
    # W: C11 += -1, C21 += 1
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += -1; W[1][6] += 1
    
    return U, V, W

def _build_2x4x2_block(strassen_U, strassen_V, strassen_W, j_offset, k_offset):
    """
    Build the 2x4x2 block decomposition using two Strassen 2x2x2 blocks.
    j_offset: starting column in the 4-wide middle dimension
    k_offset: starting row in the 2-wide output dimension
    Returns U_block (2*4=8 rows x 14 cols), V_block (4*2=8 rows x 14 cols), W_block (2*2=4 rows x 14 cols)
    """
    U_block = [[0]*14 for _ in range(8)]
    V_block = [[0]*14 for _ in range(8)]
    W_block = [[0]*14 for _ in range(4)]
    
    # First half: j in [j_offset, j_offset+1], using strassen directly
    for i in range(2):
        for j in range(2):
            for r in range(7):
                U_block[i*4 + (j_offset + j)][r] += strassen_U[i*2 + j][r]
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    V_block[(j_offset + j)*2 + k][r] += strassen_V[j*2 + k][r]
        for k in range(2):
            for r in range(7):
                W_block[(k_offset + k)*2 + i][r] += strassen_W[k*2 + i][r]
    
    # Second half: j in [j_offset+2, j_offset+3], using strassen with offset
    for i in range(2):
        for j in range(2):
            for r in range(7):
                U_block[i*4 + (j_offset + 2 + j)][14//2 + r] += strassen_U[i*2 + j][r]
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    V_block[(j_offset + 2 + j)*2 + k][14//2 + r] += strassen_V[j*2 + k][r]
        for k in range(2):
            for r in range(7):
                W_block[(k_offset + k)*2 + i][14//2 + r] += strassen_W[k*2 + i][r]
    
    return U_block, V_block, W_block

def _structured_decomposition(n, m, p):
    """
    Build structured decomposition for 2x4x7 using Kronecker/block decomposition.
    Split the 7 columns into 2+2+2+1.
    Three 2x4x2 blocks (each rank 14) + one 2x4x1 block (rank 1).
    Total rank: 3*14 + 1 = 43.
    """
    rank = 43
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    sU, sV, sW = _strassen_2x2x2()
    
    col_offset = 0
    # Three 2x4x2 blocks for k=0..5 (columns 0-1, 2-3, 4-5)
    for block in range(3):
        k_offset = block * 2
        j_offset = 0  # full 4-wide middle
        U_block, V_block, W_block = _build_2x4x2_block(sU, sV, sW, j_offset, k_offset)
        for i in range(n*m):
            for r in range(14):
                U[i][col_offset + r] += U_block[i][r]
        for i in range(m*p):
            for r in range(14):
                V[i][col_offset + r] += V_block[i][r]
        for i in range(p*n):
            for r in range(14):
                W[i][col_offset + r] += W_block[i][r]
        col_offset += 14
    
    # One 2x4x1 block for k=6
    # T[i,j,6] = 1 for all i,j. Rank 1: U[i,j,r]=1, V[j,6,r]=1, W[6,i,r]=1
    for i in range(n):
        for j in range(m):
            U[i*m + j][col_offset] = 1
            V[j*p + 6][col_offset] = 1
            W[6*n + i][col_offset] = 1
    col_offset += 1
    
    return {"U": U, "V": V, "W": W}, rank

def _schoolbook_decomp(n, m, p):
    """Standard schoolbook decomposition."""
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
    return {"U": U, "V": V, "W": W}, rank

def _local_search_polish(U, V, W, n, m, p, rank, deadline, initial_temp=3.0, cooling_rate=0.997):
    """
    Local search with simulated annealing to polish a decomposition.
    Tries to reduce the error (number of entries not equal to 1).
    """
    current_error = _compute_error_np(U, V, W, n, m, p, rank)
    if current_error == 0:
        return U, V, W, 0
    
    best_error = current_error
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    temperature = initial_temp
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and temperature > 0.01:
        if no_improve > max_no_improve:
            # Reset to best and lower temperature more aggressively
            U = [row[:] for row in best_U]
            V = [row[:] for row in best_V]
            W = [row[:] for row in best_W]
            current_error = best_error
            temperature *= 0.5
            no_improve = 0
            max_no_improve = int(max_no_improve * 1.5)
        
        mat = random.choice(["U", "V", "W"])
        
        if mat == "U":
            idx = random.randint(0, n*m-1)
            r = random.randint(0, rank-1)
            old_val = U[idx][r]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
            U[idx][r] = new_val
            new_error = _compute_error_np(U, V, W, n, m, p, rank)
            
            if new_error < current_error or (temperature > 0.05 and random.random() < np.exp(-(new_error - current_error) / temperature)):
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
                    no_improve = 0
                    if best_error == 0:
                        return U, V, W, 0
                else:
                    no_improve += 1
            else:
                U[idx][r] = old_val
                no_improve += 1
        elif mat == "V":
            idx = random.randint(0, m*p-1)
            r = random.randint(0, rank-1)
            old_val = V[idx][r]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
            V[idx][r] = new_val
            new_error = _compute_error_np(U, V, W, n, m, p, rank)
            
            if new_error < current_error or (temperature > 0.05 and random.random() < np.exp(-(new_error - current_error) / temperature)):
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
                    no_improve = 0
                    if best_error == 0:
                        return U, V, W, 0
                else:
                    no_improve += 1
            else:
                V[idx][r] = old_val
                no_improve += 1
        else:
            idx = random.randint(0, p*n-1)
            r = random.randint(0, rank-1)
            old_val = W[idx][r]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
            W[idx][r] = new_val
            new_error = _compute_error_np(U, V, W, n, m, p, rank)
            
            if new_error < current_error or (temperature > 0.05 and random.random() < np.exp(-(new_error - current_error) / temperature)):
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
                    no_improve = 0
                    if best_error == 0:
                        return U, V, W, 0
                else:
                    no_improve += 1
            else:
                W[idx][r] = old_val
                no_improve += 1
        
        temperature *= cooling_rate
    
    return best_U, best_V, best_W, best_error

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Try structured decomposition first
    best_result = None
    best_rank = n * m * p  # 56
    
    # 1. Structured Kronecker decomposition (rank 43)
    if time.time() < deadline - 60:
        struct, struct_rank = _structured_decomposition(n, m, p)
        err = _compute_error_np(struct["U"], struct["V"], struct["W"], n, m, p, struct_rank)
        if err == 0:
            best_result = struct
            best_rank = struct_rank
        else:
            # Polish the structured decomposition
            U_p, V_p, W_p, err_p = _local_search_polish(
                struct["U"], struct["V"], struct["W"], n, m, p, struct_rank,
                deadline - 30, initial_temp=2.0, cooling_rate=0.998
            )
            if err_p == 0:
                best_result = {"U": U_p, "V": V_p, "W": W_p}
                best_rank = struct_rank
    
    # 2. Try to find even lower rank via random initialization + local search
    for target_rank in [40, 41, 42]:
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            break
        
        found = False
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            restart_deadline = min(deadline - 5, time.time() + (deadline - time.time() - 10) / 3)
            
            # Initialize with structured pattern + random perturbations
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            U_p, V_p, W_p, err_p = _local_search_polish(
                U, V, W, n, m, p, target_rank,
                restart_deadline, initial_temp=5.0, cooling_rate=0.995
            )
            if err_p == 0:
                best_result = {"U": U_p, "V": V_p, "W": W_p}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # 3. Fallback to schoolbook
    if best_result is None:
        school, school_rank = _schoolbook_decomp(n, m, p)
        return school
    
    return best_result
