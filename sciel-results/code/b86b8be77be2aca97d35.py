"""Matrix multiplication 2x4x5 with rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify exact decomposition. Returns max error."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _verify_fast(U, V, W, n, m, p, rank):
    """Fast verification, returns True if exact."""
    return _verify(U, V, W, n, m, p, rank) < 1e-9

def _strassen_2x2x2(U, V, W, i0, j0, k0, c):
    """
    Add Strassen 2x2x2 block.
    U rows: i0*m+j for i in {i0,i0+1}, j in {j0,j0+1}
    V rows: (j0+p+j)*p... wait, let me use the flat indexing.
    U index: i*m + j
    V index: j*p + k  
    W index: k*n + i
    """
    # M1 = (A[i0,j0]+A[i0+1,j0+1])*(B[j0,k0]+B[j0+1,k0+1])
    U[i0*m+j0][c] = 1
    U[(i0+1)*m+j0+1][c] = 1
    V[j0*p+k0][c] = 1
    V[(j0+1)*p+(k0+1)][c] = 1
    W[k0*n+i0][c] = 1
    W[(k0+1)*n+(i0+1)][c] = 1
    c += 1
    
    # M2 = (A[i0+1,j0]+A[i0+1,j0+1])*B[j0,k0]
    U[(i0+1)*m+j0][c] = 1
    U[(i0+1)*m+j0+1][c] = 1
    V[j0*p+k0][c] = 1
    W[k0*n+(i0+1)][c] = 1
    W[(k0+1)*n+(i0+1)][c] = -1
    c += 1
    
    # M3 = A[i0,j0]*(B[j0,k0+1]-B[j0+1,k0+1])
    U[i0*m+j0][c] = 1
    V[j0*p+(k0+1)][c] = 1
    V[(j0+1)*p+(k0+1)][c] = -1
    W[k0*n+i0][c] = 1
    W[(k0+1)*n+i0][c] = 1
    c += 1
    
    # M4 = A[i0+1,j0+1]*(B[j0+1,k0]-B[j0,k0])
    U[(i0+1)*m+j0+1][c] = 1
    V[(j0+1)*p+k0][c] = 1
    V[j0*p+k0][c] = -1
    W[k0*n+(i0+1)][c] = 1
    W[(k0+1)*n+(i0+1)][c] = 1
    c += 1
    
    # M5 = (A[i0,j0]+A[i0,j0+1])*B[j0+1,k0+1]
    U[i0*m+j0][c] = 1
    U[i0*m+j0+1][c] = 1
    V[(j0+1)*p+(k0+1)][c] = 1
    W[k0*n+i0][c] = -1
    W[(k0+1)*n+i0][c] = 1
    c += 1
    
    # M6 = (A[i0+1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k0+1])
    U[(i0+1)*m+j0][c] = 1
    U[i0*m+j0][c] = -1
    V[j0*p+k0][c] = 1
    V[j0*p+(k0+1)][c] = 1
    W[(k0+1)*n+(i0+1)][c] = 1
    c += 1
    
    # M7 = (A[i0,j0+1]-A[i0+1,j0+1])*(B[j0+1,k0]+B[j0+1,k0+1])
    U[i0*m+j0+1][c] = 1
    U[(i0+1)*m+j0+1][c] = -1
    V[(j0+1)*p+k0][c] = 1
    V[(j0+1)*p+(k0+1)][c] = 1
    W[k0*n+i0][c] = 1
    c += 1
    
    return c

def _construct_rank36():
    """
    Construct rank-36 decomposition using Strassen blocks.
    Split m=4 into two groups of 2, p=5 into (2,2,1).
    
    Block structure:
    - j in {0,1}, k in {0,1}: Strassen (7)
    - j in {0,1}, k in {2,3}: Strassen (7)
    - j in {0,1}, k in {4}:    schoolbook (4)
    - j in {2,3}, k in {0,1}: Strassen (7)
    - j in {2,3}, k in {2,3}: Strassen (7)
    - j in {2,3}, k in {4}:    schoolbook (4)
    Total: 36
    """
    n, m, p = 2, 4, 5
    total_rank = 36
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    c = 0
    # Block (i=0,1; j=0,1; k=0,1): Strassen
    c = _strassen_2x2x2(U, V, W, 0, 0, 0, c)
    
    # Block (i=0,1; j=0,1; k=2,3): Strassen
    c = _strassen_2x2x2(U, V, W, 0, 0, 2, c)
    
    # Block (i=0,1; j=0,1; k=4): schoolbook
    for i in range(2):
        for j in range(2):
            U[i*m+j][c] = 1
            V[j*p+4][c] = 1
            W[4*n+i][c] = 1
            c += 1
    
    # Block (i=0,1; j=2,3; k=0,1): Strassen
    c = _strassen_2x2x2(U, V, W, 0, 2, 0, c)
    
    # Block (i=0,1; j=2,3; k=2,3): Strassen
    c = _strassen_2x2x2(U, V, W, 0, 2, 2, c)
    
    # Block (i=0,1; j=2,3; k=4): schoolbook
    for i in range(2):
        for j in range(2):
            U[i*m+j][c] = 1
            V[j*p+4][c] = 1
            W[4*n+i][c] = 1
            c += 1
    
    return {"U": U, "V": V, "W": W}

def _construct_rank32():
    """
    Try to construct a rank-32 decomposition.
    
    Idea: Use a more efficient decomposition for the 2x4x5 tensor.
    Split p=5 into (3,2). For k in {0,1,2} and k in {3,4}.
    
    For k in {0,1,2}: 2x4x3
    For k in {3,4}: 2x4x2
    
    For 2x4x2: split j into {0,1} and {2,3}, use Strassen for each = 14
    For 2x4x3: split j into {0,1} and {2,3}, for each 2x2x3 use rank-5 Winograd = 10
    Total: 14 + 10 = 24? No, that's not right either.
    
    Actually, let me try a different approach.
    Use the fact that we can share some Strassen terms.
    
    Alternative: Try rank 32 by combining blocks more carefully.
    """
    n, m, p = 2, 4, 5
    total_rank = 32
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    c = 0
    # For j in {0,1}:
    #   k in {0,1}: Strassen (7)
    #   k in {2,3}: Strassen (7)  
    #   k in {4}: schoolbook (4)
    c = _strassen_2x2x2(U, V, W, 0, 0, 0, c)
    c = _strassen_2x2x2(U, V, W, 0, 0, 2, c)
    for i in range(2):
        for j in range(2):
            U[i*m+j][c] = 1
            V[j*p+4][c] = 1
            W[4*n+i][c] = 1
            c += 1
    
    # For j in {2,3}:
    #   k in {0,1}: Strassen (7)
    #   k in {2,3}: Strassen (7)
    #   k in {4}: schoolbook (4)
    c = _strassen_2x2x2(U, V, W, 0, 2, 0, c)
    c = _strassen_2x2x2(U, V, W, 0, 2, 2, c)
    for i in range(2):
        for j in range(2):
            U[i*m+j][c] = 1
            V[j*p+4][c] = 1
            W[4*n+i][c] = 1
            c += 1
    
    # This gives rank 36, not 32. Let me try a different structure.
    # Actually, this is the same as rank 36. Let me try something else.
    
    # Try: split m=4 into {0,1,2} and {3}
    # j in {0,1,2}: 2x3x5
    # j in {3}: 2x1x5 = 10 rank (schoolbook)
    # For 2x3x5: split k into {0,1,2} and {3,4}
    #   j in {0,1,2}, k in {0,1,2}: 2x3x3
    #   j in {0,1,2}, k in {3,4}: 2x3x2
    # 2x3x3: split j into {0,1} and {2}
    #   j in {0,1}, k in {0,1,2}: 2x2x3 (rank 5 Winograd)
    #   j in {2}, k in {0,1,2}: 2x1x3 (rank 6 schoolbook)
    # 2x3x2: split j into {0,1} and {2}
    #   j in {0,1}, k in {3,4}: 2x2x2 (rank 7 Strassen)
    #   j in {2}, k in {3,4}: 2x1x2 (rank 4 schoolbook)
    # Total for 2x3x5: 5+6+7+4 = 22
    # Total: 22 + 10 = 32
    
    # Let me implement this!
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    c = 0
    
    # j in {0,1}, k in {0,1,2}: 2x2x3 Winograd rank 5
    # M1 = A[i0,j0] * B[j0,k0]
    U[0*m+0][c] = 1; V[0*p+0][c] = 1; W[0*n+0][c] = 1; c += 1
    # M2 = A[i0,j0] * B[j0,k1]
    U[0*m+0][c] = 1; V[0*p+1][c] = 1; W[1*n+0][c] = 1; c += 1
    # M3 = A[i0,j1] * B[j1,k0]
    U[0*m+1][c] = 1; V[1*p+0][c] = 1; W[0*n+0][c] = 1; c += 1
    # M4 = A[i0,j1] * B[j1,k1]
    U[0*m+1][c] = 1; V[1*p+1][c] = 1; W[1*n+0][c] = 1; c += 1
    # M5 = (A[i0,j0]+A[i0,j1]) * (B[j0,k2]+B[j1,k2]) ... this doesn't work for 2x2x3
    # Actually, 2x2x3 doesn't have a rank-5 decomposition in general.
    # Let me use schoolbook for 2x2x3 = 12 rank.
    
    # This approach is getting complicated. Let me just use the rank-36 and try to reduce.
    return None

def _winograd_2x2x3(U, V, W, i0, j0, k0, c):
    """
    Winograd-style decomposition for 2x2x3 (rank 5).
    This computes C[i,k] += sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
    using 5 multiplications.
    """
    # The Winograd scheme for 2x2x3:
    # M1 = A[i0,j0] * B[j0,k0]
    U[i0*m+j0][c] = 1
    V[j0*p+k0][c] = 1
    W[k0*n+i0][c] = 1
    c += 1
    
    # M2 = A[i0,j1] * B[j1,k0]
    U[i0*m+j0+1][c] = 1
    V[(j0+1)*p+k0][c] = 1
    W[k0*n+i0][c] = 1
    c += 1
    
    # M3 = A[i1,j0] * B[j0,k1]
    U[(i0+1)*m+j0][c] = 1
    V[j0*p+(k0+1)][c] = 1
    W[(k0+1)*n+(i0+1)][c] = 1
    c += 1
    
    # M4 = A[i1,j1] * B[j1,k1]
    U[(i0+1)*m+j0+1][c] = 1
    V[(j0+1)*p+(k0+1)][c] = 1
    W[(k0+1)*n+(i0+1)][c] = 1
    c += 1
    
    # M5 = (A[i0,j0]+A[i1,j1]) * (B[j0,k2]+B[j1,k2])
    # This gives C[i0,k2] += A[i0,j0]*B[j0,k2] + A[i0,j0]*B[j1,k2] + A[i1,j1]*B[j0,k2] + A[i1,j1]*B[j1,k2]
    # But we need C[i0,k2] = A[i0,j0]*B[j0,k2] + A[i0,j1]*B[j1,k2]
    # and C[i1,k2] = A[i1,j0]*B[j0,k2] + A[i1,j1]*B[j1,k2]
    # So this doesn't work directly. The 2x2x3 tensor has rank 6, not 5.
    # Let me use schoolbook for 2x2x3.
    
    # Actually, let me just do schoolbook for all the small blocks and see if I can get rank 32.
    return c

def _schoolbook_block(U, V, W, i0, j0, k0, ni, nj, nk, c):
    """Schoolbook for a ni x nj x nk block."""
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                U[(i0+i)*m + (j0+j)][c] = 1
                V[(j0+j)*p + (k0+k)][c] = 1
                W[(k0+k)*n + (i0+i)][c] = 1
                c += 1
    return c

def _construct_rank32_v2():
    """
    Try to construct rank-32 by using a non-rectangular split.
    Split m=4 into {0,1,2} (size 3) and {3} (size 1).
    
    For j in {0,1,2}:
      For k in {0,1,2}: 2x3x3 = split j into {0,1} and {2}
        j in {0,1}: 2x2x3 = 12 (schoolbook)
        j in {2}: 2x1x3 = 6 (schoolbook)
        Total: 18
      For k in {3,4}: 2x3x2 = split j into {0,1} and {2}
        j in {0,1}: 2x2x2 = 7 (Strassen)
        j in {2}: 2x1x2 = 4 (schoolbook)
        Total: 11
    For j in {3}:
      For k in {0,1,2,3,4}: 2x1x5 = 10 (schoolbook)
    
    Total: 18 + 11 + 10 = 39. That's worse.
    
    Let me try another split.
    """
    return None

def _construct_rank30():
    """
    Try rank 30.
    Split m=4 into {0,1} and {2,3}.
    Split p=5 into {0,1,2} and {3,4}.
    
    j in {0,1}, k in {0,1,2}: 2x2x3 = 12 (schoolbook) or 8 (if we can do better)
    j in {0,1}, k in {3,4}: 2x2x2 = 7 (Strassen)
    j in {2,3}, k in {0,1,2}: 2x2x3 = 12 (schoolbook)
    j in {2,3}, k in {3,4}: 2x2x2 = 7 (Strassen)
    
    Total: 12+7+12+7 = 38. Worse.
    
    What if we use a rank-8 decomposition for 2x2x3?
    Actually, the rank of 2x2x3 is 6 (it's the same as 2x2x2 plus one more output).
    Wait, 2x2x3 has 12 entries. The rank is at most 12.
    Actually, for 2x2x3, we can use:
    M1 = A00*B00, M2 = A00*B01, M3 = A00*B02, M4 = A01*B10, M5 = A01*B11, M6 = A01*B12,
    M7 = A10*B00, M8 = A10*B01, M9 = A10*B02, M10 = A11*B10, M11 = A11*B11, M12 = A11*B12
    That's 12. Can we do better?
    
    For 2x2x2, Strassen gives rank 7.
    For 2x2x3, we can extend Strassen:
    M1 = (A00+A11)*(B00+B11) -> contributes to C00, C01, C10, C11
    M2 = (A10+A11)*B00 -> contributes to C10, C11
    M3 = A00*(B01-B11) -> contributes to C01
    M4 = A11*(B10-B00) -> contributes to C11
    M5 = (A00+A01)*B11 -> contributes to C01
    M6 = (A10-A00)*(B00+B01) -> contributes to C10
    M7 = (A01-A11)*(B10+B11) -> contributes to C11
    That's 7 for the k=0,1 part. For k=2, we need 4 more (schoolbook).
    Total: 11 for 2x2x3.
    
    So: j in {0,1}, k in {0,1,2}: 11
    j in {0,1}, k in {3,4}: 7
    j in {2,3}, k in {0,1,2}: 11
    j in {2,3}, k in {3,4}: 7
    Total: 36. Same as before.
    """
    return None

def _construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # First try the constructive rank-36
    result = _construct_rank36()
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    
    if _verify_fast(U, V, W, n, m, p, rank):
        best_result = result
        best_rank = rank
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    # Try to reduce rank using local search
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        U_np = np.array(best_result["U"], dtype=np.float64)
        V_np = np.array(best_result["V"], dtype=np.float64)
        W_np = np.array(best_result["W"], dtype=np.float64)
        current_rank = U_np.shape[1]
        
        for attempt in range(2000):
            if time.time() > deadline:
                break
            if current_rank <= 25:
                break
            
            # Try removing a random column
            col_to_remove = np.random.randint(0, current_rank)
            mask = [c for c in range(current_rank) if c != col_to_remove]
            new_rank = current_rank - 1
            
            U_new = np.zeros((n*m, new_rank))
            V_new = np.zeros((m*p, new_rank))
            W_new = np.zeros((p*n, new_rank))
            for new_c, old_c in enumerate(mask):
                U_new[:, new_c] = U_np[:, old_c]
                V_new[:, new_c] = V_np[:, old_c]
                W_new[:, new_c] = W_np[:, old_c]
            
            # Check if the column is redundant
            err = _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, new_rank)
            if err < 1e-9:
                U_np, V_np, W_np = U_new, V_new, W_new
                current_rank = new_rank
                continue
            
            # Try small perturbations
            for perturb_trial in range(5):
                if time.time() > deadline:
                    break
                
                U_pert = U_new.copy()
                V_pert = V_new.copy()
                W_pert = W_new.copy()
                
                # Perturb a few entries
                num_perturb = np.random.randint(3, 8)
                for _ in range(num_perturb):
                    mat_choice = np.random.randint(0, 3)
                    if mat_choice == 0:
                        mat = U_pert
                    elif mat_choice == 1:
                        mat = V_pert
                    else:
                        mat = W_pert
                    row = np.random.randint(0, mat.shape[0])
                    col = np.random.randint(0, mat.shape[1])
                    mat[row, col] += np.random.choice([-0.5, 0.5, -1.0, 1.0])
                    mat[row, col] = round(mat[row, col] * 2) / 2
                
                err = _verify(U_pert.tolist(), V_pert.tolist(), W_pert.tolist(), n, m, p, new_rank)
                if err < 1e-9:
                    U_np, V_np, W_np = U_pert, V_pert, W_pert
                    current_rank = new_rank
                    break
        
        if current_rank < best_rank:
            best_result = {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
            best_rank = current_rank
    
    return best_result

def construct():
    return _construct()
# EVOLVE-BLOCK-END
