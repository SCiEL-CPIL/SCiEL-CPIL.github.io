"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify_decomposition(U, V, W, n=5, m=5, p=5):
    """Verify that U, V, W form a valid matrix multiplication decomposition."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_exact(U, V, W, n=5, m=5, p=5):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                u_row = U_frac[i * m + j]
                v_row = V_frac[j * p + k]
                w_row = W_frac[k * n + i]
                for c in range(rank):
                    s += u_row[c] * v_row[c] * w_row[c]
                if s != 1:
                    return False
    return True

def _strassen_2x2_factors():
    """Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
    
    Each term is (U_vec, V_vec, W_vec) where:
    - U_vec[li*2+lj] = coefficient for A[li,lj]
    - V_vec[lj*2+lk] = coefficient for B[lj,lk]
    - W_vec[lk*2+li] = coefficient for C[li,lk]
    
    The 7 Strassen products:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B22)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Indexing: U[li*2+lj], V[lj*2+lk], W[lk*2+li]
    # So U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)  [lk*2+li]
    
    factors = []
    
    # M1 = (A11+A22)(B11+B22): contributes +1 to C11, +1 to C22
    # U: A11=1, A22=1 -> [1,0,0,1]
    # V: B11=1, B22=1 -> [1,0,0,1]
    # W: C11=+1, C22=+1 -> W[0*2+0]=1, W[1*2+1]=1 -> [1,0,1,0]
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, 0]))
    
    # M2 = (A21+A22)B11: contributes +1 to C21, -1 to C22
    # U: A21=1, A22=1 -> [0,0,1,1]
    # V: B11=1 -> [1,0,0,0]
    # W: C21=+1 (lk=0,li=1 -> idx 0*2+1=1), C22=-1 (lk=1,li=1 -> idx 1*2+1=3)
    # W: [0,1,0,-1]
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]))
    
    # M3 = A11(B21-B22): contributes +1 to C12, +1 to C22
    # U: A11=1 -> [1,0,0,0]
    # V: B21=1, B22=-1 -> [0,0,1,-1]
    # W: C12=+1 (lk=1,li=0 -> idx 1*2+0=2), C22=+1 (lk=1,li=1 -> idx 1*2+1=3)
    # W: [0,0,1,1]
    factors.append(([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]))
    
    # M4 = A22(B12-B22): contributes +1 to C11, +1 to C21
    # U: A22=1 -> [0,0,0,1]
    # V: B12=1, B22=-1 -> [0,1,0,-1]
    # W: C11=+1 (lk=0,li=0 -> idx 0), C21=+1 (lk=0,li=1 -> idx 1)
    # W: [1,1,0,0]
    factors.append(([0, 0, 0, 1], [0, 1, 0, -1], [1, 1, 0, 0]))
    
    # M5 = (A11+A12)B22: contributes -1 to C11, +1 to C12
    # U: A11=1, A12=1 -> [1,1,0,0]
    # V: B22=1 -> [0,0,0,1]
    # W: C11=-1 (idx 0), C12=+1 (idx 2)
    # W: [-1,0,1,0]
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]))
    
    # M6 = (A21-A11)(B11+B12): contributes +1 to C22
    # U: A21=1, A11=-1 -> [-1,0,1,0]
    # V: B11=1, B12=1 -> [1,1,0,0]
    # W: C22=+1 (idx 3)
    # W: [0,0,0,1]
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    
    # M7 = (A12-A22)(B21+B22): contributes +1 to C11
    # U: A12=1, A22=-1 -> [0,1,0,-1]
    # V: B21=1, B22=1 -> [0,0,1,1]
    # W: C11=+1 (idx 0)
    # W: [1,0,0,0]
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    return factors

def _schoolbook_decomposition(n=5, m=5, p=5):
    """Return the schoolbook decomposition as fallback."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

def _construct_5x5x5_strassen():
    """
    Construct a decomposition using 2+2+1 partition with Strassen for 2x2x2 blocks.
    """
    n, m, p = 5, 5, 5
    block_sizes = [2, 2, 1]
    
    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    strassen_factors = _strassen_2x2_factors()
    
    # Calculate total rank
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen for 2x2x2 block
                    for (uf, vf, wf) in strassen_factors:
                        # Map local indices to global
                        # U local index li*2+lj -> global (offsets[i]+li)*m + (offsets[j]+lj)
                        # V local index lj*2+lk -> global (offsets[j]+lj)*p + (offsets[k]+lk)
                        # W local index lk*2+li -> global (offsets[k]+lk)*n + (offsets[i]+li)
                        
                        for li in range(2):
                            for lj in range(2):
                                local_u_idx = li * 2 + lj
                                global_u_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                if uf[local_u_idx] != 0:
                                    U[global_u_idx][col] += uf[local_u_idx]
                        
                        for lj in range(2):
                            for lk in range(2):
                                local_v_idx = lj * 2 + lk
                                global_v_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                if vf[local_v_idx] != 0:
                                    V[global_v_idx][col] += vf[local_v_idx]
                        
                        for lk in range(2):
                            for li in range(2):
                                local_w_idx = lk * 2 + li
                                global_w_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                if wf[local_w_idx] != 0:
                                    W[global_w_idx][col] += wf[local_w_idx]
                        
                        col += 1
                else:
                    # Schoolbook for other blocks
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Try Strassen-based decomposition
    try:
        result = _construct_5x5x5_strassen()
        rank = len(result["U"][0])
        
        # Quick numeric verification
        if _verify_decomposition(result["U"], result["V"], result["W"]):
            # Exact verification with fractions
            if _verify_exact(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END
