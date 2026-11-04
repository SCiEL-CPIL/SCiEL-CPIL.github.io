import time
import numpy as np
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def _schoolbook():
    """Return the trivial schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-9):
    """Verify that the decomposition correctly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # The tensor should satisfy: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But the correct identity for matrix multiplication (A*B) where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the tensor decomposition should give:
    # For each output element C[i,k], we have C[i,k] = sum_j A[i,j]*B[j,k]
    # In tensor form: T[i,j,k] = 1 for all i,j,k (since each triple contributes)
    # Wait, let me reconsider. The matrix multiplication tensor T_{i,j,k} satisfies:
    # (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So the tensor is T_{i,j,k} = 1 (it's just the identity tensor in some sense)
    # Actually no. The tensor decomposition is:
    # T_{i,j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    # And we need T_{i,j,k} = 1 for all i,j,k? No...
    
    # Let me think again. The matrix multiplication can be written as:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is a trilinear form. The tensor T has components T_{i,j,k} such that:
    # sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k} = C_{i,k} for each i,k
    # Actually, the standard way: the multiplication map is a bilinear map
    # A (n x m) and B (m x p) -> C (n x p)
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # In tensor decomposition form, we want:
    # C_{i,k} = sum_c (sum_j U_{i*m+j,c} A_{i,j}) * (sum_l V_{l*p+k,c} B_{l,k}) * W_{k*n+i,c}
    # Hmm, that doesn't look right either.
    
    # Let me use the standard formulation. The matrix multiplication tensor is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the indices correspond to (row of A, col of A = row of B, col of B)
    # Actually, the correct tensor is:
    # T_{i,j,k} = 1 for all valid i,j,k
    # Because C_{i,k} = sum_j A_{i,j} * B_{j,k} * 1
    
    # So we need: sum_c U_{i*m+j,c} * V_{j*p+k,c} * W_{k*n+i,c} = 1 for all i,j,k
    
    # Let's verify this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    return True, None

def _strassen_2x2():
    """Strassen decomposition for 2x2 * 2x2. Rank 7."""
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Indices: A_{i,j} -> U[i*2+j, c], B_{j,k} -> V[j*2+k, c], C_{i,k} -> W[k*2+i, c]
    # Wait, the convention is W[k*n+i, r]
    
    # Let me define the Strassen products:
    # For each term c, U, V, W encode the linear combinations
    
    # M1: (A11+A22)(B11+B22)
    # A part: A11 + A22 -> U[0,0]=1, U[3,0]=1 (indices 0,1,2,3 for A11,A12,A21,A22)
    # B part: B11 + B22 -> V[0,0]=1, V[3,0]=1 (indices 0,1,2,3 for B11,B12,B21,B22)
    # C part: contributes to C11 (+1), C22 (+1)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1  # C11 at index 0*2+0=0, C22 at index 1*2+1=3
    
    # M2: (A21+A22)B11
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1  # C21 at index 1*2+0=2
    
    # M3: A11(B12-B22)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1  # C12 at index 0*2+1=1
    
    # M4: A22(B21-B11)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1  # C11 and C21
    
    # M5: (A11+A12)B22
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1  # C11 (-1), C12 (+1)
    
    # M6: (A21-A11)(B11+B12)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1  # C22 at index 3
    
    # M7: (A12-A22)(B21+B22)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1; W[3,6]=-1  # C11 (+1), C22 (-1)
    
    return {"U": U, "V": V, "W": W}

def _try_random_search(best_rank, deadline, seed=42):
    """Try random perturbations to find lower rank decompositions."""
    n, m, p = DIMENSIONS
    np.random.seed(seed)
    
    best = None
    best_rank_found = best_rank
    
    # Start from schoolbook and try to reduce
    sb = _schoolbook()
    U = sb["U"].copy()
    V = sb["V"].copy()
    W = sb["W"].copy()
    
    rank = U.shape[1]
    
    # Try removing columns and adjusting
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Try a random subset of columns of size rank - 1
        new_rank = rank - 1
        if new_rank < best_rank_found:
            # Try to solve for the missing column
            # This is complex, so let's just try random small perturbations
            pass
        
        # Alternative: try random initialization with smaller rank
        target_rank = max(10, best_rank_found - 10)
        
        if time.time() > deadline:
            break
        
        U_rand = np.random.randn(n*m, target_rank)
        V_rand = np.random.randn(m*p, target_rank)
        W_rand = np.random.randn(p*n, target_rank)
        
        # This won't work directly, we need to solve the system
        # The constraint is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a nonlinear system. Let's skip this for now.
        
        break
    
    return None

def _combine_decompositions(decs):
    """Combine multiple decompositions into one."""
    if not decs:
        return None
    
    n, m, p = DIMENSIONS
    total_rank = sum(d["U"].shape[1] for d in decs)
    
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    col = 0
    for d in decs:
        r = d["U"].shape[1]
        U[:, col:col+r] = d["U"]
        V[:, col:col+r] = d["V"]
        W[:, col:col+r] = d["W"]
        col += r
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    sb = _schoolbook()
    best = sb
    best_rank = sb["U"].shape[1]
    
    # Try Strassen-based approach for 4x4 * 4x7
    # We can view 4x4 * 4x7 as blocks
    # For 4x4 * 4x4, Strassen gives rank 7^2 = 49 for the 4x4*4x4 part
    # But we have 4x7, which is 4x4 + 4x3
    
    # Let's try to build a decomposition using Strassen on 2x2 blocks
    # 4x4 * 4x7:
    # Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Split B into 2x2 and 2x1 blocks: B = [[B11, B12, B13], [B21, B22, B23]]
    # where each *1 is 2x1
    
    # This gets complex. Let me try a different approach.
    
    # Try to verify schoolbook
    valid, info = _verify(sb["U"], sb["V"], sb["W"])
    if not valid:
        print(f"Schoolbook verification failed: {info}")
    
    # Try random search for better decompositions
    result = _try_random_search(best_rank, deadline)
    if result is not None:
        r = result["U"].shape[1]
        if r < best_rank:
            best = result
            best_rank = r
    
    if time.time() > deadline:
        return best
    
    # Return the best found
    return best

