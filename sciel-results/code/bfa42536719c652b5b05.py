"""Optimized tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import random
import numpy as np
import itertools

DIMENSIONS = [3, 3, 6]
_TIME_LIMIT = 150.0  # wall-clock limit in seconds (evaluator kills at 150s)

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
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


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r]^T equals the matmul tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j2 * p + k
                    C = k * n + i
                    expected = 1.0 if (j == j2) else 0.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[A][r] * V[B][r] * W[C][r]
                    if abs(actual - expected) > tol:
                        return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j2 in range(m):
            for k in range(p):
                A = i * m + j2  # Note: j2 is the column of A that matches row of B
                # Actually, the tensor index is T[A_idx, B_idx, C_idx] where:
                # A_idx = i*m + j (row i, col j of A)
                # B_idx = j*p + k (row j, col k of B)  
                # C_idx = k*n + i (row k, col i of C)
                # And T = 1 iff j == j (the shared index)
                # So for fixed (i, j, k), T[i*m+j, j*p+k, k*n+i] = 1
                # All other entries are 0.
                pass
    
    # Let me re-verify the convention more carefully.
    # The task says: W[k*n+i, r] is the output factor.
    # The tensor T_{A,B,C} = sum_r U_{A,r} V_{B,r} W_{C,r}
    # where A = i*m + j, B = j'*p + k, C = k'*n + i
    # T = 1 iff j == j' (and implicitly the output C corresponds to correct positions)
    
    # Actually for matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T_{i,j; j',k; i',k'} = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # In the flat indexing:
    # A_idx = i*m + j
    # B_idx = j'*p + k
    # C_idx = k'*n + i'
    # T = 1 iff j == j' AND i == i' AND k == k'
    
    # Let me verify this interpretation with the schoolbook:
    # Schoolbook: for each (i,j,k): U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # So T[i*m+j, j*p+k, k*n+i] = 1 for each (i,j,k) term
    # This gives C[k*n+i] += A[i*m+j] * B[j*p+k]
    # i.e., C[k,i] += A[i,j] * B[j,k]
    # So C[k,i] = sum_j A[i,j]*B[j,k]
    # This means C is indexed as C[k,i] = (A*B)^T[i,k]... 
    # Wait, no. C[k*n+i] means row k, col i of C.
    # So C[k,i] = sum_j A[i,j] * B[j,k]
    # But (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So C[k,i] = (A*B)[i,k], meaning C = (A*B)^T
    # OR the output is stored in column-major order.
    
    # Let me just use the verify function as given and trust it.
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j2 * p + k
                    C = k * n + i
                    expected = 1.0 if (j == j2) else 0.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[A][r] * V[B][r] * W[C][r]
                    if abs(actual - expected) > tol:
                        return False
    return True


def _build_rank19_3x3x3():
    """
    Build a rank-19 decomposition for 3x3 * 3x3 matrix multiplication.
    Based on the known result that rank(3x3 matmul) = 19 (over reals/rationals).
    
    The decomposition is based on the work of de Groote et al. / Bläser et al.
    We'll construct it using a known explicit formula.
    
    For 3x3 matrices A and B:
    A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    C = A*B
    
    We need 19 products M1..M19 such that each C[i,j] is a linear combination
    of the M's.
    
    One known construction uses the following approach:
    - Use Strassen-like additions/differences
    - The key is to find 19 linearly independent rank-1 tensors that span
      the matrix multiplication tensor.
    
    Let me use a concrete known rank-19 decomposition.
    Reference: "On the rank of the 3x3 matrix multiplication tensor"
    
    Here's a construction based on the "balanced" approach:
    """
    rank = 19
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Indexing: A[i*3+j], B[j*3+k], C[k*3+i]
    # (Note: C index is k*3+i based on the trace convention W[k*n+i])
    
    # Let me use a known explicit rank-19 decomposition.
    # Based on the construction from the literature, here are the 19 terms:
    
    # Term 1: (a11+a21+a31)(b11+b21+b31)
    U[0,0] = 1; U[3,0] = 1; U[6,0] = 1
    V[0,0] = 1; V[3,0] = 1; V[6,0] = 1
    W[0,0] = 1/9; W[3,0] = 1/9; W[6,0] = 1/9
    # This contributes to all C entries via the outer product of row-sums and col-sums
    
    # This approach of trying to write down all 19 terms by hand is error-prone.
    # Let me use a different strategy: use the known fact that we can decompose
    # 3x3*3x3 into a 2x2*2x2 Strassen (rank 7) plus additional terms.
    
    # Actually, let me use a more systematic approach.
    # The rank-19 decomposition can be constructed as follows:
    # 
    # Partition the 3x3 matrices into blocks:
    # A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
    # where each Aij is a scalar.
    #
    # Use the following 19 products:
    # M1 = (a11+a21)(b11+b21)
    # M2 = (a11-a21)(b11-b21)  
    # M3 = (a12+a22)(b12+b22)
    # M4 = (a12-a22)(b12-b22)
    # M5 = (a13+a23)(b13+b23)
    # M6 = (a13-a23)(b13-b23)
    # M7 = (a31+a11)(b31+b11)
    # M8 = (a31-a11)(b31-b11)
    # M9 = (a32+a12)(b32+b12)
    # M10 = (a32-a12)(b32-b12)
    # M11 = (a33+a13)(b33+b13)
    # M12 = (a33-a13)(b33-b13)
    # M13 = (a21+a31)(b21+b31)
    # M14 = (a21-a31)(b21-b31)
    # M15 = (a22+a32)(b22+b32)
    # M16 = (a22-a32)(b22-b32)
    # M17 = (a23+a33)(b23+b33)
    # M18 = (a23-a33)(b23-b33)
    # M19 = (a11+a22+a33)(b11+b22+b33)  # trace-like term
    
    # But this gives 19 terms and I need to verify it actually works.
    # This is getting complicated. Let me use a verified approach instead.
    
    # Alternative: Use the fact that rank(3x3) <= rank(2x2) * 3 + some overhead
    # Actually, a cleaner approach: use the known rank-19 decomposition from
    # the paper "Lower bounds for matrix multiplication" by Bläser.
    
    # For now, let me return None and use a different strategy in construct().
    return None


def _construct_via_block_decomposition(n, m, p):
    """
    Construct a decomposition for 3x3 * 3x6 by viewing it as three 3x3 * 3x2 blocks
    or using other structured decompositions.
    
    Strategy: The 3x6 matrix B can be viewed as [B1 | B2] where B1, B2 are 3x3.
    Then A*B = [A*B1 | A*B2].
    
    If we have a rank-R decomposition for 3x3*3x3, we can use it twice
    to get rank 2R for 3x3*3x6. But we can do better by sharing structure.
    
    Better strategy: Use the field extension approach.
    Encode the 3x6 matrix as a 3x3 matrix over the ring R = F[t]/(t^2)
    or R = F x F.
    
    If we use R = F x F (pairs), then a 3x3 matrix over R represents a 3x6 matrix.
    A rank-R decomposition over R for 3x3*3x3 gives a rank-R decomposition for 3x6.
    But we need the decomposition to be over the base field.
    
    Key insight: If we have a rank-R decomposition of 3x3*3x3 over the base field
    F, then by applying it to each "component" of the pair, we get a rank-2R
    decomposition of 3x3*3x6. But we might be able to do better.
    
    Actually, the correct approach: 
    - A 3x6 matrix B can be written as B = B1 + t*B2 where t^2 = 0 (dual numbers)
      or B = B1 + B2 where we use the ring F x F.
    - If we have a rank-R decomposition for 3x3*3x3 that works over any commutative
      ring, then it works over F x F, giving a rank-R decomposition for 3x3*3x6
      where each "multiplication" is component-wise.
    - To convert back to the base field, each rank-1 term over F x F becomes
      two rank-1 terms over F (one for each component).
    - So rank-R over F x F gives rank-2R over F.
    
    Therefore, if rank(3x3*3x3) = 19, then rank(3x3*3x6) <= 2*19 = 38.
    But we might do better by finding a decomposition that exploits the structure.
    
    Let me try to directly search for a low-rank decomposition using the
    "linear dependence" method.
    """
    deadline = time.time() + _TIME_LIMIT - 20
    
    # Strategy 1: Use two rank-19 decompositions for the two 3x3 blocks
    # This gives rank 38.
    
    # Strategy 2: Try to find a decomposition with rank < 38 using optimization.
    
    # For now, let's implement a structured search.
    
    best_result = None
    best_rank = n * m * p  # 54
    
    # Try ranks from 30 to 53
    for rank in range(30, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try a few random restarts
        found = False
        for restart in range(3):
            if time.time() > deadline:
                break
            result = _try_rank_optimization(n, m, p, rank, seed=1000 + rank * 10 + restart)
            if result is not None:
                best_result = result
                best_rank = rank
                found = True
                break
        
        if found and best_rank <= 35:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: return schoolbook
    return _schoolbook(n, m, p)


def _try_rank_optimization(n, m, p, rank, seed=42):
    """
    Try to find a rank-`rank` decomposition using gradient-free optimization
    with structured initialization.
    """
    deadline = time.time() + _TIME_LIMIT - 30
    rng = random.Random(seed)
    
    # Number of non-zero tensor entries
    # T[i*m+j, j'*p+k, k'*n+i] = 1 iff j==j' and i==i' and k==k'
    # So there are n*m*p = 54 non-zero entries (all with value 1)
    
    # Initialize with random values
    U = np.array([[rng.gauss(0, 1) for _ in range(rank)] for _ in range(n * m)])
    V = np.array([[rng.gauss(0, 1) for _ in range(rank)] for _ in range(m * p)])
    W = np.array([[rng.gauss(0, 1) for _ in range(rank)] for _ in range(p * n)])
    
    best_err = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Precompute the list of (A, B, C, expected) tuples
    entries = []
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j2 * p + k
                    C = k * n + i
                    expected = 1.0 if (j == j2) else 0.0
                    entries.append((A, B, C, expected))
    
    def compute_error(U, V, W):
        err = 0.0
        for A, B, C, expected in entries:
            actual = np.dot(U[A] * V[B], W[C])
            err += (actual - expected) ** 2
        return err
    
    max_iter = 300
    step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random matrix and entry to perturb
        mat_idx = rng.randint(0, 2)
        if mat_idx == 0:
            row = rng.randint(0, n * m - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice(step_sizes) * rng.choice([-1, 1])
            old_val = U[row, col]
            U[row, col] += delta
        elif mat_idx == 1:
            row = rng.randint(0, m * p - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice(step_sizes) * rng.choice([-1, 1])
            old_val = V[row, col]
            V[row, col] += delta
        else:
            row = rng.randint(0, p * n - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice(step_sizes) * rng.choice([-1, 1])
            old_val = W[row, col]
            W[row, col] += delta
        
        err = compute_error(U, V, W)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        else:
            if mat_idx == 0:
                U[row, col] = old_val
            elif mat_idx == 1:
                V[row, col] = old_val
            else:
                W[row, col] = old_val
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-9:
        # Round to nearest rational with small denominator
        U_r = np.round(best_U * 1000) / 1000
        V_r = np.round(best_V * 1000) / 1000
        W_r = np.round(best_W * 1000) / 1000
        
        # Verify the rounded version
        if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-6):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Try with original values
        if _verify_fast(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-9):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None


def _construct_rank38_via_two_blocks(n, m, p):
    """
    Construct a rank-38 decomposition by using two rank-19 decompositions
    for the two 3x3 blocks of the 3x6 matrix.
    
    B = [B1 | B2] where B1, B2 are 3x3.
    A*B = [A*B1 | A*B2]
    
    If we have rank-19 decompositions for A*B1 and A*B2, we get rank-38 total.
    But we need to verify that rank-19 decompositions exist and are correct.
    
    For now, this is a placeholder. The actual rank-19 decomposition is complex
    to write down explicitly.
    """
    # This function would return a rank-38 decomposition if we had verified
    # rank-19 decompositions for 3x3*3x3.
    # Since constructing them explicitly is complex, we fall back to search.
    return None


def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    fallback = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured optimization
    result = _construct_via_block_decomposition(n, m, p)
    if result is not None:
        U, V, W = result["U"], result["V"], result["W"]
        rank = len(U[0])
        if rank < best_rank and _verify(U, V, W, n, m, p):
            return result
    
    return fallback

# EVOLVE-BLOCK-END
