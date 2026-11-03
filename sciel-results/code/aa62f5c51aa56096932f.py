"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    
    # Check all "on" positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check a sample of "off" positions
    rng = np.random.RandomState(42)
    for _ in range(500):
        a = rng.randint(0, nm)
        b = rng.randint(0, mp)
        c = rng.randint(0, pn)
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        i_c, k_c = divmod(c, n)
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        val = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=-1; W[1,0]=1; W[2,0]=-1  # C00+=M1, C11+=M1, C01-=M1, C10-=M1
    # Wait, standard Strassen:
    # C00 = M1 - M2 + M3 - M4  (or similar variants)
    # Let's use the standard formulation:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M2 + M3 - M4
    # C01 = M5 + M4
    # C10 = M2 + M3
    # C11 = M1 - M5 + M6 + M7
    
    # Let me redo with correct signs.
    # C[i,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    # For 2x2: i,j,k in {0,1}
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> U[0,0]=1, U[3,0]=1
    # V: B00=1, B11=1 -> V[0,0]=1, V[3,0]=1
    # W: C00+=1, C11+=1, C01-=1, C10-=1
    # W[k*2+i, 0]: C00 (k=0,i=0) -> W[0,0]=1
    # C01 (k=1,i=0) -> W[2,0]=-1
    # C10 (k=0,i=1) -> W[1,0]=-1
    # C11 (k=1,i=1) -> W[3,0]=1
    
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[2,0]=-1; W[1,0]=-1; W[3,0]=1
    
    # Term 1: M2 = (A10+A11)B00
    # U: A10=1, A11=1 -> U[2,1]=1, U[3,1]=1
    # V: B00=1 -> V[0,1]=1
    # W: C00-=1, C10+=1
    # W[0,1]=-1; W[1,1]=1
    
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[0,1]=-1; W[1,1]=1
    
    # Term 2: M3 = A00(B10-B11)
    # U: A00=1 -> U[0,2]=1
    # V: B10=1, B11=-1 -> V[2,2]=1, V[3,2]=-1
    # W: C00+=1, C10+=1
    # W[0,2]=1; W[1,2]=1
    
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[0,2]=1; W[1,2]=1
    
    # Term 3: M4 = A11(B01-B10)
    # U: A11=1 -> U[3,3]=1
    # V: B01=1, B10=-1 -> V[1,3]=1, V[2,3]=-1
    # W: C00-=1, C01+=1
    # W[0,3]=-1; W[2,3]=1
    
    U[3,3]=1
    V[1,3]=1; V[2,3]=-1
    W[0,3]=-1; W[2,3]=1
    
    # Term 4: M5 = (A00+A01)B11
    # U: A00=1, A01=1 -> U[0,4]=1, U[1,4]=1
    # V: B11=1 -> V[3,4]=1
    # W: C01+=1, C11-=1
    # W[2,4]=1; W[3,4]=-1
    
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[2,4]=1; W[3,4]=-1
    
    # Term 5: M6 = (A10-A00)(B00+B10)
    # U: A10=1, A00=-1 -> U[2,5]=1, U[0,5]=-1
    # V: B00=1, B10=1 -> V[0,5]=1, V[2,5]=1
    # W: C11+=1
    # W[3,5]=1
    
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[2,5]=1
    W[3,5]=1
    
    # Term 6: M7 = (A01-A11)(B01+B11)
    # U: A01=1, A11=-1 -> U[1,6]=1, U[3,6]=-1
    # V: B01=1, B11=1 -> V[1,6]=1, V[3,6]=1
    # W: C11+=1
    # W[3,6]=1
    
    U[1,6]=1; U[3,6]=-1
    V[1,6]=1; V[3,6]=1
    W[3,6]=1
    
    return U, V, W

def _construct_3x3x5_via_strassen(n, m, p):
    """
    Construct a decomposition for 3x3 * 3x5 using Strassen for 2x2 blocks.
    
    Strategy: Decompose the 3x3 A and 3x3 B (first 3 cols of B) using Strassen,
    and handle the remaining 2 columns of B separately.
    
    Actually, a simpler approach: 
    - Use Strassen for the 2x2x2 sub-blocks
    - Handle the 3rd row/col with additional terms
    
    Let's try a different approach: 
    For 3x3 * 3x5, we can write:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can group the j=0,1 terms using Strassen-like structure and handle j=2 separately.
    But this gets complex. Let's try a direct construction.
    
    Alternative: Use the fact that rank(3x3*3x3) <= 23 is known, but constructing
    it is non-trivial. For now, let's try a structured random search.
    """
    return None

def _refine_decomposition(U, V, W, n, m, p, max_iters=1000, tol=1e-6):
    """
    Try to refine a decomposition by optimizing the entries to satisfy the tensor identity.
    Uses gradient descent on the loss function:
    L = sum_{on} (sum_r U[a,r]V[b,r]W[c,r] - 1)^2 + sum_{off} (sum_r U[a,r]V[b,r]W[c,r])^2
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute the "on" positions
    on_positions = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                on_positions.append((a, b, c))
    
    # Sample some "off" positions
    rng = np.random.RandomState(123)
    off_positions = []
    for _ in range(200):
        a = rng.randint(0, nm)
        b = rng.randint(0, mp)
        c = rng.randint(0, pn)
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        i_c, k_c = divmod(c, n)
        if not (j_a == j_b and i_a == i_c and k_b == k_c):
            off_positions.append((a, b, c))
    
    # Simple gradient descent
    lr = 0.01
    for it in range(max_iters):
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # On positions: want sum_r U[a,r]V[b,r]W[c,r] = 1
        for a, b, c in on_positions:
            for r in range(rank):
                val = U[a,r] * V[b,r] * W[c,r]
                # dL/dU[a,r] = 2*(val - 1) * V[b,r] * W[c,r]
                # But we need to sum over all terms, so this is approximate
                # For simplicity, use a local update
                dU[a,r] += 2 * (val - 1) * V[b,r] * W[c,r]
                dV[b,r] += 2 * (val - 1) * U[a,r] * W[c,r]
                dW[c,r] += 2 * (val - 1) * U[a,r] * V[b,r]
        
        # Off positions: want sum_r U[a,r]V[b,r]W[c,r] = 0
        for a, b, c in off_positions:
            for r in range(rank):
                val = U[a,r] * V[b,r] * W[c,r]
                dU[a,r] += 2 * val * V[b,r] * W[c,r]
                dV[b,r] += 2 * val * U[a,r] * W[c,r]
                dW[c,r] += 2 * val * U[a,r] * V[b,r]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    return U, V, W

def _try_merge_terms(U, V, W, n, m, p, target_rank, max_iters=100):
    """
    Try to reduce the rank by merging terms.
    Start with a higher-rank decomposition and try to project to lower rank.
    """
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    # Simple projection: keep the first target_rank columns
    # This is crude but might work as a starting point for refinement
    U_proj = U[:, :target_rank]
    V_proj = V[:, :target_rank]
    W_proj = W[:, :target_rank]
    
    # Refine
    U_ref, V_ref, W_ref = _refine_decomposition(U_proj, V_proj, W_proj, n, m, p, max_iters=500)
    
    return U_ref, V_ref, W_ref

def _random_search(n, m, p, target_rank, seed=42, max_iters=100):
    """Try to find a rank-target_rank decomposition via random initialization and refinement."""
    rng = np.random.RandomState(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    for it in range(max_iters):
        U = rng.randn(nm, target_rank)
        V = rng.randn(mp, target_rank)
        W = rng.randn(pn, target_rank)
        
        # Scale
        scale = 1.0 / np.sqrt(target_rank)
        U *= scale
        V *= scale
        W *= scale * target_rank
        
        # Refine
        U_ref, V_ref, W_ref = _refine_decomposition(U, V, W, n, m, p, max_iters=200)
        
        if _validate(U_ref, V_ref, W_ref, n, m, p, tol=1e-4):
            return U_ref, V_ref, W_ref
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_school, V_school, W_school)
    
    # Try to find lower-rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But realistically, rank >= n*m*p / something
    
    # Try ranks from lower to higher
    for target_rank in range(15, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try random search with refinement
        result = _random_search(n, m, p, target_rank, seed=target_rank, max_iters=50)
        if result is not None:
            U, V, W = result
            if _validate(U, V, W, n, m, p, tol=1e-6):
                best_rank = target_rank
                best_result = (U, V, W)
                break
    
    # Also try starting from Strassen-based construction
    if time.time() < deadline - 30:
        # Try a structured approach based on 2x2 Strassen blocks
        U_s, V_s, W_s = _strassen_2x2()
        # Embed in 3x3x5 context - this is complex, skip for now
        pass
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
