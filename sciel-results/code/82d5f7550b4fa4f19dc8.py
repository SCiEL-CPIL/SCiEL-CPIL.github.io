import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """Strassen 2x2x2: U(4,7), V(4,7), W(4,7) with trace convention W[k*2+i]."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1=(A11-A10)(B00+B01)
    U[3][0]=1; U[2][0]=-1; V[0][0]=1; V[1][0]=1; W[0][0]=1; W[2][0]=-1; W[3][0]=1
    # M2=(A10+A11)(B10+B11)
    U[2][1]=1; U[3][1]=1; V[2][1]=1; V[3][1]=1; W[2][1]=1; W[3][1]=1
    # M3=(A00+A10)(B00)
    U[0][2]=1; U[2][2]=1; V[0][2]=1; W[2][2]=1; W[3][2]=1
    # M4=(A11+A01)(B11)
    U[3][3]=1; U[1][3]=1; V[3][3]=1; W[0][3]=1; W[2][3]=-1
    # M5=(A00-A10)(B01+B11)
    U[0][4]=1; U[2][4]=-1; V[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    # M6=(A01+A11)(B10-B00)
    U[1][5]=1; U[3][5]=1; V[2][5]=1; V[0][5]=-1; W[2][5]=-1; W[3][5]=-1
    # M7=(A00+A01)(B00-B10)
    U[0][6]=1; U[1][6]=1; V[0][6]=1; V[2][6]=-1; W[0][6]=1; W[2][6]=-1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify tensor decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] = delta."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            a = i * m + j
                            b = j2 * p + k
                            c = k2 * n + i2
                            expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                            actual = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _verify_fast(U, V, W, n, m, p, rank):
    """Faster verification using numpy operations."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all diagonal entries (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                actual = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(actual - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal entries (should be 0)
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We can compute this as a batch operation
    for a in range(n * m):
        for b in range(m * p):
            # For each a, b, the vector sum_r U[a,r] * V[b,r] should be such that
            # dot with W[c,:] gives 0 for all c where (j != j2 or i != i2 or k != k2)
            # This is equivalent to: for all c, sum_r U[a,r] * V[b,r] * W[c,r] = 0
            # Let X[a,b] = sum_r U[a,r] * V[b,r] ... no that's not right
            # T[a,b,c] = sum_r (U[a,r] * V[b,r]) * W[c,r]
            # For fixed a,b: T[a,b,:] = U[a,:] * V[b,:] contracted with W
            # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # = sum_r (U[a,r] * V[b,r]) * W[c,r]
            # This is not a simple dot product because of the three-way product
            
            # Let's just check a sample
            pass
    
    # Better approach: compute T as a tensor and check
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This can be written as: for each r, outer product of U[:,r], V[:,r], W[:,r]
    # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # For speed, let's check using random sampling
    np.random.seed(12345)
    num_checks = min(2000, (n*m) * (m*p) * (p*n) // 10)
    for _ in range(num_checks):
        a = np.random.randint(n * m)
        b = np.random.randint(m * p)
        c = np.random.randint(p * n)
        actual = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
        
        # Determine expected value
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
        
        if abs(actual - expected) > 1e-9:
            return False
    
    return True

def _build_strassen_4x4x8():
    """Build 4x4x8 decomposition using Strassen Kronecker + p-doubling.
    
    Step 1: 2x2x2 Strassen (rank 7)
    Step 2: Kronecker 2x2x2 ⊗ 2x2x2 -> 4x4x4 (rank 49)
    Step 3: p-doubling -> 4x4x8 (rank 98)
    """
    U2, V2, W2 = _strassen_2x2()
    r2 = 7
    
    # Step 2: Kronecker to get 4x4x4
    r4 = r2 * r2  # 49
    U4 = [[0]*r4 for _ in range(16)]
    V4 = [[0]*r4 for _ in range(16)]
    W4 = [[0]*r4 for _ in range(16)]
    
    for rb in range(r2):
        for rs in range(r2):
            col = rb * r2 + rs
            # U: a = ib*8 + i0*4 + jb*2 + j0
            for ib in range(2):
                for i0 in range(2):
                    for jb in range(2):
                        for j0 in range(2):
                            a = ib * 8 + i0 * 4 + jb * 2 + j0
                            ab = ib * 2 + jb
                            as_idx = i0 * 2 + j0
                            U4[a][col] = U2[ab][rb] * U2[as_idx][rs]
            # V: b = jb2*8 + j02*4 + kb*2 + k0
            for jb2 in range(2):
                for j02 in range(2):
                    for kb in range(2):
                        for k0 in range(2):
                            b = jb2 * 8 + j02 * 4 + kb * 2 + k0
                            bb = jb2 * 2 + kb
                            bs = j02 * 2 + k0
                            V4[b][col] = V2[bb][rb] * V2[bs][rs]
            # W: c = kb2*8 + k02*4 + ib2*2 + i02
            for kb2 in range(2):
                for k02 in range(2):
                    for ib2 in range(2):
                        for i02 in range(2):
                            c = kb2 * 8 + k02 * 4 + ib2 * 2 + i02
                            cb = kb2 * 2 + ib2
                            cs = k02 * 2 + i02
                            W4[c][col] = W2[cb][rb] * W2[cs][rs]
    
    # Step 3: p-doubling to get 4x4x8
    r8 = 2 * r4  # 98
    U8 = [[0]*r8 for _ in range(16)]
    V8 = [[0]*r8 for _ in range(32)]
    W8 = [[0]*r8 for _ in range(32)]
    
    for r in range(r4):
        for i in range(16):
            U8[i][r] = U4[i][r]
            U8[i][r + r4] = U4[i][r]
        for j in range(4):
            for k in range(4):
                V8[j * 8 + k][r] = V4[j * 4 + k][r]
                V8[j * 8 + (k + 4)][r + r4] = V4[j * 4 + k][r]
        for k in range(4):
            for i in range(4):
                W8[k * 4 + i][r] = W4[k * 4 + i][r]
                W8[(k + 4) * 4 + i][r + r4] = W4[k * 4 + i][r]
    
    return U8, V8, W8

def _try_direct_4x4x8():
    """Try to find a better decomposition for 4x4x8 directly.
    
    Approach: Use the fact that 4x4x8 can be viewed as two independent 4x4x4 multiplications,
    but try to find shared structure.
    
    Alternative: Use Laderman's algorithm for 3x3x3 (rank 23) if applicable,
    but 4x4x8 doesn't factor cleanly.
    
    For now, return None to indicate no improvement found.
    """
    return None

def _try_rank_reduction(U, V, W, n, m, p, base_rank, max_time=10):
    """Try to reduce the rank by finding linear dependencies in the factor matrices.
    
    Strategy: The rank-98 solution has U(16,98), V(32,98), W(32,98).
    If we can find linear dependencies, we can reduce the rank.
    
    Approach: Use SVD to find low-rank approximations, then verify.
    """
    start = time.time()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try SVD-based rank reduction
    # The key insight: if the columns of U, V, W have dependencies, we can reduce rank
    
    # For each factor, compute SVD and keep only top r singular vectors
    # But we need to maintain the exact decomposition
    
    # Alternative: Use the tensor structure
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is a 3-way tensor with rank r
    
    # Try to find if there's a lower-rank representation
    # This is equivalent to finding the tensor rank, which is NP-hard
    
    # For now, just return the original
    return None

def schoolbook_decomposition(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 128)
    best_result = schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen-based construction: 2x2x2 -> 4x4x4 (rank 49) -> 4x4x8 (rank 98)
    try:
        U8, V8, W8 = _build_strassen_4x4x8()
        if U8 is not None:
            rank = len(U8[0])
            # Fast verification first
            if _verify_fast(U8, V8, W8, n, m, p, rank):
                # Full verification
                if _verify_decomposition(U8, V8, W8, n, m, p, rank):
                    best_result = {"U": U8, "V": V8, "W": W8}
                    best_rank = rank
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_result
    
    # Try direct 4x4x8 construction
    try:
        result = _try_direct_4x4x8()
        if result is not None:
            U8, V8, W8 = result
            rank = len(U8[0])
            if rank < best_rank:
                if _verify_fast(U8, V8, W8, n, m, p, rank):
                    if _verify_decomposition(U8, V8, W8, n, m, p, rank):
                        best_result = {"U": U8, "V": V8, "W": W8}
                        best_rank = rank
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_result
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
