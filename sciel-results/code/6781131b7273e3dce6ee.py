"""Find rank-minimal exact tensor decomposition for 2x5 x 5x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]
_START_TIME = time.time()
DEADLINE = _START_TIME + 140

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _verify_numpy(U, V, W, n, m, p):
    """Vectorized verification."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _strassen_2x2(n, m, p):
    """
    Use Strassen's algorithm to multiply 2x2 blocks of the matrices.
    For 2x5 x 5x6, we can partition as:
    A = [[A11, A12], [A21, A22]] where A11, A21 are 1x5, A12, A22 are 1x5? No.
    A is 2x5. B is 5x6.
    We can't directly use 2x2 Strassen on the outer dimensions because A is 2x5.
    
    However, we can view the multiplication as:
    C = A * B
    A is 2x5, B is 5x6.
    
    Let's try to decompose the 5x6 multiplication into blocks.
    B is 5x6.
    We can split B into columns: B1 (5x2), B2 (5x2), B3 (5x2).
    C = [A*B1, A*B2, A*B3].
    Each A*Bi is 2x2.
    We can use Strassen for each 2x2 multiplication?
    A is 2x5. Bi is 5x2.
    This is not 2x2 x 2x2.
    
    Let's try a different approach.
    The tensor is T_{i,j,k} where i in 0..1, j in 0..4, k in 0..5.
    T_{i,j,k} = 1.
    
    We can try to find a low rank decomposition using the structure.
    Rank 54 is known to be achievable.
    Let's construct the rank 54 solution from the knowledge.
    "Structured Strassen decomposition of 2x2 sub-blocks reduces rank from 60 to 54"
    
    How?
    Maybe we can treat the 5x6 part as three 5x2 multiplications?
    Or maybe we can use the fact that 2x5 * 5x6 can be computed with fewer multiplications.
    
    Let's look at the dimensions: 2, 5, 6.
    Schoolbook: 60.
    Strassen on 2x2 blocks?
    If we pad A to 2x6 and B to 6x6?
    2x6 * 6x6.
    Rank for 2x6 * 6x6?
    Schoolbook: 72.
    Strassen?
    
    Let's try to construct the rank 54 solution directly.
    The rank 54 solution likely uses 60 - 6 = 54 multiplications.
    
    Let's try to use the following decomposition:
    We can write the 2x5 matrix A as a sum of 2 matrices?
    Or use the fact that 5 = 2+3 and 6 = 2+4?
    
    Actually, there is a known result for small matrix multiplication.
    For 2x5 x 5x6, the rank is likely 54 or lower.
    
    Let's try to find a rank 54 decomposition using ALS or random search.
    But the prompt says "Structured Strassen decomposition ... reduces rank from 60 to 54".
    So there is a specific construction.
    
    Let's try to implement a generic ALS solver and hope it finds rank 54 or lower.
    """
    return _schoolbook(n, m, p)

def _rationalize(U, V, W, n, m, p):
    """Try to round to nearest half-integers and verify."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i*m+j], V_r[j*p+k] * W_r[k*n+i])
                if abs(val - 1) > 1e-9:
                    return None
    
    rank = U.shape[1]
    U_list = [[float(U_r[i,j]) for j in range(rank)] for i in range(U_r.shape[0])]
    V_list = [[float(V_r[i,j]) for j in range(rank)] for i in range(V_r.shape[0])]
    W_list = [[float(W_r[i,j]) for j in range(rank)] for i in range(W_r.shape[0])]
    return {"U": U_list, "V": V_list, "W": W_list}

def _try_rank_als(n, m, p, rank, deadline, seed_offset=0):
    """Alternating least squares to find a rank decomposition."""
    max_restarts = 50
    max_iter = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(12345 + restart * 7919 + seed_offset)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U rows
            for i in range(n):
                for j in range(m):
                    M = np.empty((p, rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            
            # Update V rows
            for j in range(m):
                for k in range(p):
                    N_mat = np.empty((n, rank))
                    for i in range(n):
                        N_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(N_mat.T, np.ones(n), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            
            # Update W rows
            for k in range(p):
                for i in range(n):
                    P_mat = np.empty((m, rank))
                    for j in range(m):
                        P_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(P_mat.T, np.ones(m), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            
            # Check convergence
            if it % 10 == 9:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-7:
                    result = _rationalize(U, V, W, n, m, p)
                    if result:
                        return result
                elif max_err < 1e-4:
                    # Try a final polish
                    for _ in range(50):
                        for i in range(n):
                            for j in range(m):
                                M = np.empty((p, rank))
                                for k in range(p):
                                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                                try:
                                    U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                                except:
                                    pass
                        for j in range(m):
                            for k in range(p):
                                N_mat = np.empty((n, rank))
                                for i in range(n):
                                    N_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                                try:
                                    V[j * p + k, :] = np.linalg.lstsq(N_mat.T, np.ones(n), rcond=None)[0]
                                except:
                                    pass
                        for k in range(p):
                            for i in range(n):
                                P_mat = np.empty((m, rank))
                                for j in range(m):
                                    P_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                                try:
                                    W[k * n + i, :] = np.linalg.lstsq(P_mat.T, np.ones(m), rcond=None)[0]
                                except:
                                    pass
                        if time.time() > deadline:
                            break
                    result = _rationalize(U, V, W, n, m, p)
                    if result:
                        return result
    
    return None

def _try_rank_sparse(n, m, p, rank, deadline, seed_offset=0):
    """Try sparse random initializations with values in {-1, -0.5, 0, 0.5, 1}."""
    vals = [-1, -0.5, 0, 0.5, 1]
    max_restarts = 60
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(777 + restart * 31 + seed_offset)
        U = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(n*m)])
        V = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(m*p)])
        W = np.array([[rng.choice(vals) for _ in range(rank)] for _ in range(p*n)])
        
        if _verify_numpy(U, V, W, n, m, p):
            U_list = [[float(U[i,j]) for j in range(rank)] for i in range(U.shape[0])]
            V_list = [[float(V[i,j]) for j in range(rank)] for i in range(V.shape[0])]
            W_list = [[float(W[i,j]) for j in range(rank)] for i in range(W.shape[0])]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _construct_rank_54(n, m, p):
    """
    Construct a rank 54 decomposition.
    This is based on the idea that 2x5 * 5x6 can be computed with 54 multiplications.
    One way is to use the fact that 5 = 2+3 and 6 = 2+4, and use block multiplication.
    Or use the Strassen-like decomposition for 2x2 blocks.
    
    Let's try to find a specific construction.
    A is 2x5, B is 5x6.
    C = A * B.
    
    We can write A = [A1, A2] where A1 is 2x2, A2 is 2x3.
    B = [B1; B2] where B1 is 2x6, B2 is 3x6.
    C = A1*B1 + A2*B2.
    
    A1*B1 is 2x2 * 2x6 = 2x6. Rank 12.
    A2*B2 is 2x3 * 3x6 = 2x6. Rank 36.
    Total rank 48? No, the sum of ranks is not the rank of the sum.
    But if the supports are disjoint, the rank is the sum.
    Here the supports are the same (all 2x6 entries).
    So we need to combine them.
    
    Actually, the rank of the sum of two tensors is at most the sum of their ranks.
    So rank(C) <= rank(A1*B1) + rank(A2*B2) = 12 + 36 = 48.
    This would be an improvement over 54!
    
    Wait, is this valid?
    A1*B1 uses 2x2 * 2x6. The tensor is T_{i,j,k} for i in 0..1, j in 0..1, k in 0..5.
    A2*B2 uses 2x3 * 3x6. The tensor is T_{i,j,k} for i in 0..1, j in 2..4, k in 0..5.
    These are disjoint in the j index.
    So the total tensor is the sum of two disjoint tensors.
    The rank is the sum of the ranks.
    Rank(A1*B1) = 2*2*6 = 24? No, schoolbook for 2x2 * 2x6 is 2*2*6 = 24.
    Rank(A2*B2) = 2*3*6 = 36.
    Total rank = 24 + 36 = 60.
    This is the same as schoolbook.
    
    So we need a better decomposition.
    
    Let's try 2x5 * 5x6 = (2x2 + 2x3) * (2x6 + 3x6).
    If we use Strassen for the 2x2 * 2x2 part?
    But B is 2x6, not 2x2.
    
    Let's try to use the known rank 54 construction.
    It likely involves a clever combination of blocks.
    
    For now, let's just return the schoolbook solution and rely on ALS to find a better one.
    """
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from 54 down to 30
    for R in range(54, 29, -1):
        if time.time() > DEADLINE - 15:
            break
        
        # First try sparse random
        result = _try_rank_sparse(n, m, p, R, DEADLINE - 10, seed_offset=R)
        if result is not None:
            best = result
            best_rank = R
            break
        
        # Then try ALS
        if time.time() > DEADLINE - 10:
            break
        result = _try_rank_als(n, m, p, R, DEADLINE - 10, seed_offset=R * 100)
        if result is not None:
            best = result
            best_rank = R
            break
    
    return best
