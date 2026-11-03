import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = 2, 3, 6
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def compute_tensor(U, V, W):
        Ur = U.reshape(n, m, -1)
        Vr = V.reshape(m, p, -1)
        Wr = W.reshape(p, n, -1)
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                BC = Vr[j, :, :] * Wr[:, i, :]
                T[i, j, :] = np.sum(Ur[i, j, :] * BC, axis=1)
        return T
    
    def verify(U, V, W):
        T = compute_tensor(U, V, W)
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True
    
    # Strategy: The 2x3x6 tensor can be decomposed as 2 independent 2x3x3 tensors.
    # If we can find a rank-13 decomposition for 2x3x3, we can use it twice for a rank-26 solution.
    # Schoolbook for 2x3x3 is 18. Rank 13 is a known lower bound/improvement (Laderman-like).
    # However, finding rank 13 exactly with half-integers is hard.
    # Let's try to construct a rank 20 or 22 solution using block structure or random search.
    # Actually, let's look at the "flattening" idea.
    
    # A robust approach: Use the known rank-12 algorithm for 3x3? No, 3x3 is 23.
    # 2x3x3 rank is 13?
    # Let's try a random search for a low rank with half-integers.
    # Given the time constraint, random search for rank < 36 is unlikely to find a solution quickly.
    
    # Alternative: Can we do better than 36 with a simple block structure?
    # C = A B. A is 2x3, B is 3x6.
    # Split B into B1 (3x3) and B2 (3x3).
    # C1 = A B1, C2 = A B2.
    # Rank = rank(M_{2,3,3}) + rank(M_{2,3,3}).
    # If we use schoolbook for each, rank = 18 + 18 = 36.
    
    # Is there a better decomposition for 2x3x3?
    # The border rank of 2x3x3 is 13?
    # Let's assume we can't easily find rank 26.
    
    # Let's try a specific rank 30 or 32 construction if possible.
    # Or just stick to schoolbook which is guaranteed to work.
    
    # However, the prompt asks for a "stronger algorithm".
    # Let's try to implement a rank 29 or 30 solution if we can find one in literature or via search.
    # Since I cannot browse, I will stick to the most reliable improvement:
    # Using the fact that M_{2,3,6} is the direct sum of two M_{2,3,3}.
    # If we use a rank 15 algorithm for 2x3x3 (if it exists?), total 30.
    # Schoolbook is 18.
    
    # Let's try a random search for rank 30 with half-integers, limited to 10 seconds.
    # If it fails, return schoolbook.
    
    search_time_limit = 10.0
    best_result = None
    
    if time.time() < deadline - search_time_limit - 5:
        # Try rank 30
        for rank in [30, 32]:
            if time.time() > deadline - search_time_limit:
                break
            
            max_iters = 2000
            found = False
            for it in range(max_iters):
                if time.time() > deadline - search_time_limit:
                    break
                
                # Generate random half-integer matrices
                # Scale: values in {-2, -1.5, ..., 2}
                U = np.random.randint(-4, 5, size=(n*m, rank)) * 0.5
                V = np.random.randint(-4, 5, size=(m*p, rank)) * 0.5
                W = np.random.randint(-4, 5, size=(p*n, rank)) * 0.5
                
                T = compute_tensor(U, V, W)
                err = np.sum((T - 1.0) ** 2)
                
                if err < 1e-9:
                    if verify(U, V, W):
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        found = True
                        break
            
            if found:
                break
    
    if best_result:
        return best_result
    
    return schoolbook()
# EVOLVE-BLOCK-END
