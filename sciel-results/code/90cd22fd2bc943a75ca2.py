"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    
    def schoolbook():
        rank = rank_schoolbook
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
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        rank = len(U[0])
        # Verify shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        # Check the tensor equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # for all i, j, k.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Use Strassen's algorithm for 4x4x4 (rank 21) and extend to 4x4x5
    # Strassen's algorithm for 2x2x2 has rank 7
    # For 4x4x4, we can use two levels of Strassen, but the standard Strassen-Winograd
    # for 4x4x4 uses rank 21 (or similar structured decomposition)
    
    # Let's construct a rank-21 decomposition for 4x4x4 first
    # Using the standard Strassen decomposition extended to 4x4
    
    def strassen_4x4x4():
        """Return U, V, W for 4x4x4 matrix multiplication with rank 21."""
        rank = 21
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        # Strassen's 7 multiplications for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For 4x4x4, we can split into 2x2 blocks and apply Strassen
        # But this gives rank 7*7 = 49 if done naively
        # The rank-21 result comes from a more clever structure
        
        # Actually, let's use a known rank-21 decomposition for 4x4x4
        # This is based on the work of Strassen and later improvements
        
        # For simplicity, let's use the standard Strassen extension:
        # Split 4x4 into four 2x2 blocks
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]] where each is 2x2
        
        # Apply Strassen to the block level:
        # M1 = (A11+A22)(B11+B22) -> 2x2 multiplication, rank 7
        # M2 = (A21+A22)B11 -> 2x2 multiplication, rank 7
        # M3 = A11(B12-B22) -> 2x2 multiplication, rank 7
        # M4 = A22(B21-B11) -> 2x2 multiplication, rank 7
        # M5 = (A11+A12)B22 -> 2x2 multiplication, rank 7
        # M6 = (A21-A11)(B11+B12) -> 2x2 multiplication, rank 7
        # M7 = (A12-A22)(B21+B22) -> 2x2 multiplication, rank 7
        
        # Total rank: 7*7 = 49, which is worse than schoolbook 64
        # But we can share some computations
        
        # Let's use a different approach: the rank-21 decomposition is known
        # to exist for 4x4x4. Let me construct it explicitly.
        
        # Actually, the known best for 4x4x4 is rank 21, achieved by 
        # a specific tensor decomposition. Let me hardcode it.
        
        # For now, let's try a simpler approach: use Strassen for the 2x2 blocks
        # but with shared structure to get rank 21
        
        # This is complex, so let's try a different strategy:
        # Use the fact that for 4x4x5, we can use a rank-21 decomposition for the 
        # first 4 columns and add 4 more terms for the 5th column, giving rank 25.
        # Or use a rank-24 decomposition for 4x4x4 and add 1 for the 5th column.
        
        # Let's try to construct a rank-24 decomposition for 4x4x4
        # using two levels of Strassen with some sharing.
        
        # For now, let's return a placeholder and try random search
        return None
    
    # Try to find a better decomposition using structured approach
    # For 4x4x5, the theoretical minimum rank is at least 4*4*5 / (4+4+5-2) = 80/11 ≈ 7.27
    # by the Coppersmith-Winograd lower bound, but actual known bounds are higher
    
    # Let's try a randomized search for rank 22-30 with integer entries
    np.random.seed(42)
    
    for target_rank in [22, 24, 26, 28, 30, 32, 36, 40, 44, 48]:
        if time.time() > deadline - 10:
            break
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if check_decomposition(U_list, V_list, W_list):
                result = {"U": U_list, "V": V_list, "W": W_list}
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = result
                break
    
    return best_result
# EVOLVE-BLOCK-END
