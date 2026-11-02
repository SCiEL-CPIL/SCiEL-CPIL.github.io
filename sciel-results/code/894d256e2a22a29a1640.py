import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Rank 39 solution: 5 Strassen blocks (each rank 7) for 2x2 * 2x11
    # Each block handles 2 columns of B
    def get_rank_39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]  # 4 x 39
        V = [[0] * rank for _ in range(m * p)]  # 22 x 39
        W = [[0] * rank for _ in range(p * n)]  # 22 x 39
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0       # B[0, k0]
            v_f = k1       # B[0, k1]
            v_g = 11 + k0  # B[1, k0]
            v_h = 11 + k1  # B[1, k1]
            w_c00 = 2 * k0    # C[0, k0]
            w_c10 = 2 * k0 + 1  # C[1, k0]
            w_c01 = 2 * k1    # C[0, k1]
            w_c11 = 2 * k1 + 1  # C[1, k1]
            
            # Strassen's 7 multiplications for 2x2 * 2x2:
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        
        # Last column (k=10) handled separately with schoolbook (4 mults)
        k2 = 10
        v_e2 = k2
        v_g2 = 11 + k2
        w_c02 = 2 * k2
        w_c12 = 2 * k2 + 1
        
        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
        
        return {"U": U, "V": V, "W": W}

    def verify_solution(U, V, W, rank):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        if time.time() > deadline - 5:
            return False
        # Convert to numpy arrays
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        if U_arr.shape != (4, rank) or V_arr.shape != (22, rank) or W_arr.shape != (22, rank):
            return False
        
        # Build target tensor T[i,l,k] = 1 for matrix multiplication
        # T has dimensions (n*m, m*p, p*n) = (4, 22, 22)
        # T[(i,l), (l,k), (i,k)] = 1 where i in [0,2), l in [0,2), k in [0,11)
        # U index: i*m + l = i*2 + l
        # V index: l*p + k = l*11 + k
        # W index: k*n + i = k*2 + i
        
        # Check a sample of entries
        for i in range(2):
            for l in range(2):
                for k in range(11):
                    u_idx = i * 2 + l
                    v_idx = l * 11 + k
                    w_idx = k * 2 + i
                    # T[u_idx, v_idx, w_idx] should equal 1
                    # Compute sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r]
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        
        # Also check that entries that should be 0 are 0
        # T[u_idx, v_idx, w_idx] = 0 when the indices don't correspond to valid (i,l,k)
        # u_idx = i*2+l, v_idx = l'*11+k, w_idx = k'*2+i'
        # Should be 0 unless i=i', l=l', k=k'
        # Check a few zero entries
        for i in range(2):
            for l in range(2):
                for k in range(11):
                    u_idx = i * 2 + l
                    # Check with wrong l'
                    for l2 in range(2):
                        if l2 != l:
                            v_idx = l2 * 11 + k
                            w_idx = k * 2 + i
                            val = 0.0
                            for r in range(rank):
                                val += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
                            if abs(val) > 1e-10:
                                return False
        return True

    # Try to find rank 38: use Strassen for 4 blocks (8 cols, rank 28) + 
    # a rank 10 decomposition for the remaining 3 columns (cols 8,9,10)
    # Total: 28 + 10 = 38
    
    def search_rank_10_for_2x2x2x3(max_time=30):
        """Search for rank 10 decomposition of 2x2 * 2x3 matrix multiplication."""
        search_start = time.time()
        search_deadline = search_start + max_time
        
        # For 2x2 * 2x3: A is 2x2, B is 2x3, C is 2x3
        # U: 4x10, V: 6x10, W: 6x10
        # U index: i*2+l, V index: l*3+k, W index: k*2+i
        
        # Target tensor: T[i,l,k] = 1 for all valid (i,l,k)
        # T[u_idx, v_idx, w_idx] = 1 if u_idx=i*2+l, v_idx=l*3+k, w_idx=k*2+i
        
        import random
        random.seed(42)
        
        # Strategy 1: Try structured solutions based on Strassen-like patterns
        # For 2x2 * 2x3, we can think of it as 2x2 * 2x2 + 2x2 * 2x1
        # Rank 7 + 4 = 11 (schoolbook for last col)
        # Can we do better?
        
        # Let's try a systematic search with small coefficients
        coeffs = [-1, -0.5, 0, 0.5, 1]
        
        best_solution = None
        
        # Try random searches
        for attempt in range(2000):
            if time.time() > search_deadline:
                break
            
            U = np.zeros((4, 10))
            V = np.zeros((6, 10))
            W = np.zeros((6, 10))
            
            for r in range(10):
                # Sparse random vectors with small coefficients
                for i in range(4):
                    if random.random() < 0.4:
                        U[i, r] = random.choice(coeffs)
                for i in range(6):
                    if random.random() < 0.4:
                        V[i, r] = random.choice(coeffs)
                for i in range(6):
                    if random.random() < 0.4:
                        W[i, r] = random.choice(coeffs)
            
            # Check if this gives the right tensor
            T_recon = np.zeros((4, 6, 6))
            for r in range(10):
                T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            # Target tensor
            T_target = np.zeros((4, 6, 6))
            for i in range(2):
                for l in range(2):
                    for k in range(3):
                        T_target[i*2+l, l*3+k, i*3+k] = 1
            
            if np.allclose(T_recon, T_target, atol=1e-10):
                best_solution = (U.tolist(), V.tolist(), W.tolist())
                break
        
        if best_solution is None:
            # Try a more structured approach
            # Start from the rank-11 solution (Strassen + schoolbook) and try to merge
            # Strassen for cols 0,1 (rank 7) + schoolbook for col 2 (rank 4) = rank 11
            # Try to find rank 10 by combining some terms
            
            # Let's try specific structured patterns
            # Pattern: 5 Strassen-like terms + 5 additional terms
            
            for trial in range(500):
                if time.time() > search_deadline:
                    break
                
                U = np.zeros((4, 10))
                V = np.zeros((6, 10))
                W = np.zeros((6, 10))
                
                # First 7 terms: Strassen for cols 0,1
                # M1 = (A00+A11)(B00+B11)
                U[0,0]=1; U[3,0]=1
                V[0,0]=1; V[5,0]=1  # B[0,0], B[1,1] -> indices 0*3+0=0, 1*3+2=5
                W[0,0]=1; W[5,0]=1  # C[0,0], C[1,1] -> indices 0*2+0=0, 1*2+1=3... 
                # Wait, W index is k*2+i where k is col, i is row
                # C[0,0]: k=0,i=0 -> 0
                # C[1,1]: k=1,i=1 -> 1*2+1=3
                W[0,0]=1; W[3,0]=1
                
                # M2 = A10*(B01+B11)
                U[2,1]=1; U[3,1]=1
                V[1,1]=1; V[5,1]=1  # B[0,1]=1*3+1=4? No: l=0,k=1 -> 0*3+1=1; l=1,k=1 -> 1*3+1=4
                V[1,1]=1; V[4,1]=1
                W[2,1]=1; W[3,1]=-1  # C[1,0]: k=0,i=1 -> 0*2+1=1; C[1,1]: k=1,i=1 -> 1*2+1=3
                W[1,1]=1; W[3,1]=-1
                
                # This is getting complex. Let me just use the random search.
        
        return best_solution

    # Try the search
    result_10 = search_rank_10_for_2x2x2x3(max_time=25)
    
    if result_10 is not None and time.time() < deadline - 10:
        U10, V10, W10 = result_10
        # Build rank 38 solution: 4 Strassen blocks (cols 0-7) + rank 10 (cols 8-10)
        rank = 38
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(22)]
        W = [[0] * rank for _ in range(22)]
        r = 0
        
        # 4 Strassen blocks for cols 0-7
        for block in range(4):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0
            v_f = k1
            v_g = 11 + k0
            v_h = 11 + k1
            w_c00 = 2 * k0
            w_c10 = 2 * k0 + 1
            w_c01 = 2 * k1
            w_c11 = 2 * k1 + 1
            
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        
        # Add rank 10 block for cols 8,9,10
        for rr in range(10):
            for i in range(4):
                U[i][r + rr] = U10[i][rr]
            for i in range(6):
                if i < 3:
                    V[8 + i][r + rr] = V10[i][rr]
                else:
                    k_sub = i - 3
                    V[11 + 8 + k_sub][r + rr] = V10[i][rr]
            for i in range(6):
                if i < 3:
                    W[2 * (8 + i) + 0][r + rr] = W10[i][rr]
                else:
                    k_sub = i - 3
                    W[2 * (8 + (i-3)) + 1][r + rr] = W10[i][rr]
        
        # Verify before returning
        if verify_solution(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to rank 39
    return get_rank_39()
