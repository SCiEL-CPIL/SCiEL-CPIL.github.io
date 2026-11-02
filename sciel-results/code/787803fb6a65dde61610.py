import time
import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank_school = n * m * p  # 100
    deadline = time.time() + 135.0  # safety margin before 150s kill

    def build_schoolbook():
        U = [[0.0] * rank_school for _ in range(n * m)]
        V = [[0.0] * rank_school for _ in range(m * p)]
        W = [[0.0] * rank_school for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j][c] = 1.0
                    V[j * p + k][c] = 1.0
                    W[k * n + i][c] = 1.0
        return U, V, W

    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    # Start with schoolbook as best
    best_U, best_V, best_W = build_schoolbook()
    best_rank = rank_school

    # Strategy: Try structured decompositions using the algebraic structure.
    # For matrix multiplication, we can use a "sum of products" approach.
    # Key idea: try to find a decomposition of rank r by solving a system of equations.
    
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k.
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Approach: Try ranks from lower bound upward.
    # Lower bound: max(n*m, m*p, n*p) = max(20, 25, 20) = 25
    
    # For each candidate rank, try a structured pattern:
    # Use a "grid" pattern where U, V, W have specific sparsity.
    
    # Try a few structured patterns for low ranks:
    
    # Pattern 1: Rank = n*m*p / some_factor
    # Pattern 2: Use the fact that for square matrices, rank is at most (n*p + n*m + m*p - something)
    
    # Let's try a systematic approach: for each rank from 25 to 100,
    # try to construct a valid decomposition using a structured method.
    
    # Method: For a given rank r, we can try to set up the problem as:
    # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # 
    # This is a system of n*m*p = 100 equations in r^2 * (nm + mp + pn) unknowns... 
    # Actually it's more complex. Let me try a different approach.
    
    # Alternative: Use the known result that for matrix multiplication,
    # the rank is at most the schoolbook rank, and for small dimensions,
    # we can try specific algebraic identities.
    
    # For 4x5 * 5x5, let's try a block-based approach:
    # Split A (4x5) into 2 blocks of 2x5
    # Split B (5x5) into 5 blocks of 1x5 (each row)
    # Then C (4x5) = A*B has entries C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Try rank = 25 (the lower bound) with a structured pattern:
    # Let r = 25. Index c = 0..24.
    # Map c to (a, b) where a = c // 5, b = c % 5
    # U[i*m+j, c] = 1 if j == a, else 0  (U depends only on j and a)
    # V[j*p+k, c] = 1 if k == b, else 0  (V depends only on k and b)  
    # W[k*n+i, c] = 1 if i == c // 5, else 0  (W depends on i and a)
    
    # Check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # = sum_{a=0}^4 sum_{b=0}^4 [j==a] * [k==b] * [i==a]
    # = sum_{a} [j==a]*[i==a] * sum_b [k==b]
    # = [i==j] * 5  (since sum_b [k==b] = 1 for any k)
    # This gives 5 if i==j, 0 otherwise. Not 1 for all i,j,k.
    
    # That pattern doesn't work. Let me try another.
    
    # Try: U[i*m+j, c] = 1 if i == c//5
    # V[j*p+k, c] = 1 if j == c%5  
    # W[k*n+i, c] = 1 if k == c//5
    # sum_c [i==c//5]*[j==c%5]*[k==c//5]
    # = sum_c [i==c//5]*[j==c%5]*[k==c//5]
    # For a given (i,j,k), c must satisfy c//5==i, c%5==j, c//5==k
    # So i==k and j==c%5. If i==k, sum over c where c//5==i and c%5==j: exactly 1 value
    # So result is 1 if i==k, else 0. Still not right.
    
    # The correct schoolbook pattern uses c = i*m*p + j*p + k, which is rank 100.
    # To reduce rank, we need to find linear dependencies.
    
    # Let me try a numerical approach: for each candidate rank, use a random 
    # structured initialization and check if it works. If not, try variations.
    
    # Actually, let's try a more clever structured approach:
    # Use the identity: for matrix multiplication, we can use "sum over shared index"
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T[i,j,k] = 1.
    
    # A rank-r decomposition means T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is a vector in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    
    # For the multiplication tensor, u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] must sum to 1.
    
    # Let's try rank = 50 (half of schoolbook) with a structured pattern:
    # Split the 100 schoolbook terms into 50 pairs and see if each pair can be 
    # represented by a single term.
    
    # For schoolbook term (i,j,k): U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # Can we merge two terms (i1,j1,k1) and (i2,j2,k2) into one?
    # We'd need: U[i1*m+j1,c]V[j1*p+k1,c]W[k1*n+i1,c] + U[i2*m+j2,c]V[j2*p+k2,c]W[k2*n+i2,c] = 1
    # and for all other (i,j,k), the contribution should be 0.
    
    # This is getting complex. Let me just try random search more efficiently.
    
    # Efficient random search: for each rank, generate random sparse matrices and verify.
    # Use numpy for fast verification.
    
    import random
    random.seed(42)
    
    # Try ranks from 25 to 99
    for target_rank in range(25, rank_school):
        if time.time() > deadline - 15:
            break
        
        # Number of attempts per rank: more for lower ranks
        max_attempts = 20 if target_rank < 50 else 10
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Generate random sparse factors
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Each column gets a few non-zero entries
            for c in range(target_rank):
                # U: 1-3 non-zeros
                nz_u = random.randint(1, 3)
                rows_u = random.sample(range(n * m), min(nz_u, n * m))
                for r in rows_u:
                    U[r, c] = random.choice([-1.0, 1.0])
                
                # V: 1-3 non-zeros
                nz_v = random.randint(1, 3)
                rows_v = random.sample(range(m * p), min(nz_v, m * p))
                for r in rows_v:
                    V[r, c] = random.choice([-1.0, 1.0])
                
                # W: 1-3 non-zeros
                nz_w = random.randint(1, 3)
                rows_w = random.sample(range(p * n), min(nz_w, p * n))
                for r in rows_w:
                    W[r, c] = random.choice([-1.0, 1.0])
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify(U_list, V_list, W_list, target_rank):
                best_rank = target_rank
                best_U = U_list
                best_V = V_list
                best_W = W_list
                return {
                    "U": best_U,
                    "V": best_V,
                    "W": best_W
                }
    
    # If no lower rank found, return schoolbook
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
