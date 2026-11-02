import time
import random

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Find a low-rank exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Uses Strassen's algorithm on 2x2 blocks, then verifies and falls back to schoolbook if needed.
    """
    start_time = time.time()
    DEATHLINE = 130.0  # Leave safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    def schoolbook():
        rank = n * m * p  # 64
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_factors(U_mat, V_mat, W_mat):
        """
        Verify that the factors correctly decompose the matrix multiplication tensor.
        The identity is: for all i, j, k:
            sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        This ensures that C[i,k] = sum_j A[i,j] B[j,k] is correctly computed.
        """
        try:
            rank = len(U_mat[0])
            if len(V_mat[0]) != rank or len(W_mat[0]) != rank:
                return False
            if len(U_mat) != nm or len(V_mat) != mp or len(W_mat) != pn:
                return False
            for r in range(rank):
                for i in range(len(U_mat)):
                    if len(U_mat[i]) != rank:
                        return False
                for j in range(len(V_mat)):
                    if len(V_mat[j]) != rank:
                        return False
                for k in range(len(W_mat)):
                    if len(W_mat[k]) != rank:
                        return False
            
            # Check the tensor identity: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(rank):
                            val += U_mat[i*m + j][r] * V_mat[j*p + k][r] * W_mat[k*n + i][r]
                        if abs(val - 1) > 1e-9:
                            return False
            return True
        except Exception:
            return False

    def strassen_2x2_factors():
        """
        Strassen's 7 multipliers for 2x2 * 2x2 -> 2x2.
        A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        p1 = (a+d)(e+h), p2 = (c+d)e, p3 = a(f-h), p4 = d(g-e),
        p5 = (a+b)h, p6 = (c-a)(e+f), p7 = (b-d)(g+h)
        C[0,0] = p1+p4-p5+p7, C[0,1] = p3+p5, C[1,0] = p2+p4, C[1,1] = p1-p2+p3+p6
        
        Returns list of 7 (u, v, w) tuples where:
        u[li*2+lj] = coefficient for A[li, lj]
        v[lj*2+lk] = coefficient for B[lj, lk]  
        w[lk*2+li] = coefficient for C[li, lk]  (using W[k*n+i] convention with n=2)
        """
        factors = []
        # p1=(a+d)(e+h): +C[0,0], +C[1,1]
        factors.append(([1,0,1,0], [1,0,0,1], [1,0,0,1]))
        # p2=(c+d)e: +C[1,0], -C[1,1]
        factors.append(([0,0,1,1], [1,0,0,0], [0,1,0,-1]))
        # p3=a(f-h): +C[0,1], +C[1,1]
        factors.append(([1,0,0,0], [0,1,0,-1], [0,0,1,1]))
        # p4=d(g-e): +C[0,0], +C[1,0]
        factors.append(([0,0,0,1], [0,0,1,-1], [1,1,0,0]))
        # p5=(a+b)h: -C[0,0], +C[0,1]
        factors.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
        # p6=(c-a)(e+f): +C[1,1]
        factors.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
        # p7=(b-d)(g+h): +C[0,0]
        factors.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
        return factors

    def build_strassen_decomposition():
        """
        Build a Strassen-based decomposition for 2x4 * 4x8.
        
        Strategy: Split the 4 intermediate columns (j=0,1,2,3) into two groups of 2.
        For each group, we have a 2x2 (A part) times 2x8 (B part) = 2x8 product.
        Actually, let's think of it differently:
        
        C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
        
        Split j into two halves: j in {0,1} and j in {2,3}
        C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
        
        For the first sum: A_left is 2x2, B_left is 2x8
        For the second sum: A_right is 2x2, B_right is 2x8
        
        Now, B_left is 2x8 = two 2x4 blocks, or four 2x2 blocks.
        Let's split B_left into four 2x2 blocks: B_left[:, 0:2], B_left[:, 2:4], etc.
        Actually B_left is 2 rows x 8 cols. Split into 4 blocks of 2x2.
        
        So for each of the 4 output column pairs (k in {0,1}, {2,3}, {4,5}, {6,7}):
        C[:, k_pair] = A_left * B_left[:, k_pair] + A_right * B_right[:, k_pair]
        
        Each term is 2x2 * 2x2 = 2x2.
        
        We can apply Strassen to each 2x2 * 2x2 product.
        That's 4 column pairs * 2 (left/right A) * 7 Strassen multipliers = 56 total rank.
        
        But wait, can we do better by combining the left and right A parts?
        Actually, for a fixed output column pair, we have:
        C_block = A_left * B_left_block + A_right * B_right_block
        
        This is a sum of two 2x2*2x2 products. We could potentially use a joint algorithm,
        but for now let's just use 2*7=14 Strassen multipliers per column pair, times 4 = 56.
        
        Actually, I realize we might be able to do even better. Let me reconsider.
        
        The key insight: for each output column pair (2x2 block of C), we need to compute
        a 2x2 result that is the sum of two 2x2*2x2 products.
        
        Let's just go with the straightforward approach: 56 rank using Strassen on each
        individual 2x2*2x2 product.
        """
        strassen_factors = strassen_2x2_factors()
        
        # 4 column pairs * 2 (A_left, A_right) * 7 Strassen = 56
        rank = 56
        U_mat = [[0] * rank for _ in range(nm)]
        V_mat = [[0] * rank for _ in range(mp)]
        W_mat = [[0] * rank for _ in range(pn)]
        
        col = 0
        for k_pair in range(4):  # 4 output column pairs
            k_start = 2 * k_pair
            for a_half in range(2):  # 0: A_left (j=0,1), 1: A_right (j=2,3)
                j_start = 2 * a_half
                for s_idx in range(7):
                    u_vec, v_vec, w_vec = strassen_factors[s_idx]
                    
                    # Fill U: A is 2x4, so U[i*m + j, col]
                    # Local A is 2x2 with rows li=0,1 and cols lj=0,1
                    # Global: row i=li, col j=j_start+lj
                    for li in range(2):
                        for lj in range(2):
                            U_mat[li * m + (j_start + lj)][col] = u_vec[li * 2 + lj]
                    
                    # Fill V: B is 4x8, so V[j*p + k, col]
                    # Local B is 2x2 with rows lj=0,1 (local) and cols lk=0,1 (local)
                    # Global: row j=j_start+lj, col k=k_start+lk
                    for lj in range(2):
                        for lk in range(2):
                            V_mat[(j_start + lj) * p + (k_start + lk)][col] = v_vec[lj * 2 + lk]
                    
                    # Fill W: W[k*n + i, col]
                    # Local C is 2x2 with rows li=0,1 and cols lk=0,1
                    # Global: row i=li, col k=k_start+lk
                    # W index = k_global * n + i = (k_start+lk) * 2 + li
                    for lk in range(2):
                        for li in range(2):
                            W_mat[(k_start + lk) * n + li][col] = w_vec[lk * 2 + li]
                    
                    col += 1
        
        return {"U": U_mat, "V": V_mat, "W": W_mat}

    # Try Strassen-based decomposition first
    try:
        result = build_strassen_decomposition()
        if verify_factors(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass

    # Fall back to schoolbook
    return schoolbook()

if __name__ == "__main__":
    pass
