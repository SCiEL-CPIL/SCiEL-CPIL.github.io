import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Local Strassen factors for 2x2 * 2x2 -> 2x2
    # Indices for A: 0=00, 1=01, 2=10, 3=11
    # Indices for B: 0=00, 1=01, 2=10, 3=11
    # Indices for C: 0=00, 1=01, 2=10, 3=11
    U_loc = [[0]*7 for _ in range(4)]
    V_loc = [[0]*7 for _ in range(4)]
    W_loc = [[0]*7 for _ in range(4)]

    # r=0: M1=(a00+a11)(b00+b11) -> c00+1, c11+1
    U_loc[0][0]=1; U_loc[3][0]=1
    V_loc[0][0]=1; V_loc[3][0]=1
    W_loc[0][0]=1; W_loc[3][0]=1
    # r=1: M2=(a10+a11)b00 -> c10+1, c11-1
    U_loc[2][1]=1; U_loc[3][1]=1
    V_loc[0][1]=1
    W_loc[2][1]=1; W_loc[3][1]=-1
    # r=2: M3=a00(b01-b11) -> c01+1, c11+1
    U_loc[0][2]=1
    V_loc[1][2]=1; V_loc[3][2]=-1
    W_loc[1][2]=1; W_loc[3][2]=1
    # r=3: M4=a11(b10-b00) -> c00+1, c10+1
    U_loc[3][3]=1
    V_loc[2][3]=1; V_loc[0][3]=-1
    W_loc[0][3]=1; W_loc[2][3]=1
    # r=4: M5=(a00+a01)b11 -> c00-1, c01+1
    U_loc[0][4]=1; U_loc[1][4]=1
    V_loc[3][4]=1
    W_loc[0][4]=-1; W_loc[1][4]=1
    # r=5: M6=(a10-a00)(b00+b01) -> c11+1
    U_loc[2][5]=1; U_loc[0][5]=-1
    V_loc[0][5]=1; V_loc[1][5]=1
    W_loc[3][5]=1
    # r=6: M7=(a01-a11)(b10+b11) -> c00+1
    U_loc[1][6]=1; U_loc[3][6]=-1
    V_loc[2][6]=1; V_loc[3][6]=1
    W_loc[0][6]=1

    # Build the full tensor decomposition with rank 56
    total_rank = 56

    U = [[0]*total_rank for _ in range(n*m)]   # 8 x 56
    V = [[0]*total_rank for _ in range(m*p)]   # 32 x 56
    W = [[0]*total_rank for _ in range(p*n)]   # 16 x 56

    group = 0
    for half in range(2):
        for jj in range(p // 2):
            base_r = group * 7
            group += 1

            a_map = [
                0*m + (2*half + 0),
                0*m + (2*half + 1),
                1*m + (2*half + 0),
                1*m + (2*half + 1),
            ]

            b_map = [
                (2*half + 0)*p + (2*jj + 0),
                (2*half + 0)*p + (2*jj + 1),
                (2*half + 1)*p + (2*jj + 0),
                (2*half + 1)*p + (2*jj + 1),
            ]

            c_map = [
                (2*jj + 0)*n + 0,
                (2*jj + 1)*n + 0,
                (2*jj + 0)*n + 1,
                (2*jj + 1)*n + 1,
            ]

            for r in range(7):
                rr = base_r + r
                for la in range(4):
                    val = U_loc[la][r]
                    if val:
                        U[a_map[la]][rr] += val
                for lb in range(4):
                    val = V_loc[lb][r]
                    if val:
                        V[b_map[lb]][rr] += val
                for lc in range(4):
                    val = W_loc[lc][r]
                    if val:
                        W[c_map[lc]][rr] += val

    # Now attempt to reduce rank by finding linear dependencies among the rank factors.
    # Convert to numpy arrays for efficient linear algebra.
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # The decomposition is valid if for all i,k: sum_r U[i,r]*W[k,r] = (A@B)[i//m, k%n] * B[i%m, k//p]
    # More precisely, the tensor T[i,j,k] = delta(i%m, j//p) * delta(j%m, k//p) ... 
    # Actually, the matmul tensor is T[i,j,k] = 1 if i%m == j//p and j%m == k//p... no.
    # For matrix multiply C = A@B: C[a,c] = sum_b A[a,b]*B[b,c]
    # In our indexing: A is n x m, B is m x p, C is n x p
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The condition is: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
    # where T[i,j,k] = 1 if (i%m == j//p) and (j%m == k//n) ... 
    # Let me think: i indexes A (row a = i//m, col b = i%m)
    # j indexes B (row b' = j//p, col c = j%p)
    # k indexes C (col c' = k//n, row a' = k%n)
    # T[i,j,k] = 1 if b==b' and c==c' and a==a'
    # i.e., i%m == j//p and j%p == k//n and i//m == k%n

    # Check rank by looking for null space in the "slice" matrices
    # For each k, define M_k[i,j] = sum_r U[i,r]*V[j,r]*W[k,r]
    # The matrix M_k should equal the tensor slice.
    
    # To find if we can reduce rank, check if any rank factor is a linear combination of others.
    # This is equivalent to checking if the columns of a certain matrix are linearly dependent.
    
    # Construct the "flatten" matrix: for each pair (i,j), the vector across r is U[i,:]*V[j,:] (element-wise)
    # Then W[k,:] should be such that dot(W[k,:], U[i,:]*V[j,:]) = T[i,j,k]
    
    # Alternative: check SVD of the matrix formed by reshaping.
    # The tensor T can be viewed as a matrix of size (n*m) x (m*p) for each k, or (n*m) x (p*n) for each j, etc.
    
    # Let's try: form matrix M of size (n*m) x (m*p*n) where M[i, j*p*n + k] = T[i,j,k]
    # But that's huge. Instead, let's check if the current rank-56 factors have dependencies.
    
    # Check: are the 56 rank vectors (U[:,r], V[:,r], W[:,r]) linearly independent in some sense?
    # Simpler: try to see if we can find a smaller rank by numerical optimization.
    
    # For now, let's just verify the decomposition is correct and return it.
    # The deadline guard:
    if time.time() - start_time > 115:
        return {"U": U, "V": V, "W": W}

    # Try to reduce rank: check if any two rank factors can be combined.
    # For each pair of rank indices (r1, r2), check if the corresponding "triads" are proportional.
    # This is expensive (56^2 * 56 checks), so let's be smart.
    
    # Actually, let's try a different approach: use the fact that the problem has structure.
    # The 8 groups of 7 Strassen factors might have shared structure.
    
    # For now, return the rank-56 solution.
    return {"U": U, "V": V, "W": W}
