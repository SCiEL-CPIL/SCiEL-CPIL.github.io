import numpy as np
import time

def construct():
    n, m, p = 2, 5, 6
    
    # Initialize matrices with 0
    # Rank estimate: 54
    max_rank = 54
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    
    r = 0 # current rank index
    
    def add_strassen_222(i0, j0, k0):
        """
        Add rank-1 terms for 2x2x2 Strassen decomposition.
        i0: start row of A (and C)
        j0: start col of A (and row of B)
        k0: start col of B (and C)
        """
        nonlocal r
        # M1 = (A00 + A11)(B00 + B11)
        U[i0*m + j0, r] = 1
        U[(i0+1)*m + (j0+1), r] = 1
        V[j0*p + k0, r] = 1
        V[(j0+1)*p + (k0+1), r] = 1
        W[k0*n + i0, r] = 1
        W[(k0+1)*n + (i0+1), r] = 1
        r += 1
        
        # M2 = (A10 + A11)B00
        U[i0*m + (j0+1), r] = 1
        U[(i0+1)*m + (j0+1), r] = 1
        V[j0*p + k0, r] = 1
        W[k0*n + (i0+1), r] = 1
        r += 1
        
        # M3 = A00(B01 - B11)
        U[i0*m + j0, r] = 1
        V[j0*p + (k0+1), r] = 1
        V[(j0+1)*p + (k0+1), r] = -1
        W[(k0+1)*n + i0, r] = 1
        r += 1
        
        # M4 = A11(B10 - B00)
        U[(i0+1)*m + (j0+1), r] = 1
        V[(j0+1)*p + k0, r] = 1
        V[j0*p + k0, r] = -1
        W[k0*n + (i0+1), r] = 1
        r += 1
        
        # M5 = (A00 + A01)B11
        U[i0*m + j0, r] = 1
        U[i0*m + (j0+1), r] = 1
        V[(j0+1)*p + (k0+1), r] = 1
        W[(k0+1)*n + i0, r] = 1
        r += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[(i0+1)*m + j0, r] = 1
        U[i0*m + j0, r] = -1
        V[j0*p + k0, r] = 1
        V[j0*p + (k0+1), r] = 1
        W[k0*n + (i0+1), r] = 1
        W[(k0+1)*n + (i0+1), r] = 1
        r += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i0*m + (j0+1), r] = 1
        U[(i0+1)*m + (j0+1), r] = -1
        V[(j0+1)*p + k0, r] = 1
        V[(j0+1)*p + (k0+1), r] = 1
        W[k0*n + i0, r] = 1
        W[(k0+1)*n + i0, r] = 1
        r += 1

    def add_simple_212(i0, j0, k0):
        """
        Add rank-1 terms for 2x1x2 simple multiplication.
        i0: start row of A (and C)
        j0: col of A (and row of B)
        k0: start col of B (and C)
        """
        nonlocal r
        # C[i0, k0] += A[i0, j0] * B[j0, k0]
        U[i0*m + j0, r] = 1
        V[j0*p + k0, r] = 1
        W[k0*n + i0, r] = 1
        r += 1
        
        # C[i0, k0+1] += A[i0, j0] * B[j0, k0+1]
        U[i0*m + j0, r] = 1
        V[j0*p + (k0+1), r] = 1
        W[(k0+1)*n + i0, r] = 1
        r += 1
        
        # C[i0+1, k0] += A[i0+1, j0] * B[j0, k0]
        U[(i0+1)*m + j0, r] = 1
        V[j0*p + k0, r] = 1
        W[k0*n + (i0+1), r] = 1
        r += 1
        
        # C[i0+1, k0+1] += A[i0+1, j0] * B[j0, k0+1]
        U[(i0+1)*m + j0, r] = 1
        V[j0*p + (k0+1), r] = 1
        W[(k0+1)*n + (i0+1), r] = 1
        r += 1

    # Part 1: A[0:2, 0:2] * B[0:2, 0:6]
    # Three 2x2x2 blocks for B cols (0,1), (2,3), (4,5)
    add_strassen_222(0, 0, 0)
    add_strassen_222(0, 0, 2)
    add_strassen_222(0, 0, 4)
    
    # Part 2: A[0:2, 2:5] * B[2:5, 0:6]
    # Three 2x3x2 blocks for B cols (0,1), (2,3), (4,5)
    # Each 2x3x2 is 2x2x2 (B rows 2,3) + 2x1x2 (B row 4)
    
    # Block for B cols 0,1
    add_strassen_222(0, 2, 0)
    add_simple_212(0, 4, 0)
    
    # Block for B cols 2,3
    add_strassen_222(0, 2, 2)
    add_simple_212(0, 4, 2)
    
    # Block for B cols 4,5
    add_strassen_222(0, 2, 4)
    add_simple_212(0, 4, 4)

    # r should be 54
    # Trim to actual rank
    U = U[:, :r]
    V = V[:, :r]
    W = W[:, :r]
    
    # Convert to lists of lists for output
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    # Quick verification
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    n, m, p = 2, 5, 6
    rank = U.shape[1]
    
    # Check valid positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    print(f"FAIL valid ({i},{j},{k}): {s}")
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        print("All valid positions correct.")
        
    # Check some invalid positions
    import random
    for _ in range(1000):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if not (i1 == i2 and j1 == j2 and k1 == k2):
            s = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(s) > 1e-9:
                print(f"FAIL invalid ({a},{b},{c}): {s}")
                break
    else:
        print("Sampled invalid positions correct.")
        
    print(f"Rank: {rank}")
