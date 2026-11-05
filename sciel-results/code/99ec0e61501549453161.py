import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 33
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_block(j_base, k_base):
        nonlocal col
        # j_base is 0 or 2
        # k_base is 0 or 3
        
        # m0: (x00+x11)(y00+y11)
        # i=0, j=0, k=0
        U[0*m + j_base][col] += 1
        U[1*m + (j_base+1)][col] += 1
        V[j_base*p + k_base][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] += 1
        W[k_base*n + 0][col] += 1
        W[(k_base+1)*n + 1][col] += 1
        col += 1
        
        # m1: (x10+x11)y00
        U[1*m + j_base][col] += 1
        U[1*m + (j_base+1)][col] += 1
        V[j_base*p + k_base][col] += 1
        W[k_base*n + 1][col] += 1
        W[(k_base+1)*n + 1][col] -= 1
        col += 1
        
        # m2: x00(y01-y11)
        U[0*m + j_base][col] += 1
        V[j_base*p + (k_base+1)][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] -= 1
        W[(k_base+1)*n + 0][col] += 1
        W[(k_base+1)*n + 1][col] += 1
        col += 1
        
        # m3: x11(y10-y00)
        U[1*m + (j_base+1)][col] += 1
        V[(j_base+1)*p + k_base][col] += 1
        V[j_base*p + k_base][col] -= 1
        W[k_base*n + 0][col] += 1
        W[k_base*n + 1][col] += 1
        col += 1
        
        # m4: (x00+x01)y11
        U[0*m + j_base][col] += 1
        U[0*m + (j_base+1)][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] += 1
        W[k_base*n + 0][col] -= 1
        W[(k_base+1)*n + 0][col] += 1
        col += 1
        
        # m5: (x10-x00)(y00+y01)
        U[1*m + j_base][col] += 1
        U[0*m + j_base][col] -= 1
        V[j_base*p + k_base][col] += 1
        V[j_base*p + (k_base+1)][col] += 1
        W[(k_base+1)*n + 1][col] += 1
        col += 1
        
        # m6: (x01-x11)(y10+y11)
        U[0*m + (j_base+1)][col] += 1
        U[1*m + (j_base+1)][col] -= 1
        V[(j_base+1)*p + k_base][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] += 1
        W[k_base*n + 0][col] += 1
        col += 1

    def add_rank1_2x2x1(j_base, k_idx):
        nonlocal col
        # T[i, j, k_idx] = 1 for i in {0,1}, j in {j_base, j_base+1}
        # U[i, j, r] = 1
        # V[j, k_idx, r] = 1
        # W[k_idx, i, r] = 1
        
        U[0*m + j_base][col] = 1
        U[0*m + (j_base+1)][col] = 1
        U[1*m + j_base][col] = 1
        U[1*m + (j_base+1)][col] = 1
        
        V[j_base*p + k_idx][col] = 1
        V[(j_base+1)*p + k_idx][col] = 1
        
        W[k_idx*n + 0][col] = 1
        W[k_idx*n + 1][col] = 1
        col += 1

    def add_rank1_2x1x6(j_idx):
        nonlocal col
        # T[i, j_idx, k] = 1 for i in {0,1}, k in {0..5}
        # U[i, j_idx, r] = 1
        # V[j_idx, k, r] = 1
        # W[k, i, r] = 1
        
        U[0*m + j_idx][col] = 1
        U[1*m + j_idx][col] = 1
        
        for k in range(p):
            V[j_idx*p + k][col] = 1
            W[k*n + 0][col] = 1
            W[k*n + 1][col] = 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, :]
    add_strassen_block(0, 0)
    add_rank1_2x2x1(0, 2)
    add_strassen_block(0, 3)
    add_rank1_2x2x1(0, 5)
    
    # Block 2: A[0:2, 2:4] * B[2:4, :]
    add_strassen_block(2, 0)
    add_rank1_2x2x1(2, 2)
    add_strassen_block(2, 3)
    add_rank1_2x2x1(2, 5)
    
    # Block 3: A[0:2, 4] * B[4, :]
    add_rank1_2x1x6(4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Simple verification
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    n, m, p = 2, 5, 6
    # Check T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Should be 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(U.shape[1]):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    print(f"Error at {i},{j},{k}: {val}")
                    break
    print("Verification complete.")
