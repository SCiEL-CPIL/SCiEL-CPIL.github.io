import time
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit

    # Rank 7 Strassen decomposition (optimal for 2x2 matrix multiplication)
    rank = 7
    
    # Initialize U, V, W as numpy arrays for efficient verification
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((4, rank), dtype=np.float64)
    W = np.zeros((4, rank), dtype=np.float64)

    # Index conventions:
    # U[i*2+j][r] corresponds to A[i,j]
    # V[j*2+k][r] corresponds to B[j,k]
    # W[k*2+i][r] corresponds to C[i,k] output (Note: W index is k*n+i)

    # Strassen's 7 products:
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b01 - b11)
    # m4 = a11(b10 - b00)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)

    # m1: U[0][0]=1, U[3][0]=1, V[0][0]=1, V[3][0]=1, W[0][0]=1, W[3][0]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m2: U[2][1]=1, U[3][1]=1, V[0][1]=1, W[1][1]=1, W[3][1]=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # m3: U[0][2]=1, V[1][2]=1, V[3][2]=-1, W[2][2]=1, W[3][2]=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # m4: U[3][3]=1, V[2][3]=1, V[0][3]=-1, W[0][3]=1, W[1][3]=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # m5: U[0][4]=1, U[1][4]=1, V[3][4]=1, W[0][4]=-1, W[2][4]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # m6: U[2][5]=1, U[0][5]=-1, V[0][5]=1, V[1][5]=1, W[3][5]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # m7: U[1][6]=1, U[3][6]=-1, V[2][6]=1, V[3][6]=1, W[0][6]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    # Vectorized verification using NumPy
    # Tensor identity: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] == 1
    # We can reshape U, V, W to 3D tensors to check this efficiently, 
    # but for small sizes, a simple loop or einsum is fine.
    # Let's use einsum to verify:
    # U is 4x7, V is 4x7, W is 4x7
    # We want to check T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is a bit tricky with standard einsum indices because W's index depends on k and i.
    # Let's just do the explicit loop for clarity and safety, it's very fast.
    
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # Sum over r
                # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
                s = 0.0
                for r in range(rank):
                    s += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
        if not valid:
            break

    if not valid:
        # Fallback should not happen
        pass

    # Convert back to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
