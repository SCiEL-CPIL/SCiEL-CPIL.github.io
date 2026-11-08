import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    DEADLINE = t_start + 140.0

    # Rank 15 decomposition
    rank = 15
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Helper to add a Strassen block
    # local_i, local_k, local_j are 0-based indices in the 2x2x2 block
    # offset_i, offset_k, offset_j are global offsets
    def add_strassen_block(offset_i, offset_k, offset_j, start_col):
        # M1: (A11+A22)(B11+B22)
        # A: (0,0), (1,1) -> global (offset_i+0, offset_k+0), (offset_i+1, offset_k+1)
        # B: (0,0), (1,1) -> global (offset_k+0, offset_j+0), (offset_k+1, offset_j+1)
        # W: (0,0), (1,1) -> global (offset_j+0, offset_i+0), (offset_j+1, offset_i+1)
        # Note: W index is j*n + i.
        
        # Term 1: M1
        U[offset_i + 0, start_col] += 1
        U[offset_i + 1, start_col] += 1
        V[offset_k + 0, start_col] += 1
        V[offset_k + 1, start_col] += 1
        W[offset_j + 0, start_col] += 1
        W[offset_j + 1, start_col] += 1
        
        # Term 2: M2
        U[offset_i + 1, start_col + 1] += 1
        V[offset_k + 0, start_col + 1] += 1
        W[offset_j + 0, start_col + 1] += 1
        
        # Term 3: M3
        U[offset_i + 0, start_col + 2] += 1
        V[offset_k + 0, start_col + 2] += 1
        V[offset_k + 1, start_col + 2] -= 1
        W[offset_j + 1, start_col + 2] += 1
        
        # Term 4: M4
        U[offset_i + 1, start_col + 3] += 1
        V[offset_k + 1, start_col + 3] += 1
        V[offset_k + 0, start_col + 3] -= 1
        W[offset_j + 0, start_col + 3] += 1
        
        # Term 5: M5
        U[offset_i + 0, start_col + 4] += 1
        U[offset_i + 0, start_col + 4] += 1 # A12 is (0,1)
        # Wait, A12 is i=0, k=1. A11 is i=0, k=0.
        # M5 = (A11 + A12) B22
        # A11: (0,0), A12: (0,1). Both i=0.
        # So U[0] += 1, U[1] (k=1) += 1?
        # U index is i*m + k.
        # U[offset_i + 0, start_col + 4] corresponds to i=0, k=0?
        # Let's be precise.
        
        # Reset and redo Strassen carefully.
        # U index: i*m + k
        # V index: k*p + j
        # W index: j*n + i

        # Clear the block we just wrote to? No, we are building from scratch.
        # Let's just define the 7 terms explicitly for one block and then call it.
        pass

    # Redefine add_strassen_block properly
    def add_strassen_block(offset_i, offset_k, offset_j, start_col):
        # Local indices: i in {0,1}, k in {0,1}, j in {0,1}
        # Global U: (offset_i + i) * m + (offset_k + k)
        # Global V: (offset_k + k) * p + (offset_j + j)
        # Global W: (offset_j + j) * n + (offset_i + i)
        
        def u_idx(i, k): return (offset_i + i) * m + (offset_k + k)
        def v_idx(k, j): return (offset_k + k) * p + (offset_j + j)
        def w_idx(j, i): return (offset_j + j) * n + (offset_i + i)

        # M1: (A11+A22)(B11+B22)
        # A11: i=0,k=0. A22: i=1,k=1.
        U[u_idx(0,0), start_col] += 1
        U[u_idx(1,1), start_col] += 1
        # B11: k=0,j=0. B22: k=1,j=1.
        V[v_idx(0,0), start_col] += 1
        V[v_idx(1,1), start_col] += 1
        # C11: i=0,j=0. C22: i=1,j=1.
        # M1 contributes to C11 and C22?
        # C11 = M1 + M4 - M5 + M7
        # C22 = M1 - M2 + M3 + M6
        # So M1 contributes +1 to C11 and +1 to C22.
        W[w_idx(0,0), start_col] += 1
        W[w_idx(1,1), start_col] += 1

        # M2: (A21+A22) B11
        # A21: i=1,k=0. A22: i=1,k=1.
        U[u_idx(1,0), start_col + 1] += 1
        U[u_idx(1,1), start_col + 1] += 1
        # B11: k=0,j=0.
        V[v_idx(0,0), start_col + 1] += 1
        # C21: i=1,j=0. C22: i=1,j=1?
        # C21 = M2 + M4
        # C22 = M1 - M2 + ...
        # So M2 contributes +1 to C21 and -1 to C22.
        W[w_idx(0,1), start_col + 1] += 1  # j=0, i=1
        W[w_idx(1,1), start_col + 1] -= 1  # j=1, i=1

        # M3: A11 (B12 - B22)
        # A11: i=0,k=0.
        U[u_idx(0,0), start_col + 2] += 1
        # B12: k=0,j=1. B22: k=1,j=1.
        V[v_idx(0,1), start_col + 2] += 1
        V[v_idx(1,1), start_col + 2] -= 1
        # C12: i=0,j=1. C22: i=1,j=1?
        # C12 = M3 + M5
        # C22 = ... + M3 ...
        # So M3 contributes +1 to C12 and +1 to C22.
        W[w_idx(1,0), start_col + 2] += 1  # j=1, i=0
        W[w_idx(1,1), start_col + 2] += 1  # j=1, i=1

        # M4: A22 (B21 - B11)
        # A22: i=1,k=1.
        U[u_idx(1,1), start_col + 3] += 1
        # B21: k=1,j=0. B11: k=0,j=0.
        V[v_idx(1,0), start_col + 3] += 1
        V[v_idx(0,0), start_col + 3] -= 1
        # C11: i=0,j=0. C21: i=1,j=0.
        # C11 = ... + M4 ...
        # C21 = ... + M4 ...
        # So M4 contributes +1 to C11 and +1 to C21.
        W[w_idx(0,0), start_col + 3] += 1  # j=0, i=0
        W[w_idx(0,1), start_col + 3] += 1  # j=0, i=1

        # M5: (A11 + A12) B22
        # A11: i=0,k=0. A12: i=0,k=1.
        U[u_idx(0,0), start_col + 4] += 1
        U[u_idx(0,1), start_col + 4] += 1
        # B22: k=1,j=1.
        V[v_idx(1,1), start_col + 4] += 1
        # C11: i=0,j=0. C12: i=0,j=1.
        # C11 = ... - M5 ...
        # C12 = ... + M5 ...
        # So M5 contributes -1 to C11 and +1 to C12.
        W[w_idx(0,0), start_col + 4] -= 1  # j=0, i=0
        W[w_idx(1,0), start_col + 4] += 1  # j=1, i=0

        # M6: (A21 - A11) (B11 + B12)
        # A21: i=1,k=0. A11: i=0,k=0.
        U[u_idx(1,0), start_col + 5] += 1
        U[u_idx(0,0), start_col + 5] -= 1
        # B11: k=0,j=0. B12: k=0,j=1.
        V[v_idx(0,0), start_col + 5] += 1
        V[v_idx(0,1), start_col + 5] += 1
        # C22: i=1,j=1.
        # C22 = ... + M6 ...
        # So M6 contributes +1 to C22.
        W[w_idx(1,1), start_col + 5] += 1  # j=1, i=1

        # M7: (A12 - A22) (B21 + B22)
        # A12: i=0,k=1. A22: i=1,k=1.
        U[u_idx(0,1), start_col + 6] += 1
        U[u_idx(1,1), start_col + 6] -= 1
        # B21: k=1,j=0. B22: k=1,j=1.
        V[v_idx(1,0), start_col + 6] += 1
        V[v_idx(1,1), start_col + 6] += 1
        # C11: i=0,j=0.
        # C11 = ... + M7 ...
        # So M7 contributes +1 to C11.
        W[w_idx(0,0), start_col + 6] += 1  # j=0, i=0

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # offset_i=0, offset_k=0, offset_j=0
    add_strassen_block(0, 0, 0, 0)

    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    # offset_i=0, offset_k=0, offset_j=2
    add_strassen_block(0, 0, 2, 7)

    # Block 3: A[0:2, 2:3] * B[2:3, 0:4]
    # This is the rank-1 part.
    # A[0:2, 2:3] is i in {0,1}, k=2.
    # B[2:3, 0:4] is k=2, j in {0,1,2,3}.
    # Term: A[i,2] * B[2,j]
    # U[i,2] -> U[i*m + 2]
    # V[2,j] -> V[2*p + j]
    # W[j,i] -> W[j*n + i]
    # All coefficients are 1.
    U[0*m + 2, 14] += 1
    U[1*m + 2, 14] += 1
    V[2*p + 0, 14] += 1
    V[2*p + 1, 14] += 1
    V[2*p + 2, 14] += 1
    V[2*p + 3, 14] += 1
    W[0*n + 0, 14] += 1
    W[0*n + 1, 14] += 1
    W[1*n + 0, 14] += 1
    W[1*n + 1, 14] += 1
    W[2*n + 0, 14] += 1
    W[2*n + 1, 14] += 1
    W[3*n + 0, 14] += 1
    W[3*n + 1, 14] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
