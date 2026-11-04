import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # The multiplication is 2x4 * 4x8.
    # We can decompose this into 4 independent 2x4 * 4x2 multiplications.
    # Each 2x4 * 4x2 multiplication has rank 14 (sum of two 2x2 * 2x2 Strassen multiplications).
    # Total rank = 4 * 14 = 56.
    
    rank = 56
    U = np.zeros((n * m, rank))  # 8 x 56
    V = np.zeros((m * p, rank))  # 32 x 56
    W = np.zeros((p * n, rank))  # 16 x 56
    
    # Helper to add a Strassen term for a 2x2 * 2x2 block
    # i0, j0, k0 are the starting indices for the 2x2 blocks of A, B, C
    # A is 2x2 at rows i0, i0+1 and cols j0, j0+1
    # B is 2x2 at rows j0, j0+1 and cols k0, k0+1
    # C is 2x2 at rows k0, k0+1 and cols i0, i0+1
    def add_strassen(i0, j0, k0, base_col):
        # M1 = (A00 + A11)(B00 + B11)
        U[i0*m + j0, base_col] += 1
        U[(i0+1)*m + (j0+1), base_col] += 1
        V[j0*p + k0, base_col] += 1
        V[(j0+1)*p + (k0+1), base_col] += 1
        W[k0*n + i0, base_col] += 1
        W[(k0+1)*n + (i0+1), base_col] += 1
        base_col += 1
        
        # M2 = (A10 + A11)B00
        U[(i0+1)*m + j0, base_col] += 1
        U[(i0+1)*m + (j0+1), base_col] += 1
        V[j0*p + k0, base_col] += 1
        W[k0*n + (i0+1), base_col] += 1
        W[(k0+1)*n + (i0+1), base_col] -= 1
        base_col += 1
        
        # M3 = A00(B01 - B11)
        U[i0*m + j0, base_col] += 1
        V[j0*p + (k0+1), base_col] += 1
        V[(j0+1)*p + (k0+1), base_col] -= 1
        W[(k0+1)*n + i0, base_col] += 1
        W[(k0+1)*n + (i0+1), base_col] += 1
        base_col += 1
        
        # M4 = A11(B10 - B00)
        U[(i0+1)*m + (j0+1), base_col] += 1
        V[(j0+1)*p + k0, base_col] += 1
        V[j0*p + k0, base_col] -= 1
        W[k0*n + i0, base_col] += 1
        W[k0*n + (i0+1), base_col] += 1
        base_col += 1
        
        # M5 = (A00 + A01)B11
        U[i0*m + j0, base_col] += 1
        U[i0*m + (j0+1), base_col] += 1
        V[(j0+1)*p + (k0+1), base_col] += 1
        W[k0*n + i0, base_col] -= 1
        W[(k0+1)*n + i0, base_col] += 1
        base_col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[(i0+1)*m + j0, base_col] += 1
        U[i0*m + j0, base_col] -= 1
        V[j0*p + k0, base_col] += 1
        V[j0*p + (k0+1), base_col] += 1
        W[(k0+1)*n + (i0+1), base_col] += 1
        base_col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i0*m + (j0+1), base_col] += 1
        U[(i0+1)*m + (j0+1), base_col] -= 1
        V[(j0+1)*p + k0, base_col] += 1
        V[(j0+1)*p + (k0+1), base_col] += 1
        W[k0*n + i0, base_col] += 1
        base_col += 1
        
        return base_col

    col = 0
    # We have 4 blocks of columns in B (size 2 each).
    # For each column block k0 in [0, 2, 4, 6]:
    #   We split A into two 2x2 blocks (j0 in [0, 2]).
    #   This corresponds to C = A_L B_top + A_R B_bot.
    #   Wait, the decomposition of 2x4 * 4x2 is:
    #   A = [A_L, A_R], B = [B_T; B_B].
    #   C = A_L B_T + A_R B_B.
    #   A_L is 2x2 (rows 0,1 of A, cols 0,1 of A).
    #   B_T is 2x2 (rows 0,1 of B, cols k0, k0+1 of B).
    #   A_R is 2x2 (rows 0,1 of A, cols 2,3 of A).
    #   B_B is 2x2 (rows 2,3 of B, cols k0, k0+1 of B).
    #   
    #   So for each k0:
    #     Add Strassen for (i0=0, j0=0, k0=k0) -> A_L * B_T
    #     Add Strassen for (i0=0, j0=2, k0=k0+2) -> A_R * B_B
    #     Note: B_B rows are 2,3. So j0 for B is 2.
    #     But in our Strassen helper, j0 is the row index in B?
    #     No, V indices are (j, k). j is row of B.
    #     So for A_L * B_T: A rows 0,1; A cols 0,1. B rows 0,1; B cols k0, k0+1.
    #     Helper args: i0=0 (A row), j0=0 (A col / B row), k0=k0 (B col).
    #     For A_R * B_B: A rows 0,1; A cols 2,3. B rows 2,3; B cols k0, k0+1.
    #     Helper args: i0=0 (A row), j0=2 (A col / B row), k0=k0 (B col).
    
    for k0 in [0, 2, 4, 6]:
        # First 2x2 * 2x2: A[0:2, 0:2] * B[0:2, k0:k0+2]
        col = add_strassen(0, 0, k0, col)
        # Second 2x2 * 2x2: A[0:2, 2:4] * B[2:4, k0:k0+2]
        col = add_strassen(0, 2, k0, col)
        
    # Convert to lists of lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
