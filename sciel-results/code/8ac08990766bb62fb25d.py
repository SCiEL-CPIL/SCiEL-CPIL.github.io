"""Rewritten program: try to project the 4x4x4 rank-23 decomposition onto 2x3x10."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Strategy: Try to use Strassen-based block decomposition
    # Split the middle dimension (m=3) into blocks
    # 2x3x10 = 2x(2+1)x10 = (2x2x10) + (2x1x10)
    # 2x2x10 can use Strassen on 2x2 blocks: 5 Strassen multiplications per 2x2x2 block
    # 10 = 2*5, so 5 blocks of 2x2x2 -> 5*7 = 35 rank for 2x2x10
    # 2x1x10 is just 20 scalar multiplications -> rank 20
    # Total: 35 + 20 = 55
    
    # Let's implement the Strassen-based decomposition
    # For 2x2x2 Strassen:
    # C = A*B where A is 2x2, B is 2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2x10: A is 2x2, B is 2x10
    # Split B into 5 blocks of 2x2: B = [B0, B1, B2, B3, B4]
    # For each block, use Strassen (7 multiplications)
    # Total: 5 * 7 = 35 multiplications
    
    # For 2x1x10: A is 2x1, B is 1x10
    # Just 20 scalar multiplications
    
    # Let's build this decomposition
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (30, rank)
    # W has shape (p*n, rank) = (20, rank)
    
    # Indexing:
    # U[i*m+j, c] where i in [0,2), j in [0,3)
    # V[j*p+k, c] where j in [0,3), k in [0,10)
    # W[k*n+i, c] where k in [0,10), i in [0,2)
    
    # Block 1: 2x2x10 using Strassen on 5 2x2x2 blocks
    # A is 2x2 (rows i=0,1; cols j=0,1)
    # B is 2x10 (rows j=0,1; cols k=0..9)
    
    # Strassen for 2x2x2:
    # For block b (b in 0..4), B block is 2x2 with cols 2b, 2b+1
    
    # Let's define the Strassen decomposition for 2x2x2:
    # U_A (2x2 input) -> 7 intermediate values
    # U rows: i*m+j for i in {0,1}, j in {0,1}
    # V rows: j*p+k for j in {0,1}, k in {2b, 2b+1}
    # W rows: k*n+i for k in {2b, 2b+1}, i in {0,1}
    
    # Strassen intermediates:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # For each Strassen intermediate, we need to express it as a product of a linear
    # combination of A entries and a linear combination of B entries.
    
    # This is getting complex. Let me use a different approach:
    # Directly construct the factor matrices by enumerating the Strassen operations.
    
    # Actually, let me try a simpler approach: use the known Strassen decomposition
    # and map it to the 2x3x10 problem.
    
    # For the 2x2x10 part (j=0,1; all k):
    # We split into 5 2x2x2 blocks.
    
    # Let's build U, V, W for the 2x2x10 Strassen part:
    
    rank_2x2x10 = 35
    U_2x2x10 = np.zeros((4, rank_2x2x10))  # i in {0,1}, j in {0,1} -> 4 rows
    V_2x2x10 = np.zeros((20, rank_2x2x10))  # j in {0,1}, k in {0..9} -> 20 rows
    W_2x2x10 = np.zeros((20, rank_2x2x10))  # k in {0..9}, i in {0,1} -> 20 rows
    
    # Strassen for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # A00 = U[0, c] (i=0, j=0)
    # A01 = U[1, c] (i=0, j=1)
    # A10 = U[2, c] (i=1, j=0)
    # A11 = U[3, c] (i=1, j=1)
    
    # For block b, B00 = V[0*p+2b, c] = V[2b, c]
    # B01 = V[0*p+2b+1, c] = V[2b+1, c]
    # B10 = V[1*p+2b, c] = V[10+2b, c]
    # B11 = V[1*p+2b+1, c] = V[10+2b+1, c]
    
    # C00 = M1 + M4 - M5 + M7 -> W[2b*n+0, c] = W[2b, c]
    # C01 = M3 + M5 -> W[2b+1, c]
    # C10 = M2 + M4 -> W[10+2b, c]
    # C11 = M1 - M2 + M3 + M6 -> W[10+2b+1, c]
    
    col = 0
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        # B indices
        B00_idx = 0 * p + k0  # j=0, k=k0
        B01_idx = 0 * p + k1  # j=0, k=k1
        B10_idx = 1 * p + k0  # j=1, k=k0
        B11_idx = 1 * p + k1  # j=1, k=k1
        # W indices
        W00_idx = k0 * n + 0  # k=k0, i=0
        W01_idx = k1 * n + 0  # k=k1, i=0
        W10_idx = k0 * n + 1  # k=k0, i=1
        W11_idx = k1 * n + 1  # k=k1, i=1
        
        # M1 = (A00+A11)*(B00+B11)
        U_2x2x10[0, col] = 1  # A00
        U_2x2x10[3, col] = 1  # A11
        V_2x2x10[B00_idx, col] = 1
        V_2x2x10[B11_idx, col] = 1
        W_2x2x10[W00_idx, col] = 1
        W_2x2x10[W11_idx, col] = -1
        col += 1
        
        # M2 = (A10+A11)*B00
        U_2x2x10[2, col] = 1  # A10
        U_2x2x10[3, col] = 1  # A11
        V_2x2x10[B00_idx, col] = 1
        W_2x2x10[W10_idx, col] = 1
        W_2x2x10[W11_idx, col] = -1
        col += 1
        
        # M3 = A00*(B01-B11)
        U_2x2x10[0, col] = 1  # A00
        V_2x2x10[B01_idx, col] = 1
        V_2x2x10[B11_idx, col] = -1
        W_2x2x10[W01_idx, col] = 1
        W_2x2x10[W11_idx, col] = 1
        col += 1
        
        # M4 = A11*(B10-B00)
        U_2x2x10[3, col] = 1  # A11
        V_2x2x10[B10_idx, col] = 1
        V_2x2x10[B00_idx, col] = -1
        W_2x2x10[W00_idx, col] = 1
        W_2x2x10[W10_idx, col] = 1
        col += 1
        
        # M5 = (A00+A01)*B11
        U_2x2x10[0, col] = 1  # A00
        U_2x2x10[1, col] = 1  # A01
        V_2x2x10[B11_idx, col] = 1
        W_2x2x10[W00_idx, col] = -1
        W_2x2x10[W01_idx, col] = 1
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U_2x2x10[2, col] = 1  # A10
        U_2x2x10[0, col] = -1  # -A00
        V_2x2x10[B00_idx, col] = 1
        V_2x2x10[B01_idx, col] = 1
        W_2x2x10[W11_idx, col] = 1
        col += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U_2x2x10[1, col] = 1  # A01
        U_2x2x10[3, col] = -1  # -A11
        V_2x2x10[B10_idx, col] = 1
        V_2x2x10[B11_idx, col] = 1
        W_2x2x10[W00_idx, col] = 1
        col += 1
    
    # Now the 2x1x10 part (j=2; all k):
    # A is 2x1, B is 1x10
    # 20 scalar multiplications
    rank_2x1x10 = 20
    U_2x1x10 = np.zeros((2, rank_2x1x10))  # i in {0,1}, j=2 -> 2 rows
    V_2x1x10 = np.zeros((10, rank_2x1x10))  # j=2, k in {0..9} -> 10 rows
    W_2x1x10 = np.zeros((20, rank_2x1x10))  # k in {0..9}, i in {0,1} -> 20 rows
    
    col = 0
    for i in range(2):
        for k in range(10):
            U_2x1x10[i, col] = 1
            V_2x1x10[k, col] = 1
            W_2x1x10[k * n + i, col] = 1
            col += 1
    
    # Now combine into full U, V, W
    # U full: (n*m, rank) = (6, 55)
    # Rows: i*m+j for i in {0,1}, j in {0,1,2}
    # j=0,1 -> rows 0-3 (from U_2x2x10)
    # j=2 -> rows 4-5 (from U_2x1x10)
    
    rank_total = rank_2x2x10 + rank_2x1x10  # 35 + 20 = 55
    U_full = np.zeros((6, rank_total))
    V_full = np.zeros((30, rank_total))
    W_full = np.zeros((20, rank_total))
    
    # Copy 2x2x10 part (columns 0-34)
    U_full[0:4, 0:35] = U_2x2x10
    V_full[0:20, 0:35] = V_2x2x10
    W_full[0:20, 0:35] = W_2x2x10
    
    # Copy 2x1x10 part (columns 35-54)
    U_full[4:6, 35:55] = U_2x1x10
    V_full[20:30, 35:55] = V_2x1x10
    W_full[0:20, 35:55] = W_2x1x10
    
    # Verify the decomposition
    # The multiplication tensor T[i,j,k] = 1 if and only if (i,j,k) is a valid index
    # The decomposition should satisfy: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k)
    
    # Let's verify by checking a few entries
    # Actually, let's verify the full decomposition
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank_total):
                    val += U_full[i * m + j, c] * V_full[j * p + k, c] * W_full[k * n + i, c]
                # For matrix multiplication, this should be 1
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)
    
    if error < 1e-9 and rank_total < best_rank:
        best_U, best_V, best_W = U_full, V_full, W_full
        best_rank = rank_total
    
    # Check deadline
    if time.time() > deadline:
        # Return best found so far
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to improve further: maybe we can do better than 55?
    # The theoretical minimum is unknown, but 55 is a good starting point.
    # Let's see if we can find a better decomposition using optimization.
    
    # For now, return the 55-rank decomposition if it's valid
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
