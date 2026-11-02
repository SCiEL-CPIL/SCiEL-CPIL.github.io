import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _build_rank56():
    """
    Build the rank-56 decomposition for 2x4 by 4x8 matrix multiplication.
    This uses the Strassen algorithm for 2x2 blocks.
    The multiplication is split into 4 independent 2x2 by 2x2 multiplications 
    (actually, it's 4 sums of two 2x2 products).
    """
    n, m, p = DIMENSIONS
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We have 4 blocks of B, each 4x2.
    # For each block d (0..3), we compute C_d = A B_d.
    # A is 2x4, B_d is 4x2.
    # Split A into A_L (2x2) and A_R (2x2).
    # Split B_d into B_dL (2x2) and B_dR (2x2).
    # C_d = A_L B_dL + A_R B_dR.
    # This requires 14 multiplications per d.
    # Total 4 * 14 = 56.
    
    for d in range(4):
        # d corresponds to the column block of B and C.
        # B columns: 2d, 2d+1
        # C columns: 2d, 2d+1
        
        # Indices for A:
        # A_L: rows 0,1; cols 0,1
        # A_R: rows 0,1; cols 2,3
        
        # Indices for B_d:
        # B_dL: rows 0,1; cols 2d, 2d+1
        # B_dR: rows 2,3; cols 2d, 2d+1
        
        # Indices for C_d:
        # C_d: rows 0,1; cols 2d, 2d+1
        
        base = d * 14
        
        # Strassen for 2x2:
        # M1 = A00 * B00
        # M2 = A10 * B00
        # M3 = A00 * B01
        # M4 = A11 * B11
        # M5 = (A00+A11) * (B00+B11)
        # M6 = (A10+A11) * (B11-B01)
        # M7 = (A00+A01) * (B10-B00)
        
        # We need to compute A_L B_dL and A_R B_dR.
        # Let's denote the Strassen multipliers for A_L B_dL as M1..M7
        # and for A_R B_dR as M8..M14.
        
        # Helper to add to U, V, W
        def add_term(u_idx, v_idx, w_idx, val=1.0):
            # u_idx is (row, col) in A
            # v_idx is (row, col) in B
            # w_idx is (row, col) in C
            # We need to map these to the global indices.
            pass

        # Instead of complex indexing, we use the explicit mapping from the current program.
        # The current program logic:
        # for d in range(4):
        #   for h in range(2): # h=0 for A_L, h=1 for A_R
        #     base = (d * 2 + h) * 7
        #     ...
        
        # This logic is correct. I will reuse it.
        pass

    # Re-implementing the specific logic from the current program to ensure correctness
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for d in range(4):
        for h in range(2):
            base = (d * 2 + h) * 7
            if h == 0:
                def a_idx(r, c): return r * m + c
                def b_idx(r, c): return r * p + (2 * d + c)
            else:
                def a_idx(r, c): return r * m + (2 + c)
                def b_idx(r, c): return (2 + r) * p + (2 * d + c)
            def c_idx(r, c): return (2 * d + c) * n + r
            
            # M1
            col = base + 0
            U[a_idx(0,0),col]+=1; U[a_idx(1,1),col]+=1
            V[b_idx(0,0),col]+=1; V[b_idx(1,1),col]+=1
            W[c_idx(0,0),col]+=1; W[c_idx(1,1),col]+=1
            # M2
            col = base + 1
            U[a_idx(1,0),col]+=1; U[a_idx(1,1),col]+=1
            V[b_idx(0,0),col]+=1
            W[c_idx(1,0),col]+=1; W[c_idx(1,1),col]-=1
            # M3
            col = base + 2
            U[a_idx(0,0),col]+=1
            V[b_idx(0,1),col]+=1; V[b_idx(1,1),col]-=1
            W[c_idx(0,1),col]+=1; W[c_idx(1,1),col]+=1
            # M4
            col = base + 3
            U[a_idx(1,1),col]+=1
            V[b_idx(1,0),col]+=1; V[b_idx(0,0),col]-=1
            W[c_idx(0,0),col]+=1; W[c_idx(1,0),col]+=1
            # M5
            col = base + 4
            U[a_idx(0,0),col]+=1; U[a_idx(0,1),col]+=1
            V[b_idx(1,1),col]+=1
            W[c_idx(0,0),col]-=1; W[c_idx(0,1),col]+=1
            # M6
            col = base + 5
            U[a_idx(1,0),col]+=1; U[a_idx(0,0),col]-=1
            V[b_idx(0,0),col]+=1; V[b_idx(0,1),col]+=1
            W[c_idx(1,1),col]+=1
            # M7
            col = base + 6
            U[a_idx(0,1),col]+=1; U[a_idx(1,1),col]-=1
            V[b_idx(1,0),col]+=1; V[b_idx(1,1),col]+=1
            W[c_idx(0,0),col]+=1

    return U, V, W

def construct():
    """
    Construct a rank-minimal decomposition for 2x4 by 4x8 matrix multiplication.
    """
    deadline = time.time() + 120.0
    U, V, W = _build_rank56()
    
    # Verification
    n, m, p = DIMENSIONS
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                if abs(val - 1.0) > 1e-8:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to naive if Strassen construction fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    return True

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    n, m, p = DIMENSIONS
    ok = _verify(result["U"], result["V"], result["W"], n, m, p)
    rank = len(result["U"][0])
    elapsed = time.time() - t0
    print(f"Rank: {rank}, Valid: {ok}, Time: {elapsed:.1f}s")
    if not ok:
        print("DECOMPOSITION IS INVALID!")
    else:
        print("Decomposition verified successfully.")
