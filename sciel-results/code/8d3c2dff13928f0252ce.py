import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 10.0 # 10 seconds is plenty for this
    
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize U, V, W
    # U: (n*m, rank) -> (4, 7)
    # V: (m*p, rank) -> (4, 7)
    # W: (p*n, rank) -> (4, 7)
    
    # Use int dtype to ensure entries are strictly integers as per "integer/half-integer" requirement
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    # Strassen's Algorithm for 2x2 multiplication
    # A is 2x2, B is 2x2
    # Indices:
    # U row index: i*m + j  (i in 0..1, j in 0..1)
    # V row index: j*p + k  (j in 0..1, k in 0..1)
    # W row index: k*n + i  (k in 0..1, i in 0..1) -> Note: W depends on k and i, NOT j.
    # The sum is over j and r.
    # Result term for C[i, k] is sum_j sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # Wait, W index is k*n+i. This corresponds to C[i, k] if we view W as a coefficient for C[i,k]?
    # Let's check the trace convention: W[k*n+i, r].
    # Usually matrix mult is C[i, k] = sum_j A[i, j] B[j, k].
    # Here we have U[i, j, r], V[j, k, r], W[?, r].
    # If W[k, i, r] is the weight, then the term is U[i, j, r] V[j, k, r] W[k, i, r].
    # This sums over j and r to give a value associated with (i, k).
    # Let's call this value X[i, k].
    # We want X[i, k] == C[i, k].
    
    # Let's define the factors for each of the 7 Strassen products.
    # Each product M_r corresponds to a rank-1 tensor slice.
    # M_r = (linear combo of A) * (linear combo of B)
    # Let A_part = sum_i,j u_{i,j,r} A[i,j]
    # Let B_part = sum_j,k v_{j,k,r} B[j,k]
    # Then contribution to C[i, k] is w_{k,i,r} * A_part * B_part.
    
    # Strassen Factors:
    # u, v, w are vectors of coefficients for A, B, and output C[i,k] respectively.
    
    # Product 1: M1 = (A00 + A11) * (B00 + B11)
    # u: A00 coeff 1, A11 coeff 1 -> u[0,0]=1, u[1,1]=1
    # v: B00 coeff 1, B11 coeff 1 -> v[0,0]=1, v[1,1]=1
    # Contribution to C[i,k]: w[i,k] * M1
    # C00 gets +1, C11 gets +1 from M1?
    # Strassen:
    # C00 = M1 - M5 + M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 + M3 - M2 + M6
    
    # So for M1 (r=0):
    # u[0,0]=1, u[1,1]=1
    # v[0,0]=1, v[1,1]=1
    # w: C00 coeff 1, C11 coeff 1 -> w[0,0]=1 (idx 0*2+0), w[1,1]=1 (idx 1*2+1)
    
    # Product 2: M2 = (A10 + A11) * B00
    # u: A10=1, A11=1 -> u[1,0]=1, u[1,1]=1
    # v: B00=1 -> v[0,0]=1
    # w: C10 coeff 1, C11 coeff -1 -> w[0,1]=1, w[1,1]=-1
    # Note: W index is k*n+i.
    # C10: i=1, k=0 -> idx 0*2+1 = 1. w[1]=1.
    # C11: i=1, k=1 -> idx 1*2+1 = 3. w[3]=-1.
    
    # Product 3: M3 = A00 * (B01 - B11)
    # u: A00=1 -> u[0,0]=1
    # v: B01=1, B11=-1 -> v[0,1]=1, v[1,1]=-1
    # w: C01 coeff 1, C11 coeff 1 -> w[1,0]=1, w[1,1]=1
    # C01: i=0, k=1 -> idx 1*2+0 = 2. w[2]=1.
    # C11: i=1, k=1 -> idx 1*2+1 = 3. w[3]=1.
    
    # Product 4: M4 = A11 * (B10 - B00)
    # u: A11=1 -> u[1,1]=1
    # v: B10=1, B00=-1 -> v[1,0]=1, v[0,0]=-1
    # w: C00 coeff 1, C10 coeff 1 -> w[0,0]=1, w[0,1]=1
    # C00: i=0, k=0 -> idx 0*2+0 = 0. w[0]=1.
    # C10: i=1, k=0 -> idx 0*2+1 = 1. w[1]=1.
    
    # Product 5: M5 = (A00 + A01) * B11
    # u: A00=1, A01=1 -> u[0,0]=1, u[0,1]=1
    # v: B11=1 -> v[1,1]=1
    # w: C00 coeff -1, C01 coeff 1 -> w[0,0]=-1, w[1,0]=1
    # C00: i=0, k=0 -> idx 0. w[0]=-1.
    # C01: i=0, k=1 -> idx 2. w[2]=1.
    
    # Product 6: M6 = (A10 - A00) * (B00 + B01)
    # u: A10=1, A00=-1 -> u[1,0]=1, u[0,0]=-1
    # v: B00=1, B01=1 -> v[0,0]=1, v[0,1]=1
    # w: C11 coeff 1 -> w[1,1]=1
    # C11: i=1, k=1 -> idx 3. w[3]=1.
    
    # Product 7: M7 = (A01 - A11) * (B10 + B11)
    # u: A01=1, A11=-1 -> u[0,1]=1, u[1,1]=-1
    # v: B10=1, B11=1 -> v[1,0]=1, v[1,1]=1
    # w: C00 coeff 1 -> w[0,0]=1
    # C00: i=0, k=0 -> idx 0. w[0]=1.

    factors = [
        # (u_coords, u_vals, v_coords, v_vals, w_coords, w_vals)
        # u_coords: list of (i, j), v_coords: list of (j, k), w_coords: list of (k, i)
        # M1
        ([ (0,0), (1,1) ], [1, 1], 
         [ (0,0), (1,1) ], [1, 1], 
         [ (0,0), (1,1) ], [1, 1]),
        # M2
        ([ (1,0), (1,1) ], [1, 1], 
         [ (0,0) ], [1], 
         [ (0,1), (1,1) ], [1, -1]),
        # M3
        ([ (0,0) ], [1], 
         [ (0,1), (1,1) ], [1, -1], 
         [ (1,0), (1,1) ], [1, 1]),
        # M4
        ([ (1,1) ], [1], 
         [ (1,0), (0,0) ], [1, -1], 
         [ (0,0), (0,1) ], [1, 1]),
        # M5
        ([ (0,0), (0,1) ], [1, 1], 
         [ (1,1) ], [1], 
         [ (0,0), (1,0) ], [-1, 1]),
        # M6
        ([ (1,0), (0,0) ], [1, -1], 
         [ (0,0), (0,1) ], [1, 1], 
         [ (1,1) ], [1]),
        # M7
        ([ (0,1), (1,1) ], [1, -1], 
         [ (1,0), (1,1) ], [1, 1], 
         [ (0,0) ], [1]),
    ]
    
    for r, (u_c, u_v, v_c, v_v, w_c, w_v) in enumerate(factors):
        for (i, j), val in zip(u_c, u_v):
            U[i*m + j, r] = val
        for (j, k), val in zip(v_c, v_v):
            V[j*p + k, r] = val
        for (k, i), val in zip(w_c, w_v):
            W[k*n + i, r] = val
            
    # Return lists of integers
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if time.time() > deadline - 0.1:
         return {"U": U_list, "V": V_list, "W": W_list}
         
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
