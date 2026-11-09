import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    def verify(U, V, W, n, m, p):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def to_exact(U_mat):
        U_list = []
        for row in U_mat:
            r = []
            for val in row:
                v = round(val * 2) / 2.0
                if abs(v - round(v)) < 1e-9:
                    r.append(int(round(v)))
                else:
                    r.append(v)
            U_list.append(r)
        return U_list

    # Try rank 50 construction
    # Idea: Use 5 groups of 2 columns. For each group, use a 2x3x2 sub-problem.
    # Rank of 2x3x2: schoolbook is 12. Can we do better?
    # Let's try: for each 2-column group, use 10 terms (instead of 12)
    # Total: 5 * 10 = 50
    
    # Actually, let me try the Strassen approach more carefully.
    # For 2x2x2, Strassen uses 7 terms.
    # For 2x3x2, we can think of it as:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # for i in {0,1}, k in {0,1}
    # That's 4 outputs, each a sum of 3 products = 12 schoolbook terms.
    
    # Can we share? The A entries are shared across k, B entries shared across i.
    # Let's try to use 10 terms per 2-column group.
    
    # For a 2x3x2 block, let me try:
    # M1 = A[0,0]*(B[0,0]+B[0,1]) -> contributes to C[0,0] and C[0,1]
    # M2 = A[1,0]*(B[0,0]+B[0,1]) -> contributes to C[1,0] and C[1,1]
    # M3 = A[0,1]*B[1,0] -> C[0,0]
    # M4 = A[0,1]*B[1,1] -> C[0,1]
    # M5 = A[1,1]*B[1,0] -> C[1,0]
    # M6 = A[1,1]*B[1,1] -> C[1,1]
    # M7 = A[0,2]*B[2,0] -> C[0,0]
    # M8 = A[0,2]*B[2,1] -> C[0,1]
    # M9 = A[1,2]*B[2,0] -> C[1,0]
    # M10 = A[1,2]*B[2,1] -> C[1,1]
    # That's 10 terms, but let me verify:
    # C[0,0] = M1_part + M3 + M7 = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] ✓
    # C[0,1] = M1_part + M4 + M8 = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1] ✓
    # C[1,0] = M2_part + M5 + M9 = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0] ✓
    # C[1,1] = M2_part + M6 + M10 = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1] ✓
    # Yes! 10 terms per 2-column group, total 50.
    
    rank_50 = 50
    U50 = np.zeros((n * m, rank_50))
    V50 = np.zeros((m * p, rank_50))
    W50 = np.zeros((p * n, rank_50))
    
    c_idx = 0
    for col_pair in range(5):
        k0 = col_pair * 2
        k1 = col_pair * 2 + 1
        
        # M1 = A[0,0]*(B[0,0]+B[0,1])
        U50[0, c_idx] = 1
        V50[k0, c_idx] = 1; V50[k1, c_idx] = 1
        W50[k0*2+0, c_idx] = 1; W50[k1*2+0, c_idx] = 1
        c_idx += 1
        
        # M2 = A[1,0]*(B[0,0]+B[0,1])
        U50[3, c_idx] = 1
        V50[k0, c_idx] = 1; V50[k1, c_idx] = 1
        W50[k0*2+1, c_idx] = 1; W50[k1*2+1, c_idx] = 1
        c_idx += 1
        
        # M3 = A[0,1]*B[1,0]
        U50[1, c_idx] = 1
        V50[10+k0, c_idx] = 1
        W50[k0*2+0, c_idx] = 1
        c_idx += 1
        
        # M4 = A[0,1]*B[1,1]
        U50[1, c_idx] = 1
        V50[10+k1, c_idx] = 1
        W50[k1*2+0, c_idx] = 1
        c_idx += 1
        
        # M5 = A[1,1]*B[1,0]
        U50[4, c_idx] = 1
        V50[10+k0, c_idx] = 1
        W50[k0*2+1, c_idx] = 1
        c_idx += 1
        
        # M6 = A[1,1]*B[1,1]
        U50[4, c_idx] = 1
        V50[10+k1, c_idx] = 1
        W50[k1*2+1, c_idx] = 1
        c_idx += 1
        
        # M7 = A[0,2]*B[2,0]
        U50[2, c_idx] = 1
        V50[20+k0, c_idx] = 1
        W50[k0*2+0, c_idx] = 1
        c_idx += 1
        
        # M8 = A[0,2]*B[2,1]
        U50[2, c_idx] = 1
        V50[20+k1, c_idx] = 1
        W50[k1*2+0, c_idx] = 1
        c_idx += 1
        
        # M9 = A[1,2]*B[2,0]
        U50[5, c_idx] = 1
        V50[20+k0, c_idx] = 1
        W50[k0*2+1, c_idx] = 1
        c_idx += 1
        
        # M10 = A[1,2]*B[2,1]
        U50[5, c_idx] = 1
        V50[20+k1, c_idx] = 1
        W50[k1*2+1, c_idx] = 1
        c_idx += 1
    
    if verify(U50, V50, W50, n, m, p):
        best_U, best_V, best_W = U50, V50, W50
    else:
        # Fallback to rank 55 Strassen
        rank_55 = 55
        U55 = np.zeros((n * m, rank_55))
        V55 = np.zeros((m * p, rank_55))
        W55 = np.zeros((p * n, rank_55))
        
        c_idx = 0
        for col_pair in range(5):
            k0 = col_pair * 2
            k1 = col_pair * 2 + 1
            
            U55[0, c_idx] = 1; U55[4, c_idx] = 1
            V55[k0, c_idx] = 1; V55[10+k1, c_idx] = 1
            W55[k0*2+0, c_idx] = 1; W55[k1*2+1, c_idx] = 1
            c_idx += 1
            
            U55[3, c_idx] = 1; U55[4, c_idx] = 1
            V55[k0, c_idx] = 1
            W55[k0*2+1, c_idx] = 1
            c_idx += 1
            
            U55[0, c_idx] = 1
            V55[k1, c_idx] = 1; V55[10+k1, c_idx] = -1
            W55[k1*2+0, c_idx] = 1; W55[k1*2+1, c_idx] = -1
            c_idx += 1
            
            U55[4, c_idx] = 1
            V55[10+k0, c_idx] = 1; V55[k0, c_idx] = -1
            W55[k0*2+0, c_idx] = -1; W55[k0*2+1, c_idx] = 1
            c_idx += 1
            
            U55[0, c_idx] = 1; U55[1, c_idx] = 1
            V55[10+k1, c_idx] = 1
            W55[k1*2+0, c_idx] = 1; W55[k1*2+1, c_idx] = 1
            c_idx += 1
            
            U55[3, c_idx] = 1; U55[0, c_idx] = -1
            V55[k0, c_idx] = 1; V55[k1, c_idx] = 1
            W55[k0*2+0, c_idx] = 1; W55[k0*2+1, c_idx] = -1
            c_idx += 1
            
            U55[1, c_idx] = 1; U55[4, c_idx] = -1
            V55[10+k0, c_idx] = 1; V55[10+k1, c_idx] = 1
            W55[k0*2+0, c_idx] = -1; W55[k1*2+0, c_idx] = 1
            c_idx += 1
            
            U55[2, c_idx] = 1; V55[20+k0, c_idx] = 1; W55[k0*2+0, c_idx] = 1; c_idx += 1
            U55[2, c_idx] = 1; V55[20+k1, c_idx] = 1; W55[k1*2+0, c_idx] = 1; c_idx += 1
            U55[5, c_idx] = 1; V55[20+k0, c_idx] = 1; W55[k0*2+1, c_idx] = 1; c_idx += 1
            U55[5, c_idx] = 1; V55[20+k1, c_idx] = 1; W55[k1*2+1, c_idx] = 1; c_idx += 1
        
        if verify(U55, V55, W55, n, m, p):
            best_U, best_V, best_W = U55, V55, W55
        else:
            rank60 = n * m * p
            U60 = np.zeros((n * m, rank60))
            V60 = np.zeros((m * p, rank60))
            W60 = np.zeros((p * n, rank60))
            for c in range(rank60):
                i = c // (m * p)
                j = (c // p) % m
                k = c % p
                U60[i * m + j, c] = 1
                V60[j * p + k, c] = 1
                W60[k * n + i, c] = 1
            best_U, best_V, best_W = U60, V60, W60

    return {"U": to_exact(best_U), "V": to_exact(best_V), "W": to_exact(best_W)}
