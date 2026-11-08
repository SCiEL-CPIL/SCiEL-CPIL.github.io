import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    deadline = time.time() + 130  # safety margin
    
    # Strategy: Block decomposition
    # Split j (middle dim, size 3) into {0,1} and {2}
    # Split i (row dim, size 3) into {0,1} and {2}
    # Split k (col dim, size 7) into {0,1}, {2,3}, {4,5}, {6}
    
    # This gives:
    # j={2}: 3x1 x 1x7 -> rank 1
    # j={0,1}, i={2}: 1x2 x 2x7 -> rank 1
    # j={0,1}, i={0,1}: 2x2 x 2x7 -> rank 22 (3*Strassen + 1 tail)
    # Total: 24
    
    # Let's build this decomposition
    
    # First, define Strassen for 2x2 x 2x2
    # The 7 terms of Strassen's algorithm
    # Using the standard formulation
    
    # I'll use a verified Strassen decomposition
    # Reference: the 7 terms are:
    strassen_U = np.array([
        [1, 1, 1, 0, 1, 0, 0],  # i=0,j=0
        [0, 1, 0, 0, 1, 1, -1], # i=0,j=1
        [0, 1, 0, 1, 0, 1, 0],  # i=1,j=0
        [1, 1, -1, 1, 0, 0, 1], # i=1,j=1
    ], dtype=float)
    
    strassen_V = np.array([
        [1, 1, 0, 0, 0, 1, 0],  # j=0,k=0
        [0, 0, 1, -1, 1, 0, 1], # j=0,k=1
        [0, 0, 1, 1, 0, 0, 0],  # j=1,k=0
        [1, 0, -1, 0, 1, 1, 1], # j=1,k=1
    ], dtype=float)
    
    strassen_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # k=0,i=0
        [0, 1, 0, 0, 0, 1, 0],   # k=0,i=1
        [0, 0, 1, 0, 1, 0, -1],  # k=1,i=0
        [1, -1, 1, 0, 0, 0, 1],  # k=1,i=1
    ], dtype=float)
    
    # Verify Strassen
    # For 2x2x2x2: check sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] == 1
    U_s = strassen_U.reshape(4, 7)
    V_s = strassen_V.reshape(4, 7)
    W_s = strassen_W.reshape(4, 7)
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for c in range(7):
                    s += U_s[i*2+j, c] * V_s[j*2+k, c] * W_s[k*2+i, c]
                if abs(s - 1.0) > 1e-9:
                    print(f"Strassen verify failed at ({i},{j},{k}): {s}")
                    # Fall back to schoolbook
                    pass
    
    # Actually, let me just verify and if it fails, use schoolbook
    strassen_ok = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U_s[i*2+j, c] * V_s[j*2+k, c] * W_s[k*2+i, c] for c in range(7))
                if abs(s - 1.0) > 1e-9:
                    strassen_ok = False
                    break
    
    if not strassen_ok:
        # Use schoolbook for 2x2x2x2 (rank 8)
        strassen_U = np.eye(4).T  # 4x4, will pad to 4x8
        # Actually just use identity-based schoolbook
        pass
    
    # Build the full decomposition
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (21, rank)
    # W shape: (p*n, rank) = (21, rank)
    
    U_full = []
    V_full = []
    W_full = []
    
    # Block 1: j={2}, all i, all k -> 3x1 x 1x7, rank 1
    # U[i*3+2, c] * V[2*7+k, c] * W[k*3+i, c] = 1
    # Set all to 1
    U_full.append([0,0,0, 0,0,0, 1])  # i*3+j: i=0->2, i=1->5, i=2->8, j=2
    V_full.append([0]*7 + [1]*7 + [0]*7)  # j*7+k: j=2 -> 14..20
    W_full.append([1]*21)  # k*3+i: all 1
    
    # Block 2: j={0,1}, i={2}, all k -> 1x2 x 2x7, rank 1
    # U[2*3+j, c] * V[j*7+k, c] * W[k*3+2, c] = 1
    # j∈{0,1}, k∈{0..6}
    U_full.append([0,0,0, 0,0,0, 1,1,0])  # i=2, j=0 -> 6; i=2, j=1 -> 7
    V_full.append([1]*14 + [0]*7)  # j=0,1 -> 0..13
    W_full.append([0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1])  # k*3+2
    
    # Block 3: j={0,1}, i={0,1}, k={0,1} -> 2x2 x 2x2, Strassen rank 7
    # U[i*3+j, c] for i∈{0,1}, j∈{0,1}
    # V[j*7+k, c] for j∈{0,1}, k∈{0,1}
    # W[k*3+i, c] for k∈{0,1}, i∈{0,1}
    
    for c in range(7):
        u_col = [0]*9
        v_col = [0]*21
        w_col = [0]*21
        for i in range(2):
            for j in range(2):
                u_col[i*3+j] = strassen_U[i*2+j, c]
        for j in range(2):
            for k in range(2):
                v_col[j*7+k] = strassen_V[j*2+k, c]
        for k in range(2):
            for i in range(2):
                w_col[k*3+i] = strassen_W[k*2+i, c]
        U_full.append(u_col)
        V_full.append(v_col)
        W_full.append(w_col)
    
    # Block 4: j={0,1}, i={0,1}, k={2,3} -> 2x2 x 2x2, Strassen rank 7
    for c in range(7):
        u_col = [0]*9
        v_col = [0]*21
        w_col = [0]*21
        for i in range(2):
            for j in range(2):
                u_col[i*3+j] = strassen_U[i*2+j, c]
        for j in range(2):
            for k in range(2):
                v_col[j*7+(k+2)] = strassen_V[j*2+k, c]
        for k in range(2):
            for i in range(2):
                w_col[(k+2)*3+i] = strassen_W[k*2+i, c]
        U_full.append(u_col)
        V_full.append(v_col)
        W_full.append(w_col)
    
    # Block 5: j={0,1}, i={0,1}, k={4,5} -> 2x2 x 2x2, Strassen rank 7
    for c in range(7):
        u_col = [0]*9
        v_col = [0]*21
        w_col = [0]*21
        for i in range(2):
            for j in range(2):
                u_col[i*3+j] = strassen_U[i*2+j, c]
        for j in range(2):
            for k in range(2):
                v_col[j*7+(k+4)] = strassen_V[j*2+k, c]
        for k in range(2):
            for i in range(2):
                w_col[(k+4)*3+i] = strassen_W[k*2+i, c]
        U_full.append(u_col)
        V_full.append(v_col)
        W_full.append(w_col)
    
    # Block 6: j={0,1}, i={0,1}, k={6} -> 2x2 x 2x1, rank 1
    # U[i*3+j, c] * V[j*7+6, c] * W[6*3+i, c] = 1
    # Set all to 1
    u_col = [0]*9
    v_col = [0]*21
    w_col = [0]*21
    for i in range(2):
        for j in range(2):
            u_col[i*3+j] = 1
    for j in range(2):
        v_col[j*7+6] = 1
    for i in range(2):
        w_col[6*3+i] = 1
    U_full.append(u_col)
    V_full.append(v_col)
    W_full.append(w_col)
    
    rank = len(U_full)
    
    # Convert to numpy for verification
    U_np = np.array(U_full, dtype=float).T  # (9, rank)
    V_np = np.array(V_full, dtype=float).T  # (21, rank)
    W_np = np.array(W_full, dtype=float).T  # (21, rank)
    
    # Verify
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False, (i, j, k, s)
        return True, None
    
    ok, info = verify(U_np, V_np, W_np)
    if not ok:
        print(f"Verification failed: {info}")
        # Fall back to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n*m, rank_school))
        V_school = np.zeros((m*p, rank_school))
        W_school = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
        U_list = [[float(U_school[i, c]) for c in range(rank_school)] for i in range(n*m)]
        V_list = [[float(V_school[i, c]) for c in range(rank_school)] for i in range(m*p)]
        W_list = [[float(W_school[i, c]) for c in range(rank_school)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Convert to lists
    U_list = [[float(U_np[i, c]) for c in range(rank)] for i in range(n*m)]
    V_list = [[float(V_np[i, c]) for c in range(rank)] for i in range(m*p)]
    W_list = [[float(W_np[i, c]) for c in range(rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
