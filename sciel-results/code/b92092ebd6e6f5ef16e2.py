"""Program for 2x3x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 9]

import time
import random
import numpy as np

def construct():
    """Return a low-rank exact tensor decomposition for 2x3x9 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 9
    schoolbook_rank = n * m * p  # 54

    def build_schoolbook():
        """Build the schoolbook (rank 54) decomposition."""
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify_full(U, V, W, rank, n=n, m=m, p=p):
        """Verify the decomposition is exact for the full 2x3x9 tensor."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Build target tensor T of shape (n*m, m*p, p*n)
        T = np.zeros((n * m, m * p, p * n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0

        # Compute decomposition tensor
        D = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)

        return np.allclose(D, T, atol=1e-9)

    def verify_233(U, V, W, R):
        """Verify 2x3x3 decomposition."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        T = np.zeros((6, 9, 6), dtype=float)
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    u = i * 3 + j
                    v = j * 3 + k
                    w = k * 2 + i
                    T[u, v, w] = 1.0

        D = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
        return np.allclose(D, T, atol=1e-9)

    def verify_222(U, V, W, R):
        """Verify 2x2x2 decomposition (Strassen-style)."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        T = np.zeros((4, 4, 4), dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = i * 2 + j
                    v = j * 2 + k
                    w = k * 2 + i
                    T[u, v, w] = 1.0

        D = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
        return np.allclose(D, T, atol=1e-9)

    def build_strassen_222():
        """Build Strassen's 2x2x2 decomposition (rank 7)."""
        # Strassen's algorithm for 2x2 matrix multiplication
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = A*B = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
        #
        # Strassen's 7 products:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-d)(e+f)
        # M7 = (b-c)(g+h)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        R = 7
        U = np.zeros((4, R))
        V = np.zeros((4, R))
        W = np.zeros((4, R))
        
        # Index mapping: U[i*2+j], V[j*2+k], W[k*2+i]
        # i=0,1; j=0,1; k=0,1
        # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        # M1 = (a+d)(e+h)
        # U: a and d are non-zero -> U[0]=1, U[3]=1
        # V: e and h are non-zero -> V[0]=1, V[1]=1
        # W: contributes to C00 (k=0,i=0) and C11 (k=1,i=1)
        # C00 = M1+M4-M5+M7, C11 = M1-M2+M3+M6
        # So W for M1: W[0]=1 (C00), W[3]=1 (C11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)e
        # U: c and d -> U[2]=1, U[3]=1
        # V: e -> V[0]=1
        # W: contributes to C10 (k=0,i=1) and C11 (k=1,i=1)
        # C10 = M2+M4, C11 = M1-M2+M3+M6
        # W[2] = 1 (C10), W[3] = -1 (C11)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = a(f-h)
        # U: a -> U[0]=1
        # V: f and h -> V[1]=1, V[3]=-1 (since f-h)
        # W: contributes to C01 (k=1,i=0) and C11 (k=1,i=1)
        # C01 = M3+M5, C11 = M1-M2+M3+M6
        # W[1] = 1 (C01), W[3] = 1 (C11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = d(g-e)
        # U: d -> U[3]=1
        # V: g and e -> V[2]=1, V[0]=-1 (since g-e)
        # W: contributes to C00 (k=0,i=0) and C10 (k=0,i=1)
        # C00 = M1+M4-M5+M7, C10 = M2+M4
        # W[0] = 1 (C00), W[2] = 1 (C10)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (a+b)h
        # U: a and b -> U[0]=1, U[1]=1
        # V: h -> V[3]=1
        # W: contributes to C00 (k=0,i=0) and C01 (k=1,i=0)
        # C00 = M1+M4-M5+M7, C01 = M3+M5
        # W[0] = -1 (C00), W[1] = 1 (C01)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (c-d)(e+f)
        # U: c and d -> U[2]=1, U[3]=-1
        # V: e and f -> V[0]=1, V[1]=1
        # W: contributes to C11 (k=1,i=1)
        # C11 = M1-M2+M3+M6
        # W[3] = 1 (C11)
        U[2, 5] = 1; U[3, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (b-c)(g+h)
        # U: b and c -> U[1]=1, U[2]=-1
        # V: g and h -> V[2]=1, V[3]=1
        # W: contributes to C00 (k=0,i=0)
        # C00 = M1+M4-M5+M7
        # W[0] = 1 (C00)
        U[1, 6] = 1; U[2, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_239_from_222(decomp_222):
        """
        Build 2x3x9 decomposition using 2x2x2 Strassen blocks.
        
        Strategy:
        - Split j dimension into {0,1} and {2}
        - For j={0,1}: use 2x2x2 Strassen for each pair of k columns
        - For j=2: use schoolbook for each k column
        
        For p=9:
        - j={0,1}: 4 pairs of k columns (0,1), (2,3), (4,5), (6,7) -> 4*7=28
        - j={0,1}, k=8: 2x2x1 schoolbook -> 4
        - j=2: 9 columns * 2 (i values) = 18
        Total: 28 + 4 + 18 = 50
        """
        R_222 = len(decomp_222["U"][0])  # 7
        total_rank = 4 * R_222 + 4 + 18  # 28 + 4 + 18 = 50
        
        U = [[0] * total_rank for _ in range(6)]
        V = [[0] * total_rank for _ in range(27)]
        W = [[0] * total_rank for _ in range(18)]
        
        U_222 = decomp_222["U"]
        V_222 = decomp_222["V"]
        W_222 = decomp_222["W"]
        
        col = 0
        
        # Part 1: j={0,1}, k in pairs
        for block in range(4):
            k_start = block * 2
            for s in range(R_222):
                # U: same for all blocks (A is 2x2, shared)
                for u_idx in range(4):
                    # u_idx maps to (i, j) where i in {0,1}, j in {0,1}
                    # In 2x3 matrix, j=0,1 are the first two columns
                    U[u_idx][col] = U_222[u_idx][s]
                
                # V: j in {0,1}, k in {k_start, k_start+1}
                for v_idx in range(4):
                    j = v_idx // 2
                    k_local = v_idx % 2
                    k_actual = k_start + k_local
                    V[j * 9 + k_actual][col] = V_222[v_idx][s]
                
                # W: k in {k_start, k_start+1}, i in {0,1}
                for w_idx in range(4):
                    k_local = w_idx // 2
                    i = w_idx % 2
                    k_actual = k_start + k_local
                    W[k_actual * 2 + i][col] = W_222[w_idx][s]
                
                col += 1
        
        # Part 2: j={0,1}, k=8 (single column)
        # Schoolbook 2x2x1: rank 4
        for i in range(2):
            for j in range(2):
                U[i * 3 + j][col] = 1
                V[j * 9 + 8][col] = 1
                W[8 * 2 + i][col] = 1
                col += 1
        
        # Part 3: j=2, k in 0..8
        # Schoolbook 2x1x9: rank 18
        for i in range(2):
            for k in range(9):
                U[i * 3 + 2][col] = 1
                V[2 * 9 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
        
        return {"U": U, "V": V, "W": W}

    # Try to find a better 2x2x2 decomposition (Strassen gives 7, which is optimal)
    # Verify Strassen
    strassen_222 = build_strassen_222()
    if verify_222(strassen_222["U"], strassen_222["V"], strassen_222["W"], 7):
        result = build_239_from_222(strassen_222)
        if verify_full(result["U"], result["V"], result["W"], 50):
            return result
        else:
            # Fallback to schoolbook
            return build_schoolbook()
    else:
        # Strassen verification failed, try schoolbook
        return build_schoolbook()

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
