import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Helper to combine decompositions
    def combine_decomps(decomps):
        """
        decomps: list of dicts {'U': np.array, 'V': np.array, 'W': np.array}
        Returns a single dict with concatenated matrices.
        """
        U_list = []
        V_list = []
        W_list = []
        for d in decomps:
            U_list.append(d['U'])
            V_list.append(d['V'])
            W_list.append(d['W'])
        
        if not U_list:
            return {'U': np.array([]), 'V': np.array([]), 'W': np.array([])}
            
        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        return {'U': U, 'V': V, 'W': W}

    def schoolbook_decomp(n, m, p):
        """
        Returns schoolbook decomposition for M_{n,m,p}.
        U: (n*m, n*m*p)
        V: (m*p, n*m*p)
        W: (p*n, n*m*p)
        """
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}

    def strassen_222_decomp():
        """
        Returns rank-7 Strassen decomposition for M_{2,2,2}.
        Indices:
        U: (4, 7) for A (2x2)
        V: (4, 7) for B (2x2)
        W: (4, 7) for C (2x2)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C00, C11
        
        # m2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C10
        
        # m3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C01
        
        # m4 = d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1 # C00, C10
        
        # m5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # C00, C01
        
        # m6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C11
        
        # m7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = -1 # C00, C11
        
        return {'U': U, 'V': V, 'W': W}

    def schoolbook_212_decomp():
        """
        Returns schoolbook decomposition for M_{2,1,2} (Rank 4).
        U: (2, 4)
        V: (2, 4)
        W: (2, 4)
        """
        return schoolbook_decomp(2, 1, 2)

    def schoolbook_213_decomp():
        """
        Returns schoolbook decomposition for M_{2,1,3} (Rank 6).
        U: (2, 6)
        V: (3, 6)
        W: (6, 6)
        """
        return schoolbook_decomp(2, 1, 3)

    def combine_232_decomp():
        """
        Constructs M_{2,3,2} decomposition using M_{2,2,2} (Rank 7) and M_{2,1,2} (Rank 4).
        Total Rank 11.
        A (2x3) = [A1 (2x2), A2 (2x1)]
        B (3x2) = [B1 (2x2); B2 (1x2)]
        C = A1 B1 + A2 B2
        """
        # M_{2,2,2} part
        d_222 = strassen_222_decomp()
        # M_{2,1,2} part
        d_212 = schoolbook_212_decomp()
        
        # We need to map the indices correctly.
        # For M_{2,2,2}:
        # U indices: i (0,1), j (0,1) -> 0..3
        # V indices: j (0,1), k (0,1) -> 0..3
        # W indices: i (0,1), k (0,1) -> 0..3
        
        # For M_{2,1,2}:
        # U indices: i (0,1), j (2) -> mapped to 0..1 in local, but 0..1 in global j=2?
        # We need to embed these into the full 2x3x2 space.
        
        U_full = np.zeros((6, 11)) # 2*3 = 6
        V_full = np.zeros((6, 11)) # 3*2 = 6
        W_full = np.zeros((4, 11)) # 2*2 = 4
        
        # Part 1: M_{2,2,2} (j=0,1)
        # U: i in 0,1; j in 0,1. Global U index: i*3 + j.
        # V: j in 0,1; k in 0,1. Global V index: j*2 + k.
        # W: i in 0,1; k in 0,1. Global W index: k*2 + i.
        
        U_part1 = d_222['U']
        V_part1 = d_222['V']
        W_part1 = d_222['W']
        
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U_full[i*3 + j, c] = U_part1[i*2 + j, c]
            for j in range(2):
                for k in range(2):
                    V_full[j*2 + k, c] = V_part1[j*2 + k, c]
            for i in range(2):
                for k in range(2):
                # W index in d_222 is k*2 + i? 
                # d_222 W shape (4,7). Index k*2 + i.
                # Global W index: k*2 + i.
                    W_full[k*2 + i, c] = W_part1[k*2 + i, c]

        # Part 2: M_{2,1,2} (j=2)
        # U: i in 0,1; j=2. Global U index: i*3 + 2.
        # V: j=2; k in 0,1. Global V index: 2*2 + k = 4+k.
        # W: i in 0,1; k in 0,1. Global W index: k*2 + i.
        
        U_part2 = d_212['U'] # (2, 4)
        V_part2 = d_212['V'] # (2, 4)
        W_part2 = d_212['W'] # (4, 4)
        
        for c in range(4):
            for i in range(2):
                U_full[i*3 + 2, c + 7] = U_part2[i, c]
            for k in range(2):
                V_full[4 + k, c + 7] = V_part2[k, c]
            for i in range(2):
                for k in range(2):
                    W_full[k*2 + i, c + 7] = W_part2[k*2 + i, c]

        return {'U': U_full, 'V': V_full, 'W': W_full}

    def combine_233_decomp():
        """
        Constructs M_{2,3,3} decomposition using M_{2,2,3} (Rank 11) and M_{2,1,3} (Rank 6).
        Total Rank 17.
        A (2x3) = [A1 (2x2), A2 (2x1)]
        B (3x3) = [B1 (2x3); B2 (1x3)]
        C = A1 B1 + A2 B2
        """
        # We need M_{2,2,3} (Rank 11).
        # Construct M_{2,2,3} using M_{2,2,2} (Rank 7) and M_{2,2,1} (Rank 4).
        # M_{2,2,1} is schoolbook rank 4.
        
        def schoolbook_221_decomp():
            return schoolbook_decomp(2, 2, 1)
            
        d_222 = strassen_222_decomp()
        d_221 = schoolbook_221_decomp()
        
        # M_{2,2,3}
        U_223 = np.zeros((4, 11))
        V_223 = np.zeros((6, 11))
        W_223 = np.zeros((6, 11)) # p*n = 3*2 = 6
        
        # Part 1: M_{2,2,2} (k=0,1)
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U_223[i*2 + j, c] = d_222['U'][i*2 + j, c]
            for j in range(2):
                for k in range(2):
                    V_223[j*2 + k, c] = d_222['V'][j*2 + k, c]
            for i in range(2):
                for k in range(2):
                    W_223[k*2 + i, c] = d_222['W'][k*2 + i, c]

        # Part 2: M_{2,2,1} (k=2)
        # U: i, j (0,1). Same as U_222.
        # V: j (0,1), k=2. Global V index: j*3 + 2.
        # W: i (0,1), k=2. Global W index: 2*2 + i = 4+i.
        
        for c in range(4):
            for i in range(2):
                for j in range(2):
                    U_223[i*2 + j, c + 7] = d_221['U'][i*2 + j, c]
            for j in range(2):
                V_223[j*3 + 2, c + 7] = d_221['V'][j, c]
            for i in range(2):
                W_223[4 + i, c + 7] = d_221['W'][2 + i, c] # W index in 221 is k*n+i = 0*2+i = i? 
                # Wait, schoolbook_decomp(2,2,1) W shape is (1*2, 4) = (2,4). Index k*n+i = 0*2+i = i.
                # So W_221[i, c].
                # Global W index for k=2 is 2*2+i = 4+i.
                W_223[4 + i, c + 7] = d_221['W'][i, c]

        d_223 = {'U': U_223, 'V': V_223, 'W': W_223}
        
        # Now combine M_{2,2,3} (j=0,1) and M_{2,1,3} (j=2)
        d_213 = schoolbook_213_decomp()
        
        U_full = np.zeros((6, 17)) # 2*3
        V_full = np.zeros((9, 17)) # 3*3
        W_full = np.zeros((6, 17)) # 3*2
        
        # Part 1: M_{2,2,3}
        for c in range(11):
            for i in range(2):
                for j in range(2):
                    U_full[i*3 + j, c] = d_223['U'][i*2 + j, c]
            for j in range(2):
                for k in range(3):
                    V_full[j*3 + k, c] = d_223['V'][j*3 + k, c]
            for i in range(2):
                for k in range(3):
                    W_full[k*2 + i, c] = d_223['W'][k*2 + i, c]

        # Part 2: M_{2,1,3}
        for c in range(6):
            for i in range(2):
                U_full[i*3 + 2, c + 11] = d_213['U'][i, c]
            for k in range(3):
                V_full[2*3 + k, c + 11] = d_213['V'][k, c]
            for i in range(2):
                for k in range(3):
                    W_full[k*2 + i, c + 11] = d_213['W'][k*2 + i, c]

        return {'U': U_full, 'V': V_full, 'W': W_full}

    def final_decomp():
        d_232 = combine_232_decomp()
        d_233 = combine_233_decomp()
        
        # Combine M_{2,3,2} and M_{2,3,3}
        # A (2x3), B (3x5) = [B1 (3x2), B2 (3x3)]
        # C = A B1 + A B2
        
        U_full = np.zeros((6, 28))
        V_full = np.zeros((15, 28))
        W_full = np.zeros((10, 28))
        
        for c in range(11):
            for i in range(2):
                for j in range(3):
                    U_full[i*3 + j, c] = d_232['U'][i*3 + j, c]
            for j in range(3):
                for k in range(2):
                    V_full[j*5 + k, c] = d_232['V'][j*2 + k, c]
            for i in range(2):
                for k in range(2):
                    W_full[k*2 + i, c] = d_232['W'][k*2 + i, c]

        for c in range(17):
            for i in range(2):
                for j in range(3):
                    U_full[i*3 + j, c + 11] = d_233['U'][i*3 + j, c]
            for j in range(3):
                for k in range(3):
                    V_full[j*5 + (2 + k), c + 11] = d_233['V'][j*3 + k, c]
            for i in range(2):
                for k in range(3):
                    W_full[(2 + k)*2 + i, c + 11] = d_233['W'][k*2 + i, c]

        return {'U': U_full, 'V': V_full, 'W': W_full}

    try:
        result = final_decomp()
        # Verify
        U, V, W = result['U'], result['V'], result['W']
        T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
        
        T = np.zeros((6, 15, 10))
        for i in range(2):
            for j in range(3):
                for k in range(5):
                    T[i*3+j, j*5+k, k*2+i] = 1.0
        
        err = np.linalg.norm(T_calc - T)
        if err < 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to schoolbook if verification fails
            sb = schoolbook_decomp(2, 3, 5)
            return {"U": sb['U'].tolist(), "V": sb['V'].tolist(), "W": sb['W'].tolist()}
    except Exception:
        sb = schoolbook_decomp(2, 3, 5)
        return {"U": sb['U'].tolist(), "V": sb['V'].tolist(), "W": sb['W'].tolist()}

# EVOLVE-BLOCK-END
