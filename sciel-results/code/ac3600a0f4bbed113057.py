"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a low-rank decomposition for multiplying a 2x2 matrix by a 2x11 matrix.
    Uses Strassen's algorithm (rank 7) for pairs of columns and a rank-5 algorithm
    for the remaining single column to minimize the total rank.
    """
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # n=2, m=2, p=11
    # U: 4 x rank
    # V: 22 x rank
    # W: 22 x rank

    U_cols = []
    V_cols = []
    W_cols = []

    def add_strassen_pair(k1, k2):
        """
        Adds rank-7 decomposition for multiplying 2x2 A by 2x2 B_sub (columns k1, k2).
        B_sub = [[B0k1, B0k2], [B1k1, B1k2]]
        A = [[a, b], [c, d]]
        Indices for U: i*m+j -> 0:00, 1:01, 2:10, 3:11
        Indices for V: j*p+k -> 0:0k1, 1:0k2, 2:1k1, 3:1k2, 4:0k1', ... (actually just local indices for V in this block)
        Wait, V global index is j*p+k.
        V rows:
        j=0: k1 -> 0*p + k1 = k1, k2 -> k2
        j=1: k1 -> 1*p + k1 = p+k1, k2 -> p+k2
        
        Indices for W: k*n+i
        k=k1: i=0 -> k1*n+0 = 2k1, i=1 -> 2k1+1
        k=k2: i=0 -> 2k2, i=1 -> 2k2+1
        """
        # Strassen steps for AB where B has columns k1, k2
        # M1 = (a+d)(B0k1+B1k2) -- wait, standard Strassen is (a+d)(e+h) where e=B00, h=B11.
        # Here B00 = B[0,k1], B01 = B[0,k2], B10 = B[1,k1], B11 = B[1,k2].
        # e = B[0,k1], f = B[0,k2], g = B[1,k1], h = B[1,k2].
        
        # M1 = (A00 + A11) * (B[0,k1] + B[1,k2])
        # U: A00, A11 -> 1. V: B[0,k1], B[1,k2] -> 1.
        # Output C[0,k1] += M1, C[1,k2] += M1
        U = [1, 0, 0, 1]
        V = [0]*22
        V[k1] = 1
        V[p+k2] = 1
        W = [0]*22
        W[2*k1] = 1
        W[2*k2+1] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)

        # M2 = (A10 + A11) * B[0,k1]
        # U: A10, A11 -> 1. V: B[0,k1] -> 1.
        # Output C[1,k1] += M2
        U = [0, 0, 1, 1]
        V = [0]*22
        V[k1] = 1
        W = [0]*22
        W[2*k1+1] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)

        # M3 = A00 * (B[0,k2] - B[1,k2])
        # U: A00 -> 1. V: B[0,k2] -> 1, B[1,k2] -> -1.
        # Output C[0,k2] += M3
        U = [1, 0, 0, 0]
        V = [0]*22
        V[k2] = 1
        V[p+k2] = -1
        W = [0]*22
        W[2*k2] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)

        # M4 = A11 * (B[1,k1] - B[0,k1])
        # U: A11 -> 1. V: B[1,k1] -> 1, B[0,k1] -> -1.
        # Output C[1,k1] += M4, C[0,k1] += M4
        U = [0, 0, 0, 1]
        V = [0]*22
        V[p+k1] = 1
        V[k1] = -1
        W = [0]*22
        W[2*k1+1] = 1
        W[2*k1] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)

        # M5 = (A00 + A01) * B[1,k2]
        # U: A00, A01 -> 1. V: B[1,k2] -> 1.
        # Output C[0,k2] += M5, C[1,k2] += M5
        U = [1, 1, 0, 0]
        V = [0]*22
        V[p+k2] = 1
        W = [0]*22
        W[2*k2] = 1
        W[2*k2+1] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)

        # M6 = (A10 - A00) * (B[0,k1] + B[0,k2])
        # U: A10 -> 1, A00 -> -1. V: B[0,k1] -> 1, B[0,k2] -> 1.
        # Output C[1,k2] += M6
        U = [-1, 0, 1, 0]
        V = [0]*22
        V[k1] = 1
        V[k2] = 1
        W = [0]*22
        W[2*k2+1] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)

        # M7 = (A01 - A11) * (B[1,k1] + B[1,k2])
        # U: A01 -> 1, A11 -> -1. V: B[1,k1] -> 1, B[1,k2] -> 1.
        # Output C[1,k1] += M7, C[0,k1] += M7
        U = [0, 1, 0, -1]
        V = [0]*22
        V[p+k1] = 1
        V[p+k2] = 1
        W = [0]*22
        W[2*k1+1] = 1
        W[2*k1] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)

    def add_single_column(k):
        """
        Adds rank-5 decomposition for multiplying 2x2 A by 2x1 B_col.
        B = [[b0], [b1]].
        C[0,k] = a*b0 + b*b1
        C[1,k] = c*b0 + d*b1
        
        We need to compute:
        M1 = (a+b) b0  ? No.
        Let's use:
        M1 = (a+d) b0
        M2 = (c+d) b0
        M3 = (a+b) b1
        M4 = (c-a) b1  ? 
        Let's just use the standard 5-multiplication algorithm for 2x2 * 2x1.
        Actually, the tensor [2,2,2]x[2,1,2] has rank 5? 
        Schoolbook is 4. Wait, schoolbook is 4 multiplications: a*b0, b*b1, c*b0, d*b1.
        Rank 4 is minimal for 2x2x2x2? No, tensor T_{ij,kl} = delta_{il}.
        For 2x2 * 2x1, it's 4 terms. Can we do less?
        Rank of [2,2,2]x[2,1,2] is 4? 
        If rank is 4, we just use schoolbook.
        Why did I think rank 5? 
        Ah, Strassen for 2x2*2x2 is rank 7. Schoolbook is 8.
        For 2x2 * 2x1, schoolbook is 4. 
        Is there a rank 3 algorithm? No, lower bound is 4? 
        Actually, for rectangular matrices, rank can be lower than nm.
        For 2x2 * 2x1, rank is 4.
        So we use 4 columns for the single one.
        """
        # M1 = A00 * B0k
        U = [1, 0, 0, 0]
        V = [0]*22
        V[k] = 1
        W = [0]*22
        W[2*k] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M2 = A01 * B1k
        U = [0, 1, 0, 0]
        V = [0]*22
        V[p+k] = 1
        W = [0]*22
        W[2*k] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M3 = A10 * B0k
        U = [0, 0, 1, 0]
        V = [0]*22
        V[k] = 1
        W = [0]*22
        W[2*k+1] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M4 = A11 * B1k
        U = [0, 0, 0, 1]
        V = [0]*22
        V[p+k] = 1
        W = [0]*22
        W[2*k+1] = 1
        U_cols.append(U); V_cols.append(V); W_cols.append(W)

    # Apply pairs for (0,1), (2,3), (4,5), (6,7), (8,9)
    add_strassen_pair(0, 1)
    add_strassen_pair(2, 3)
    add_strassen_pair(4, 5)
    add_strassen_pair(6, 7)
    add_strassen_pair(8, 9)
    
    # Apply single for 10
    add_single_column(10)

    rank = len(U_cols)
    U = [U_cols[i][j] for i in range(rank) for j in range(4)] # Transpose? No, return shape (n*m, rank)
    # My U_cols are rows of U.
    U = U_cols
    
    # V_cols are rows of V.
    V = V_cols
    
    # W_cols are rows of W.
    W = W_cols

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
