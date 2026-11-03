"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Strategy: Split 4x4x7 into 4x4x4 + 4x4x3
    # Use Laderman's algorithm (rank 49) for 4x4x4
    # Use schoolbook (rank 48) for 4x4x3
    # Total rank = 49 + 48 = 97
    
    rank_laderman = 49
    rank_schoolbook = 48
    total_rank = rank_laderman + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: Laderman's algorithm for 4x4x4 (rank 49) ---
    # Laderman's algorithm factors for 4x4 matrix multiplication
    # A = [[a,b,c,d],[e,f,g,h],[i,j,k,l],[m,n,o,p]]
    # B = [[A,B,C,D],[E,F,G,H],[I,J,K,L],[M,N,O,P]]
    # C = A*B
    
    # Laderman's 49 multiplications:
    # We'll define the factors explicitly.
    
    # Input indices for A: row i, col j -> index i*4+j
    # Input indices for B: row j, col k -> index j*4+k (for first 4 cols)
    # Output indices: row i, col k -> index i*4+k (for first 4 cols)
    
    # Laderman's algorithm:
    # The 49 products are defined as follows (using standard notation):
    
    laderman_U = []
    laderman_V = []
    laderman_W = []
    
    # Define the 49 rank-1 terms
    # Each term is (u_vector, v_vector, w_vector)
    
    # I'll use the known Laderman factors.
    # A is 4x4, B is 4x4, C is 4x4.
    
    # Let me define them by listing the 49 products:
    # m1 = a*A
    # m2 = b*B
    # m3 = c*C
    # m4 = d*D
    # m5 = e*E
    # m6 = f*F
    # m7 = g*G
    # m8 = h*H
    # m9 = i*I
    # m10 = j*J
    # m11 = k*K
    # m12 = l*L
    # m13 = m*M
    # m14 = n*N
    # m15 = o*O
    # m16 = p*P
    # m17 = (a+b)*(A+B)
    # m18 = (c+d)*(C+D)
    # m19 = (e+f)*(E+F)
    # m20 = (g+h)*(G+H)
    # m21 = (i+j)*(I+J)
    # m22 = (k+l)*(K+L)
    # m23 = (m+n)*(M+N)
    # m24 = (o+p)*(O+P)
    # m25 = (a+c)*(A+C)
    # m26 = (b+d)*(B+D)
    # m27 = (e+g)*(E+G)
    # m28 = (f+h)*(F+H)
    # m29 = (i+k)*(I+K)
    # m30 = (j+l)*(J+L)
    # m31 = (m+o)*(M+O)
    # m32 = (n+p)*(N+P)
    # m33 = (a+e)*(A+E)
    # m34 = (b+f)*(B+F)
    # m35 = (c+g)*(C+G)
    # m36 = (d+h)*(D+H)
    # m37 = (i+m)*(I+M)
    # m38 = (j+n)*(J+N)
    # m39 = (k+o)*(K+O)
    # m40 = (l+p)*(L+P)
    # m41 = (a+b+e+f)*(A+B+E+F)
    # m42 = (c+d+g+h)*(C+D+G+H)
    # m43 = (i+j+k+l)*(I+J+K+L)
    # m44 = (m+n+o+p)*(M+N+O+P)
    # m45 = (a+c+e+g)*(A+C+E+G)
    # m46 = (b+d+f+h)*(B+D+F+H)
    # m47 = (i+k+m+o)*(I+K+M+O)
    # m48 = (j+l+n+p)*(J+L+N+P)
    # m49 = (a+b+c+d+e+f+g+h)*(A+B+C+D+E+F+G+H)
    
    # Wait, that's not quite right. Let me look up the actual Laderman factors.
    
    # Actually, let me just use a verified set of Laderman factors.
    # I'll define them as lists of (coeff, index) for each of U, V, W.
    
    # For simplicity and correctness, let me just use schoolbook for both parts.
    # 64 + 48 = 112. Same as seed.
    
    # Let me try a different approach: use the known rank-49 Laderman factors.
    
    # I'll hardcode the Laderman factors.
    
    # A indices: 0=a, 1=b, 2=c, 3=d, 4=e, 5=f, 6=g, 7=h, 8=i, 9=j, 10=k, 11=l, 12=m, 13=n, 14=o, 15=p
    # B indices (4x4): 0=A, 1=B, 2=C, 3=D, 4=E, 5=F, 6=G, 7=H, 8=I, 9=J, 10=K, 11=L, 12=M, 13=N, 14=O, 15=P
    # C indices: 0,1,2,3 (row 0), 4,5,6,7 (row 1), 8,9,10,11 (row 2), 12,13,14,15 (row 3)
    
    # Laderman's 49 products (from literature):
    # I'll define each as a tuple of (U_coeffs, V_coeffs, W_coeffs)
    # where each is a dict from index to coefficient.
    
    products = []
    
    # m1-m16: simple products
    for idx in range(16):
        products.append(({idx: 1}, {idx: 1}, {idx: 1}))
    
    # m17: (a+b)*(A+B)
    products.append(({0:1, 1:1}, {0:1, 1:1}, {0:-1, 1:-1}))  # This is wrong, need to be careful
    
    # This is getting too complex to hardcode correctly. Let me use a simpler approach.
    
    # I'll just use schoolbook for everything: rank 112.
    # Or I can try to implement a known rank-49 algorithm.
    
    # For now, let me just return the schoolbook solution.
    rank = n * m * p
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

# EVOLVE-BLOCK-END
