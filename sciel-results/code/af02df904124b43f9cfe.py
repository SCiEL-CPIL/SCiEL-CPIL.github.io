import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    # Rank 36 decomposition: 3 blocks of rank 12 each
    rank = 36
    U = [[0] * rank for _ in range(n_rows)]
    V = [[0] * rank for _ in range(m_rows)]
    W = [[0] * rank for _ in range(p_rows)]
    
    col = 0
    
    # For each block of 3 columns (k = 3*block to 3*block+2)
    for block in range(3):
        # B indices for this block: B[j, 3*block+k] for j=0,1,2, k=0,1,2
        # V index: j*p + (3*block + k)
        # C indices: C[i, 3*block+k] -> W index: (3*block+k)*n + i
        
        # Rank 12 decomposition of 2x3x3:
        # We use 12 products to compute all 6 entries of C
        
        # Let A = [[a,b,c], [d,e,f]] where a=A[0,0], b=A[0,1], c=A[0,2], d=A[1,0], e=A[1,1], f=A[1,2]
        # Let B block = [[g,h,i], [j,k,l], [m,n,o]] where g=B[0,3b], h=B[1,3b], i=B[2,3b], 
        #                 j=B[0,3b+1], k=B[1,3b+1], l=B[2,3b+1], 
        #                 m=B[0,3b+2], n=B[1,3b+2], o=B[2,3b+2]
        
        # U indices: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
        # V indices in this block:
        #   B[0,3b]=0*p+3b, B[1,3b]=1*p+3b, B[2,3b]=2*p+3b
        #   B[0,3b+1]=0*p+3b+1, B[1,3b+1]=1*p+3b+1, B[2,3b+1]=2*p+3b+1
        #   B[0,3b+2]=0*p+3b+2, B[1,3b+2]=1*p+3b+2, B[2,3b+2]=2*p+3b+2
        
        # W indices in this block:
        #   C[0,3b]=(3b)*2+0=6b, C[1,3b]=(3b)*2+1=6b+1
        #   C[0,3b+1]=(3b+1)*2+0=6b+2, C[1,3b+1]=(3b+1)*2+1=6b+3
        #   C[0,3b+2]=(3b+2)*2+0=6b+4, C[1,3b+2]=(3b+2)*2+1=6b+5
        
        # 12 products for 2x3x3 (rank 12):
        # This is a known optimal decomposition
        
        # P1 = (a+d)*(g+j)  -> contributes to C[0,0], C[1,0]
        U[0][col] += 1
        U[3][col] += 1
        V[0*p + 3*block][col] += 1
        V[1*p + 3*block][col] += 1
        W[6*block][col] += 1    # C[0,3b]
        W[6*block+1][col] += 1  # C[1,3b]
        
        # P2 = (a-d)*(h-k)  -> contributes to C[0,0], C[1,0]
        U[0][col+1] += 1
        U[3][col+1] -= 1
        V[0*p + 3*block][col+1] += 1
        V[1*p + 3*block][col+1] -= 1
        W[6*block][col+1] += 1
        W[6*block+1][col+1] -= 1
        
        # P3 = (b+e)*(h+k)  -> contributes to C[0,0], C[1,0]
        U[1][col+2] += 1
        U[4][col+2] += 1
        V[0*p + 3*block][col+2] += 1
        V[1*p + 3*block][col+2] += 1
        W[6*block][col+2] += 1
        W[6*block+1][col+2] += 1
        
        # P4 = (b-e)*(g-j)  -> contributes to C[0,0], C[1,0]
        U[1][col+3] += 1
        U[4][col+3] -= 1
        V[0*p + 3*block][col+3] += 1
        V[1*p + 3*block][col+3] -= 1
        W[6*block][col+3] += 1
        W[6*block+1][col+3] -= 1
        
        # P5 = (c+f)*(i+l)  -> contributes to C[0,0], C[1,0]
        U[2][col+4] += 1
        U[5][col+4] += 1
        V[0*p + 3*block][col+4] += 1
        V[2*p + 3*block][col+4] += 1
        W[6*block][col+4] += 1
        W[6*block+1][col+4] += 1
        
        # P6 = (c-f)*(m-n)  -> contributes to C[0,0], C[1,0]
        U[2][col+5] += 1
        U[5][col+5] -= 1
        V[0*p + 3*block+2][col+5] += 1
        V[1*p + 3*block+2][col+5] -= 1
        W[6*block][col+5] += 1
        W[6*block+1][col+5] -= 1
        
        # P7 = (a+d)*(h+k)  -> contributes to C[0,1], C[1,1]
        U[0][col+6] += 1
        U[3][col+6] += 1
        V[0*p + 3*block][col+6] += 1
        V[1*p + 3*block][col+6] += 1
        W[6*block+2][col+6] += 1
        W[6*block+3][col+6] += 1
        
        # P8 = (a-d)*(g-j)  -> contributes to C[0,1], C[1,1]
        U[0][col+7] += 1
        U[3][col+7] -= 1
        V[0*p + 3*block][col+7] += 1
        V[1*p + 3*block][col+7] -= 1
        W[6*block+2][col+7] += 1
        W[6*block+3][col+7] -= 1
        
        # P9 = (b+e)*(g+j)  -> contributes to C[0,1], C[1,1]
        U[1][col+8] += 1
        U[4][col+8] += 1
        V[0*p + 3*block][col+8] += 1
        V[1*p + 3*block][col+8] += 1
        W[6*block+2][col+8] += 1
        W[6*block+3][col+8] += 1
        
        # P10 = (b-e)*(h-k)  -> contributes to C[0,1], C[1,1]
        U[1][col+9] += 1
        U[4][col+9] -= 1
        V[0*p + 3*block][col+9] += 1
        V[1*p + 3*block][col+9] -= 1
        W[6*block+2][col+9] += 1
        W[6*block+3][col+9] -= 1
        
        # P11 = (c+f)*(m+n)  -> contributes to C[0,1], C[1,1]
        U[2][col+10] += 1
        U[5][col+10] += 1
        V[0*p + 3*block+2][col+10] += 1
        V[1*p + 3*block+2][col+10] += 1
        W[6*block+2][col+10] += 1
        W[6*block+3][col+10] += 1
        
        # P12 = (c-f)*(i-l)  -> contributes to C[0,1], C[1,1]
        U[2][col+11] += 1
        U[5][col+11] -= 1
        V[0*p + 3*block][col+11] += 1
        V[2*p + 3*block][col+11] -= 1
        W[6*block+2][col+11] += 1
        W[6*block+3][col+11] -= 1
        
        col += 12
    
    # We need to handle C[:, 3b+2] for each block
    # The above only handled C[:, 3b] and C[:, 3b+1]
    # Let me redo this more carefully
    
    # Actually, let me use a simpler approach: direct rank 54 schoolbook but optimized
    # Or better, use the structure properly
    
    # Let me restart with a cleaner approach
    # 2x3x9: We can decompose as sum over j=0,1,2 of (2x1x9) * (1x1x9)
    # Each 2x1x9 has rank 18 (2*9), total 54 - not helpful
    
    # Better: Use the fact that rank(2x3x9) <= rank(2x3x3) * 3
    # But we need to be careful
    
    # Let me just return the current working solution with rank 50
    # and verify it's correct
    
    # Actually, let me try a different decomposition
    # Split 9 = 3+3+3, use rank 12 for each 2x3x3
    
    # The issue is that my rank 12 construction above is incorrect
    # Let me use a known correct rank 12 for 2x3x3
    
    # For 2x3x3, one valid rank 12 decomposition:
    # Uses the following 12 products to compute all 6 outputs
    
    # I'll use a verified construction
    
    rank = 36
    U = [[0] * rank for _ in range(n_rows)]
    V = [[0] * rank for _ in range(m_rows)]
    W = [[0] * rank for _ in range(p_rows)]
    
    col = 0
    
    for block in range(3):
        # For this block, B columns are 3*block, 3*block+1, 3*block+2
        # C columns are 3*block, 3*block+1, 3*block+2
        
        # 12 products for 2x3x3:
        # Let me use a standard construction
        
        # A = [[a,b,c],[d,e,f]], B = [[g,h,i],[j,k,l],[m,n,o]]
        # C[0,0] = ag+bj+cm, C[0,1] = ah+bk+cn, C[0,2] = ai+bl+co
        # C[1,0] = dg+ej+fm, C[1,1] = dh+ek+fn, C[1,2] = di+el+fo
        
        # Rank 12 construction:
        # P1 = (a+d)g -> C[0,0]+=a*g, C[1,0]+=d*g
        U[0][col] = 1; U[3][col] = 1
        V[0*p + 3*block][col] = 1
        W[6*block][col] = 1; W[6*block+1][col] = 1
        
        # P2 = (b+e)h -> C[0,0]+=b*h? No, that's wrong
        # Let me be more careful
        
        # I'll use a direct but suboptimal rank 18 per block (total 54)
        # to ensure correctness, then try to optimize
        
        # Actually, let me just return the original rank 50 solution which is known to work
        pass
    
    # Fallback: return the original working rank 50 solution
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; U[u_d][col] += 1
        V[v_e][col] += 1; V[v_h][col] += 1
        U[u_c][col+1] += 1; U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1; V[v_h][col+2] -= 1
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1; V[v_e][col+3] -= 1
        U[u_a][col+4] += 1; U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1
        V[v_e][col+5] += 1; V[v_f][col+5] += 1
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1
        V[v_g][col+6] += 1; V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
        W[w_01][col+2] += 1; W[w_01][col+4] += 1
        W[w_10][col+1] += 1; W[w_10][col+3] += 1
        W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    U[u_a][col] += 1; V[v_e][col] += 1
    U[u_b][col+1] += 1; V[v_g][col+1] += 1
    U[u_c][col+2] += 1; V[v_e][col+2] += 1
    U[u_d][col+3] += 1; V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1; W[w_08][col+1] += 1
    W[w_18][col+2] += 1; W[w_18][col+3] += 1
    
    col += 4
    
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2
    
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2
    
    return {"U": U, "V": V, "W": W}
