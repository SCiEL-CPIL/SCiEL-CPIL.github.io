"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 11
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for base in range(0, 10, 2):
        # Strassen for 2x2 block of columns base, base+1
        # P1 = (a+d)(e+h)
        U[0][col] = 1; U[3][col] = 1
        V[base][col] = 1; V[base+2][col] = 1
        W[base*2][col] = 1; W[base*2+1][col] = 1
        col += 1
        
        # P2 = (c+d)(e)
        U[2][col] = 1; U[3][col] = 1
        V[base][col] = 1
        W[base*2][col] = 1
        col += 1
        
        # P3 = a(f-h)
        U[0][col] = 1
        V[base+1][col] = 1; V[base+3][col] = -1
        W[base*2+1][col] = 1
        col += 1
        
        # P4 = d(g-e)
        U[3][col] = 1
        V[base+2][col] = 1; V[base][col] = -1
        W[base*2][col] = 1
        col += 1
        
        # P5 = (a+b)(h)
        U[0][col] = 1; U[1][col] = 1
        V[base+3][col] = 1
        W[base*2+1][col] = 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[2][col] = 1; U[0][col] = -1
        V[base][col] = 1; V[base+2][col] = 1
        W[base*2+1][col] = 1
        col += 1
        
        # P7 = (b-d)(g+h)
        U[1][col] = 1; U[3][col] = -1
        V[base+2][col] = 1; V[base+3][col] = 1
        W[base*2][col] = 1
        col += 1
    
    # Last column (base=10), single column, rank 1
    # C_{i,0} = sum_j A_{i,j} B_{j,0}
    # This is just 4 terms: A00*B00 + A01*B10 + A10*B00 + A11*B10
    # But we can factor: (A00+A01+B00+B10)? No, that's not right.
    # Actually for 2x2 x 2x1, the tensor is T_{i,j,k=0} = 1.
    # Rank of 2x2x1 tensor with all ones is 1.
    # T_{i,j,0} = 1 = 1 * 1 * 1
    # So U[i,j,0]=1, V[j,0,0]=1, W[0,i,0]=1
    # Wait, W[k*n+i, 0] where k=0.
    # W[0*n + i, 0] = W[i, 0] = 1
    # V[j*p + 0, 0] = V[j, 0] = 1
    # U[i*m + j, 0] = U[i*2+j, 0] = 1
    
    # But wait, does this work?
    # C_{i,k} = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # For k=0: C_{i,0} = sum_j U[i,j,0] V[j,0,0] W[0,i,0]
    # = sum_j 1 * 1 * 1 = 2
    # But we want C_{i,0} = sum_j A_{i,j} B_{j,0}
    # The decomposition is C = U * A * V * B * W? No.
    # The convention is:
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # C_{i,k} = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This computes the bilinear form.
    # If we set U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1, then C_{i,k} = sum_j 1 = m = 2.
    # This is not correct.
    
    # Let's re-read the task: "exactly decompose the matrix multiplication tensor"
    # The tensor is T_{i,j,k} such that (AB)_{ik} = sum_j T_{i,j,k} A_{ij} B_{jk}
    # So T_{i,j,k} = 1.
    # The rank decomposition is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # Wait, the indices of U, V, W in the task are:
    # U: (n*m, rank) -> U[i*m+j, r]
    # V: (m*p, rank) -> V[j*p+k, r]
    # W: (p*n, rank) -> W[k*n+i, r]
    # And the equation is: (AB)_{ik} = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] ?
    # No, that's not how it works.
    # The standard decomposition is:
    # (AB)_{ik} = sum_r sum_j U_{i,j,r} V_{j,k,r} W_{k,i,r} ? No.
    
    # Let's look at the seed program.
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This implies the term is A_{i,j} * B_{j,k} * 1.
    # So the product is sum_j sum_r A_{i,j} B_{j,k} * (U[i*m+j,r] V[j*p+k,r] W[k*n+i,r])
    # For this to equal AB, we need U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 if the term is A_{i,j}B_{j,k}.
    # In the seed, for each (i,j,k), there is a rank-1 term with U=1, V=1, W=1.
    # So the "tensor" being decomposed is T_{i,j,k} = 1, where the product is
    # sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}.
    # And the decomposition is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    
    # So for the last column (k=10), we need T_{i,j,10} = 1 for all i,j.
    # T_{i,j,10} = sum_r U_{i,j,r} V_{j,10,r} W_{10,i,r}
    # Can we do this with rank 1?
    # Set U_{i,j,r}=1, V_{j,10,r}=1, W_{10,i,r}=1.
    # Then T_{i,j,10} = 1 * 1 * 1 = 1.
    # Yes!
    # So for the last column, we add one rank term.
    
    U[0][col] = 1; U[1][col] = 1; U[2][col] = 1; U[3][col] = 1
    V[10][col] = 1; V[11][col] = 1  # j*p+k = j*2+10 = 2j+10. j=0->10, j=1->11
    W[10*2+0][col] = 1  # k*n+i = 10*2+0 = 20
    W[10*2+1][col] = 1  # k*n+i = 10*2+1 = 21
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

