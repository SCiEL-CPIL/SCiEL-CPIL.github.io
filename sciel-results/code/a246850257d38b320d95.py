"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _kronecker_decomp(n, m, p):
    """
    Try to decompose into Kronecker products of smaller matrix multiplications.
    We look for factors (n1,m1,p1) and (n2,m2,p2) such that:
      n = n1*n2, m = m1*m2, p = p1*p2
    Then rank = rank1 * rank2 where rank1 = n1*m1*p1 (schoolbook for first),
    rank2 = n2*m2*p2 (schoolbook for second).
    """
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    best_UVW = None
    
    # Try all factorizations n = n1*n2, m = m1*m2, p = p1*p2
    for n1 in range(1, n + 1):
        if n % n1 != 0:
            continue
        n2 = n // n1
        for m1 in range(1, m + 1):
            if m % m1 != 0:
                continue
            m2 = m // m1
            for p1 in range(1, p + 1):
                if p % p1 != 0:
                    continue
                p2 = p // p1
                
                # Schoolbook ranks for the two sub-problems
                rank1 = n1 * m1 * p1
                rank2 = n2 * m2 * p2
                total_rank = rank1 * rank2
                
                if total_rank >= best_rank:
                    continue
                
                # Build Kronecker product decomposition
                # U[i*m + j, c] where i = i1*n2 + i2, j = j1*m2 + j2
                # c = c1 * rank2 + c2
                U = [[0] * total_rank for _ in range(n * m)]
                V = [[0] * total_rank for _ in range(m * p)]
                W = [[0] * total_rank for _ in range(p * n)]
                
                for i1 in range(n1):
                    for j1 in range(m1):
                        for k1 in range(p1):
                            for i2 in range(n2):
                                for j2 in range(m2):
                                    for k2 in range(p2):
                                        # Indices in full problem
                                        i = i1 * n2 + i2
                                        j = j1 * m2 + j2
                                        k = k1 * p2 + k2
                                        
                                        # Column index in Kronecker product
                                        c1 = i1 * m1 * p1 + j1 * p1 + k1
                                        c2 = i2 * m2 * p2 + j2 * p2 + k2
                                        c = c1 * rank2 + c2
                                        
                                        U[i * m + j][c] = 1
                                        V[j * p + k][c] = 1
                                        W[k * n + i][c] = 1
                                        
                best_rank = total_rank
                best_UVW = (U, V, W)
    
    if best_UVW is not None:
        return best_UVW[0], best_UVW[1], best_UVW[2], best_rank
    else:
        return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    U, V, W, rank = _kronecker_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
