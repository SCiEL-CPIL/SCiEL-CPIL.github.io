# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T has components T_{(ij), (jk), (ki)} = 1
    # for all i, j, k.
    # This tensor is rank 1: T = u \otimes v \otimes w where u, v, w are all-ones vectors.
    # U_{ij, 0} = 1, V_{jk, 0} = 1, W_{ki, 0} = 1.
    # Sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 * 1 * 1 = 1.
    
    R = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
