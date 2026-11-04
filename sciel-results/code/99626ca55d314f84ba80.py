"""Rewritten program for 6x6 matrix multiplication with rank 189 decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block matrices
    alphas = [
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    betas = [
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    gammas = [
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for t in range(7):
        for ip in range(3):
            for jp in range(3):
                for kp in range(3):
                    c = t * 27 + ip * 9 + jp * 3 + kp
                    
                    for (I, J), val in alphas[t].items():
                        U[(I * 3 + ip) * 6 + (J * 3 + jp)][c] = val
                        
                    for (Jp, K), val in betas[t].items():
                        V[(Jp * 3 + jp) * 6 + (K * 3 + kp)][c] = val
                        
                    for (I, K), val in gammas[t].items():
                        W[(K * 3 + kp) * 6 + (I * 3 + ip)][c] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
