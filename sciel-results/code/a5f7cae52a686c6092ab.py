"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return 7 rank-1 factor tuples for M_{2,2,2} Strassen decomposition.
    Each tuple is (U_vec, V_vec, W_vec) with 4 entries each.
    U indexed by (i,j) -> i*2+j, V by (j,k) -> j*2+k, W by (k,i) -> k*2+i.
    """
    def make(U_d, V_d, W_d):
        U = [0.0]*4; V = [0.0]*4; W = [0.0]*4
        for (i,j),c in U_d.items(): U[i*2+j] = c
        for (j,k),c in V_d.items(): V[j*2+k] = c
        for (k,i),c in W_d.items(): W[k*2+i] = c
        return (U, V, W)
    
    t1 = make({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1})
    t2 = make({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1})
    t3 = make({(0,0):1}, {(1,0):1,(1,1):-1}, {(0,1):1,(1,1):1})
    t4 = make({(1,1):1}, {(0,1):1,(0,0):-1}, {(0,0):1,(1,0):1})
    t5 = make({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1})
    t6 = make({(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}, {(1,1):1})
    t7 = make({(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}, {(0,0):1})
    return [t1,t2,t3,t4,t5,t6,t7]

def _m122_terms():
    """4 rank-1 terms for M_{1,2,2}. U:(1,2), V:(2,2), W:(2,1)."""
    terms = []
    for j in range(2):
        for k in range(2):
            U = [0.0, 0.0]
            V = [0.0, 0.0, 0.0, 0.0]
            W = [0.0, 0.0]
            U[j] = 1.0
            V[j*2+k] = 1.0
            W[k] = 1.0
            terms.append((U, V, W))
    return terms

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    # Rank = 2 (j_blocks) * 4 (k_blocks) * (7+4) = 88
    total_rank = 88
    U = [[0.0]*total_rank for _ in range(n*m)]  # 12
    V = [[0.0]*total_rank for _ in range(m*p)]  # 32
    W = [[0.0]*total_rank for _ in range(p*n)]  # 24
    
    strassen = _strassen_2x2x2()
    m122 = _m122_terms()
    
    col = 0
    for jb in range(2):      # j block: j_global = jb*2 + jl
        for kb in range(4):  # k block: k_global = kb*2 + kl
            # Strassen part for i=0,1
            for U_s, V_s, W_s in strassen:
                for i in range(2):
                    for jl in range(2):
                        v = U_s[i*2+jl]
                        if v: U[i*m + jb*2+jl][col] = v
                for jl in range(2):
                    for kl in range(2):
                        v = V_s[jl*2+kl]
                        if v: V[(jb*2+jl)*p + kb*2+kl][col] = v
                for kl in range(2):
                    for i in range(2):
                        v = W_s[kl*2+i]
                        if v: W[(kb*2+kl)*n + i][col] = v
                col += 1
            # M_{1,2,2} part for i=2
            for U_s, V_s, W_s in m122:
                for jl in range(2):
                    v = U_s[jl]
                    if v: U[2*m + jb*2+jl][col] = v
                for jl in range(2):
                    for kl in range(2):
                        v = V_s[jl*2+kl]
                        if v: V[(jb*2+jl)*p + kb*2+kl][col] = v
                for kl in range(2):
                    v = W_s[kl]
                    if v: W[(kb*2+kl)*n + 2][col] = v
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
