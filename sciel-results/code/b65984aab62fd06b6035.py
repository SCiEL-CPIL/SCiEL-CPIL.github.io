"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 1]
    block_starts = [0, 2, 4]
    
    # Strassen terms: (U00, U01, U10, U11, V00, V01, V10, V11, W00, W01, W10, W11)
    strassen_terms = [
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),   # m1
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0),   # m2
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 0),  # m3
        (0, 0, 0, 1, -1, 0, 1, 0, 0, 0, 1, 0),  # m4
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),  # m5
        (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),  # m6
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 1),  # m7
    ]
    
    terms = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                sa, sb, sc = block_sizes[a], block_sizes[b], block_sizes[c]
                ia, ib, ic = block_starts[a], block_starts[b], block_starts[c]
                
                if sa == 2 and sb == 2 and sc == 2:
                    for term in strassen_terms:
                        U00, U01, U10, U11, V00, V01, V10, V11, W00, W01, W10, W11 = term
                        U_dict = {}
                        V_dict = {}
                        W_dict = {}
                        if U00 != 0: U_dict[(ia)*m + (ib)] = U00
                        if U01 != 0: U_dict[(ia)*m + (ib+1)] = U01
                        if U10 != 0: U_dict[(ia+1)*m + (ib)] = U10
                        if U11 != 0: U_dict[(ia+1)*m + (ib+1)] = U11
                        if V00 != 0: V_dict[(ib)*p + (ic)] = V00
                        if V01 != 0: V_dict[(ib)*p + (ic+1)] = V01
                        if V10 != 0: V_dict[(ib+1)*p + (ic)] = V10
                        if V11 != 0: V_dict[(ib+1)*p + (ic+1)] = V11
                        if W00 != 0: W_dict[(ic)*n + (ia)] = W00
                        if W01 != 0: W_dict[(ic)*n + (ia+1)] = W01
                        if W10 != 0: W_dict[(ic+1)*n + (ia)] = W10
                        if W11 != 0: W_dict[(ic+1)*n + (ia+1)] = W11
                        terms.append((U_dict, V_dict, W_dict))
                else:
                    U_dict = {}
                    V_dict = {}
                    W_dict = {}
                    for ii in range(sa):
                        for jj in range(sb):
                            U_dict[(ia+ii)*m + (ib+jj)] = 1
                    for jj in range(sb):
                        for kk in range(sc):
                            V_dict[(ib+jj)*p + (ic+kk)] = 1
                    for kk in range(sc):
                        for ii in range(sa):
                            W_dict[(ic+kk)*n + (ia+ii)] = 1
                    terms.append((U_dict, V_dict, W_dict))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_dict, V_dict, W_dict) in enumerate(terms):
        for idx, val in U_dict.items():
            U[idx][col] = val
        for idx, val in V_dict.items():
            V[idx][col] = val
        for idx, val in W_dict.items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
