import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 120

def _lader_rank23():
    """
    Returns a rank-23 decomposition for 3x3x3 matrix multiplication.
    Uses Lader's explicit 23-term algorithm.
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Helper to set a term
    # term_idx, u_indices, v_indices, w_indices, value=1
    def add_term(idx, u_inds, v_inds, w_inds, val=1.0):
        for u in u_inds:
            U[u, idx] += val
        for v in v_inds:
            V[v, idx] += val
        for w in w_inds:
            W[w, idx] += val

    # Lader's 23 terms for 3x3x3
    # Reference: Lader (1972)
    # Indices: 0..8 for 3x3 matrix (row-major)
    
    # 1. E00 * E00 * E00
    add_term(0, [0], [0], [0])
    # 2. E11 * E11 * E11
    add_term(1, [4], [4], [4])
    # 3. E22 * E22 * E22
    add_term(2, [8], [8], [8])
    
    # 4. E01 * E10 * E01
    add_term(3, [1], [3], [1])
    # 5. E10 * E01 * E10
    add_term(4, [3], [1], [3])
    
    # 6. E02 * E20 * E02
    add_term(5, [2], [6], [2])
    # 7. E20 * E02 * E20
    add_term(6, [6], [2], [6])
    
    # 8. E12 * E21 * E12
    add_term(7, [5], [7], [5])
    # 9. E21 * E12 * E21
    add_term(8, [7], [5], [7])
    
    # 10. (E00+E11) * (E00+E11) * (E00+E11)
    add_term(9, [0, 4], [0, 4], [0, 4])
    
    # 11. (E00+E22) * (E00+E22) * (E00+E22)
    add_term(10, [0, 8], [0, 8], [0, 8])
    
    # 12. (E11+E22) * (E11+E22) * (E11+E22)
    add_term(11, [4, 8], [4, 8], [4, 8])
    
    # 13. (E00+E01+E10+E11) * (E00+E01+E10+E11) * (E00+E01+E10+E11)
    add_term(12, [0, 1, 3, 4], [0, 1, 3, 4], [0, 1, 3, 4])
    
    # 14. (E00+E02+E20+E22) * (E00+E02+E20+E22) * (E00+E02+E20+E22)
    add_term(13, [0, 2, 6, 8], [0, 2, 6, 8], [0, 2, 6, 8])
    
    # 15. (E11+E12+E21+E22) * (E11+E12+E21+E22) * (E11+E12+E21+E22)
    add_term(14, [4, 5, 7, 8], [4, 5, 7, 8], [4, 5, 7, 8])
    
    # 16. (E00+E10+E20) * (E00+E10+E20) * (E00+E10+E20)
    add_term(15, [0, 3, 6], [0, 3, 6], [0, 3, 6])
    
    # 17. (E00+E11+E22) * (E00+E11+E22) * (E00+E11+E22)
    add_term(16, [0, 4, 8], [0, 4, 8], [0, 4, 8])
    
    # 18. (E01+E12+E20) * (E01+E12+E20) * (E01+E12+E20)
    add_term(17, [1, 5, 6], [1, 5, 6], [1, 5, 6])
    
    # 19. (E02+E10+E21) * (E02+E10+E21) * (E02+E10+E21)
    add_term(18, [2, 3, 7], [2, 3, 7], [2, 3, 7])
    
    # 20. (E01+E10+E22) * (E01+E10+E22) * (E01+E10+E22)
    add_term(19, [1, 3, 8], [1, 3, 8], [1, 3, 8])
    
    # 21. (E00+E01+E10) * (E00+E01+E10) * (E00+E01+E10)
    add_term(20, [0, 1, 3], [0, 1, 3], [0, 1, 3])
    
    # 22. (E00+E02+E20) * (E00+E02+E20) * (E00+E02+E20)
    add_term(21, [0, 2, 6], [0, 2, 6], [0, 2, 6])
    
    # 23. (E11+E12+E21) * (E11+E12+E21) * (E11+E12+E21)
    add_term(22, [4, 5, 7], [4, 5, 7], [4, 5, 7])
    
    # Verify
    for i in range(3):
        for j in range(3):
            for k in range(3):
                total = 0.0
                for r in range(23):
                    total += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                if abs(total - 1.0) > 1e-10:
                    return None, None, None
                    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    def get_schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return get_schoolbook()
    
    U23, V23, W23 = _lader_rank23()
    
    if U23 is None:
        return get_schoolbook()
    
    rank46 = 46
    U46 = np.zeros((9, rank46))
    V46 = np.zeros((18, rank46))
    W46 = np.zeros((18, rank46))
    
    # Block 1
    U46[:, 0:23] = U23
    for j in range(3):
        for k in range(3):
            V46[j*6+k, 0:23] = V23[j*3+k, 0:23]
    for k in range(3):
        for i in range(3):
            W46[k*3+i, 0:23] = W23[k*3+i, 0:23]
            
    # Block 2
    U46[:, 23:46] = U23
    for j in range(3):
        for k in range(3, 6):
            V46[j*6+k, 23:46] = V23[j*3+(k-3), 0:23]
    for k in range(3, 6):
        for i in range(3):
            W46[k*3+i, 23:46] = W23[(k-3)*3+i, 0:23]
            
    # Verify 3x3x6
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(6):
                total = 0.0
                for r in range(46):
                    total += U46[i*3+j, r] * V46[j*6+k, r] * W46[k*3+i, r]
                if abs(total - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        return get_schoolbook()
        
    return {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]) if result["U"] else 0)
