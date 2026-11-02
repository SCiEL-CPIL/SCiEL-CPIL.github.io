DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(idx, u, v, w):
        for i, val in u.items():
            U[i][idx] = val
        for j, val in v.items():
            V[j][idx] = val
        for k, val in w.items():
            W[k][idx] = val
            
    # Term 0: P1
    add_term(0, {0:1, 3:1}, {0:1, 5:1}, {0:1, 3:1})
    # Term 1: P2
    add_term(1, {2:1, 3:1}, {0:1, 5:1}, {1:1, 3:-1})
    # Term 2: P3
    add_term(2, {0:1}, {1:1, 5:-1}, {2:1, 3:1})
    # Term 3: P4
    add_term(3, {3:1}, {4:1, 0:-1}, {0:1, 1:1})
    # Term 4: P5
    add_term(4, {0:1, 1:1}, {5:1}, {0:-1, 2:1})
    # Term 5: P6
    add_term(5, {2:1, 3:1}, {1:1}, {3:1})
    # Term 6: P7
    add_term(6, {0:1, 2:-1}, {0:1, 4:1}, {0:1})
    
    # Term 7: Q1
    add_term(7, {0:1, 3:1}, {2:1, 7:1}, {4:1, 7:1})
    # Term 8: Q2
    add_term(8, {2:1, 3:1}, {2:1, 7:1}, {5:1, 7:-1})
    # Term 9: Q3
    add_term(9, {0:1}, {3:1, 7:-1}, {6:1, 7:1})
    # Term 10: Q4
    add_term(10, {3:1}, {6:1, 2:-1}, {4:1, 5:1})
    # Term 11: Q5
    add_term(11, {0:1, 1:1}, {7:1}, {4:-1, 6:1})
    # Term 12: Q6
    add_term(12, {2:1, 3:1}, {3:1}, {7:1})
    # Term 13: Q7
    add_term(13, {0:1, 2:-1}, {2:1, 6:1}, {4:1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
