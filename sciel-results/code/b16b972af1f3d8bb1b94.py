DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = 3, 4, 4
    result_terms = []
    
    def emit_term(u_entries, v_entries, w_entries):
        if all(c == 0 for c in u_entries.values()): return
        if all(c == 0 for c in v_entries.values()): return
        if all(c == 0 for c in w_entries.values()): return
        result_terms.append((dict(u_entries), dict(v_entries), dict(w_entries)))
    
    # Row 0 of C: schoolbook, 16 terms
    for c in range(4):
        for j in range(4):
            emit_term({j: 1}, {j*4+c: 1}, {c: 1})
    
    def strassen_2x2(u_pos, v_pos, w_pos):
        u00,u01 = u_pos[0]
        u10,u11 = u_pos[1]
        v00,v01 = v_pos[0]
        v10,v11 = v_pos[1]
        w00,w01 = w_pos[0]
        w10,w11 = w_pos[1]
        
        emit_term({u00:1, u11:1}, {v00:1, v11:1}, {w00:1, w11:1})
        emit_term({u10:1, u11:1}, {v00:1}, {w10:1, w11:-1})
        emit_term({u00:1}, {v01:1, v11:-1}, {w01:1, w11:1})
        emit_term({u11:1}, {v10:1, v00:-1}, {w00:1, w10:1})
        emit_term({u00:1, u01:1}, {v11:1}, {w00:-1, w01:1})
        emit_term({u10:1, u00:-1}, {v00:1, v01:1}, {w11:1})
        emit_term({u01:1, u11:-1}, {v10:1, v11:1}, {w00:1})
    
    # C[1:3, 0:2]
    strassen_2x2(
        [[4,5],[8,9]],
        [[0,1],[4,5]],
        [[1,4],[2,5]]
    )
    strassen_2x2(
        [[6,7],[10,11]],
        [[8,9],[12,13]],
        [[1,4],[2,5]]
    )
    # C[1:3, 2:4]
    strassen_2x2(
        [[4,5],[8,9]],
        [[2,3],[6,7]],
        [[7,10],[8,11]]
    )
    strassen_2x2(
        [[6,7],[10,11]],
        [[10,11],[14,15]],
        [[7,10],[8,11]]
    )
    
    rank = len(result_terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for col, (ud, vd, wd) in enumerate(result_terms):
        for row, coef in ud.items():
            U[row][col] += coef
        for row, coef in vd.items():
            V[row][col] += coef
        for row, coef in wd.items():
            W[row][col] += coef
    
    return {"U": U, "V": V, "W": W}
