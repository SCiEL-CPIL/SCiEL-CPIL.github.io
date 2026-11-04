def add_2x2_strassen(u_idx, v_idx, w_idx, r_offset):
    # u_idx: [A11, A12, A21, A22]
    # v_idx: [B11, B12, B21, B22]
    # w_idx: [C11, C12, C21, C22]
    
    # 1. (A11+A22)(B11+B22) -> C11, C22
    U[u_idx[0]][r_offset] += 1
    U[u_idx[3]][r_offset] += 1
    V[v_idx[0]][r_offset] += 1
    V[v_idx[3]][r_offset] += 1
    W[w_idx[0]][r_offset] += 1
    W[w_idx[3]][r_offset] += 1
    
    # 2. A22(B12-B22) -> C21
    U[u_idx[3]][r_offset+1] += 1
    V[v_idx[1]][r_offset+1] += 1
    V[v_idx[3]][r_offset+1] -= 1
    W[w_idx[2]][r_offset+1] += 1
    
    # 3. (A11+A12)B22 -> C12
    U[u_idx[0]][r_offset+2] += 1
    U[u_idx[1]][r_offset+2] += 1
    V[v_idx[3]][r_offset+2] += 1
    W[w_idx[1]][r_offset+2] += 1
    
    # 4. A11(B21-B11) -> C21
    U[u_idx[0]][r_offset+3] += 1
    V[v_idx[2]][r_offset+3] += 1
    V[v_idx[0]][r_offset+3] -= 1
    W[w_idx[2]][r_offset+3] += 1
    
    # 5. (A21+A22)B11 -> C11
    U[u_idx[2]][r_offset+4] += 1
    U[u_idx[3]][r_offset+4] += 1
    V[v_idx[0]][r_offset+4] += 1
    W[w_idx[0]][r_offset+4] += 1
    
    # 6. (A12-A22)(B11+B21) -> C11
    U[u_idx[1]][r_offset+5] += 1
    U[u_idx[3]][r_offset+5] -= 1
    V[v_idx[0]][r_offset+5] += 1
    V[v_idx[2]][r_offset+5] += 1
    W[w_idx[0]][r_offset+5] += 1
    
    # 7. (A11-A21)(B12+B22) -> C11
    U[u_idx[0]][r_offset+6] += 1
    U[u_idx[2]][r_offset+6] -= 1
    V[v_idx[1]][r_offset+6] += 1
    V[v_idx[3]][r_offset+6] += 1
    W[w_idx[0]][r_offset+6] += 1
