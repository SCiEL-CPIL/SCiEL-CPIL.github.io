for i in range(4):
    for c in range(7):
        U[u_rows[i]][col_offset + c] += U2[i][c]
        V[v_rows[i]][col_offset + c] += V2[i][c]
        W[w_rows[i]][col_offset + c] += W2[i][c]
