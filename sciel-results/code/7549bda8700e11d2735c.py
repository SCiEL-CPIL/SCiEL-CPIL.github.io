for u in range(2):
    for v in range(2):
        # ...
        for r in range(4):
            if alpha[s].get(r, 0) != 0:
                ar, br = r // 2, r % 2
                u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                U[u_idx, col] += alpha[s][r]
        
        for c in range(4):
            if beta[s].get(c, 0) != 0:
                ac, bc = c // 2, c % 2
                v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                V[v_idx, col] += beta[s][c]
        
        for a, b, gamma in recon[s]:
            w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
            W[w_idx, col] += gamma
        
        col += 1
