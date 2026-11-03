    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for c in range(r3):
                    s += U33[i*3+j][c] * V33[j*3+k][c] * W33[k*3+i][c]
                if s != 1:
                    # This decomposition is wrong. Let me fall back to schoolbook.
                    pass
