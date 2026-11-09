            for (i, r, sa) in AF[m]:
                U[i * M + r][c] += sa
            for (r, k, sb) in BF[m]:
                V[r * P + k][c] += sb
            for (i, k, sc) in CF[m]:
                W[k * N + i][c] += sc
