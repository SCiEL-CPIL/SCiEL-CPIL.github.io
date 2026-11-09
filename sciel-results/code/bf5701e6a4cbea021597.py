                # dF/dU[ui, r] = -V[vj, r] * W[wk, r]
                J[t, ui * R + r] = -vwr * wwr
                
                # dF/dV[vj, r] = -U[i*m+j, r] * W[k*n+i, r]
                J[t, nm * R + vj * R + r] = -uwr * wwr
                
                # dF/dW[wk, r] = -U[i*m+j, r] * V[j*p+k, r]
                J[t, nm * R + mp * R + wk * R + r] = -uwr * vwr
