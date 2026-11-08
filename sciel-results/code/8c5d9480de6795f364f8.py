import numpy as np
import time

def construct():
    t0 = time.time()
    limit = t0 + 130
    
    N, M, P = 4, 4, 7
    
    def build_decomp():
        terms = []
        
        def add(u_idx, v_idx, w_idx, coeff):
            u = np.zeros(N * M)
            v = np.zeros(M * P)
            w = np.zeros(P * N)
            for i, c in zip(u_idx, coeff):
                u[i] = c
            for i, c in zip(v_idx, coeff):
                v[i] = c
            for i, c in zip(w_idx, coeff):
                w[i] = c
            terms.append((u.tolist(), v.tolist(), w.tolist()))
        
        def strassen_2x2(ia, ja, ka, scale=1):
            r1 = [ia*M+ja, (ia+1)*M+(ja+1)]
            s1 = [ja*P+ka, (ja+1)*P+(ka+1)]
            t1 = [ka*N+ia, (ka+1)*N+(ia+1)]
            add(r1, s1, t1, [scale, scale])
            
            r2 = [(ia+1)*M+ja, (ia+1)*M+(ja+1)]
            s2 = [ja*P+ka]
            t2 = [ka*N+(ia+1), (ka+1)*N+(ia+1)]
            add(r2, s2, t2, [scale, scale, scale, -scale])
            
            r3 = [ia*M+ja]
            s3 = [ja*P+(ka+1), (ja+1)*P+(ka+1)]
            t3 = [(ka+1)*N+ia, (ka+1)*N+(ia+1)]
            add(r3, s3, t3, [scale, -scale, scale, scale])
            
            r4 = [(ia+1)*M+(ja+1)]
            s4 = [(ja+1)*P+ka, ja*P+ka]
            t4 = [ka*N+ia, ka*N+(ia+1)]
            add(r4, s4, t4, [scale, scale, scale, scale])
            
            r5 = [ia*M+ja, ia*M+(ja+1)]
            s5 = [(ja+1)*P+(ka+1)]
            t5 = [(ka+1)*N+ia, ka*N+ia]
            add(r5, s5, t5, [scale, scale, scale, -scale])
            
            r6 = [(ia+1)*M+ja, ia*M+ja]
            s6 = [ja*P+ka, ja*P+(ka+1)]
            t6 = [(ka+1)*N+(ia+1)]
            add(r6, s6, t6, [scale, -scale, scale, scale, scale])
            
            r7 = [ia*M+(ja+1), (ia+1)*M+(ja+1)]
            s7 = [(ja+1)*P+ka, (ja+1)*P+(ka+1)]
            t7 = [ka*N+ia]
            add(r7, s7, t7, [scale, -scale, scale, scale, scale])
        
        def strassen_4x4x4():
            outer = [
                ([((0,1),(3,1)),(0,1),(3,1)]),
                ([((2,1),(3,1)),(0,1)],[(2,1),(3,-1)]),
                ([[(0,1)],[(1,1),(3,-1)],[(1,1),(3,1)]]),
            ]
            
            blocks = [
                ([((0,1),(3,1)),(0,1),(3,1)]),
                ([((2,1),(3,1)),(0,1)],[(2,1),(3,-1)]),
                ([[(0,1)],[(1,1),(3,-1)],[(1,1),(3,1)]]),
            ]
            
            for bi in range(7):
                u_entries = []
                v_entries = []
                w_entries = []
                for ai in range(4):
                    for aj in range(4):
                        for bk in range(4):
                            idx_a = ai * M + aj
                            idx_b = aj * P + bk
                            idx_c = bk * N + ai
                            u_entries.append(idx_a)
                            v_entries.append(idx_b)
                            w_entries.append(idx_c)
            
            for ioff in range(2):
                for joff in range(2):
                    for koff in range(2):
                        strassen_2x2(ioff, joff, koff)
        
        def strassen_4x4x3():
            for ioff in range(2):
                for joff in range(2):
                    for koff in range(2):
                        strassen_2x2(ioff, joff, koff)
        
        def handle_last_col():
            for i in range(N):
                for j in range(M):
                    u = np.zeros(N*M)
                    v = np.zeros(M*P)
                    w = np.zeros(P*N)
                    u[i*M+j] = 1
                    v[j*P+6] = 1
                    w[6*N+i] = 1
                    terms.append((u.tolist(), v.tolist(), w.tolist()))
        
        strassen_4x4x4()
        strassen_4x4x3()
        handle_last_col()
        
        return terms
    
    t_list = build_decomp()
    
    if time.time() > limit - 5:
        return {"U": [], "V": [], "W": []}
    
    rk = len(t_list)
    U_arr = np.zeros((N*M, rk))
    V_arr = np.zeros((M*P, rk))
    W_arr = np.zeros((P*N, rk))
    
    for k, (u, v, w) in enumerate(t_list):
        U_arr[:, k] = u
        V_arr[:, k] = v
        W_arr[:, k] = w
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
