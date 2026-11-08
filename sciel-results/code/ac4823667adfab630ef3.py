    def construct():
        n, m, p = 2, 3, 7
        rank = 39
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Helper to add a term
        def add_term(u_indices, v_indices, w_indices, coeff=1.0):
            nonlocal col
            for idx, val in u_indices:
                U[idx][col] += val * coeff
            for idx, val in v_indices:
                V[idx][col] += val * coeff
            for idx, val in w_indices:
                W[idx][col] += val * coeff
            col += 1
            
        # Helper for Strassen 2x2x2
        # A[i0:i0+2, j0:j0+2], B[j0:j0+2, k0:k0+2], C[i0:i0+2, k0:k0+2]
        def strassen_222(i0, j0, k0):
            # m1 = (a+d)(e+h)
            add_term(
                [(i0*m+j0, 1), ((i0+1)*m+(j0+1), 1)],
                [(j0*p+k0, 1), ((j0+1)*p+(k0+1), 1)],
                [(k0*n+i0, 1), ((k0+1)*n+(i0+1), 1)]
            )
            # m2 = (c+d)(e)
            add_term(
                [((i0+1)*m+j0, 1), ((i0+1)*m+(j0+1), 1)],
                [(j0*p+k0, 1)],
                [(k0*n+(i0+1), 1), ((k0+1)*n+(i0+1), 1)]
            )
            # m3 = (a)(f-h)
            add_term(
                [(i0*m+j0, 1)],
                [(j0*p+(k0+1), 1), ((j0+1)*p+(k0+1), -1)],
                [(k0*n+i0, 1), ((k0+1)*n+i0, 1)]
            )
            # m4 = (d)(g-e)
            add_term(
                [((i0+1)*m+(j0+1), 1)],
                [((j0+1)*p+k0, 1), (j0*p+k0, -1)],
                [(k0*n+i0, 1), ((k0+1)*n+(i0+1), 1)]
            )
            # m5 = (a+b)(h)
            add_term(
                [(i0*m+j0, 1), (i0*m+(j0+1), 1)],
                [((j0+1)*p+(k0+1), 1)],
                [(k0*n+i0, -1), ((k0+1)*n+(i0+1), 1)]
            )
            # m6 = (c-a)(e+f)
            add_term(
                [((i0+1)*m+j0, 1), (i0*m+j0, -1)],
                [(j0*p+k0, 1), (j0*p+(k0+1), 1)],
                [((k0+1)*n+(i0+1), 1)]
            )
            # m7 = (b-d)(g+h)
            add_term(
                [(i0*m+(j0+1), 1), ((i0+1)*m+(j0+1), -1)],
                [((j0+1)*p+k0, 1), ((j0+1)*p+(k0+1), 1)],
                [(k0*n+i0, 1)]
            )
            
        # Helper for Schoolbook 2x2x1
        # A[i0:i0+2, j0:j0+2], B[j0:j0+2, k0:k0+1], C[i0:i0+2, k0:k0+1]
        def schoolbook_221(i0, j0, k0):
            for i in range(2):
                for j in range(2):
                    add_term(
                        [((i0+i)*m + (j0+j), 1)],
                        [((j0+j)*p + k0, 1)],
                        [((k0)*n + (i0+i), 1)]
                    )
                    
        # Helper for Schoolbook 2x1x3
        # A[i0:i0+2, j0:j0+1], B[j0:j0+1, k0:k0+3], C[i0:i0+2, k0:k0+3]
        def schoolbook_213(i0, j0, k0):
            for i in range(2):
                for k in range(3):
                    add_term(
                        [((i0+i)*m + (j0), 1)],
                        [((j0)*p + (k0+k), 1)],
                        [((k0+k)*n + (i0+i), 1)]
                    )
                    
        # Helper for Schoolbook 2x1x2
        # A[i0:i0+2, j0:j0+1], B[j0:j0+1, k0:k0+2], C[i0:i0+2, k0:k0+2]
        def schoolbook_212(i0, j0, k0):
            for i in range(2):
                for k in range(2):
                    add_term(
                        [((i0+i)*m + (j0), 1)],
                        [((j0)*p + (k0+k), 1)],
                        [((k0+k)*n + (i0+i), 1)]
                    )

        # Part 1: M_2,3,3 (B cols 0,1,2)
        # 1a: M_2,2,3 (A cols 0,1; B rows 0,1)
        # 1a1: M_2,2,2 (A 0,1; B 0,1; C 0,1)
        strassen_222(0, 0, 0)
        # 1a2: M_2,2,1 (A 0,1; B 0,1; C 2)
        schoolbook_221(0, 0, 2)
        # 1b: M_2,1,3 (A 2; B 2; C 0,1,2)
        schoolbook_213(0, 2, 0)
        
        # Part 2: M_2,3,4 (B cols 3,4,5,6)
        # 2a: M_2,3,2 (B cols 3,4)
        # 2a1: M_2,2,2 (A 0,1; B 0,1; C 3,4)
        strassen_222(0, 0, 3)
        # 2a2: M_2,1,2 (A 2; B 2; C 3,4)
        schoolbook_212(0, 2, 3)
        
        # 2b: M_2,3,2 (B cols 5,6)
        # 2b1: M_2,2,2 (A 0,1; B 0,1; C 5,6)
        strassen_222(0, 0, 5)
        # 2b2: M_2,1,2 (A 2; B 2; C 5,6)
        schoolbook_212(0, 2, 5)
        
        return {"U": U, "V": V, "W": W}
    