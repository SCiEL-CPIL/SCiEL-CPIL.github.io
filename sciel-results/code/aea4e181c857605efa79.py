    def strassen_plus_trivial(n_b, m_b, p_b):
        if n_b == 2 and m_b == 2 and p_b == 2:
            return strassen_222()
        
        if n_b == 2 and m_b == 2 and p_b == 3:
            U1, V1, W1 = strassen_222()
            U2, V2, W2 = trivial_block(2, 2, 1)
            
            rank = 11
            U = np.zeros((4, rank), dtype=np.int8)
            V = np.zeros((6, rank), dtype=np.int8)
            W = np.zeros((6, rank), dtype=np.int8)
            
            U[:, :7] = U1
            U[:, 7:] = U2
            
            # V mapping: V1 (j*2+k, k in 0,1) -> V (j*3+k)
            # V1 rows: 0(j0k0), 1(j0k1), 2(j1k0), 3(j1k1)
            # V rows: 0(j0k0), 1(j0k1), 2(j0k2), 3(j1k0), 4(j1k1), 5(j1k2)
            V[0, :7] = V1[0, :]
            V[1, :7] = V1[1, :]
            V[3, :7] = V1[2, :]
            V[4, :7] = V1[3, :]
            
            # V2 (j*1+k', k'=0->k2) -> V (j*3+2)
            # V2 rows: 0(j0), 1(j1)
            V[2, 7:] = V2[0, :]
            V[5, 7:] = V2[1, :]
            
            # W mapping: W1 (k*2+i, k in 0,1) -> W (k*2+i)
            # W1 rows: 0(k0i0), 1(k0i1), 2(k1i0), 3(k1i1)
            # W rows: 0(k0i0), 1(k0i1), 2(k1i0), 3(k1i1), 4(k2i0), 5(k2i1)
            W[0, :7] = W1[0, :]
            W[1, :7] = W1[1, :]
            W[2, :7] = W1[2, :]
            W[3, :7] = W1[3, :]
            
            # W2 (k'*2+i, k'=0->k2) -> W (2*2+i)
            # W2 rows: 0(i0), 1(i1)
            W[4, 7:] = W2[0, :]
            W[5, 7:] = W2[1, :]
            
            return U, V, W
        
        if n_b == 2 and m_b == 3 and p_b == 2:
            U1, V1, W1 = strassen_222()
            U2, V2, W2 = trivial_block(2, 1, 2)
            
            rank = 11
            U = np.zeros((6, rank), dtype=np.int8)
            V = np.zeros((6, rank), dtype=np.int8)
            W = np.zeros((4, rank), dtype=np.int8)
            
            # U mapping: U1 (i*2+j, j in 0,1) -> U (i*3+j)
            # U1 rows: 0(i0j0), 1(i0j1), 2(i1j0), 3(i1j1)
            # U rows: 0(i0j0), 1(i0j1), 2(i0j2), 3(i1j0), 4(i1j1), 5(i1j2)
            U[0, :7] = U1[0, :]
            U[1, :7] = U1[1, :]
            U[3, :7] = U1[2, :]
            U[4, :7] = U1[3, :]
            
            # U2 (i*1+j', j'=0->j2) -> U (i*3+2)
            # U2 rows: 0(i0), 1(i1)
            U[2, 7:] = U2[0, :]
            U[5, 7:] = U2[1, :]
            
            # V mapping: V1 (j*2+k, j in 0,1) -> V (j*2+k)
            # V1 rows: 0(j0k0), 1(j0k1), 2(j1k0), 3(j1k1)
            # V rows: 0(j0k0), 1(j0k1), 2(j1k0), 3(j1k1), 4(j2k0), 5(j2k1)
            V[0, :7] = V1[0, :]
            V[1, :7] = V1[1, :]
            V[2, :7] = V1[2, :]
            V[3, :7] = V1[3, :]
            
            # V2 (j'*2+k, j'=0->j2) -> V (2*2+k)
            # V2 rows: 0(k0), 1(k1)
            V[4, 7:] = V2[0, :]
            V[5, 7:] = V2[1, :]
            
            # W mapping: W1 (k*2+i, k in 0,1) -> W (k*2+i)
            # W1 rows: 0(k0i0), 1(k0i1), 2(k1i0), 3(k1i1)
            # W rows: 0(k0i0), 1(k0i1), 2(k1i0), 3(k1i1)
            W[0, :7] = W1[0, :]
            W[1, :7] = W1[1, :]
            W[2, :7] = W1[2, :]
            W[3, :7] = W1[3, :]
            
            # W2 (k*1+i, k in 0,1) -> W (k*2+i)
            # W2 rows: 0(i0), 1(i1) for k=0? No, trivial_block(2,1,2) W is (p*n, rank) = (2*2, 4) = (4,4).
            # Wait, trivial_block(2,1,2): n=2, m=1, p=2. W is (p*n, rank) = (4, 4).
            # Rows are k*n+i. k in 0..1, i in 0..1.
            # W2 rows: 0(k0i0), 1(k0i1), 2(k1i0), 3(k1i1)
            # This matches W rows exactly.
            W[:, 7:] = W2
            
            return U, V, W

        if n_b == 2 and m_b == 3 and p_b == 3:
            U1, V1, W1 = strassen_222() # 2x2x2 (j0-1, k0-1)
            U2, V2, W2 = trivial_block(2, 2, 1) # 2x2x1 (j0-1, k2)
            U3, V3, W3 = trivial_block(2, 1, 2) # 2x1x2 (j2, k0-1)
            U4, V4, W4 = trivial_block(2, 1, 1) # 2x1x1 (j2, k2)
            
            rank = 17
            U = np.zeros((6, rank), dtype=np.int8)
            V = np.zeros((9, rank), dtype=np.int8)
            W = np.zeros((6, rank), dtype=np.int8)
            
            # U
            # U1 (i*2+j, j0-1) -> U (i*3+j)
            U[0, 0:7] = U1[0, :]
            U[1, 0:7] = U1[1, :]
            U[3, 0:7] = U1[2, :]
            U[4, 0:7] = U1[3, :]
            
            # U2 (i*2+j, j0-1) -> U (i*3+j)
            U[0, 7:11] = U2[0, :]
            U[1, 7:11] = U2[1, :]
            U[3, 7:11] = U2[2, :]
            U[4, 7:11] = U2[3, :]
            
            # U3 (i*1+j, j=0->j2) -> U (i*3+2)
            U[2, 11:15] = U3[0, :]
            U[5, 11:15] = U3[1, :]
            
            # U4 (i*1+j, j=0->j2) -> U (i*3+2)
            U[2, 15:17] = U4[0, :]
            U[5, 15:17] = U4[1, :]
            
            # V
            # V1 (j*2+k, j0-1, k0-1) -> V (j*3+k)
            V[0, 0:7] = V1[0, :] # j0k0
            V[1, 0:7] = V1[1, :] # j0k1
            V[3, 0:7] = V1[2, :] # j1k0
            V[4, 0:7] = V1[3, :] # j1k1
            
            # V2 (j*1+k, j0-1, k=0->k2) -> V (j*3+2)
            V[2, 7:11] = V2[0, :] # j0k2
            V[5, 7:11] = V2[1, :] # j1k2
            
            # V3 (j*2+k, j=0->j2, k0-1) -> V (2*3+k)
            V[6, 11:15] = V3[0, :] # j2k0
            V[7, 11:15] = V3[1, :] # j2k1
            
            # V4 (j*1+k, j=0->j2, k=0->k2) -> V (2*3+2)
            V[8, 15:17] = V4[0, :] # j2k2
            
            # W
            # W1 (k*2+i, k0-1, i0-1) -> W (k*2+i)
            W[0, 0:7] = W1[0, :]
            W[1, 0:7] = W1[1, :]
            W[2, 0:7] = W1[2, :]
            W[3, 0:7] = W1[3, :]
            
            # W2 (k*2+i, k=0->k2, i0-1) -> W (2*2+i)
            W[4, 7:11] = W2[0, :]
            W[5, 7:11] = W2[1, :]
            
            # W3 (k*2+i, k0-1, i0-1) -> W (k*2+i)
            # Wait, W3 is from trivial_block(2,1,2). n=2, m=1, p=2. W is (4,4).
            # Rows k*2+i. k in 0..1, i in 0..1.
            W[0, 11:15] = W3[0, :]
            W[1, 11:15] = W3[1, :]
            W[2, 11:15] = W3[2, :]
            W[3, 11:15] = W3[3, :]
            
            # W4 (k*2+i, k=0->k2, i0-1) -> W (2*2+i)
            # W4 is from trivial_block(2,1,1). n=2, m=1, p=1. W is (2,2).
            # Rows k*2+i. k=0, i in 0..1.
            W[4, 15:17] = W4[0, :]
            W[5, 15:17] = W4[1, :]
            
            return U, V, W
        
        return trivial_block(n_b, m_b, p_b)
