import numpy as np
import time

_start_time = time.time()

def construct():
    n, m, p = 3, 5, 6
    
    # Global matrices
    total_rank = 0
    U = np.zeros((n*m, 100)) # Pre-allocate enough, will slice
    V = np.zeros((m*p, 100))
    W = np.zeros((p*n, 100))
    
    # Helper to get block decomposition
    def get_block(n_s, m_s, p_s):
        # Returns U, V, W, rank
        # Base cases
        if n_s == 1 and m_s == 1 and p_s == 1:
            return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]]), 1
        
        if n_s == 1:
            rank = m_s * p_s
            U_b = np.zeros((m_s, rank))
            V_b = np.zeros((m_s*p_s, rank))
            W_b = np.zeros((p_s, rank))
            for j in range(m_s):
                for k in range(p_s):
                    r = j * p_s + k
                    U_b[j, r] = 1.0
                    V_b[j*p_s + k, r] = 1.0
                    W_b[k, r] = 1.0
            return U_b, V_b, W_b, rank
            
        if m_s == 1:
            rank = n_s * p_s
            U_b = np.zeros((n_s, rank))
            V_b = np.zeros((p_s, rank))
            W_b = np.zeros((p_s*n_s, rank))
            for i in range(n_s):
                for k in range(p_s):
                    r = i * p_s + k
                    U_b[i, r] = 1.0
                    V_b[k, r] = 1.0
                    W_b[k*n_s + i, r] = 1.0
            return U_b, V_b, W_b, rank
            
        if p_s == 1:
            rank = n_s * m_s
            U_b = np.zeros((n_s*m_s, rank))
            V_b = np.zeros((m_s, rank))
            W_b = np.zeros((n_s, rank))
            for i in range(n_s):
                for j in range(m_s):
                    r = i * m_s + j
                    U_b[i*m_s + j, r] = 1.0
                    V_b[j, r] = 1.0
                    W_b[i, r] = 1.0
            return U_b, V_b, W_b, rank
            
        if n_s == 2 and m_s == 2 and p_s == 2:
            rank = 7
            U_b = np.zeros((4, rank))
            V_b = np.zeros((4, rank))
            W_b = np.zeros((4, rank))
            
            # M1
            U_b[0, 0] = 1; U_b[3, 0] = 1
            V_b[0, 0] = 1; V_b[3, 0] = 1
            W_b[0, 0] = 1; W_b[3, 0] = 1
            
            # M2
            U_b[2, 1] = 1; U_b[3, 1] = 1
            V_b[0, 1] = 1
            W_b[2, 1] = 1
            
            # M3
            U_b[0, 2] = 1
            V_b[2, 2] = 1; V_b[3, 2] = -1
            W_b[1, 2] = 1
            
            # M4
            U_b[3, 3] = 1
            V_b[1, 3] = 1; V_b[0, 3] = -1
            W_b[0, 3] = 1; W_b[2, 3] = 1
            
            # M5
            U_b[0, 4] = 1; U_b[1, 4] = 1
            V_b[3, 4] = 1
            W_b[0, 4] = -1; W_b[1, 4] = 1
            
            # M6
            U_b[2, 5] = 1; U_b[0, 5] = -1
            V_b[0, 5] = 1; V_b[1, 5] = 1
            W_b[3, 5] = 1
            
            # M7
            U_b[1, 6] = 1; U_b[3, 6] = -1
            V_b[2, 6] = 1; V_b[3, 6] = 1
            W_b[0, 6] = 1; W_b[1, 6] = 1
            
            return U_b, V_b, W_b, rank

        # Recursive cases
        # Try splitting n
        if n_s > 1:
            # Split n into 1 and n_s-1
            n1 = 1
            n2 = n_s - 1
            U1, V1, W1, r1 = get_block(n1, m_s, p_s)
            U2, V2, W2, r2 = get_block(n2, m_s, p_s)
            rank = r1 + r2
            U_b = np.zeros((n_s*m_s, rank))
            V_b = np.zeros((m_s*p_s, rank))
            W_b = np.zeros((p_s*n_s, rank))
            
            # Place 1
            U_b[:m_s, :r1] = U1
            V_b[:m_s*p_s, :r1] = V1
            W_b[:p_s, :r1] = W1 # W index k*n+i. For n1=1, i=0. k*1+0 = k.
            
            # Place 2
            # n2 starts at index 1
            U_b[m_s:m_s*n_s, r1:r1+r2] = U2
            V_b[:m_s*p_s, r1:r1+r2] = V2
            # W index for n2: k*n2 + i. Global i = 1 + i_local.
            # W_b global index: k*n_s + (1 + i_local) = k*n_s + 1 + i_local.
            # W2 local index: k*n2 + i_local.
            # We need to map W2 to W_b.
            W2_mapped = np.zeros((p_s*n_s, r2))
            for k in range(p_s):
                for i in range(n2):
                    W2_mapped[k*n_s + 1 + i, :] = W2[k*n2 + i, :]
            W_b[:, r1:r1+r2] = W2_mapped
            
            return U_b, V_b, W_b, rank

        # Try splitting m
        if m_s > 1:
            # Heuristic split for m
            if m_s == 5:
                m_parts = [1, 2, 2]
            elif m_s == 2:
                return get_block(n_s, 2, p_s) # Should be handled by base or p split
            else:
                m_parts = [1, m_s-1]
            
            rank = 0
            U_b = np.zeros((n_s*m_s, 100))
            V_b = np.zeros((m_s*p_s, 100))
            W_b = np.zeros((p_s*n_s, 100))
            
            m_offset = 0
            for mp in m_parts:
                U1, V1, W1, r1 = get_block(n_s, mp, p_s)
                
                # Place U
                # U index: i*m_s + j.
                # Local U index: i*mp + j_local.
                # Global j = m_offset + j_local.
                # Global U index: i*m_s + m_offset + j_local.
                U1_mapped = np.zeros((n_s*m_s, r1))
                for i in range(n_s):
                    for j in range(mp):
                        U1_mapped[i*m_s + m_offset + j, :] = U1[i*mp + j, :]
                U_b[:, rank:rank+r1] = U1_mapped
                
                # Place V
                # V index: j*p_s + k.
                # Local V index: j_local*p_s + k.
                # Global j = m_offset + j_local.
                # Global V index: (m_offset + j_local)*p_s + k.
                V1_mapped = np.zeros((m_s*p_s, r1))
                for j in range(mp):
                    for k in range(p_s):
                        V1_mapped[(m_offset + j)*p_s + k, :] = V1[j*p_s + k, :]
                V_b[:, rank:rank+r1] = V1_mapped
                
                # Place W
                # W index: k*n_s + i. Same as local.
                W_b[:, rank:rank+r1] = W1
                
                rank += r1
                m_offset += mp
                
            return U_b[:, :rank], V_b[:, :rank], W_b[:, :rank], rank

        # Try splitting p
        if p_s > 1:
            if p_s == 6:
                p_parts = [2, 2, 2]
            else:
                p_parts = [1, p_s-1]
            
            rank = 0
            U_b = np.zeros((n_s*m_s, 100))
            V_b = np.zeros((m_s*p_s, 100))
            W_b = np.zeros((p_s*n_s, 100))
            
            p_offset = 0
            for pp in p_parts:
                U1, V1, W1, r1 = get_block(n_s, m_s, pp)
                
                # Place U
                # U index: i*m_s + j. Same as local.
                U_b[:, rank:rank+r1] = U1
                
                # Place V
                # V index: j*p_s + k.
                # Local V index: j*pp + k_local.
                # Global k = p_offset + k_local.
                # Global V index: j*p_s + p_offset + k_local.
                V1_mapped = np.zeros((m_s*p_s, r1))
                for j in range(m_s):
                    for k in range(pp):
                        V1_mapped[j*p_s + p_offset + k, :] = V1[j*pp + k, :]
                V_b[:, rank:rank+r1] = V1_mapped
                
                # Place W
                # W index: k*n_s + i.
                # Local W index: k_local*n_s + i.
                # Global k = p_offset + k_local.
                # Global W index: (p_offset + k_local)*n_s + i.
                W1_mapped = np.zeros((p_s*n_s, r1))
                for k in range(pp):
                    for i in range(n_s):
                        W1_mapped[(p_offset + k)*n_s + i, :] = W1[k*n_s + i, :]
                W_b[:, rank:rank+r1] = W1_mapped
                
                rank += r1
                p_offset += pp
                
            return U_b[:, :rank], V_b[:, :rank], W_b[:, :rank], rank

        # Fallback schoolbook
        rank = n_s * m_s * p_s
        U_b = np.zeros((n_s*m_s, rank))
        V_b = np.zeros((m_s*p_s, rank))
        W_b = np.zeros((p_s*n_s, rank))
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    r = (i*m_s + j)*p_s + k
                    U_b[i*m_s + j, r] = 1.0
                    V_b[j*p_s + k, r] = 1.0
                    W_b[k*n_s + i, r] = 1.0
        return U_b, V_b, W_b, rank

    U_final, V_final, W_final, final_rank = get_block(n, m, p)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
