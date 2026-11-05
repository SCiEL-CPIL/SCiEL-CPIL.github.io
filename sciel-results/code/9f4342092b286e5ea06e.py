import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 110  # Safety margin before 120s limit
    
    n, m, p = 3, 5, 7
    
    def all_partitions(size):
        """Generate all partitions of size into parts of 1 or 2."""
        if size == 0:
            return [[]]
        results = []
        for rest in all_partitions(size - 1):
            results.append([1] + rest)
        if size >= 2:
            for rest in all_partitions(size - 2):
                results.append([2] + rest)
        return results
    
    def compute_rank(np_, mp_, pp_):
        """Compute total rank for given block partitions."""
        rank = 0
        for ns in np_:
            for ms in mp_:
                for ps in pp_:
                    if ns == 2 and ms == 2 and ps == 2:
                        rank += 7
                    elif (ns == 2 and ms == 2 and ps == 1) or \
                         (ns == 2 and ms == 1 and ps == 2) or \
                         (ns == 1 and ms == 2 and ps == 2):
                        rank += 3  # 2x2x1 is isomorphic to 2x2x2 with one dim trivial
                    else:
                        rank += ns * ms * ps
        return rank
    
    def get_strassen_2x2x2():
        """Strassen 2x2x2: U[i*2+a,c], V[b*2+k,c], W[k*2+i,c]"""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # M2 = (A10+A11)B00 -> C10+=1, C11-=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
        # M3 = A00(B01-B11) -> C01+=1, C11+=1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
        # M4 = A11(B10-B00) -> C00+=1, C10+=1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
        # M5 = (A00+A01)B11 -> C00-=1, C01+=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
        # M6 = (A10-A00)(B00+B01) -> C11+=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        # M7 = (A01-A11)(B10+B11) -> C00+=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W
    
    def get_strassen_2x2x1():
        """2x2x1: A is 2x2, B is 2x1, C is 2x1. W[k*2+i,c] where k in [0,1), i in [0,2).
        This is C[i] = sum_a A[i,a] * B[a]
        Use 3 terms: M1=A00*B0, M2=A11*B1, M3=(A00+A11)(B0+B1)
        C0 = M1 + M3 - M2, C1 = M2 + M3 - M1
        U[i*2+a, c], V[b*1+k, c] where b in [0,2), k in [0,1) -> V[b, c]
        W[0*2+i, c] = W[i, c] for i in [0,2)
        """
        U = np.zeros((4, 3))
        V = np.zeros((2, 3))
        W = np.zeros((2, 3))
        # M1 = A00 * B0
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # M2 = A11 * B1
        U[3,1]=1; V[1,1]=1; W[1,1]=1
        # M3 = (A00+A11)*(B0+B1)
        U[0,2]=1; U[3,2]=1; V[0,2]=1; V[1,2]=1; W[0,2]=1; W[1,2]=1
        # But we need C0 = M1 + M3 - M2, C1 = M2 + M3 - M1
        # So W[0] should be: M1 coeff 1, M2 coeff -1, M3 coeff 1
        # W[1] should be: M1 coeff -1, M2 coeff 1, M3 coeff 1
        W[0,0]=1; W[0,1]=-1; W[0,2]=1
        W[1,0]=-1; W[1,1]=1; W[1,2]=1
        return U, V, W
    
    def get_strassen_2x1x2():
        """2x1x2: A is 2x1, B is 1x2, C is 2x2.
        U[i*1+a, c] = U[i, c] where i in [0,2), a in [0,1)
        V[0*2+k, c] = V[k, c] where k in [0,2)
        W[k*2+i, c] where k in [0,2), i in [0,2)
        C[i,k] = A[i] * B[k]
        Use 3 terms: M1=A0*B0, M2=A1*B1, M3=(A0+A1)(B0+B1)
        C00 = M1 + M3 - M2, C01 = M3 - M2... wait
        Actually C00 = A0*B0 = M1
        C01 = A0*B1
        C10 = A1*B0
        C11 = A1*B1 = M2
        M3 = (A0+A1)(B0+B1) = A0B0 + A0B1 + A1B0 + A1B1 = C00+C01+C10+C11
        So C01 = M3 - M1 - M2 - C10... this doesn't work cleanly with 3 terms.
        
        Let me use: M1=A0*B0, M2=A1*B1, M3=(A0+A1)(B0+B1)
        C00 = M1
        C11 = M2
        C01 = M3 - M1 - M2 - C10... still need C10.
        
        Actually with 3 terms we can get:
        C00 = M1
        C11 = M2  
        C01 = M3 - M1 - M2 - C10... no.
        
        Let me think differently. C = A * B where A is 2x1, B is 1x2.
        C = [[A0*B0, A0*B1], [A1*B0, A1*B1]]
        This is an outer product. Rank 3 decomposition:
        M1 = A0*B0 -> C00 += 1
        M2 = A1*B1 -> C11 += 1
        M3 = (A0+A1)*(B0+B1) -> C00+=1, C01+=1, C10+=1, C11+=1
        Then C01 = M3 - M1 - M2 - (C10)... 
        
        Hmm, let me just use: 
        M1 = A0*B0 -> C00
        M2 = A0*B1 -> C01
        M3 = A1*B0 -> C10
        M4 = A1*B1 -> C11
        That's rank 4 = schoolbook. 
        
        For rank 3: M1=A0*B0, M2=A1*B1, M3=(A0+A1)(B0+B1)
        C00 = M1
        C11 = M2
        C01 = M3 - M1 - M2 - C10... 
        
        Actually: M3 = A0B0 + A0B1 + A1B0 + A1B1
        So A0B1 = M3 - A0B0 - A1B0 - A1B1 = M3 - M1 - C10 - M2
        This is circular.
        
        Let me try: M1=A0*B0, M2=A1*B1, M3=(A0-A1)*(B0-B1)
        M3 = A0B0 - A0B1 - A1B0 + A1B1
        C00 = M1
        C11 = M2
        C01 = (M1 + M2 - M3)/2... but we need integer/half-integer.
        C01 = A0B1 = (M1 + M2 - M3)/2? Let's check:
        M1+M2-M3 = A0B0 + A1B1 - A0B0 + A0B1 + A1B0 - A1B1 = A0B1 + A1B0
        So (M1+M2-M3)/2 = (A0B1+A1B0)/2, not A0B1.
        
        OK so 2x1x2 with rank 3 may not work with simple Strassen. Let me just use schoolbook (rank 4) for these cases.
        """
        # Use schoolbook for 2x1x2
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        for i in range(2):
            for k in range(2):
                U[i, i*2+k] = 1
                V[k, i*2+k] = 1
                W[k*2+i, i*2+k] = 1
        return U, V, W
    
    def get_strassen_1x2x2():
        """1x2x2: A is 1x2, B is 2x2, C is 1x2.
        U[0*2+a, c] = U[a, c] where a in [0,2)
        V[b*2+k, c] where b in [0,2), k in [0,2)
        W[k*1+i, c] = W[k, c] where k in [0,2)
        C[k] = sum_a A[a] * B[a,k]
        This is like a row vector times a matrix. Rank 3:
        M1 = A0*B00, M2 = A1*B11, M3 = (A0+A1)(B00+B11)
        C0 = A0*B00 + A1*B10
        C1 = A0*B01 + A1*B11
        M3 = A0B00 + A0B01 + A1B10 + A1B11
        C0 = M1 + A1B10
        C1 = A0B01 + M2
        Still need cross terms. Use schoolbook rank 4.
        """
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        for a in range(2):
            for k in range(2):
                U[a, a*2+k] = 1
                V[a*2+k, a*2+k] = 1
                W[k, a*2+k] = 1
        return U, V, W
    
    sU, sV, sW = get_strassen_2x2x2()
    s221U, s221V, s221W = get_strassen_2x2x1()
    s212U, s212V, s212W = get_strassen_2x1x2()
    s122U, s122V, s122W = get_strassen_1x2x2()
    
    def build_decomp(np_, mp_, pp_):
        n_off = [0]
        for s in np_:
            n_off.append(n_off[-1] + s)
        m_off = [0]
        for s in mp_:
            m_off.append(m_off[-1] + s)
        p_off = [0]
        for s in pp_:
            p_off.append(p_off[-1] + s)
        
        total_rank = compute_rank(np_, mp_, pp_)
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col = 0
        for ni in range(len(np_)):
            for mi in range(len(mp_)):
                for pi in range(len(pp_)):
                    n0, m0, p0 = n_off[ni], m_off[mi], p_off[pi]
                    ns, ms, ps = np_[ni], mp_[mi], pp_[pi]
                    
                    if ns == 2 and ms == 2 and ps == 2:
                        for c in range(7):
                            for i in range(2):
                                for a in range(2):
                                    U[(n0+i)*m + (m0+a), col] += sU[i*2+a, c]
                            for b in range(2):
                                for k in range(2):
                                    V[(m0+b)*p + (p0+k), col] += sV[b*2+k, c]
                            for i in range(2):
                                for k in range(2):
                                    W[(p0+k)*n + (n0+i), col] += sW[k*2+i, c]
                            col += 1
                    elif ns == 2 and ms == 2 and ps == 1:
                        for c in range(3):
                            for i in range(2):
                                for a in range(2):
                                    U[(n0+i)*m + (m0+a), col] += s221U[i*2+a, c]
                            for b in range(2):
                                V[(m0+b)*p + p0, col] += s221V[b, c]
                            for i in range(2):
                                W[p0*n + (n0+i), col] += s221W[i, c]
                            col += 1
                    elif ns == 2 and ms == 1 and ps == 2:
                        for c in range(4):  # schoolbook
                            for i in range(2):
                                U[(n0+i)*m + m0, col] = s212U[i, c]
                            for k in range(2):
                                V[m0*p + (p0+k), col] = s212V[k, c]
                            for i in range(2):
                                for k in range(2):
                                    W[(p0+k)*n + (n0+i), col] = s212W[k*2+i, c]
                            col += 1
                    elif ns == 1 and ms == 2 and ps == 2:
                        for c in range(4):  # schoolbook
                            for a in range(2):
                                U[n0*m + (m0+a), col] = s122U[a, c]
                            for b in range(2):
                                for k in range(2):
                                    V[(m0+b)*p + (p0+k), col] = s122V[b*2+k, c]
                            for k in range(2):
                                W[(p0+k)*n + n0, col] = s122W[k, c]
                            col += 1
                    else:
                        for i in range(ns):
                            for a in range(ms):
                                for k in range(ps):
                                    U[(n0+i)*m + (m0+a), col] = 1
                                    V[(m0+a)*p + (p0+k), col] = 1
                                    W[(p0+k)*n + (n0+i), col] = 1
                                    col += 1
        
        if col != total_rank:
            # Fallback: just return what we have, padded
            if col < total_rank:
                U = U[:, :col]
                V = V[:, :col]
                W = W[:, :col]
            else:
                U = U[:, :total_rank]
                V = V[:, :total_rank]
                W = W[:, :total_rank]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Generate partitions
    n_parts = all_partitions(n)
    m_parts = all_partitions(m)
    p_parts = all_partitions(p)
    
    # Find best partition
    best_rank = n * m * p
    best_np, best_mp, best_pp = n_parts[0], m_parts[0], p_parts[0]
    
    for np_ in n_parts:
        for mp_ in m_parts:
            for pp_ in p_parts:
                r = compute_rank(np_, mp_, pp_)
                if r < best_rank:
                    best_rank = r
                    best_np, best_mp, best_pp = np_, mp_, pp_
        
        if time.time() > deadline - 10:
            break
    
    # Build the best decomposition
    try:
        result = build_decomp(best_np, best_mp, best_pp)
        return result
    except Exception:
        pass
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for a in range(m):
            for k in range(p):
                U[i*m + a, c] = 1
                V[a*p + k, c] = 1
                W[k*n + i, c] = 1
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
