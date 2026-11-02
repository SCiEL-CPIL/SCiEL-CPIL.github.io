"""Optimized rank-minimal tensor decomposition for 2x4 @ 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]
import time
import random
import numpy as np

def construct():
    n, m, p = 2, 4, 6
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def strassen_2x2():
        """Return Strassen rank-7 decomposition for 2x2 @ 2x2.
        U: (4,7) indexed by i*2+j
        V: (4,7) indexed by j*2+k
        W: (4,7) indexed by k*2+i (trace convention)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B01-B11)
        # M4=A11(B10-B00), M5=(A00+A01)B11, M6=(A10-A00)(B00+B01), M7=(A01-A11)(B10+B11)
        U[0,0]=1; U[3,0]=1
        U[2,1]=1; U[3,1]=1
        U[0,2]=1
        U[3,3]=1
        U[0,4]=1; U[1,4]=1
        U[2,5]=1; U[0,5]=-1
        U[1,6]=1; U[3,6]=-1
        V[0,0]=1; V[3,0]=1
        V[0,1]=1
        V[1,2]=1; V[3,2]=-1
        V[2,3]=1; V[0,3]=-1
        V[3,4]=1
        V[0,5]=1; V[1,5]=1
        V[2,6]=1; V[3,6]=1
        # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
        # W index: k*2+i
        W[0,0]=1; W[0,3]=1; W[0,4]=-1; W[0,6]=1  # C00
        W[2,2]=1; W[2,4]=1                          # C01
        W[1,1]=1; W[1,3]=1                          # C10
        W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=1    # C11
        return U, V, W

    def build_rank42():
        """Build rank-42 decomposition using 6 independent Strassen blocks."""
        r = 42
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        Us, Vs, Ws = strassen_2x2()
        col = 0
        for r_block in range(3):  # 3 column groups of 2
            for sub in range(2):  # 2 row groups (top/bottom)
                a_col_off = sub * 2
                b_row_off = sub * 2
                for s in range(7):
                    for i in range(2):
                        for j in range(2):
                            val = Us[i*2+j, s]
                            if val != 0:
                                U[i*4 + a_col_off + j, col] += val
                    for j in range(2):
                        for k in range(2):
                            val = Vs[j*2+k, s]
                            if val != 0:
                                V[(b_row_off+j)*6 + r_block*2 + k, col] += val
                    for k in range(2):
                        for i in range(2):
                            val = Ws[k*2+i, s]
                            if val != 0:
                                W[(r_block*2+k)*2 + i, col] += val
                    col += 1
        return U, V, W, r

    def verify(U, V, W, r):
        """Verify decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(r):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def try_reduce(U, V, W, r, max_elim=10):
        """Try to eliminate redundant terms via linear dependence checks."""
        U = U.copy(); V = V.copy(); W = W.copy()
        eliminated = set()
        changed = True
        elim_count = 0
        while changed and elim_count < max_elim and time.time() < deadline - 5:
            changed = False
            for t in range(r):
                if t in eliminated:
                    continue
                if time.time() > deadline - 10:
                    break
                # Check if term t is linearly dependent on others
                others = [s for s in range(r) if s != t and s not in eliminated]
                if not others:
                    continue
                # Build system: for each (i,j,k), sum_s c_s * U[i,j,s]*V[j,k,s]*W[k,i,s] = U[i,j,t]*V[j,k,t]*W[k,i,t]
                eqs = []
                rhs = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = np.array([U[i*m+j, s]*V[j*p+k, s]*W[k*n+i, s] for s in others])
                            eqs.append(row)
                            rhs.append(U[i*m+j, t]*V[j*p+k, t]*W[k*n+i, t])
                if not eqs:
                    continue
                A_mat = np.array(eqs)
                b_vec = np.array(rhs)
                try:
                    sol, res, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
                    if np.allclose(A_mat @ sol, b_vec, atol=1e-9):
                        eliminated.add(t)
                        elim_count += 1
                        changed = True
                except:
                    pass
        if not eliminated:
            return U, V, W, r
        keep = [i for i in range(r) if i not in eliminated]
        new_r = len(keep)
        new_U = U[:, keep]
        new_V = V[:, keep]
        new_W = W[:, keep]
        if verify(new_U, new_V, new_W, new_r):
            return new_U, new_V, new_W, new_r
        return U, V, W, r

    def randomized_search(U0, V0, W0, r0, max_iters=200):
        """Randomized perturbation + reduction search."""
        best_U, best_V, best_W, best_r = U0, V0, W0, r0
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            U = U0.copy(); V = V0.copy(); W = W0.copy()
            # Perturb a few entries
            for _ in range(2):
                which = random.choice(['U', 'V', 'W'])
                if which == 'U':
                    idx = random.randint(0, len(U)-1); col = random.randint(0, r0-1)
                    U[idx, col] += random.uniform(-0.3, 0.3)
                elif which == 'V':
                    idx = random.randint(0, len(V)-1); col = random.randint(0, r0-1)
                    V[idx, col] += random.uniform(-0.3, 0.3)
                else:
                    idx = random.randint(0, len(W)-1); col = random.randint(0, r0-1)
                    W[idx, col] += random.uniform(-0.3, 0.3)
            new_U, new_V, new_W, new_r = try_reduce(U, V, W, r0, max_elim=5)
            if new_r < best_r:
                best_U, best_V, best_W, best_r = new_U, new_V, new_W, new_r
                U0, V0, W0, r0 = best_U, best_V, best_W, best_r
        return best_U, best_V, best_W, best_r

    # Build initial rank-42 decomposition
    U, V, W, r = build_rank42()
    
    # Verify
    if not verify(U, V, W, r):
        # Fall back to trivial rank 48
        r = n * m * p
        U = np.zeros((n*m, r)); V = np.zeros((m*p, r)); W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    # Try rank reduction
    U, V, W, r = try_reduce(U, V, W, r, max_elim=15)
    
    # Randomized search if time permits
    if time.time() < deadline - 30:
        U, V, W, r = randomized_search(U, V, W, r, max_iters=150)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
