import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        # M2 = (A10+A11)B00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=-1
        # M3 = A00(B01-B11)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        # M4 = A11(B10-B00)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        # M5 = (A00+A01)B11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=1; W[2,4]=-1
        # M6 = (A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        # M7 = (A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=-1
        return U, V, W

    def schoolbook_2x2x2():
        U = np.zeros((4, 8))
        V = np.zeros((4, 8))
        W = np.zeros((4, 8))
        for i in range(4):
            U[i,i]=1; V[i,i]=1; W[i,i]=1
        return U, V, W

    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        U[3,3]=1; V[1,3]=1; W[1,3]=1
        return U, V, W

    def schoolbook_1x2x2():
        U = np.zeros((2, 2))
        V = np.zeros((4, 2))
        W = np.zeros((2, 2))
        U[0,0]=1
        V[0,0]=1; V[1,0]=1
        W[0,0]=1; W[1,0]=1
        U[1,1]=1
        V[2,1]=1; V[3,1]=1
        W[0,1]=1; W[1,1]=1
        return U, V, W

    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        return U, V, W

    U_parts = []
    V_parts = []
    W_parts = []

    def add_block(U_blk, V_blk, W_blk, u_idx, v_idx, w_idx):
        for r in range(U_blk.shape[1]):
            U_part = np.zeros(n * m)
            for i, idx in enumerate(u_idx):
                U_part[idx] = U_blk[i, r]
            U_parts.append(U_part)
            V_part = np.zeros(m * p)
            for i, idx in enumerate(v_idx):
                V_part[idx] = V_blk[i, r]
            V_parts.append(V_part)
            W_part = np.zeros(p * n)
            for i, idx in enumerate(w_idx):
                W_part[idx] = W_blk[i, r]
            W_parts.append(W_part)

    # Block 1: A rows 0,1 x B rows 0,1 x cols 0,1
    add_block(strassen_2x2x2(), None, None, [0,1,4,5], [0,1,5,6], [0,3,1,4])
    # Block 2: A rows 0,1 x B rows 0,1 x cols 2,3
    add_block(strassen_2x2x2(), None, None, [0,1,4,5], [2,3,7,8], [6,9,7,10])
    # Block 3: A rows 0,1 x B rows 0,1 x col 4
    add_block(schoolbook_2x2x1(), None, None, [0,1,4,5], [4,9], [12,13])
    # Block 4: A rows 0,1 x B rows 2,3 x cols 0,1
    add_block(strassen_2x2x2(), None, None, [0,1,4,5], [10,11,15,16], [0,3,1,4])
    # Block 5: A rows 0,1 x B rows 2,3 x cols 2,3
    add_block(strassen_2x2x2(), None, None, [0,1,4,5], [12,13,17,18], [6,9,7,10])
    # Block 6: A rows 0,1 x B rows 2,3 x col 4
    add_block(schoolbook_2x2x1(), None, None, [0,1,4,5], [14,19], [12,13])
    # Block 7: A row 2 x B rows 0,1 x cols 0,1
    add_block(schoolbook_1x2x2(), None, None, [8,9], [0,1,5,6], [2,5])
    # Block 8: A row 2 x B rows 0,1 x cols 2,3
    add_block(schoolbook_1x2x2(), None, None, [8,9], [2,3,7,8], [8,11])
    # Block 9: A row 2 x B rows 0,1 x col 4
    add_block(schoolbook_1x2x1(), None, None, [8,9], [4,9], [14])
    # Block 10: A row 2 x B rows 2,3 x cols 0,1
    add_block(schoolbook_1x2x2(), None, None, [8,9], [10,11,15,16], [2,5])
    # Block 11: A row 2 x B rows 2,3 x cols 2,3
    add_block(schoolbook_1x2x2(), None, None, [8,9], [12,13,17,18], [8,11])
    # Block 12: A row 2 x B rows 2,3 x col 4
    add_block(schoolbook_1x2x1(), None, None, [8,9], [14,19], [14])

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T

    def verify(U, V, W, tol=1e-7):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    acc = np.dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
                    if abs(acc - 1.0) > tol:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]

    def fast_verify(U, V, W, tol=1e-7):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    acc = np.dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
                    if abs(acc - 1.0) > tol:
                        return False
        return True

    for attempt in range(current_rank):
        if time.time() > deadline - 15:
            break
        reduced = False
        for col in range(current_rank):
            if time.time() > deadline - 15:
                break
            keep = [c for c in range(current_rank) if c != col]
            U_new = current_U[:, keep]
            V_new = current_V[:, keep]
            W_new = current_W[:, keep]
            if fast_verify(U_new, V_new, W_new):
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                reduced = True
                break
        if not reduced:
            break

    return {
        "U": current_U.tolist(),
        "V": current_V.tolist(),
        "W": current_W.tolist()
    }
