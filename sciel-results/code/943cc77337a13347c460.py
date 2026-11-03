"""Matrix multiplication 3x4x5 - rank-minimal decomposition."""
import time
import numpy as np
from scipy.linalg import lstsq

DIMENSIONS = [3, 4, 5]
DEADLINE_SECONDS = 140

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            if not (ip == i and jp == j and kp == k):
                                a = i * m + j
                                b = jp * p + kp
                                c = kp * n + ip
                                s = np.sum(U_np[a] * V_np[b] * W_np[c])
                                if abs(s) > tol:
                                    return False
    return True

def _build_rank56(n, m, p):
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    def add(u, v, w):
        nonlocal col
        U[:, col] = u; V[:, col] = v; W[:, col] = w; col += 1
    s = [([1,0,0,1],[1,0,0,1],[1,0,0,1]),([0,0,1,1],[1,0,0,0],[0,0,1,-1]),
         ([1,0,0,0],[0,0,0,1],[0,1,0,1]),([0,0,0,1],[0,1,0,0],[-1,0,1,0]),
         ([1,1,0,0],[0,1,0,0],[1,0,0,0]),([-1,0,1,0],[1,0,0,1],[0,0,0,-1]),
         ([0,1,0,-1],[0,1,0,1],[0,1,0,-1])]
    def e22(u,v,w,ar,ac,br,bc,cr,cc):
        Ue=np.zeros(12);Ve=np.zeros(20);We=np.zeros(15)
        for i in range(2):
            for j in range(2): Ue[(ar+i)*4+(ac+j)]=u[i*2+j]
        for j in range(2):
            for k in range(2): Ve[(br+j)*5+(bc+k)]=v[j*2+k]
        for k in range(2):
            for i in range(2): We[(cc+k)*3+(cr+i)]=w[k*2+i]
        return Ue,Ve,We
    def e23(u,v,w,ar,ac,br,bc,cr,cc):
        Ue=np.zeros(12);Ve=np.zeros(20);We=np.zeros(15)
        for i in range(2):
            for j in range(2): Ue[(ar+i)*4+(ac+j)]=u[i*2+j]
        for j in range(2):
            for k in range(3): Ve[(br+j)*5+(bc+k)]=v[j*3+k]
        for k in range(3):
            for i in range(2): We[(cc+k)*3+(cr+i)]=w[k*2+i]
        return Ue,Ve,We
    def e12(u,v,w,ar,ac,br,bc,cr,cc):
        Ue=np.zeros(12);Ve=np.zeros(20);We=np.zeros(15)
        for j in range(2): Ue[ar*4+(ac+j)]=u[j]
        for j in range(2):
            for k in range(2): Ve[(br+j)*5+(bc+k)]=v[j*2+k]
        for k in range(2): We[(cc+k)*3+cr]=w[k]
        return Ue,Ve,We
    def e13(u,v,w,ar,ac,br,bc,cr,cc):
        Ue=np.zeros(12);Ve=np.zeros(20);We=np.zeros(15)
        for j in range(2): Ue[ar*4+(ac+j)]=u[j]
        for j in range(2):
            for k in range(3): Ve[(br+j)*5+(bc+k)]=v[j*3+k]
        for k in range(3): We[(cc+k)*3+cr]=w[k]
        return Ue,Ve,We
    for u,v,w in s: add(*e22(u,v,w,0,0,0,0,0,0))
    for u,v,w in s: add(*e22(u,v,w,0,2,2,0,0,0))
    r11=[]
    for u,v,w in s: r11.append((list(u),[v[0],v[1],0,v[2],v[3],0],[w[0],w[1],0,w[2],w[3],0]))
    r11+=[([1,0,0,0],[0,0,1,0,0,0],[1,0,0,0,0,0]),([0,1,0,0],[0,0,0,0,0,1],[1,0,0,0,0,0]),
           ([0,0,1,0],[0,0,1,0,0,0],[0,0,0,1,0,0]),([0,0,0,1],[0,0,0,0,0,1],[0,0,0,1,0,0])]
    for u,v,w in r11: add(*e23(u,v,w,0,0,0,2,0,2))
    for u,v,w in r11: add(*e23(u,v,w,0,2,2,2,0,2))
    r4=[([1,0],[1,0,0,0],[1,0]),([1,0],[0,1,0,0],[0,1]),([0,1],[0,0,1,0],[1,0]),([0,1],[0,0,0,1],[0,1])]
    for u,v,w in r4: add(*e12(u,v,w,2,0,0,0,2,0))
    for u,v,w in r4: add(*e12(u,v,w,2,2,2,0,2,0))
    r6=[([1,0],[1,0,0,0,0,0],[1,0,0]),([1,0],[0,1,0,0,0,0],[0,1,0]),([1,0],[0,0,1,0,0,0],[0,0,1]),
        ([0,1],[0,0,0,1,0,0],[1,0,0]),([0,1],[0,0,0,0,1,0],[0,1,0]),([0,1],[0,0,0,0,0,1],[0,0,1])]
    for u,v,w in r6: add(*e13(u,v,w,2,0,0,2,2,2))
    for u,v,w in r6: add(*e13(u,v,w,2,2,2,2,2,2))
    return U, V, W

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    U56, V56, W56 = _build_rank56(n, m, p)
    return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}

# EVOLVE-BLOCK-END
