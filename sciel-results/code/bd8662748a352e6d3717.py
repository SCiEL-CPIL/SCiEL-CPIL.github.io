"""Tensor decomposition search for 2x4 x 4x4 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]
N, M, P = DIMENSIONS
DEADLINE = 150.0
START = time.time()

def _remaining():
    return DEADLINE - (time.time() - START)

def _tensor_3d():
    T = np.zeros((N*P, N*M, M*P), dtype=object)
    for i in range(N):
        for r in range(P):
            for j in range(M):
                T[i*P + r, i*M + j, j*P + r] = 1
    return T

def _verify(U, V, W, T):
    nC, nA, nB = T.shape
    rank = len(U[0])
    S = np.zeros((nC, nA, nB), dtype=object)
    for c in range(rank):
        uc = U[:, c]; vc = V[:, c]; wc = W[:, c]
        for i in range(nC):
            ui = uc[i]
            if ui == 0: continue
            for j in range(nA):
                vj = vc[j]
                if vj == 0: continue
                base = ui * vj
                for k in range(nB):
                    wk = wc[k]
                    if wk == 0: continue
                    S[i, j, k] += base * wk
    for i in range(nC):
        for j in range(nA):
            for k in range(nB):
                if S[i, j, k] != T[i, j, k]:
                    return False
    return True

def _schoolbook():
    rank = N * M * P
    U = [[0]*rank for _ in range(N*M)]
    V = [[0]*rank for _ in range(M*P)]
    W = [[0]*rank for _ in range(P*N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M + j][col] = 1
                V[j*P + k][col] = 1
                W[k*N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(R, T):
    nC, nA, nB = (N*P, N*M, M*P)
    Tflat = T.reshape(nC*nA*nB).astype(float)
    Mmat = np.zeros((nC*nA*nB, R), dtype=float)
    for attempt in range(4000):
        if _remaining() < 8:
            return None
        rng = np.random.default_rng(attempt)
        U = rng.integers(-3, 4, size=(nC, R))
        V = rng.integers(-3, 4, size=(nA, R))
        for c in range(R):
            outer = np.outer(U[:, c].astype(float), V[:, c].astype(float))
            Mmat[:, c] = np.repeat(outer.reshape(-1), nB)
        try:
            Wflat, *_ = np.linalg.lstsq(Mmat, Tflat, rcond=None)
        except np.linalg.LinAlgError:
            continue
        W = Wflat.reshape(nB, R)
        # check exactness with half-integer rounding
        Wr = np.round(W * 2) / 2.0
        Uo = U.astype(object)
        Vo = V.astype(object)
        Wo = np.array([[Fraction(int(x*2)) / 2 for x in row] for row in Wr])
        if _verify(Uo, Vo, Wo, T):
            return Uo, Vo, Wo
    return None

def construct():
    T = _tensor_3d()
    best = _schoolbook()
    for R in range(28, N*M*P):
        if _remaining() < 12:
            break
        res = _try_rank(R, T)
        if res is not None:
            Uo, Vo, Wo = res
            U = [[int(x) if float(x).is_integer() else float(x) for x in row] for row in Uo]
            V = [[int(x) if float(x).is_integer() else float(x) for x in row] for row in Vo]
            W = [[x.numerator / x.denominator for x in row] for row in Wo]
            return {"U": U, "V": V, "W": W}
    return best

if __name__ == "__main__":
    print(construct())
