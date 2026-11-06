import time
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 135.0

def _build():
    sx = 0.5*np.array([[0.,1.],[1.,0.]],complex)
    sy = 0.5*np.array([[0.,-1j],[1j,0.]],complex)
    sz = 0.5*np.array([[1.,0.],[0.,-1.]],complex)
    I2 = np.eye(2,complex)
    N = 6; dim = 2**N
    def kron_ops(ops):
        r = np.array([[1.0]],complex)
        for op in ops:
            r = np.kron(r, op)
        return r
    def op_pair(i,j,oi,oj):
        ops=[I2]*N; ops[i]=oi; ops[j]=oj
        return kron_ops(ops)
    def op_site(s,op):
        ops=[I2]*N; ops[s]=op
        return kron_ops(ops)
    D=0.8; h=0.3
    odd=[(0,1),(2,3),(4,5)]; even=[(1,2),(3,4)]
    A=np.zeros((dim,dim),complex); B=np.zeros((dim,dim),complex)
    for (i,j) in odd:
        A += op_pair(i,j,sx,sx)+op_pair(i,j,sy,sy)+D*op_pair(i,j,sz,sz)
    for (i,j) in even:
        B += op_pair(i,j,sx,sx)+op_pair(i,j,sy,sy)+D*op_pair(i,j,sz,sz)
    field = h*sum(op_site(s,sz) for s in range(N))
    A += 0.5*field; B += 0.5*field
    return A, B

_cache = {}
def _get():
    if 'A' not in _cache:
        _cache['A'], _cache['B'] = _build()
    return _cache['A'], _cache['B']

def _strang():
    a = [1.0/12.0]*12
    b = [1.0/6.0, 0.0]*6
    return a, b

def coefficients(stages):
    try:
        A, B = _get()
        H = A + B
        dim = len(A)
        tvals = (0.25, 0.5, 1.0)
        lA, VA = eigh(A); lB, VB = eigh(B); lH, VH = eigh(H)
        VAH = VA.conj().T; VBH = VB.conj().T; VHH = VH.conj().T
        def expA(x):
            return VA @ (VAH * np.exp(-1j*x*lA)[:,None])
        def expB(x):
            return VB @ (VBH * np.exp(-1j*x*lB)[:,None])
        Uex = [VH @ (VHH * np.exp(-1j*t*lH)[:,None]) for t in tvals]
        I = np.eye(dim, complex)
        def prod(a, b, t):
            U = I
            for k in range(12):
                U = U @ expB(b[k]*t) @ expA(a[k]*t)
            return U
        def err(a, b):
            s = 0.0
            for k, t in enumerate(tvals):
                s += np.linalg.norm(prod(a, b, t) - Uex[k], 2)
            return s/3.0
        lam = 100.0
        def obj(v):
            a = v[:12]; b = v[12:]
            return err(a, b) + lam*((a.sum()-1.0)**2 + (b.sum()-1.0)**2)
        a0 = np.full(12, 1.0/12.0)
        b0 = np.array([1.0/6.0, 0.0]*6)
        v0 = np.concatenate([a0, b0])
        best = None; bestv = np.inf
        rng = np.random.default_rng(7)
        def run(vi, mf):
            nonlocal best, bestv
            if time.time() > _DEADLINE - 10:
                return
            try:
                r = minimize(obj, vi, method='L-BFGS-B',
                             bounds=[(-50.0,50.0)]*24,
                             options={'maxiter':400,'maxfun':mf,'ftol':1e-14})
                if r.fun < bestv:
                    bestv = r.fun; best = r.x
            except Exception:
                pass
        run(v0, 1500)
        for _ in range(5):
            if time.time() > _DEADLINE - 10: break
            a = rng.uniform(0,1,12); a/=a.sum()
            b = rng.uniform(0,1,12); b/=b.sum()
            run(np.concatenate([a,b]), 300)
        if best is None:
            best = v0
        a = np.clip(best[:12], -50, 50); b = np.clip(best[12:], -50, 50)
        sa = a.sum(); sb = b.sum()
        if abs(sa) > 1e-12: a = a/sa
        if abs(sb) > 1e-12: b = b/sb
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return _strang()
        return [float(x) for x in a], [float(x) for x in b]
    except Exception:
        return _strang()
