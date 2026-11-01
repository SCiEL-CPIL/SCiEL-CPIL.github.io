import numpy as np
import time

N = 11

def _hex_verts(x, y, th):
    a = th + np.arange(6)*(np.pi/3.0)
    return np.column_stack([x+np.cos(a), y+np.sin(a)])

def _inward_normals(phi):
    ang = phi + (np.arange(6)*(np.pi/3.0) + np.pi/6.0) + np.pi
    return np.column_stack([np.cos(ang), np.sin(ang)])

def _verts(p):
    x=p[0:3*N:3]; y=p[1:3*N:3]; th=p[2:3*N:3]
    v=np.empty((N,6,2))
    for i in range(N):
        v[i]=_hex_verts(x[i],y[i],th[i])
    return v

def _overlap_depth(A,B):
    eA=np.roll(A,-1,axis=0)-A; eB=np.roll(B,-1,axis=0)-B
    nA=np.column_stack([-eA[:,1],eA[:,0]]); nB=np.column_stack([-eB[:,1],eB[:,0]])
    nA/=(np.linalg.norm(nA,axis=1,keepdims=True)+1e-15)
    nB/=(np.linalg.norm(nB,axis=1,keepdims=True)+1e-15)
    nm=np.vstack([nA,nB])
    pA=A@nm.T; pB=B@nm.T
    gap=np.maximum(pA.min(0)-pB.max(0), pB.min(0)-pA.max(0))
    return max(0.0,-gap.min())

def _overlap_penalty(v):
    s=0.0
    for i in range(N):
        for j in range(i+1,N):
            d=_overlap_depth(v[i],v[j]); s+=d*d
    return s

def _Lc_soft(v, phi, t):
    m=_inward_normals(phi)
    vals=-np.einsum('kc,nvc->nkv', m, v)
    return (2.0/np.sqrt(3.0))*(t*np.log(np.mean(np.exp(vals/t))))

def _Lc_exact(v, phi):
    m=_inward_normals(phi)
    vals=-np.einsum('kc,nvc->nkv', m, v)
    return (2.0/np.sqrt(3.0))*vals.max()

W=1000.0
def _obj(p, t):
    v=_verts(p); phi=p[-1]
    return _Lc_soft(v,phi,t) + W*_overlap_penalty(v)

def construct():
    t0=time.time(); DEAD=t0+135.0
    try:
        from scipy.optimize import minimize
        have_scipy=True
    except Exception:
        have_scipy=False
    s3=np.sqrt(3.0)
    def init(seed, phi0):
        rng=np.random.default_rng(seed)
        pos=[(0.0,0.0)]
        for k in range(6):
            a=k*np.pi/3.0; pos.append((s3*np.cos(a),s3*np.sin(a)))
        for (i,j) in [(1,1),(-1,1),(-1,-1),(1,-1)]:
            pos.append((i*s3+j*s3/2.0, j*1.5))
        p=np.zeros(3*N+1)
        for i in range(N):
            p[3*i]=pos[i][0]+rng.normal(0,0.08)
            p[3*i+1]=pos[i][1]+rng.normal(0,0.08)
            p[3*i+2]=rng.normal(0,0.08)
        p[-1]=phi0
        return p
    best=None; bestval=np.inf
    scales=[0.3,0.1,0.03,0.01]
    nrestart=0
    while time.time()<DEAD-25:
        seed=nrestart*7+1
        phi0=(nrestart%6)*(np.pi/6.0)
        p0=init(seed,phi0)
        p=p0
        if have_scipy:
            for t in scales:
                if time.time()>DEAD-25: break
                try:
                    r=minimize(lambda q:_obj(q,t), p, method='L-BFGS-B',
                               options={'maxiter':250,'ftol':1e-13,'gtol':1e-9})
                    p=r.x
                except Exception:
                    pass
        v=_verts(p); phi=p[-1]
        op=_overlap_penalty(v)
        if op<1e-5:
            Lc=_Lc_exact(v,phi)
            if Lc<bestval:
                bestval=Lc; best=p
        nrestart+=1
    if best is None:
        # fallback valid large
        p0=init(0,0.0)
        return [[float(p0[3*i]),float(p0[3*i+1]),float(np.degrees(p0[3*i+2]))] for i in range(N)],(0.0,0.0),40.0,0.0
    # heavy polish
    if have_scipy and time.time()<DEAD-8:
        for t in [0.01,0.005,0.002]:
            if time.time()>DEAD-8: break
            try:
                r=minimize(lambda q:_obj(q,t), best, method='L-BFGS-B',
                           options={'maxiter':800,'ftol':1e-15,'gtol':1e-11})
                p=r.x
                v=_verts(p); phi=p[-1]; op=_overlap_penalty(v)
                if op<1e-5:
                    Lc=_Lc_exact(v,phi)
                    if Lc<bestval:
                        bestval=Lc; best=p
            except Exception:
                pass
    p=best
    v=_verts(p); phi=p[-1]
    Lc=_Lc_exact(v,phi)
    Lc*=1.000001
    x=p[0:3*N:3]; y=p[1:3*N:3]; th=p[2:3*N:3]
    inner=[[float(x[i]),float(y[i]),float(np.degrees(th[i]))] for i in range(N)]
    return inner,(0.0,0.0),float(Lc),float(np.degrees(phi))
