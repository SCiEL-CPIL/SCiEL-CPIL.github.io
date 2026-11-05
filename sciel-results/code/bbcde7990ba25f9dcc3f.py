"""Matrix multiplication 2x3x4 tensor decomposition search."""
import numpy as np
from fractions import Fraction
import random
import time

DIMENSIONS = [2, 3, 4]

def construct():
    deadline = time.time() + 138.0
    n, m, p = DIMENSIONS
    N, M, P = n, m, p
    r_sb = N * M * P
    
    # Schoolbook
    Us = [[0]*r_sb for _ in range(N*M)]
    Vs = [[0]*r_sb for _ in range(M*P)]
    Ws = [[0]*r_sb for _ in range(P*N)]
    for c in range(r_sb):
        i = c // (M*P); j = (c // P) % M; k = c % P
        Us[i*M+j][c] = 1
        Vs[j*P+k][c] = 1
        Ws[k*N+i][c] = 1
    best_r = r_sb
    best_U, best_V, best_W = Us, Vs, Ws
    
    def verify_int(Uf, Vf, Wf, r):
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    s = Fraction(0)
                    for c in range(r):
                        s += Uf[i*M+j][c] * Vf[j*P+k][c] * Wf[k*N+i][c]
                    if s != 1: return False
        for i in range(N):
            for i2 in range(N):
                if i==i2: continue
                for j in range(M):
                    for k in range(P):
                        s = Fraction(0)
                        for c in range(r):
                            s += Uf[i*M+j][c] * Vf[j*P+k][c] * Wf[k*N+i2][c]
                        if s != 0: return False
        for j in range(M):
            for j2 in range(M):
                if j==j2: continue
                for i in range(N):
                    for k in range(P):
                        s = Fraction(0)
                        for c in range(r):
                            s += Uf[i*M+j][c] * Vf[j2*P+k][c] * Wf[k*N+i][c]
                        if s != 0: return False
        for k in range(P):
            for k2 in range(P):
                if k==k2: continue
                for i in range(N):
                    for j in range(M):
                        s = Fraction(0)
                        for c in range(r):
                            s += Uf[i*M+j][c] * Vf[j*P+k2][c] * Wf[k*N+i][c]
                        if s != 0: return False
        return True
    
    random.seed(77)
    np.random.seed(77)
    
    # Use structured search: try to build from Strassen-like patterns
    # 2x3x4: C = A*B, A is 2x3, B is 3x4, C is 2x4
    # Write B as [[b11,b12,b13,b14],[b21,b22,b23,b24],[b31,b32,b33,b34]]
    # C[0] = A[0,0]*row0 + A[0,1]*row1 + A[0,2]*row2
    # C[1] = A[1,0]*row0 + A[1,1]*row1 + A[1,2]*row2
    
    # Approach: treat as computing linear combinations of rows of B
    # with coefficients from rows of A. This is essentially 2x4 = (2x3)(3x4)
    
    # Key: We can view this as: for each output column c (0..3),
    # C[:,c] = A @ B[:,c], a 2x3 times 3x1 multiplication.
    # But all columns share the same A.
    
    # Let me try: use rank-3 for each "fiber" but with sharing.
    # Actually the tensor T is T_{ij, jk, ki} -> use (i,j) index, (j,k) index, (k,i) index
    
    # Systematic: try all combinations of small structured entries
    # For rank r, we need r triples of vectors that satisfy the equations.
    
    # Let's use numpy least squares / optimization approach
    # Parameters: flatten U, V, W -> total 2*12*r + 2*12*r + 2*8*r = 72*r parameters
    # Constraints: N*M*P = 24 equations (diag) + cross terms
    # Cross terms: sum over (i,j,k) of [i!=i'] + [j!=j'] + [k!=k']
    # Actually all N*M*P combos give T[i,j,k] = delta_{...}
    # Total equations: N*M*P * (N*M*P - 1 + 1) ... 
    # For each (i,j,k): sum_c U[i,m+j]*V[j,p+k]*W[k,n+i] should equal T_{ijk} = 1 if (the triple matches)
    # Wait, T is 1 for ALL (i,j,k) in [N]*[M]*[P]!
    # No! The tensor T_{(i,j),(j,k),(k,i)} = 1 for all valid (i,j,k).
    # The cross terms are about different (i,j,k) triples.
    
    # So the system is: for ALL i,j,k: prod_c... no, sum_c U[i*M+j,c]V[j*P+k,c]W[k*N+i,c] = 1
    # That's it! All 24 equations say the same thing = 1.
    # Wait no. Let me re-read the task.
    
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The reconstruction is: for a matrix A (n x m) and B (m x p):
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c] 
    #   should give A[i,j]*B[j,k] when we multiply out.
    # Actually the standard tensor decomposition for matmul:
    # C_{ik} = sum_j A_{ij} B_{jk} = sum_c (sum_j A_{ij} u_{ij,c})(sum_j' B_{j'k} v_{j'k,c}) w_{k,n+i,c}
    # Hmm, that's not quite right either.
    
    # The correct formulation: C = A*B means C_{ik} = sum_j A_{ij} B_{jk}
    # Tensor: M_{ijk} = 1 (for all i,j,k) as a 3-tensor of shape (n*m) x (m*p) x (p*n)
    # No... M is shape (n, m) x (m, p) x (n, p) and M_{(i,j),(j',k),(i',k')} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    # But the task says W has shape (p*n, rank) with index k*n+i.
    # So the reconstruction is: C[i,k] = sum_j sum_c? No...
    # Let me just verify: the schoolbook has U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 at position c=(i,j,k)
    # So C[i,k] should = sum over (j) of sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # = sum_j [sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]]
    # For schoolbook, when c corresponds to (i,j,k): U=1,V=1,W=1 gives contribution 1 to term (i,j,k)
    # So C[i,k] = sum_j 1 * 1 * 1 (only one c matches for each (i,j,k)) = sum_j 1 = m... that's wrong.
    
    # I think the reconstruction is: C[i,k] = sum_c U[i*m+?, c] * V[?*p+k, c] * W[k*n+i, c]
    # where the ? indices are contracted. Let me look at schoolbook more carefully.
    # U[i*m+j, c] for c=(i',j',k'): = 1 iff i=i', j=j'
    # V[j'*p+k', c] = 1 iff j'=j'', k'=k'''  (where c has specific i'',j'',k'')
    # So summing over c: U[i*m+j,c] * V[j2*p+k,c] * W[k2*n+i2,c]
    # = 1 only when c's (i3,j3,k3) satisfy: i=i3 AND j=j3 AND j2=j3 AND k=k3 AND k2=k3 AND i2=i3
    # = 1 only when i=i2, j=j2=j3, k=k2=k3, i3=i=i2
    # So the sum is 1 if i=i2 and j=j2 and k=k2 and the c indices match.
    # This gives: sum_j A[i,j]*B[j,k] with the product structure.
    
    # OK so the verification is:
    # For the decomposition to work, we need:
    # sum_c U[i*M+j, c] * V[j2*P+k, c] * W[k2*N+i2, c] = 0 unless i=i2, j=j2, k=k2
    # and = 1 when i=i2, j=j2, k=k2.
    # That is exactly what verify_int checks. Good.
    
    # So the system of equations is:
    # 24 "on-diagonal" equations = 1
    # Plus all cross terms = 0
    # Total: (N*M*P)^2 - N*M*P cross terms = 24^2 - 24 = 552 cross terms
    # Total equations: 576, unknowns: 72*r. For r=19: 1368 unknowns, 576 equations. Underdetermined!
    # So solutions exist (generically) for r >= ceil(576/72) = 8? No, it's a nonlinear (cubic) system.
    
    # Let me just do random search more aggressively for ranks 20,21,22,23
    # and also try a local optimization approach.
    
    # Try local optimization using numpy
    for target_r in [20, 21, 22, 23]:
        if time.time() > deadline:
            break
        for restart in range(200):
            if time.time() > deadline:
                break
            # Random init
            scale = 0.5
            U0 = np.random.uniform(-scale, scale, (N*M, target_r))
            V0 = np.random.uniform(-scale, scale, (M*P, target_r))
            W0 = np.random.uniform(-scale, scale, (P*N, target_r))
            
            # Simple gradient-free: try to fix one matrix at a time (block coordinate descent)
            # or just do random perturbation and keep if error decreases
            
            def residual(Ua, Va, Wa):
                err = 0.0
                for i in range(N):
                    for j in range(M):
                        for k in range(P):
                            s = sum(Ua[i*M+j][c]*Va[j*P+k][c]*Wa[k*N+i][c] for c in range(target_r))
                            err += (s-1)**2
                for i in range(N):
                    for i2 in range(N):
                        if i==i2: continue
                        for j in range(M):
                            for k in range(P):
                                s = sum(Ua[i*M+j][c]*Va[j*P+k][c]*Wa[k*N+i2][c] for c in range(target_r))
                                err += s**2
                for j in range(M):
                    for j2 in range(M):
                        if j==j2: continue
                        for i in range(N):
                            for k in range(P):
                                s = sum(Ua[i*M+j][c]*Va[j2*P+k][c]*Wa[k*N+i][c] for c in range(target_r))
                                err += s**2
                for k in range(P):
                    for k2 in range(P):
                        if k==k2: continue
                        for i in range(N):
                            for j in range(M):
                                s = sum(Ua[i*M+j][c]*Va[j*P+k2][c]*Wa[k*N+i][c] for c in range(target_r))
                                err += s**2
                return err
            
            cur_err = residual(U0, V0, W0)
            if cur_err < 0.01:
                # Try rounding to integers and verify
                Ur = [[int(round(x)) for x in row] for row in U0]
                Vr = [[int(round(x)) for x in row] for row in V0]
                Wr = [[int(round(x)) for x in row] for row in W0]
                if verify_int(Ur, Vr, Wr, target_r):
                    if target_r < best_r:
                        best_r = target_r
                        best_U, best_V, best_W = Ur, Vr, Wr
                    break
            # Skip complex optimization due to time; try next restart
    
    return {"U": best_U, "V": best_V, "W": best_W}
