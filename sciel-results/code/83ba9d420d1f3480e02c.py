"""Packing unit hexagons in a regular hexagon via gradient-based optimization."""
import numpy as np
from scipy.optimize import minimize

N = 11

def _hex_vertices(s=1.0, theta=0.0):
    angles = np.pi / 3.0 * np.arange(6) + theta
    return s * np.column_stack([np.cos(angles), np.sin(angles)])

def _outer_face_normals(phi=0.0):
    angles = np.pi / 6.0 + phi + np.pi / 3.0 * np.arange(6)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _compute_containment_s(p, thetas, c, phi):
    apothem_factor = np.sqrt(3.0) / 2.0
    normals = _outer_face_normals(phi)
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        v = _hex_vertices(1.0, thetas[i])
        verts[i] = p[i] + v
    diff = verts - c[np.newaxis, np.newaxis, :]
    dots = np.einsum('nij,kj->nik', diff, normals)
    max_dot = dots.max()
    return max_dot / apothem_factor

def _compute_overlap_violation(p, thetas):
    max_viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(p[i] - p[j])
            viol = 2.0 - d
            if viol > max_viol:
                max_viol = viol
    return max_viol

def _objective(x):
    M = 100.0
    p = np.zeros((N, 2))
    thetas = np.zeros(N)
    for i in range(N):
        p[i] = x[3*i:3*i+2]
        thetas[i] = x[3*i+2]
    c = x[3*N:3*N+2]
    phi = x[3*N+2]
    s = x[3*N+3]
    overlap_v = _compute_overlap_violation(p, thetas)
    containment_s = _compute_containment_s(p, thetas, c, phi)
    penalty = M * max(0.0, overlap_v) + M * max(0.0, containment_s - s)
    return s + penalty

def _gradient(x):
    eps = 1e-6
    grad = np.zeros_like(x)
    f0 = _objective(x)
    for i in range(len(x)):
        x[i] += eps
        f1 = _objective(x)
        x[i] -= eps
        grad[i] = (f1 - f0) / eps
    return grad

def construct():
    best_s = 40.0
    best_x = None
    np.random.seed(42)
    for trial in range(40):
        p = np.zeros((N, 2))
        thetas = np.zeros(N)
        if trial % 2 == 0:
            for i in range(N):
                p[i] = [2.0 * (i - N // 2) + np.random.randn() * 0.1, np.random.randn() * 0.1]
            thetas[:] = np.random.randn(N) * 0.1
        else:
            count = 0
            for ring in range(0, 3):
                if ring == 0:
                    positions = [(0.0, 0.0)]
                else:
                    positions = []
                    for k in range(6):
                        angle = np.pi / 3.0 * k
                        for t in range(ring):
                            pos = (ring * 2.0 * np.cos(angle) + t * 2.0 * np.cos(angle + np.pi/6),
                                   ring * 2.0 * np.sin(angle) + t * 2.0 * np.sin(angle + np.pi/6))
                            positions.append(pos)
                for pos in positions:
                    if count < N:
                        p[count] = [pos[0] + np.random.randn() * 0.05, pos[1] + np.random.randn() * 0.05]
                        thetas[count] = np.random.randn() * 0.05
                        count += 1
                if count >= N:
                    break
        c = np.random.randn(2) * 0.5
        phi = np.random.rand() * np.pi / 3.0
        s_init = 40.0
        x0 = np.zeros(3 * N + 4)
        for i in range(N):
            x0[3*i:3*i+2] = p[i]
            x0[3*i+2] = thetas[i]
        x0[3*N:3*N+2] = c
        x0[3*N+2] = phi
        x0[3*N+3] = s_init
        res = minimize(_objective, x0, method='L-BFGS-B', jac=_gradient,
                       options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-7})
        if res.fun < _objective(x0):
            s_val = res.x[3*N+3]
            if s_val < best_s:
                best_s = s_val
                best_x = res.x.copy()
    if best_x is None:
        inner = [[2.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    p = np.zeros((N, 2))
    thetas = np.zeros(N)
    for i in range(N):
        p[i] = best_x[3*i:3*i+2]
        thetas[i] = best_x[3*i+2]
    c = best_x[3*N:3*N+2]
    phi = best_x[3*N+2]
    s = best_x[3*N+3]
    s_actual = _compute_containment_s(p, thetas, c, phi)
    s = max(s, s_actual)
    thetas_deg = np.degrees(thetas) % 360.0
    phi_deg = np.degrees(phi) % 360.0
    inner = [[float(p[i, 0]), float(p[i, 1]), float(thetas_deg[i])] for i in range(N)]
    return inner, (float(c[0]), float(c[1])), float(s), float(phi_deg)

