# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys

_START = time.time()
_DEADLINE = 600.0  # safety margin before 630s kill

def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(ang), cy + d * np.sin(ang)])

def _outer_vertices(outer):
    cx, cy, a = outer
    return _hex_vertices(cx, cy, 1.0, a)  # placeholder d=1, scaled below

def _separation_ok(inner_verts, outer_verts):
    """Check all inner hexagons pairwise separated and inside outer.
    inner_verts: list of (6,2) arrays. outer_verts: (6,2) array."""
    # Pairwise separation via SAT
    m = len(inner_verts)
    for i in range(m):
        vi = inner_verts[i]
        for j in range(i + 1, m):
            vj = inner_verts[j]
            # edges of both
            edges = []
            for v in (vi, vj):
                e = np.roll(v, -1, axis=0) - v
                edges.append(e)
            ok = False
            for e in edges:
                # normal
                nrm = np.column_stack([-e[:, 1], e[:, 0]])
                nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-15)
                pi = vi @ nrm.T
                pj = vj @ nrm.T
                # intervals
                if pi.max() <= pj.min() + 1e-6 or pj.max() <= pi.min() + 1e-6:
                    ok = True
                    break
            if not ok:
                return False
    # inside outer: for every outer edge, cross(p2-p1, v-p1) >= -1e-6
    for vi in inner_verts:
        for k in range(6):
            p1 = outer_verts[k]
            p2 = outer_verts[(k + 1) % 6]
            e = p2 - p1
            for v in vi:
                cross = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross < -1e-6:
                    return False
    return True

def _min_side_for_centers(centers, outer_angle=0.0):
    """Given inner centers (12,2) with unit hexagons at angle 0, find minimal
    outer side length s (outer centered at origin) such that all fit."""
    # All inner hexagons have circumradius 1. The outer hexagon must contain all
    # their vertices. We want minimal s.
    # Outer hexagon centered at origin with side s, angle outer_angle.
    # A point p is inside outer hexagon of side s iff for each of 6 edge half-planes
    # the projection onto the outward normal <= s * apothem = s * sqrt(3)/2.
    # Outward normals of a regular hexagon with circumradius R=s, angle a:
    # vertices at angles a+60i, edge normals at angles a+60i+30.
    a = np.deg2rad(outer_angle)
    nrm_ang = a + np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(nrm_ang), np.sin(nrm_ang)])
    # gather all inner vertices
    allv = []
    for (cx, cy) in centers:
        allv.append(_hex_vertices(cx, cy, 1.0, 0.0))
    allv = np.vstack(allv)
    proj = allv @ normals.T  # (N,6)
    maxproj = proj.max(axis=0)  # (6,)
    # need maxproj <= s * sqrt(3)/2  => s >= maxproj * 2/sqrt(3)
    s = (maxproj * 2.0 / np.sqrt(3.0)).max()
    return s

def _initial_centers():
    """Compact flower + extras arrangement."""
    r = np.sqrt(3.0)  # center-to-center for adjacent unit hexes
    dirs = np.deg2rad(60.0 * np.arange(6))
    dvec = np.column_stack([np.cos(dirs), np.sin(dirs)])
    centers = np.zeros((12, 2))
    centers[0] = [0, 0]
    for i in range(6):
        centers[1 + i] = r * dvec[i]
    # 5 more: place at the 6 "second ring" positions that are gaps, pick 5
    # second ring positions are at distance 2r in directions 30,90,... and r*sqrt(3) at 0,60..
    # Actually the 6 ring positions already used. Add positions at 2r in 6 dirs (offset 30 deg)
    dirs2 = np.deg2rad(30.0 + 60.0 * np.arange(6))
    dvec2 = np.column_stack([np.cos(dirs2), np.sin(dirs2)])
    # choose 5 of the 6 to keep symmetry-ish
    for i in range(5):
        centers[7 + i] = 2.0 * r * dvec2[i]
    return centers

def hexagon_packing_12():
    n = 12
    centers = _initial_centers()
    best_centers = centers.copy()
    best_s = _min_side_for_centers(best_centers)
    best_outer = np.array([0.0, 0.0, 0.0])

    def objective(c):
        c = c.reshape(12, 2)
        return _min_side_for_centers(c)

    # Nelder-Mead optimization on 24 variables
    x0 = centers.ravel()
    best_val = best_s
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-7, 'fatol': 1e-9,
                                'adaptive': True})
        if time.time() > _DEADLINE - 30:
            raise RuntimeError("deadline")
        if res.fun < best_val:
            best_val = res.fun
            best_centers = res.x.reshape(12, 2)
    except Exception:
        pass

    # If still have time, try a few random restarts with different initial arrangements
    if time.time() < _DEADLINE - 60:
        rng = np.random.default_rng(0)
        for restart in range(3):
            if time.time() > _DEADLINE - 60:
                break
            # perturb initial
            c = _initial_centers() + rng.normal(0, 0.3, size=(12, 2))
            try:
                res = minimize(objective, c.ravel(), method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9,
                                        'adaptive': True})
                if time.time() > _DEADLINE - 60:
                    break
                if res.fun < best_val:
                    best_val = res.fun
                    best_centers = res.x.reshape(12, 2)
            except Exception:
                break

    # Build final result: outer hexagon centered at origin, side = best_val, angle 0.
    # But we should center outer at the centroid of inner for tightness? The evaluator
    # shifts centers by outer center, so outer center matters. Use origin.
    s = float(best_val)
    # Ensure s >= sqrt(12)
    if s < np.sqrt(12.0):
        s = float(np.sqrt(12.0))
    outer = np.array([0.0, 0.0, 0.0])

    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = best_centers[i, 0]
        inner[i, 1] = best_centers[i, 1]
        inner[i, 2] = 0.0

    return inner, outer, s

# EVOLVE-BLOCK-END
