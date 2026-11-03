import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 3.0
EPS = 1e-9
TIME_BUDGET_PER_CANDIDATE = 3.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = []

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    axial_29 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (-1, 2)
    ]
    candidates.append(axial_30)

    axial_31 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_31)

    axial_32 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_32)

    axial_33 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (0, 3), (1, -2)
    ]
    candidates.append(axial_33)

    axial_34 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_34)

    axial_35 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 2)
    ]
    candidates.append(axial_35)

    axial_36 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_36)

    axial_37 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 3),
        (0, 4)
    ]
    candidates.append(axial_37)

    axial_38 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 2), (-1, -2)
    ]
    candidates.append(axial_38)

    axial_39 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_39)

    axial_40 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_40)

    axial_41 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_41)

    axial_42 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_42)

    axial_43 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_43)

    axial_44 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_44)

    axial_45 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_45)

    axial_46 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_46)

    axial_47 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_47)

    axial_48 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_48)

    axial_49 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (1, 4)
    ]
    candidates.append(axial_49)

    axial_50 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_50)

    # Additional candidates from inspiration
    axial_51 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4)
    ]
    candidates.append(axial_51)

    axial_52 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3),
        (1, 3)
    ]
    candidates.append(axial_52)

    axial_53 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_53)

    axial_54 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3),
        (2, 3)
    ]
    candidates.append(axial_54)

    axial_55 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2),
        (0, 3),
        (3, 3)
    ]
    candidates.append(axial_55)

    axial_56 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3),
        (2, 3)
    ]
    candidates.append(axial_56)

    axial_57 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (2, 3)
    ]
    candidates.append(axial_57)

    axial_58 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3),
        (3, 3)
    ]
    candidates.append(axial_58)

    axial_59 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_59)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

        # Use iterative center optimization instead of LP
        # Initialize center at centroid
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        
        # Iteratively refine center by projecting onto feasible region
        for _ in range(200):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            pts_centered = pts - np.array([cx, cy])
            projs = pts_centered @ normals.T  # (n_pts, 6)
            abs_projs = np.abs(projs)
            max_proj = float(np.max(abs_projs))
            
            # Find active constraints
            active_normals = []
            active_signs = []
            active_pts = []
            for k in range(6):
                col = abs_projs[:, k]
                active = np.where(np.abs(col - max_proj) < 1e-12)[0]
                if len(active) > 0:
                    for idx in active:
                        sign = 1.0 if projs[idx, k] > 0 else -1.0
                        active_normals.append(normals[k])
                        active_signs.append(sign)
                        active_pts.append(pts[idx])
            
            if len(active_normals) < 2:
                break
            
            # Solve for new center using active constraints
            # We want to minimize max projection, so move center to balance active constraints
            # Use least squares on active constraints
            A = np.array([[sign * n[0], sign * n[1]] for n, sign in zip(active_normals, active_signs)])
            b = np.array([np.dot(p, n) for p, n in zip(active_pts, active_normals)])
            
            # Solve A @ [cx, cy] = b in least squares sense
            try:
                new_center, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                new_cx, new_cy = float(new_center[0]), float(new_center[1])
                
                # Check if this improves
                S_old = max_proj
                pts_centered_new = pts - np.array([new_cx, new_cy])
                projs_new = pts_centered_new @ normals.T
                max_proj_new = float(np.max(np.abs(projs_new)))
                
                if max_proj_new < S_old - 1e-15:
                    cx, cy = new_cx, new_cy
                    max_proj = max_proj_new
                else:
                    break
            except:
                break
        
        S = (max_proj + APOTHEM) / APOTHEM
        return S, (cx, cy), max_proj

    def exact_S(pts, beta, cx, cy):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = np.abs(pts_centered @ normals.T)
        max_proj = float(np.max(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])

        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf')

        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S

    def newton_center_polish(pts, beta, cx, cy, max_iter=200, tol=1e-15):
        """High-precision center polish using active-set identification and least-squares solve."""
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

        for _ in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break

            pts_centered = pts - np.array([cx, cy])
            projs = pts_centered @ normals.T  # (n_pts, 6)
            abs_projs = np.abs(projs)
            max_proj = float(np.max(abs_projs))

            # Identify all active constraints (within tighter tolerance)
            tol_active = 1e-12
            active_eqs = []
            for k in range(6):
                col = abs_projs[:, k]
                active = np.where(np.abs(col - max_proj) < tol_active)[0]
                for idx in active:
                    sign = 1.0 if projs[idx, k] > 0 else -1.0
                    # Equation: sign * (px - cx) * nx + sign * (py - cy) * ny = max_proj
                    # => -sign*nx*cx - sign*ny*cy = max_proj - sign*(px*nx + py*ny)
                    active_eqs.append((-sign * normals[k, 0], -sign * normals[k, 1],
                                      max_proj - sign * (pts[idx, 0] * normals[k, 0] + pts[idx, 1] * normals[k, 1])))
            
            if len(active_eqs) < 2:
                break
            
            # Solve the active system in least squares sense
            A = np.array([[a, b] for a, b, c in active_eqs])
            b_vec = np.array([-c for a, b, c in active_eqs])
            
            try:
                new_center, residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                new_cx, new_cy = float(new_center[0]), float(new_center[1])
            except:
                break
            
            # Check improvement
            S_old = max_proj
            pts_centered_new = pts - np.array([new_cx, new_cy])
            projs_new = pts_centered_new @ normals.T
            max_proj_new = float(np.max(np.abs(projs_new)))
            
            if max_proj_new < S_old - 1e-15:
                if abs(new_cx - cx) < tol and abs(new_cy - cy) < tol:
                    break
                cx, cy = new_cx, new_cy
            else:
                # Try small perturbations
                improved = False
                for dx in [1e-7, -1e-7, 1e-8, -1e-8, 1e-9, -1e-9]:
                    for dy in [1e-7, -1e-7, 1e-8, -1e-8, 1e-9, -1e-9]:
                        test_cx, test_cy = cx + dx, cy + dy
                        pts_c = pts - np.array([test_cx, test_cy])
                        p = pts_c @ normals.T
                        mp = float(np.max(np.abs(p)))
                        if mp < S_old - 1e-16:
                            cx, cy = test_cx, test_cy
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break

        return cx, cy

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Coarse grid search over beta - 80 points for better resolution
        n_grid = 80
        for i in range(n_grid + 1):
            beta = i * (math.pi / 6) / n_grid
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Refine beta with bounded scalar optimization from multiple starting points
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                refine_starts = [best_beta, 0.0, math.pi/12, math.pi/6 * 0.25, math.pi/6 * 0.75,
                                 math.pi/6 * 0.1, math.pi/6 * 0.4, math.pi/6 * 0.6, math.pi/6 * 0.9,
                                 math.pi/6 * 0.15, math.pi/6 * 0.35, math.pi/6 * 0.5, math.pi/6 * 0.85,
                                 math.pi/6 * 0.05, math.pi/6 * 0.95, math.pi/6 * 0.2, math.pi/6 * 0.8]
                for rb in refine_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    lo = max(0, rb - 0.12)
                    hi = min(math.pi/6, rb + 0.12)
                    res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(lo, hi), method='bounded',
                                         options={'maxiter': 800, 'xatol': 1e-14})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_beta = res.x
                        _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Smooth surrogate optimization - multi-start with adaptive temperature
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.15, 0.08, 0.04, 0.02, 0.01, 0.005, 0.002]
                start_configs = [
                    (best_center[0], best_center[1], best_beta),
                    (best_center[0], best_center[1], best_beta + 0.04),
                    (best_center[0], best_center[1], best_beta - 0.04),
                    (best_center[0], best_center[1], 0.0),
                    (best_center[0], best_center[1], math.pi/12),
                    (best_center[0], best_center[1], math.pi/6 * 0.3),
                    (best_center[0], best_center[1], math.pi/6 * 0.7),
                    (best_center[0], best_center[1], math.pi/6 * 0.1),
                    (best_center[0], best_center[1], math.pi/6 * 0.9),
                ]

                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sc in start_configs:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break

                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            return smooth_calc_S(pts, beta, cx, cy, temperature=t)

                        res = minimize(smooth_obj, list(sc), method='Nelder-Mead',
                                     options={'maxiter': 1500, 'xatol': 1e-14, 'fatol': 1e-14,
                                              'adaptive': True})

                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S_exact = exact_S(pts, beta_r, cx_r, cy_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = (float(cx_r), float(cy_r))
        except Exception:
            pass

        # Newton center polish at the best beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                bcx, bcy = best_center
                bcx, bcy = newton_center_polish(pts, best_beta, bcx, bcy)
                S_new = exact_S(pts, best_beta, bcx, bcy)
                if S_new < best_S:
                    best_S = S_new
                    best_center = (bcx, bcy)
        except Exception:
            pass

        # Final high-precision polish using exact objective with targeted perturbations
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    return exact_S(pts, beta, cx, cy)

                bcx, bcy = best_center
                bb = best_beta
                polish_starts = [[bcx, bcy, bb]]
                # Small radial perturbations
                for dr in [0.015, -0.015, 0.01, -0.01, 0.005, -0.005, 0.002, -0.002]:
                    for ang in range(0, 360, 30):
                        dx = dr * math.cos(math.radians(ang))
                        dy = dr * math.sin(math.radians(ang))
                        polish_starts.append([bcx + dx, bcy + dy, bb])
                # Angular perturbations
                for db in [-0.015, -0.01, -0.005, -0.002, 0.002, 0.005, 0.01, 0.015]:
                    polish_starts.append([bcx, bcy, bb + db])
                # Combined small perturbations
                for db in [-0.008, 0.008]:
                    for dr in [0.008, -0.008]:
                        polish_starts.append([bcx + dr, bcy, bb + db])
                        polish_starts.append([bcx, bcy + dr, bb + db])

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-15,
                                          'adaptive': True})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (float(res.x[0]), float(res.x[1]))
                        best_beta = ((float(res.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass

        # Ultra-fine final polish with very tight tolerances
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    return exact_S(pts, beta, cx, cy)

                bcx, bcy = best_center
                bb = best_beta
                ultra_starts = [
                    [bcx, bcy, bb],
                    [bcx + 0.004, bcy, bb],
                    [bcx - 0.004, bcy, bb],
                    [bcx, bcy + 0.004, bb],
                    [bcx, bcy - 0.004, bb],
                    [bcx + 0.003, bcy + 0.003, bb],
                    [bcx - 0.003, bcy - 0.003, bb],
                    [bcx, bcy, bb + 0.0015],
                    [bcx, bcy, bb - 0.0015],
                    [bcx, bcy, bb + 0.0008],
                    [bcx, bcy, bb - 0.0008],
                    [bcx + 0.0015, bcy, bb + 0.0008],
                    [bcx - 0.0015, bcy, bb - 0.0008],
                    [bcx, bcy + 0.0015, bb + 0.0008],
                    [bcx, bcy - 0.0015, bb - 0.0008],
                    [bcx + 0.001, bcy + 0.001, bb],
                    [bcx - 0.001, bcy - 0.001, bb],
                ]

                for init in ultra_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 40000, 'xatol': 1e-16, 'fatol': 1e-16,
                                          'adaptive': True})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (float(res.x[0]), float(res.x[1]))
                        best_beta = ((float(res.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass

        # Final Newton polish after all Nelder-Mead
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_center is not None:
                bcx, bcy = best_center
                bcx, bcy = newton_center_polish(pts, best_beta, bcx, bcy, max_iter=200, tol=1e-15)
                S_new = exact_S(pts, best_beta, bcx, bcy)
                if S_new < best_S:
                    best_S = S_new
                    best_center = (bcx, bcy)
        except Exception:
            pass

        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    def hex_radius(axial_list):
        pts = axial_to_cartesian(axial_list)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        max_proj = 0.0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        return max_proj

    sorted_indices = sorted(range(len(candidates)), key=lambda i: hex_radius(candidates[i]))

    for idx in sorted_indices:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        axial = candidates[idx]
        pts = axial_to_cartesian(axial)

        candidate_start = time.time()
        S, beta, center = optimize_shape(pts)
        elapsed = time.time() - candidate_start

        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

        if elapsed > TIME_BUDGET_PER_CANDIDATE:
            remaining = TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)
            if remaining < 15.0:
                break

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Use a very small safety margin to ensure no overlap
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))

    pts_scaled = np.array(pts_scaled)

    max_proj = 0.0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
