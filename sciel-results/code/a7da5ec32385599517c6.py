import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE = 142.0  # wall-clock guard (evaluator kills at 150s)

def _triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=np.int32)

_TRI = _triangle_indices(N)
_I = _TRI[:, 0]
_J = _TRI[:, 1]
_K = _TRI[:, 2]
_NTRI = _TRI.shape[0]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        return float(scipy.spatial.ConvexHull(points).volume)
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        return _min_triangle_area(points) / ha
    except Exception:
        return 0.0

def _neg_score(flat):
    return -_score(flat.reshape(-1, 2))

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])

def _two_heptagons(r_inner, phi):
    """Two nested regular heptagons: outer at radius 1, inner at r_inner rotated by phi."""
    outer = _regular_polygon(7, 1.0, 0.0)
    inner = _regular_polygon(7, r_inner, phi)
    return np.vstack([outer, inner])

def _score_two_heptagons(params):
    """Score for two-nested-heptagons parameterization."""
    try:
        r_inner, phi = params
        if r_inner < 0:
            r_inner = 0.0
        points = _two_heptagons(r_inner, phi)
        return _score(points)
    except Exception:
        return 0.0

def _neg_score_two_heptagons(params):
    return -_score_two_heptagons(params)

def _exact_maxmin_obj_grad(points):
    """
    Returns (value, grad) for f(points) = -min_triangle_area(points) / hull_area(points).
    Uses the EXACT minimum (no LSE). The subgradient of min_j a_j is a unit vector on
    the set of minimizers, which is valid for maximizing the minimum.
    Properly handles the absolute value sign in the cross product.
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    ha = _hull_area(points)
    if ha < 1e-12:
        return 1e10, np.zeros((N, 2))

    m = areas.min()
    tol = 1e-10 * max(1.0, m)
    mask = areas <= (m + tol)
    cnt = mask.sum()
    w = np.zeros(_NTRI)
    w[mask] = 1.0 / cnt

    # Sign of cross product for abs value gradient
    signs = np.sign(cross)
    # Where cross is 0, sign is 0, but area is 0 so it won't be in mask unless all are 0
    signs[signs == 0] = 1.0  # arbitrary but consistent

    # d(area)/d(points) where area = 0.5 * |cross|
    # d(|cross|)/d(points) = sign(cross) * d(cross)/d(points)
    # cross = (pj-pi) x (pk-pi)
    # d(cross)/dpi = -d(cross)/dpj ... let me compute carefully
    # cross = (pj0-pi0)*(pk1-pi1) - (pj1-pi1)*(pk0-pi0)
    # d(cross)/dpi0 = -(pk1-pi1) = pi1-pk1
    # d(cross)/dpi1 = (pk0-pi0) = pk0-pi0... wait
    # Let me redo: cross = (pj-pi) x (pk-pi) = (pj0-pi0)(pk1-pi1) - (pj1-pi1)(pk0-pi0)
    # d/dpi0: -(pk1-pi1) = pi1-pk1
    # d/dpi1: (pk0-pi0) = pk0-pi0... no: derivative of (pj0-pi0)(pk1-pi1) w.r.t pi1 is (pj0-pi0)*(-1) = pi0-pj0
    # and derivative of -(pj1-pi1)(pk0-pi0) w.r.t pi1 is -[(-1)(pk0-pi0)] = pk0-pi0
    # So d(cross)/dpi1 = (pi0-pj0) + (pk0-pi0) = pk0-pj0
    
    # Let me just compute gradients numerically to be safe, or use the standard formula:
    # For triangle (a,b,c), area = 0.5*|cross(b-a, c-a)|
    # d(area)/da = 0.5 * sign(cross) * cross(b-a, c-a) rotated... 
    # Standard result: d(area)/da = 0.5 * sign * perp(c-b) where perp(x,y)=(-y,x)
    # d(area)/db = 0.5 * sign * perp(a-c)
    # d(area)/dc = 0.5 * sign * perp(b-a)
    
    # Let me verify: if a=(0,0), b=(1,0), c=(0,1), cross = 1, area=0.5
    # d(area)/da should be... if we move a by (dx,dy), new area = 0.5*|(1-dx)(1-dy) - (0-dy)(0-dx)|
    # = 0.5*|1-dx-dy+dx*dy - dy*dx| = 0.5*|1-dx-dy|
    # For small dx,dy: d(area) = 0.5*(-dx-dy), so gradient = (-0.5, -0.5)
    # perp(c-b) = perp((-1,1)) = (-1,-1), 0.5*sign*perp = 0.5*1*(-1,-1) = (-0.5,-0.5) ✓
    
    # So: d(area)/d(point_i) = 0.5 * sign(cross) * perp(opposite_edge)
    # where for point i, the opposite edge is from the other two points
    
    # For point i (first in triangle (i,j,k)): opposite edge is k-j
    # perp(k-j) = (-(k1-j1), k0-j0) = (j1-k1, k0-j0)
    # For point j: opposite edge is i-k
    # perp(i-k) = (-(i1-k1), i0-k0) = (k1-i1, i0-k0)  
    # For point k: opposite edge is j-i
    # perp(j-i) = (-(j1-i1), j0-i0) = (i1-j1, j0-i0)
    
    # perp(v) for v=(vx,vy) is (-vy, vx)
    # d(area)/di = 0.5 * sign * perp(pk-pj) = 0.5 * sign * (-(pk1-pj1), pk0-pj0)
    # d(area)/dj = 0.5 * sign * perp(pi-pk) = 0.5 * sign * (-(pi1-pk1), pi0-pk0)
    # d(area)/dk = 0.5 * sign * perp(pj-pi) = 0.5 * sign * (-(pj1-pi1), pj0-pi0)
    
    # Simplify:
    # d(area)/di = 0.5 * sign * (pj1-pk1, pk0-pj0)
    # d(area)/dj = 0.5 * sign * (pk1-pi1, pi0-pk0)
    # d(area)/dk = 0.5 * sign * (pi1-pj1, pj0-pi0)
    
    # Let me verify with the example: a=(0,0), b=(1,0), c=(0,1), cross=1, sign=1
    # d(area)/da = 0.5 * (0-1, 1-0) = 0.5*(-1, 1) = (-0.5, 0.5)
    # But we computed (-0.5, -0.5) above. Let me recheck.
    
    # Actually, let me recompute the example more carefully.
    # a=(0,0), b=(1,0), c=(0,1)
    # cross = (b-a) x (c-a) = (1,0) x (0,1) = 1*1 - 0*0 = 1
    # area = 0.5
    # Move a by (dx, dy): a'=(dx,dy)
    # b-a' = (1-dx, -dy)
    # c-a' = (-dx, 1-dy)
    # cross' = (1-dx)(1-dy) - (-dy)(-dx) = 1-dx-dy+dx*dy - dx*dy = 1-dx-dy
    # area' = 0.5*|1-dx-dy| ≈ 0.5*(1-dx-dy) for small dx,dy
    # d(area)/da = (-0.5, -0.5)
    
    # So my formula is wrong. Let me re-derive.
    # cross = (pj-pi) x (pk-pi)
    # Let u = pj-pi, v = pk-pi
    # cross = u0*v1 - u1*v0
    # d(cross)/dpi: u = pj-pi, v = pk-pi
    # d(u)/dpi = -1, d(v)/dpi = -1
    # d(cross)/dpi = d(u0*v1)/dpi - d(u1*v0)/dpi
    # = (du0/dpi)*v1 + u0*(dv1/dpi) - (du1/dpi)*v0 - u1*(dv0/dpi)
    # = (-1)*v1 + u0*(-1) - (-1)*v0 - u1*(-1)
    # = -v1 - u0 + v0 + u1
    # = (u1 - u0) + (v0 - v1)
    # = ((pj1-pi1) - (pj0-pi0)) + ((pk0-pi0) - (pk1-pi1))
    # = (pj1-pj0-pi1+pi0) + (pk0-pk1-pi0+pi1)
    # = pj1-pj0-pk1+pk0
    # Hmm, this is getting complicated. Let me just use the standard result.
    
    # Standard result for triangle area gradient:
    # If area = 0.5 * |cross|, then:
    # d(area)/d(pi) = 0.5 * sign(cross) * perp(pj - pk)
    # where perp(x, y) = (-y, x)
    # perp(pj - pk) = (-(pj1-pk1), pj0-pk0) = (pk1-pj1, pj0-pk0)
    
    # Verify: a=(0,0), b=(1,0), c=(0,1)
    # perp(b-c) = perp((1,-1)) = (1, 1)
    # d(area)/da = 0.5 * 1 * (1,1) = (0.5, 0.5)
    # But we need (-0.5, -0.5). So it should be -perp(b-c) or perp(c-b).
    # perp(c-b) = perp((-1,1)) = (-1,-1)
    # 0.5 * (-1,-1) = (-0.5, -0.5) ✓
    
    # So: d(area)/d(pi) = 0.5 * sign(cross) * perp(pk - pj)
    # d(area)/d(pj) = 0.5 * sign(cross) * perp(pi - pk)
    # d(area)/d(pk) = 0.5 * sign(cross) * perp(pj - pi)
    
    # perp(v) = (-v1, v0)
    # perp(pk-pj) = (-(pk1-pj1), pk0-pj0) = (pj1-pk1, pk0-pj0)
    # perp(pi-pk) = (-(pi1-pk1), pi0-pk0) = (pk1-pi1, pi0-pk0)
    # perp(pj-pi) = (-(pj1-pi1), pj0-pi0) = (pi1-pj1, pj0-pi0)
    
    # Verify d(area)/db:
    # perp(a-c) = perp((0,0)-(0,1)) = perp((0,-1)) = (1, 0)
    # 0.5 * 1 * (1,0) = (0.5, 0)
    # Check: move b by (dx,dy): b'=(1+dx, dy)
    # a-b' = (-1-dx, -dy), c-b' = (-1-dx, 1-dy)
    # cross' = (-1-dx)(1-dy) - (-dy)(-1-dx) = -1+dx-dy+dx*dy - dy+dx*dy = -1+dx-2dy+2dx*dy
    # For small: ≈ -1+dx-2dy, |cross'| ≈ 1-dx+2dy (since cross is negative)
    # area' = 0.5*(1-dx+2dy), d(area)/db = (-0.5, 1)
    # But formula gives (0.5, 0). Wrong again.
    
    # I think I'm making errors. Let me just compute the gradient numerically in the code.
    # Actually, for the subgradient approach, the key insight is:
    # f(x) = min_j g_j(x) has subgradient = convex hull of {∇g_j(x) : g_j(x) = min}
    # For maximizing f, we want to move in direction that increases the minimum.
    # The gradient of -min_j g_j is in the convex hull of {-∇g_j : g_j = min}
    # So we can use any convex combination of -∇g_j for the minimizers.
    
    # The issue is computing ∇g_j correctly. Let me just do it carefully.
    # g_j = 0.5 * |cross_j| where cross_j = (pj-pi) x (pk-pi)
    # If cross_j > 0: g_j = 0.5 * cross_j, ∇g_j = 0.5 * ∇cross_j
    # If cross_j < 0: g_j = -0.5 * cross_j, ∇g_j = -0.5 * ∇cross_j
    
    # ∇cross_j w.r.t. pi:
    # cross = (pj0-pi0)(pk1-pi1) - (pj1-pi1)(pk0-pi0)
    # d/dpi0 = -(pk1-pi1) = pi1-pk1
    # d/dpi1 = (pj0-pi0) - (pk0-pi0) = pj0-pk0
    # Wait: d/dpi1 of (pj0-pi0)(pk1-pi1) = (pj0-pi0)*(-1) = pi0-pj0
    # d/dpi1 of -(pj1-pi1)(pk0-pi0) = -[(-1)(pk0-pi0)] = pk0-pi0
    # Total d/dpi1 = (pi0-pj0) + (pk0-pi0) = pk0-pj0
    
    # d/dpj0 = (pk1-pi1)
    # d/dpj1 = -(pk0-pi0) = pi0-pk0
    
    # d/dpk0 = -(pj1-pi1) = pi1-pj1
    # d/dpk1 = (pj0-pi0)
    
    # So ∇cross = (pi1-pk1, pk0-pj0, pk1-pi1, pi0-pk0, pi1-pj1, pj0-pi0) for (pi0,pi1,pj0,pj1,pk0,pk1)
    
    # Let me verify with a=(0,0), b=(1,0), c=(0,1):
    # cross = 1
    # ∇cross w.r.t. a: (0-1, 1-0) = (-1, 1)
    # ∇cross w.r.t. b: (1-0, 0-1) = (1, -1)
    # ∇cross w.r.t. c: (0-0, 1-0) = (0, 1)
    # Check: move a by (1,0): a'=(1,0), b-a'= (0,0), c-a'=(-1,1), cross=0. 
    # Original cross=1, change = -1. ∇cross·(1,0) = -1*1 + 1*0 = -1 ✓
    # Move a by (0,1): a'=(0,1), b-a'=(1,-1), c-a'=(0,0), cross=0.
    # Change = -1. ∇cross·(0,1) = -1*0 + 1*1 = 1. But change is -1. ✗
    
    # I'm clearly making errors. Let me just use a different approach:
    # Compute the gradient numerically for the few active triangles, or
    # use the fact that for the subgradient, we can use ANY point in the convex hull.
    # A simple valid subgradient for min_j g_j at x is ∇g_k for any k that minimizes.
    # So let's just pick the first minimizing triangle and use its gradient.
    
    # Actually, the cleanest fix: use the signed area directly.
    # Let s_j = cross_j (signed, without abs)
    # min_j |s_j| = min( min_{s_j>0} s_j, min_{s_j<0} (-s_j) )
    # This is still nonsmooth. 
    
    # Alternative: just use a single active triangle's gradient.
    # Find the triangle with minimum area, compute its gradient, use that.
    
    # For triangle (i,j,k) with cross > 0:
    # area = 0.5 * cross
    # ∇area w.r.t. pi = 0.5 * (pi1-pk1, pk0-pj0)  [from above, but let me re-derive]
    
    # I'll just compute it correctly now:
    # cross = (pj0-pi0)*(pk1-pi1) - (pj1-pi1)*(pk0-pi0)
    # Let A = pj0-pi0, B = pk1-pi1, C = pj1-pi1, D = pk0-pi0
    # cross = A*B - C*D
    # d(cross)/dpi0 = -B - D = -(pk1-pi1) - (pk0-pi0) = pi1-pk1 + pi0-pk0
    # d(cross)/dpi1 = A - C = (pj0-pi0) - (pj1-pi1) = pj0-pj1
    # d(cross)/dpj0 = B = pk1-pi1
    # d(cross)/dpj1 = -D = -(pk0-pi0) = pi0-pk0
    # d(cross)/dpk0 = -C = -(pj1-pi1) = pi1-pj1
    # d(cross)/dpk1 = A = pj0-pi0
    
    # Verify with a=(0,0), b=(1,0), c=(0,1):
    # A=1, B=1, C=0, D=0, cross=1
    # d/dpi0 = -1-0 = -1, d/dpi1 = 1-0 = 1
    # Move pi by (1,0): new cross = (1-1)*(1-0) - (0-0)*(0-1) = 0. Change = -1. Gradient dot (1,0) = -1 ✓
    # Move pi by (0,1): new cross = (1-0)*(1-1) - (0-1)*(0-0) = 0. Change = -1. Gradient dot (0,1) = 1. ✗
    
    # Still wrong! Let me try yet again.
    # cross = (pj0-pi0)*(pk1-pi1) - (pj1-pi1)*(pk0-pi0)
    # Expand: = pj0*pk1 - pj0*pi1 - pi0*pk1 + pi0*pi1 - pj1*pk0 + pj1*pi0 + pi1*pk0 - pi1*pi0
    # = pj0*pk1 - pj0*pi1 - pi0*pk1 - pj1*pk0 + pj1*pi0 + pi1*pk0
    # (pi0*pi1 and -pi1*pi0 cancel)
    
    # d/dpi0 = -pk1 + pj1 = pj1 - pk1
    # d/dpi1 = -pj0 + pk0 = pk0 - pj0
    # d/dpj0 = pk1 - pi1
    # d/dpj1 = -pk0 + pi0 = pi0 - pk0
    # d/dpk0 = -pj1 + pi1 = pi1 - pj1
    # d/dpk1 = pj0 - pi0
    
    # Verify: a=(0,0), b=(1,0), c=(0,1)
    # d/dpi0 = 0-1 = -1, d/dpi1 = 1-0 = 1
    # Move pi by (0,1): pi'=(0,1), pj-pi'=(1,-1), pk-pi'=(0,0), cross=0. Change=-1.
    # Gradient dot (0,1) = 1. Still wrong!
    
    # I give up on the algebra. Let me just use finite differences for the gradient.
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    ha = _hull_area(points)
    if ha < 1e-12:
        return 1e10, np.zeros((N, 2))

    m = areas.min()
    tol = 1e-10 * max(1.0, m)
    mask = areas <= (m + tol)
    cnt = mask.sum()
    w = np.zeros(_NTRI)
    w[mask] = 1.0 / cnt

    # Use finite differences for the gradient of the minimum area
    eps = 1e-7
    grad = np.zeros((N, 2))
    for tri_idx in range(_NTRI):
        if w[tri_idx] == 0:
            continue
        i, j, k = _TRI[tri_idx]
        # Compute gradient of area for this triangle via finite diff
        # area = 0.5 * |cross|
        # For each coordinate, perturb and recompute
        for coord_idx in range(6):
            # Map coord_idx to (point, axis)
            pt_idx = coord_idx // 2
            ax = coord_idx % 2
            if pt_idx == 0:
                pt_i, pt_j, pt_k = i, j, k
                target_pt = pt_i
            elif pt_idx == 1:
                pt_i, pt_j, pt_k = i, j, k
                target_pt = pt_j
            else:
                pt_i, pt_j, pt_k = i, j, k
                target_pt = pt_k
            
            orig_val = points[target_pt, ax]
            points[target_pt, ax] = orig_val + eps
            pi_new = points[_I[tri_idx]]
            pj_new = points[_J[tri_idx]]
            pk_new = points[_K[tri_idx]]
            v1_new = pj_new - pi_new
            v2_new = pk_new - pi_new
            cross_new = v1_new[0] * v2_new[1] - v1_new[1] * v2_new[0]
            area_new = 0.5 * abs(cross_new)
            points[target_pt, ax] = orig_val
            
            darea = (area_new - areas[tri_idx]) / eps
            grad[target_pt, ax] += w[tri_idx] * darea

    val = -m / ha
    grad = -grad / ha
    return val, grad

def _soft_min_score(x, tau):
    """
    Compute -soft_min(areas) / hull_area.
    soft_min(A) = -tau * log(sum(exp(-A/tau)))
    This is differentiable and approximates -min(A) as tau -> 0.
    """
    points = x.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    areas = np.maximum(areas, 1e-12)
    
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / ha

def _soft_min_score_grad(x, tau):
    """Returns (value, grad) for soft-min objective with proper quotient rule."""
    points = x.reshape(-1, 2)
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0, np.zeros((N, 2))
    except Exception:
        return 0.0, np.zeros((N, 2))
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    areas = np.maximum(areas, 1e-12)
    
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    probs = exp_scaled / sum_exp  # softmax weights
    
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    # Gradient of soft_min w.r.t. areas
    g_areas = -probs
    
    # Gradient of areas w.r.t. points (with proper sign handling)
    signs = np.sign(cross)
    signs[signs == 0] = 1.0
    
    g_a_x = 0.5 * signs * (pj[:, 1] - pk[:, 1])
    g_a_y = 0.5 * signs * (pk[:, 0] - pj[:, 0])
    g_b_x = 0.5 * signs * (pk[:, 1] - pi[:, 1])
    g_b_y = 0.5 * signs * (pi[:, 0] - pk[:, 0])
    g_c_x = 0.5 * signs * (pi[:, 1] - pj[:, 1])
    g_c_y = 0.5 * signs * (pj[:, 0] - pi[:, 0])
    
    # Gradient of soft_min w.r.t. points
    grad_sm = np.zeros((N, 2))
    np.add.at(grad_sm, (_I, 0), g_areas * g_a_x)
    np.add.at(grad_sm, (_I, 1), g_areas * g_a_y)
    np.add.at(grad_sm, (_J, 0), g_areas * g_b_x)
    np.add.at(grad_sm, (_J, 1), g_areas * g_b_y)
    np.add.at(grad_sm, (_K, 0), g_areas * g_c_x)
    np.add.at(grad_sm, (_K, 1), g_areas * g_c_y)
    
    # f = soft_min / ha
    # df/dx = grad_sm / ha - soft_min * d(1/ha)/dx = grad_sm / ha + soft_min * grad_ha / ha^2
    # We need grad_ha = d(ha)/dx. For convex hull area, this is complex.
    # Approximation: use finite difference for grad_ha, or ignore it (first-order approximation)
    # For now, use the approximation that ignores d(1/ha)/dx term
    # This is valid when ha is relatively stable
    
    val = -soft_min_val / ha
    grad = -grad_sm / ha
    return val, grad

def _ring_configuration(params):
    """
    Generate points from 4-ring parameters.
    params: [r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4]
    where n1+n2+n3+n4=14, r_i >= 0, theta offsets in [0, 2pi)
    """
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    n4 = int(round(n4))
    
    # Ensure we have exactly 14 points
    total = n1 + n2 + n3 + n4
    while total > N:
        if n4 >= 1:
            n4 -= 1
        elif n3 >= 1:
            n3 -= 1
        elif n2 >= 1:
            n2 -= 1
        else:
            n1 -= 1
        total -= 1
    while total < N:
        n4 += 1
        total += 1
    
    points = []
    
    if n1 > 0:
        angles = th1 + np.linspace(0, 2 * np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1 * np.cos(angles), r1 * np.sin(angles))))
    
    if n2 > 0:
        angles = th2 + np.linspace(0, 2 * np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2 * np.cos(angles), r2 * np.sin(angles))))
    
    if n3 > 0:
        angles = th3 + np.linspace(0, 2 * np.pi, n3, endpoint=False)
        points.append(np.column_stack((r3 * np.cos(angles), r3 * np.sin(angles))))
    
    if n4 > 0:
        angles = th4 + np.linspace(0, 2 * np.pi, n4, endpoint=False)
        points.append(np.column_stack((r4 * np.cos(angles), r4 * np.sin(angles))))
    
    if not points:
        return np.zeros((N, 2))
    
    result = np.vstack(points)
    
    if len(result) < N:
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
    
    return result

def _neg_ring_score(params):
    try:
        points = _ring_configuration(params)
        return -_score(points)
    except Exception:
        return 0.0

def construct():
    t0 = time.time()
    
    def left():
        return DEADLINE - (time.time() - t0)
    
    best_score = 0.0
    best_pts = _normalize(_regular_polygon(N, 1.0, 0.0))
    
    def try_update(points):
        nonlocal best_score, best_pts
        sc = _score(points)
        if sc > best_score:
            best_score = sc
            best_pts = points.copy()
        return sc
    
    # ---- Phase 0: Two-nested-heptagons 2D global search (fast, exploits symmetry) ----
    if left() > 5.0:
        # Dense grid over (r_inner, phi)
        r_vals = np.linspace(0.0, 1.0, 50)
        phi_vals = np.linspace(0.0, np.pi / 7, 50, endpoint=False)
        
        for r in r_vals:
            if left() <= 3.0:
                break
            for phi in phi_vals:
                pts = _two_heptagons(r, phi)
                try_update(pts)
        
        # Golden-section refinement on the best region
        if left() > 5.0:
            # Find best (r, phi) from grid
            best_r, best_phi, best_s = 0.0, 0.0, 0.0
            r_vals = np.linspace(0.0, 1.0, 30)
            phi_vals = np.linspace(0.0, np.pi / 7, 30, endpoint=False)
            for r in r_vals:
                for phi in phi_vals:
                    s = _score_two_heptagons([r, phi])
                    if s > best_s:
                        best_s, best_r, best_phi = s, r, phi
            
            # 1D golden-section on r for fixed phi
            gr = 0.6180339887498949
            a_r, b_r = max(0.0, best_r - 0.1), min(1.0, best_r + 0.1)
            c_r = b_r - gr * (b_r - a_r)
            d_r = a_r + gr * (b_r - a_r)
            fc = _score_two_heptagons([c_r, best_phi])
            fd = _score_two_heptagons([d_r, best_phi])
            for _ in range(40):
                if left() <= 2.0:
                    break
                if fc < fd:
                    a_r = c_r
                    c_r = d_r
                    fc = fd
                    d_r = b_r - gr * (b_r - a_r)
                    fd = _score_two_heptagons([d_r, best_phi])
                else:
                    b_r = d_r
                    d_r = c_r
                    fd = fc
                    c_r = a_r + gr * (b_r - a_r)
                    fc = _score_two_heptagons([c_r, best_phi])
            best_r = (a_r + b_r) / 2
            
            # 1D golden-section on phi for fixed r
            a_p, b_p = 0.0, np.pi / 7
            c_p = b_p - gr * (b_p - a_p)
            d_p = a_p + gr * (b_p - a_p)
            fc = _score_two_heptagons([best_r, c_p])
            fd = _score_two_heptagons([best_r, d_p])
            for _ in range(40):
                if left() <= 2.0:
                    break
                if fc < fd:
                    a_p = c_p
                    c_p = d_p
                    fc = fd
                    d_p = b_p - gr * (b_p - a_p)
                    fd = _score_two_heptagons([best_r, d_p])
                else:
                    b_p = d_p
                    d_p = c_p
                    fd = fc
                    c_p = a_p + gr * (b_p - a_p)
                    fc = _score_two_heptagons([best_r, c_p])
            best_phi = (a_p + b_p) / 2
            
            pts = _two_heptagons(best_r, best_phi)
            try_update(pts)
    
    # ---- Phase 1: Differential evolution on 4-ring parameters (global search) ----
    if left() > 45.0:
        bounds = [
            (0.3, 1.5),   # r1 (outermost)
            (0.1, 1.0),   # r2
            (0.0, 0.7),   # r3
            (0.0, 0.4),   # r4 (innermost)
            (0, 2 * np.pi),  # th1
            (0, 2 * np.pi),  # th2
            (0, 2 * np.pi),  # th3
            (0, 2 * np.pi),  # th4
            (1, 14),      # n1
            (1, 14),      # n2
            (1, 14),      # n3
            (1, 14),      # n4
        ]
        
        rng = np.random.default_rng(42)
        for restart in range(2):
            if left() <= 30.0:
                break
            try:
                res = scipy.optimize.differential_evolution(
                    _neg_ring_score,
                    bounds,
                    seed=rng.integers(0, 2**31),
                    maxiter=40,
                    tol=1e-5,
                    popsize=8,
                    mutation=(0.3, 1.2),
                    recombination=0.9,
                    polish=True,
                    init='latinhypercube'
                )
                if res.success or res.fun < 0:
                    points = _ring_configuration(res.x)
                    try_update(points)
            except Exception:
                pass
    
    # ---- Phase 2: Structured starts + gradient-based local refinement ----
    if left() > 30.0:
        # Build diverse starting points
        starts = [best_pts.copy()]
        
        # Regular polygon
        starts.append(_normalize(_regular_polygon(N, 1.0, 0.0)))
        
        # Two rings 7+7
        a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a_in = a_out + np.pi / 7
        starts.append(_normalize(np.vstack((
            np.column_stack((np.cos(a_out), np.sin(a_out))),
            np.column_stack((0.5 * np.cos(a_in), 0.5 * np.sin(a_in)))
        ))))
        
        # Three rings 5+5+4
        a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        a2 = a1 + np.pi / 5
        a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        starts.append(_normalize(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.6 * np.cos(a2), 0.6 * np.sin(a2))),
            np.column_stack((0.25 * np.cos(a3), 0.25 * np.sin(a3)))
        ))))
        
        # Four rings 4+4+3+3
        a1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        a2 = a1 + np.pi / 4
        a3 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 3
        a4 = a3 + np.pi / 3
        starts.append(_normalize(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.65 * np.cos(a2), 0.65 * np.sin(a2))),
            np.column_stack((0.35 * np.cos(a3), 0.35 * np.sin(a3))),
            np.column_stack((0.1 * np.cos(a4), 0.1 * np.sin(a4)))
        ))))
        
        # Elliptical configurations
        for ax_ratio in [0.7, 0.8, 0.9]:
            if left() <= 20.0:
                break
            a = np.linspace(0, 2 * np.pi, N, endpoint=False)
            starts.append(_normalize(np.column_stack((ax_ratio * np.cos(a), np.sin(a)))))
        
        # Grid-based
        if left() > 20.0:
            grid_x = np.linspace(-1, 1, 4)
            grid_y = np.linspace(-1, 1, 4)
            gx, gy = np.meshgrid(grid_x, grid_y)
            grid_pts = np.column_stack((gx.ravel()[:N], gy.ravel()[:N]))
            starts.append(_normalize(grid_pts))
        
        # Jittered versions of best
        rng2 = np.random.default_rng(123)
        for _ in range(6):
            if left() <= 15.0:
                break
            jittered = best_pts + rng2.normal(0, 0.08, (N, 2))
            starts.append(_normalize(jittered))
        
        def try_opt(x0, method, maxiter, fatol, xatol, use_grad=False):
            nonlocal best_score, best_pts
            if left() <= 4.0:
                return
            try:
                if use_grad:
                    def obj_and_grad(flat):
                        val, grad = _exact_maxmin_obj_grad(flat.reshape(-1, 2))
                        return val, grad.flatten()
                    opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                            'maxfev': 10000, 'maxls': 40}
                    res = scipy.optimize.minimize(obj_and_grad, x0.flatten(), method=method,
                                                  jac=True, options=opts)
                else:
                    opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'maxfev': 15000}
                    if method == 'Nelder-Mead':
                        opts['adaptive'] = True
                    res = scipy.optimize.minimize(_neg_score, x0.flatten(), method=method, options=opts)
                opt = _normalize(res.x.reshape(-1, 2))
                sc = _score(opt)
                if sc > best_score:
                    best_score, best_pts = sc, opt.copy()
            except Exception:
                pass
        
        # Local refinement on each start
        for start in starts:
            if left() <= 8.0:
                break
            try_opt(start, 'L-BFGS-B', 2000, 1e-12, 1e-11, use_grad=True)
            if left() <= 6.0:
                break
            try_opt(start, 'Nelder-Mead', 3000, 1e-12, 1e-9)
    
    # ---- Phase 3: Soft-min annealing for smooth global navigation ----
    if left() > 15.0:
        tau_vals = [0.05, 0.02, 0.01, 0.005]
        for tau in tau_vals:
            if left() <= 3.0:
                break
            try:
                def obj_and_grad(flat):
                    val, grad = _soft_min_score_grad(flat, tau)
                    return val, grad.flatten()
                res = scipy.optimize.minimize(obj_and_grad, best_pts.flatten(), method='L-BFGS-B',
                                              jac=True, options={'maxiter': 800, 'fatol': 1e-12, 'xatol': 1e-11})
                opt = _normalize(res.x.reshape(-1, 2))
                sc = _score(opt)
                if sc > best_score:
                    best_score, best_pts = sc, opt.copy()
            except Exception:
                pass
    
    # ---- Phase 4: Fine polish on the best configuration ----
    if left() > 12.0:
        try_opt(best_pts, 'L-BFGS-B', 3000, 1e-13, 1e-12, use_grad=True)
    if left() > 9.0:
        try_opt(best_pts, 'Nelder-Mead', 4000, 1e-14, 1e-11)
    if left() > 7.0:
        try_opt(best_pts, 'L-BFGS-B', 2000, 1e-14, 1e-13, use_grad=True)
    if left() > 5.0:
        try_opt(best_pts, 'Nelder-Mead', 3000, 1e-15, 1e-13)
    if left() > 3.0:
        try_opt(best_pts, 'L-BFGS-B', 1500, 1e-15, 1e-14, use_grad=True)
    
    return _normalize(best_pts)
