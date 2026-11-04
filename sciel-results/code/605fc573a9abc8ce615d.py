import numpy as np
import time
from scipy.optimize import minimize
import math

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = math.radians(angle_deg)
    # Regular hexagon vertices at angles 0,60,...,300 offset by a
    # Use "flat-top" vs "pointy-top": standard pointy-top has vertex at 90 deg.
    # We'll define with first vertex at angle a, step 60 deg.
    verts = []
    for k in range(6):
        th = a + math.radians(60.0 * k)
        verts.append((cx + s * math.cos(th), cy + s * math.sin(th)))
    return np.array(verts)

def poly_area_signed(pts):
    x = pts[:,0]; y = pts[:,1]
    return 0.5*np.sum(x*np.roll(y,-1) - np.roll(x,-1)*y)

def poly_contains_point(pts, p):
    """Ray casting. Returns True if p strictly inside (or on edge counts as inside for our tolerance)."""
    x0,y0 = p
    n = len(pts)
    inside = False
    j = n-1
    for i in range(n):
        xi,yi = pts[i]; xj,yj = pts[j]
        if ((yi > y0) != (yj > y0)) and (x0 < (xj-xi)*(y0-yi)/(yj-yi)+xi):
            inside = not inside
        j = i
    return inside

def seg_intersect(p1,p2,p3,p4):
    """Return True if segments p1-p2 and p3-p4 intersect (including touching)."""
    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    def on_seg(a,b,p):
        return min(a[0],b[0])-1e-9 <= p[0] <= max(a[0],b[0])+1e-9 and min(a[1],b[1])-1e-9 <= p[1] <= max(a[1],b[1])+1e-9
    d1 = cross(p3,p4,p1)
    d2 = cross(p3,p4,p2)
    d3 = cross(p1,p2,p3)
    d4 = cross(p1,p2,p4)
    if ((d1>0 and d2<0) or (d1<0 and d2>0)) and ((d3>0 and d4<0) or (d3<0 and d4>0)):
        return True
    if abs(d1)<1e-9 and on_seg(p3,p4,p1): return True
    if abs(d2)<1e-9 and on_seg(p3,p4,p2): return True
    if abs(d3)<1e-9 and on_seg(p1,p2,p3): return True
    if abs(d4)<1e-9 and on_seg(p1,p2,p4): return True
    return False

def polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap (including edge contact counts as overlap for strictness).
    Use Separating Axis Theorem for convex polys."""
    # SAT: for each edge normal, project both polys; if gap > tol, no overlap
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]; p2 = v[(i+1)%len(v)]
            # edge vector
            ex = p2[0]-p1[0]; ey = p2[1]-p1[1]
            # normal
            nx, ny = -ey, ex
            nlen = math.hypot(nx,ny)
            if nlen < 1e-12: continue
            nx/=nlen; ny/=nlen
            # project
            min1 = min(dot(p,nx,ny) for p in v1)
            max1 = max(dot(p,nx,ny) for p in v1)
            min2 = min(dot(p,nx,ny) for p in v2)
            max2 = max(dot(p,nx,ny) for p in v2)
            if max1 < min2 - tol or max2 < min1 - tol:
                return False
    return True

def dot(p, nx, ny):
    return p[0]*nx + p[1]*ny

def min_dist_convex(v1, v2):
    """Minimum distance between two convex polygons (0 if overlapping/touching)."""
    # Approximate by sampling vertices + edge midpoints; good enough for penalty
    best = float('inf')
    for p in list(v1) + [ (v1[i]+v1[(i+1)%len(v1)])/2 for i in range(len(v1)) ]:
        for q in list(v2) + [ (v2[j]+v2[(j+1)%len(v2)])/2 for j in range(len(v2)) ]:
            d = math.hypot(p[0]-q[0], p[1]-q[1])
            if d < best: best = d
    # Also check edge-edge
    for i in range(len(v1)):
        for j in range(len(v2)):
            d = seg_min_dist(v1[i],v1[(i+1)%len(v1)], v2[j],v2[(j+1)%len(v2)])
            if d < best: best = d
    return best

def seg_min_dist(p1,p2,p3,p4):
    # min distance between two segments
    def dist_point_seg(p, a, b):
        ax,ay=a; bx,by=b; px,py=p
        dx,dy=bx-ax,by-ay
        if dx==0 and dy==0:
            return math.hypot(px-ax,py-ay)
        t = ((px-ax)*dx+(py-ay)*dy)/(dx*dx+dy*dy)
        t = max(0,min(1,t))
        cx,cy = ax+t*dx, ay+t*dy
        return math.hypot(px-cx,py-cy)
    # check endpoints
    d = min(dist_point_seg(p1,p3,p4), dist_point_seg(p2,p3,p4),
            dist_point_seg(p3,p1,p2), dist_point_seg(p4,p1,p2))
    if seg_intersect(p1,p2,p3,p4):
        return 0.0
    return d

def outer_hex_side_to_contain(inner_pts_list, outer_center, outer_angle_deg):
    """Given outer hexagon center and angle, find minimum side length such that all inner vertices are inside.
    Outer hexagon is regular; we need side s such that all points lie within.
    For a regular hexagon with 'pointy' orientation, the width in direction theta is s*(1+cos?) ...
    Simpler: for each inner point, compute required s such that point is inside, take max.
    A point is inside regular hexagon (center c, side s, angle a) iff for each of 6 edge normals,
    the projection of (p - c) onto normal <= s * apothem_factor.
    Apothem = s * sqrt(3)/2.
    Edge normals at angles a + 30 + 60*k (for flat) or a + 60*k (for pointy). We'll match vertex definition.
    """
    a = math.radians(outer_angle_deg)
    # vertices at a + 60k. Edge between vertex k and k+1 has normal at a + 60k + 30 deg.
    max_s = 0.0
    apothem_factor = math.sqrt(3.0)/2.0
    for (cx,cy,ang) in inner_pts_list:
        # we need to consider all vertices of this inner hexagon
        verts = hex_vertices(cx,cy,1.0,ang)
        for p in verts:
            dx = p[0]-outer_center[0]; dy = p[1]-outer_center[1]
            for k in range(6):
                th = a + math.radians(30.0 + 60.0*k)
                nx, ny = math.cos(th), math.sin(th)
                proj = dx*nx + dy*ny
                # need proj <= s * apothem
                if proj > max_s * apothem_factor:
                    s_req = proj / apothem_factor
                    if s_req > max_s: max_s = s_req
    return max_s

def evaluate_config(positions, angles, outer_center, outer_angle):
    """Return outer side length if valid, else inf."""
    inner_verts = [hex_vertices(cx,cy,1.0,ang) for (cx,cy,ang) in positions]
    # check pairwise non-overlap
    for i in range(len(inner_verts)):
        for j in range(i+1, len(inner_verts)):
            if polygons_overlap(inner_verts[i], inner_verts[j], tol=-1e-9):
                # allow tiny negative tolerance for touching
                # recheck with strict
                if polygons_overlap(inner_verts[i], inner_verts[j], tol=1e-8):
                    return float('inf')
    # check all inside outer
    s = outer_hex_side_to_contain([(p[0],p[1],p[2]) for p in positions], outer_center, outer_angle)
    # verify containment
    outer_verts = hex_vertices(outer_center[0],outer_center[1],s,outer_angle)
    for iv in inner_verts:
        for p in iv:
            if not poly_contains_point(outer_verts, p):
                return float('inf')
    return s

def make_symmetric_templates():
    """Generate several symmetric initial configurations of 12 unit hexagons.
    Returns list of (positions, angles) where positions is list of (x,y,angle_deg)."""
    templates = []
    
    # Template 1: 2 rows of 3, 2 rows of 3, 2 rows of 3 (rectangular-ish)
    # Hex grid spacing: for pointy-top hexes side 1, center-to-center horizontal = sqrt(3), vertical = 1.5
    # But we allow rotation; keep angle 0 for simplicity.
    sp_x = math.sqrt(3.0) + 0.001  # slight gap
    sp_y = 1.5 + 0.001
    pos = []
    # 3x4 grid
    for r in range(4):
        for c in range(3):
            x = (c - 1.0) * sp_x
            y = (r - 1.5) * sp_y
            pos.append((x, y, 0.0))
    templates.append(pos)
    
    # Template 2: hexagonal cluster: center + ring of 6 + ring of 5 (12 total)
    # center
    pos = [(0.0, 0.0, 0.0)]
    r1 = 2.0 + 0.001  # distance to first ring centers (touching: 2*1 = 2 for edge-to-edge? actually center dist = 2*apothem = sqrt(3) for flat? Let's use 2.0 for safety)
    # Actually for two unit hexagons edge-to-edge, center distance = 2 * apothem = sqrt(3) ~ 1.732 if flat sides touch, or 2 if pointy tips? 
    # To be safe use 2.0.
    for k in range(6):
        th = math.radians(60.0*k + 30.0)
        pos.append((r1*math.cos(th), r1*math.sin(th), 0.0))
    # second ring: 5 more at distance ~ 2*r1
    r2 = 2*r1
    for k in range(5):
        th = math.radians(60.0*k + 60.0)
        pos.append((r2*math.cos(th), r2*math.sin(th), 0.0))
    templates.append(pos)
    
    # Template 3: 3 columns of 4, staggered
    pos = []
    for c in range(3):
        for r in range(4):
            x = (c - 1.0) * (sp_x)
            y = (r - 1.5) * sp_y + (0.5*sp_y if c%2==1 else 0.0)
            pos.append((x, y, 0.0))
    templates.append(pos)
    
    # Template 4: two rows of 6
    pos = []
    for r in range(2):
        for c in range(6):
            x = (c - 2.5) * sp_x
            y = (r - 0.5) * sp_y
            pos.append((x, y, 0.0))
    templates.append(pos)
    
    # Template 5: compact 3x4 with slight rotation
    pos = []
    for r in range(4):
        for c in range(3):
            x = (c - 1.0) * sp_x
            y = (r - 1.5) * sp_y
            pos.append((x, y, 15.0))
    templates.append(pos)
    
    return templates

def optimize_template(pos0, outer_center0, outer_angle0, time_budget):
    """Optimize a template using scipy minimize on (positions, outer_center, outer_angle, outer_side).
    Objective: outer_side + penalty for overlaps and containment violations."""
    n = len(pos0)
    # variables: for each hex: x,y,angle (3n), outer: cx,cy,angle (3), outer_side (1) => 3n+4
    x0 = []
    for (x,y,a) in pos0:
        x0 += [x,y,a]
    x0 += [outer_center0[0], outer_center0[1], outer_angle0, 20.0]  # initial outer side guess
    
    def objective(vars):
        if time.time() > DEADLINE:
            return 1e10
        pos = []
        for i in range(n):
            pos.append((vars[3*i], vars[3*i+1], vars[3*i+2]))
        ocx, ocy, oang = vars[3*n], vars[3*n+1], vars[3*n+2]
        s = vars[3*n+3]
        if s < 0.5:
            return 1e10
        # compute actual required outer side
        s_req = outer_hex_side_to_contain(pos, (ocx,ocy), oang)
        s_eff = max(s, s_req)
        # overlap penalty
        pen = 0.0
        inner_verts = [hex_vertices(cx,cy,1.0,ang) for (cx,cy,ang) in pos]
        for i in range(n):
            for j in range(i+1, n):
                # use min distance
                d = min_dist_convex(inner_verts[i], inner_verts[j])
                if d < 1e-6:
                    pen += (1e-6 - d)**2 * 1000.0
        # containment penalty: ensure s >= s_req
        if s < s_req:
            pen += (s_req - s)**2 * 100.0
        return s_eff + pen
    
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
    if time.time() > DEADLINE:
        return None
    if not res.success and res.fun > 1e6:
        return None
    vars = res.x
    pos = []
    for i in range(n):
        pos.append((vars[3*i], vars[3*i+1], vars[3*i+2]))
    ocx, ocy, oang = vars[3*n], vars[3*n+1], vars[3*n+2]
    s = vars[3*n+3]
    s_req = outer_hex_side_to_contain(pos, (ocx,ocy), oang)
    s = max(s, s_req)
    # validate
    score = evaluate_config(pos, [a for (_,_,a) in pos], (ocx,ocy), oang)
    if score == float('inf'):
        # try to fix by increasing s
        s = s_req + 0.01
        score = evaluate_config(pos, [a for (_,_,a) in pos], (ocx,ocy), oang)
    return pos, (ocx,ocy), s, oang, score

def construct():
    start = time.time()
    best = None
    best_score = float('inf')
    
    templates = make_symmetric_templates()
    outer_center0 = (0.0, 0.0)
    outer_angle0 = 0.0
    
    for ti, tmpl in enumerate(templates):
        if time.time() > DEADLINE:
            break
        # quick initial eval
        pos = tmpl
        angles = [a for (_,_,a) in pos]
        s_init = outer_hex_side_to_contain(pos, outer_center0, outer_angle0)
        score_init = evaluate_config(pos, angles, outer_center0, outer_angle0)
        if score_init < best_score:
            best_score = score_init
            best = (pos, outer_center0, s_init, outer_angle0)
        
        # optimize
        remaining = DEADLINE - time.time() - 5.0
        if remaining < 5.0:
            break
        time_budget = min(remaining, 25.0)
        try:
            res = optimize_template(pos, outer_center0, outer_angle0, time_budget)
        except Exception:
            res = None
        if res is not None:
            pos_opt, oc, s_opt, ang_opt, sc = res
            if sc < best_score:
                best_score = sc
                best = (pos_opt, oc, s_opt, ang_opt)
    
    if best is None:
        # fallback: return a valid (if large) config
        pos = templates[0]
        s = outer_hex_side_to_contain(pos, outer_center0, outer_angle0)
        return pos, outer_center0, s, outer_angle0
    
    pos, oc, s, ang = best
    # Final safety: ensure validity by slightly increasing s
    s = s + 0.001
    return pos, oc, s, ang

# Keep the original interface: construct() -> (inner, center, side, angle)
# inner is list of [x, y, angle_degrees]
def _wrap_construct():
    res = construct()
    inner, center, side, angle = res
    # inner must be list of [x,y,angle]
    inner_out = [[p[0], p[1], p[2]] for p in inner]
    return inner_out, center, side, angle

# Override construct to return proper format
construct = _wrap_construct
