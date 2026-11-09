"""Hexagon packing: place 11 unit hexagons in a minimal regular hexagon."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = math.sqrt(3)
    half_sqrt3 = sqrt3 / 2.0
    
    # Unit hexagon vertices (side=1, centered at origin)
    unit_hex_verts = np.array([
        [math.cos(math.radians(k*60)), math.sin(math.radians(k*60))]
        for k in range(6)
    ])
    
    # Unit hexagon outward normals (perpendicular to edges, pointing out)
    # Edge k goes from vert k to vert (k+1)%6
    # Normal is rotated 90 deg clockwise from edge direction
    unit_hex_normals = np.zeros((6, 2))
    for k in range(6):
        e = unit_hex_verts[(k+1)%6] - unit_hex_verts[k]
        n = np.array([-e[1], e[0]])
        unit_hex_normals[k] = n / np.linalg.norm(n)
    
    def get_hex_verts(center, angle_deg):
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        rot = np.array([[c, -s], [s, c]])
        return center + (unit_hex_verts @ rot.T)
    
    def get_hex_normals(angle_deg):
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        rot = np.array([[c, -s], [s, c]])
        return rot @ unit_hex_normals.T
    
    def compute_min_side(inner_positions, outer_center, outer_angle):
        cx, cy = outer_center
        a = math.radians(outer_angle)
        cos_a, sin_a = math.cos(-a), math.sin(-a)
        rot_outer = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Outer hexagon face normals in local frame
        local_normals = np.array([
            [math.cos(math.radians(30+k*60)), math.sin(math.radians(30+k*60))]
            for k in range(6)
        ])
        world_normals = rot_outer @ local_normals.T
        
        max_side = 0.0
        for ix, iy, iang in inner_positions:
            verts = get_hex_verts(np.array([ix, iy]), iang)
            rel = verts - np.array([cx, cy])
            rotated = rot_outer @ rel.T
            projs = rotated @ world_normals  # (6, 6)
            max_proj = np.max(projs, axis=0)  # (6,)
            sides = max_proj / half_sqrt3
            ms = np.max(sides)
            if ms > max_side:
                max_side = ms
        return max_side + 1e-10
    
    def overlap_depth(p1, p2):
        dx, dy = p1[0]-p2[0], p1[1]-p2[1]
        if dx*dx+dy*dy > 4.0:
            return 0.0
        v1 = get_hex_verts(np.array([p1[0],p1[1]]), p1[2])
        v2 = get_hex_verts(np.array([p2[0],p2[1]]), p2[2])
        n1 = get_hex_normals(p1[2])
        n2 = get_hex_normals(p2[2])
        min_ov = float('inf')
        for normals, va, vb in [(n1,v1,v2),(n2,v2,v1)]:
            for k in range(6):
                nx, ny = normals[k]
                pa = va[:,0]*nx + va[:,1]*ny
                pb = vb[:,0]*nx + vb[:,1]*ny
                if pa.max() <= pb.min() or pb.max() <= pa.min():
                    return 0.0
                ov = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
                if ov < min_ov:
                    min_ov = ov
            if min_ov == 0:
                break
        return min_ov if min_ov < float('inf') else 0.0
    
    # Honeycomb layout
    d = sqrt3
    axial = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2)]
    base = []
    for q,r in axial:
        x = d*(q+r/2.0)
        y = d*(1.5*r)
        base.append([x,y,0.0])
    
    # Phase 1: Optimize outer hexagon
    inner0 = base[:]
    best_side = compute_min_side(inner0, (0,0), 0)
    best_outer = [0.0,0.0,0.0]
    
    # Coarse grid on angle
    for ang in np.arange(-30,30.5,2.0):
        s = compute_min_side(inner0, (0,0), ang)
        if s < best_side:
            best_side = s
            best_outer = [0.0,0.0,float(ang)]
    
    # SA on outer
    cur = best_outer[:]
    cs = best_side
    t = 0.05
    for _ in range(3000):
        if time.time() > deadline-120: break
        i = random.randint(0,2)
        old = cur[i]
        cur[i] += random.gauss(0,0.01)
        ns = compute_min_side(inner0, (cur[0],cur[1]), cur[2])
        dl = ns-cs
        if dl<0 or random.random()<math.exp(-dl/t):
            cs=ns
            if ns<best_side:
                best_side=ns
                best_outer=cur[:]
        else:
            cur[i]=old
        t*=0.9999
    
    # Phase 2: Optimize inner with fixed outer
    oc = (best_outer[0], best_outer[1])
    oa = best_outer[2]
    
    def obj2(inner):
        s = compute_min_side(inner, oc, oa)
        p = 0.0
        for i in range(N):
            for j in range(i+1,N):
                ov = overlap_depth(inner[i],inner[j])
                if ov>0: p+=ov*ov
        return s+100*p
    
    ci = base[:]
    co = obj2(ci)
    bi = ci[:]
    bo = co
    t=0.03; st=0.015
    for it in range(30000):
        if time.time()>deadline-10: break
        hi=random.randint(0,N-1)
        pi=random.randint(0,2)
        old=ci[hi][pi]
        ci[hi][pi]+=random.gauss(0,st)
        no=obj2(ci)
        dl=no-co
        if dl<0 or random.random()<math.exp(-dl/t):
            co=no
            if no<bo:
                bo=no
                bi=ci[:]
        else:
            ci[hi][pi]=old
        t*=0.99997
        st*=0.99999
    
    inner=bi
    cx,cy=oc
    ca=oa
    fs=compute_min_side(inner,(cx,cy),ca)
    return inner,(cx,cy),fs,ca

# EVOLVE-BLOCK-END
