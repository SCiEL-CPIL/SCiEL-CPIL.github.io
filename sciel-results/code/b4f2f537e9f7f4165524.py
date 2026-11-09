"""Fresh design: minimize covering radius of 50 points on the unit sphere.

The covering radius is the maximum, over all points y on the sphere, of the
angular distance from y to its nearest configuration point. This equals the
largest circumradius among the Delaunay triangles (facets of the convex hull).
We compute the spherical circumcenter of every triangle formed by the points
and assign each circumcenter to the triangle whose vertices are its three
nearest points (this is the spherical Delaunay triangulation). We then perform
a gradient-style local optimization: each point is moved toward the
circumcenters of the facets that contain it, weighted by how far those
circumcenters are. A time budget guard returns the best result found.
"""
import time
import numpy as np

N = 50
_DEADLINE_FRACTION = 0.90  # return 10% before the 150s wall-clock limit
_DEADLINE_SECONDS = 135.0  # hard cap derived from 150s * 0.90


def _normalize_rows(P):
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return P / norms


def _spherical_circumcenters(P):
    """Return (centers, radii, tri) for all triples.

    centers: (K, 3) normalized circumcenter of each triangle
    radii:   (K,)  angular distance from triangle normal to circumcenter
    tri:     (K, 3) indices of the three vertices
    """
    n = P.shape[0]
    idx = np.array([np.triu_indices(n, k=1)])
    i = idx[0]
    j = idx[1]
    # build triples
    ii, jj = np.meshgrid(i, j, indexing='ij')
    k = np.arange(n)[None, None, :]
    k = np.broadcast_to(k, ii.shape)
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)
    kk = k.reshape(-1)
    tri = np.column_stack([ii, jj, kk])

    A = P[tri[:, 0]]
    B = P[tri[:, 1]]
    C = P[tri[:, 2]]

    # triangle normal
    nrm = np.cross(B - A, C - A)
    nrm = _normalize_rows(nrm)

    # circumcenter via cross products of edges
    u = np.cross(B - A, B + A)
    v = np.cross(C - A, C + A)
    c = np.cross(u, v)
    cnorm = np.linalg.norm(c, axis=1, keepdims=True)
    cnorm = np.where(cnorm < 1e-12, 1.0, cnorm)
    centers = c / cnorm
    # ensure center is on the same side as the normal
    dot = np.sum(centers * nrm, axis=1)
    centers[dot < 0] *= -1

    radii = np.arccos(np.clip(np.sum(centers * nrm, axis=1), -1.0, 1.0))
    return centers, radii, tri


def _assign_delaunay(P, centers, radii, tri):
    """Assign each circumcenter to the triangle of its 3 nearest points.

    Returns (valid_mask, assigned_tri) where assigned_tri[k] is the index into
    tri of the triangle that owns circumcenter k, or -1 if not owned.
    """
    K = centers.shape[0]
    n = P.shape[0]
    # distance from each center to each point
    # centers (K,3), P (n,3)
    # Use chunked computation to limit memory
    chunk = max(1, min(2000, K))
    own = np.full(K, -1, dtype=np.int64)
    for s in range(0, K, chunk):
        e = min(s + chunk, K)
        C = centers[s:e]
        # d2 = |c|^2 + |p|^2 - 2 c.p ; both unit so d2 = 2 - 2 c.p
        dots = C @ P.T  # (chunk, n)
        d2 = 2.0 - 2.0 * dots
        # nearest 3
        nnz = np.argpartition(d2, 3, axis=1)[:, :3]
        # sort those 3
        nnz = nnz[np.arange(e - s)[:, None], nnz]
        nnz = np.sort(nnz, axis=1)
        # for each, check if the triangle matches one in tri
        # build a lookup from sorted triple -> tri index
        # We'll just compare against all triangles in this chunk's relevant set.
        # Simpler: for each k, check its 3 nearest; if they form a triangle in tri, assign.
        for local in range(e - s):
            k = s + local
            a, b, c = nnz[local]
            # find tri index with this sorted triple
            match = np.where((tri[:, 0] == a) & (tri[:, 1] == b) & (tri[:, 2] == c))[0]
            if match.size > 0:
                own[k] = match[0]
            else:
                # try all permutations of the 3 nearest
                found = -1
                for p in range(6):
                    perm = _perm3(p)
                    ta, tb, tc = a * 0 + nnz[local][perm]
                    m = np.where((tri[:, 0] == ta) & (tri[:, 1] == tb) & (tri[:, 2] == tc))[0]
                    if m.size > 0:
                        found = m[0]
                        break
                own[k] = found
    return own


def _perm3(p):
    return [
        (0, 1, 2), (0, 2, 1), (1, 0, 2),
        (1, 2, 0), (2, 0, 1), (2, 1, 0),
    ][p]


def _covering_radius(P):
    """Compute covering radius in radians via Delaunay facets."""
    centers, radii, tri = _spherical_circumcenters(P)
    own = _assign_delaunay(P, centers, radii, tri)
    valid = own >= 0
    if not np.any(valid):
        # fallback: max over all circumradii
        return float(np.max(radii))
    return float(np.max(radii[valid]))


def _move_toward(P, centers, radii, tri, own, step):
    """Move each point toward the circumcenters of facets that contain it."""
    n = P.shape[0]
    P_new = P.copy()
    # For each point, gather the circumcenters of triangles that contain it
    # and whose center is far (high radius). Weight by radius.
    # Build per-point list.
    for p in range(n):
        # find triangles containing p
        mask = (tri[:, 0] == p) | (tri[:, 1] == p) | (tri[:, 2] == p)
        # also only owned facets
        owned = own >= 0
        sel = mask & owned
        if not np.any(sel):
            continue
        sel_idx = np.where(sel)[0]
        # weight by radius
        w = radii[sel_idx]
        # only move toward the farthest ones
        if w.max() <= 0:
            continue
        # threshold: top facets
        thr = w.max() - 1e-6
        sel_idx = sel_idx[w > thr]
        if sel_idx.size == 0:
            continue
        C = centers[sel_idx]
        # average direction
        avg = np.mean(C, axis=0)
        anorm = np.linalg.norm(avg)
        if anorm < 1e-12:
            continue
        avg = avg / anorm
        # slerp from P[p] toward avg by step
        dot = np.clip(np.dot(P[p], avg), -1.0, 1.0)
        theta = np.arccos(dot)
        if theta < 1e-12:
            continue
        t = min(step, theta)
        s1 = np.sin((1.0 - t / theta) * theta) / np.sin(theta)
        s2 = np.sin((t / theta) * theta) / np.sin(theta)
        P_new[p] = s1 * P[p] + s2 * avg
    return _normalize_rows(P_new)


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE_SECONDS

    # Fibonacci spiral start
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    P = _normalize_rows(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

    best_P = P.copy()
    best_r = _covering_radius(best_P)

    # Precompute triangles (vertex indices) once; they don't change with P.
    centers, radii, tri = _spherical_circumcenters(P)
    own = _assign_delaunay(P, centers, radii, tri)

    step = 0.05  # initial step size in radians
    stall = 0
    it = 0
    while time.time() < deadline:
        it += 1
        # recompute centers/radii/own (they depend on P)
        centers, radii, tri = _spherical_circumcenters(P)
        own = _assign_delaunay(P, centers, radii, tri)

        P_new = _move_toward(P, centers, radii, tri, own, step)
        r_new = _covering_radius(P_new)

        if r_new < best_r:
            best_r = r_new
            best_P = P_new.copy()
            P = P_new
            stall = 0
            # increase step slightly
            step = min(step * 1.1, 0.2)
        else:
            P = P_new  # still move to explore
            stall += 1
            step *= 0.85
            if stall > 10:
                step = 0.05
                stall = 0
                # occasional random perturbation to escape local minima
                if it % 20 == 0:
                    noise = np.random.randn(N, 3) * 0.01
                    P = _normalize_rows(P + noise)
                    r_new = _covering_radius(P)
                    if r_new < best_r:
                        best_r = r_new
                        best_P = P.copy()

        # safety margin check
        if time.time() > deadline - 2.0:
            break

    # final validation: ensure not in a closed hemisphere and points are finite
    if not np.all(np.isfinite(best_P)):
        # fallback to fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        best_P = _normalize_rows(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

    # check hemisphere: if all points lie in a closed hemisphere, the covering
    # radius is at least pi/2; we want to avoid that. Quick check via centroid.
    centroid = np.sum(best_P, axis=0)
    cnorm = np.linalg.norm(centroid)
    if cnorm > 0.5:  # suspiciously concentrated
        # push one point opposite
        opp = -centroid / cnorm
        # find the point closest to centroid direction
        dots = best_P @ (centroid / cnorm)
        idx = np.argmax(dots)
        best_P[idx] = _normalize_rows(opp)[0] if False else opp
        best_P = _normalize_rows(best_P)

    return best_P
