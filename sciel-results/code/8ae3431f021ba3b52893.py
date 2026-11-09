import time
import random
import numpy as np

_DEADLINE = 150.0  # evaluator wall-clock limit
_SAFETY = 15.0     # margin so we return well before the kill


def _build_edges(d):
    """Return list of (u, v, dim) for all edges of Q_d, u < v."""
    edges = []
    n = 1 << d
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y, i))
    return edges


def _build_faces(d):
    """
    For each 2D face (identified by a vertex x with bits a,b = 0 and dims a<b),
    return the 4 edge indices forming that face.
    Also build edge->faces adjacency.
    """
    n = 1 << d
    # Map (u,v) to edge index
    edge_idx = {}
    all_edges = _build_edges(d)
    for idx, (u, v, dim) in enumerate(all_edges):
        edge_idx[(u, v)] = idx

    faces = []  # list of list of 4 edge indices
    edge_to_faces = [[] for _ in all_edges]

    for x in range(n):
        for a in range(d):
            if (x >> a) & 1:
                continue
            for b in range(a + 1, d):
                if (x >> b) & 1:
                    continue
                # Face vertices: x, x^ea, x^eb, x^ea^eb
                x1 = x ^ (1 << a)
                x2 = x ^ (1 << b)
                x3 = x ^ (1 << a) ^ (1 << b)
                # 4 edges: (x,x1) dim a, (x,x2) dim b, (x1,x3) dim b, (x2,x3) dim a
                e0 = edge_idx[(x, x1)]
                e1 = edge_idx[(x, x2)]
                e2 = edge_idx[(x1, x3)]
                e3 = edge_idx[(x2, x3)]
                faces.append([e0, e1, e2, e3])
                for e in [e0, e1, e2, e3]:
                    edge_to_faces[e].append(len(faces) - 1)

    return all_edges, faces, edge_to_faces


def _is_valid(selected, faces):
    """Check if no face has all 4 edges selected."""
    for f in faces:
        if all(selected[e] for e in f):
            return False
    return True


def _count_edges(selected):
    """Count number of selected edges."""
    return sum(selected)


def construct():
    t0 = time.time()
    d = 7
    n = 1 << d

    all_edges, faces, edge_to_faces = _build_faces(d)
    num_edges = len(all_edges)
    num_faces = len(faces)

    # Convert faces to numpy for faster checking
    faces_np = np.array(faces, dtype=np.int32)  # shape (num_faces, 4)

    # Initial valid seed: use a known C4-free construction
    # Use the "parity" construction: select edge (u,v) in dimension i if 
    # the popcount of u (with bit i cleared) is even
    selected = [False] * num_edges
    for idx, (u, v, dim) in enumerate(all_edges):
        # source is endpoint with bit dim = 0
        src = u if ((u >> dim) & 1) == 0 else v
        if (src.bit_count()) % 2 == 0:
            selected[idx] = True

    # Verify initial seed is valid
    if not _is_valid(selected, faces):
        # Fallback: start with empty set and greedily add
        selected = [False] * num_edges

    best_count = _count_edges(selected)
    best_selected = selected[:]

    # Greedy add: try to add unselected edges in random order
    edge_order = list(range(num_edges))
    random.shuffle(edge_order)

    # Multiple passes of greedy add
    for _pass in range(10):
        if (time.time() - t0) > _DEADLINE - _SAFETY:
            break
        random.shuffle(edge_order)
        for idx in edge_order:
            if (time.time() - t0) > _DEADLINE - _SAFETY:
                break
            if selected[idx]:
                continue
            # Try adding edge idx: check if it would complete any face
            can_add = True
            for fi in edge_to_faces[idx]:
                f = faces[fi]
                # Check if the other 3 edges in this face are all selected
                if all(selected[e] for e in f if e != idx):
                    can_add = False
                    break
            if can_add:
                selected[idx] = True
                if _count_edges(selected) > best_count:
                    best_count = _count_edges(selected)
                    best_selected = selected[:]

    # Local search: try swaps (remove one, add another)
    for _iter in range(50):
        if (time.time() - t0) > _DEADLINE - _SAFETY:
            break
        improved = False
        edge_order2 = list(range(num_edges))
        random.shuffle(edge_order2)
        for idx in edge_order2:
            if (time.time() - t0) > _DEADLINE - _SAFETY:
                break
            if not selected[idx]:
                continue
            # Try removing idx and adding some other edge
            selected[idx] = False
            # Try to add edges that were previously blocked
            added = False
            for jdx in edge_order2:
                if (time.time() - t0) > _DEADLINE - _SAFETY:
                    break
                if selected[jdx] or jdx == idx:
                    continue
                can_add = True
                for fi in edge_to_faces[jdx]:
                    f = faces[fi]
                    if all(selected[e] for e in f if e != jdx):
                        can_add = False
                        break
                if can_add:
                    selected[jdx] = True
                    added = True
                    # Check if this improved our count (it shouldn't, but structure changed)
                    if _count_edges(selected) > best_count:
                        best_count = _count_edges(selected)
                        best_selected = selected[:]
                    improved = True
                    break
            if not added:
                # Restore
                selected[idx] = True

        if not improved:
            break

    # Final validation
    if not _is_valid(best_selected, faces):
        # Fallback to initial valid seed if we somehow corrupted it
        # Find a valid subset
        best_selected = [False] * num_edges
        for idx, (u, v, dim) in enumerate(all_edges):
            src = u if ((u >> dim) & 1) == 0 else v
            if (src.bit_count()) % 2 == 0:
                best_selected[idx] = True
        best_count = _count_edges(best_selected)

    # Build output
    out = []
    for idx, (u, v, dim) in enumerate(all_edges):
        if best_selected[idx]:
            out.append([u, v])

    return out
