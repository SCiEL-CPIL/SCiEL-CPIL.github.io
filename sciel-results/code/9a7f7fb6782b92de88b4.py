import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    d = 7
    n = 1 << d
    m = d * n // 2
    
    # Build edges
    edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_idx[(x, y)] = idx
    
    # Build faces and face-edge incidence
    faces = []
    edge_faces = [[] for _ in range(m)]
    for x in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                xi = x ^ (1 << i)
                xj = x ^ (1 << j)
                xij = x ^ (1 << i) ^ (1 << j)
                e1 = edge_idx[(min(x, xi), max(x, xi))]
                e2 = edge_idx[(min(x, xj), max(x, xj))]
                e3 = edge_idx[(min(xi, xij), max(xi, xij))]
                e4 = edge_idx[(min(xj, xij), max(xj, xij))]
                face = (e1, e2, e3, e4)
                faces.append(face)
                for e in face:
                    edge_faces[e].append(len(faces) - 1)
    
    # Start with all edges selected
    selected = np.ones(m, dtype=bool)
    
    # Count violations: number of faces with all 4 edges selected
    def count_violations():
        count = 0
        for face in faces:
            if all(selected[e] for e in face):
                count += 1
        return count
    
    # Greedy removal: repeatedly remove the edge in the most violated faces
    while count_violations() > 0 and time.time() < deadline - 5:
        # Find edge that's in the most violated faces
        best_edge = -1
        best_count = -1
        for e in range(m):
            if not selected[e]:
                continue
            violated = 0
            for fi in edge_faces[e]:
                face = faces[fi]
                if all(selected[fe] for fe in face):
                    violated += 1
            if violated > best_count:
                best_count = violated
                best_edge = e
        if best_edge == -1:
            break
        selected[best_edge] = False
    
    # Now try to add edges back: local search
    # For each unselected edge, check if adding it creates a violation
    for e in range(m):
        if selected[e]:
            continue
        if time.time() > deadline - 5:
            break
        # Check if adding e creates any violated face
        can_add = True
        for fi in edge_faces[e]:
            face = faces[fi]
            if all(selected[fe] for fe in face if fe != e):
                can_add = False
                break
        if can_add:
            selected[e] = True
    
    # Iterative local search: try to swap edges
    for iteration in range(50):
        if time.time() > deadline - 10:
            break
        improved = False
        for e_remove in range(m):
            if not selected[e_remove]:
                continue
            for e_add in range(m):
                if selected[e_add] or e_add == e_remove:
                    continue
                if time.time() > deadline - 10:
                    break
                # Check if we can swap
                # First, check if removing e_remove and adding e_add keeps us C4-free
                # This is complex, so let's just try adding e_add if possible
                can_add = True
                for fi in edge_faces[e_add]:
                    face = faces[fi]
                    if all(selected[fe] for fe in face if fe != e_add):
                        can_add = False
                        break
                if can_add:
                    selected[e_add] = True
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    out = [[edges[i][0], edges[i][1]] for i in range(m) if selected[i]]
    return out
