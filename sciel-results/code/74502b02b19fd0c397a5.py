import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = d * (2 ** (d - 1))

    # Generate all edges
    edges = []
    edge_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                edge_index[(x, y)] = len(edges) - 1
                edge_index[(y, x)] = len(edges) - 1

    # Precompute faces (C4s)
    # A face is defined by a vertex and two distinct bits i, j
    face_edges = []
    seen_faces = set()
    
    for v in range(n_vertices):
        for i in range(d):
            for j in range(i+1, d):
                verts = [v, v ^ (1<<i), v ^ (1<<j), v ^ (1<<i ^ (1<<j))]
                verts.sort()
                key = tuple(verts)
                
                if key in seen_faces:
                    continue
                seen_faces.add(key)
                
                e1 = edge_index[(min(v, v^(1<<i)), max(v, v^(1<<i)))]
                e2 = edge_index[(min(v^(1<<i), v^(1<<i^(1<<j))), max(v^(1<<i), v^(1<<i^(1<<j))))]
                e3 = edge_index[(min(v^(1<<j), v^(1<<i^(1<<j))), max(v^(1<<j), v^(1<<i^(1<<j))))]
                e4 = edge_index[(min(v, v^(1<<j)), max(v, v^(1<<j)))]
                
                face_edges.append([e1, e2, e3, e4])

    n_faces = len(face_edges)
    
    # ILP Formulation
    # Maximize sum(x_e)
    # Subject to sum(x_e for e in face) <= 3 for all faces
    # x_e in {0, 1}
    
    # Objective coefficients (all 1)
    c = np.ones(total_edges)
    
    # Constraints
    # A_matrix * x <= b
    # A is n_faces x total_edges
    A = np.zeros((n_faces, total_edges))
    for i, face in enumerate(face_edges):
        for e in face:
            A[i, e] = 1.0
    
    b = np.full(n_faces, 3.0)
    
    # Bounds for x: [0, 1]
    bounds = Bounds(lb=np.zeros(total_edges), ub=np.ones(total_edges))
    
    # Integrality: all variables are integer
    integrality = np.ones(total_edges)
    
    # Solve
    start_time = time.time()
    time_limit = 115
    
    try:
        res = milp(c=c, constraints=LinearConstraint(A, ub=b), bounds=bounds, integrality=integrality)
        
        if res.status == 0: # Optimal
            x = res.x
            # Convert to list of edges
            result = []
            for i in range(total_edges):
                if x[i] > 0.5:
                    result.append(list(edges[i]))
            return result
        else:
            # If ILP fails or times out, fallback to a simple heuristic
            # But milp usually returns quickly for this size.
            # If it doesn't, we might get a partial solution or status != 0.
            # Let's check if we can use the incumbent if available?
            # milp returns res.x even if status is not 0? No, usually res.x is None or partial.
            # Let's assume it works. If not, we need a fallback.
            
            # Fallback: Random Greedy
            import random
            rng = random.Random(42)
            selected = [False] * total_edges
            # Start with all selected, remove conflicts
            selected = [True] * total_edges
            
            # Greedy removal
            face_violated = [True] * n_faces
            # Count violations
            # Actually, let's just do a simple pass
            for _ in range(100): # Iterations
                changed = False
                for f_idx, face in enumerate(face_edges):
                    # Check if violated
                    count = sum(1 for e in face if selected[e])
                    if count == 4:
                        # Remove one edge
                        # Pick the one with highest degree in remaining violations?
                        # Simple: pick random
                        e_to_remove = rng.choice(face)
                        selected[e_to_remove] = False
                        changed = True
                if not changed:
                    break
            
            result = []
            for i in range(total_edges):
                if selected[i]:
                    result.append(list(edges[i]))
            return result

    except Exception:
        # Fallback on exception
        import random
        rng = random.Random(42)
        selected = [True] * total_edges
        for _ in range(100):
            changed = False
            for f_idx, face in enumerate(face_edges):
                count = sum(1 for e in face if selected[e])
                if count == 4:
                    e_to_remove = rng.choice(face)
                    selected[e_to_remove] = False
                    changed = True
            if not changed:
                break
        result = []
        for i in range(total_edges):
            if selected[i]:
                result.append(list(edges[i]))
        return result

