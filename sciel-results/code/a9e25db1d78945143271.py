import time
import random
import numpy as np

def construct():
    """
    Optimized Tensor-slice approach for C4-free edge set in Q7.
    Uses vectorized NumPy operations for constraint checking.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    d = 7
    n_vertices = 1 << d  # 128
    
    def time_left():
        return deadline - time.time()
    
    # Precompute edge representation
    edge_list = []
    axis_vertex_to_eidx = {}
    for axis in range(d):
        bit = 1 << axis
        for v in range(n_vertices):
            if v & bit:
                continue
            u = v
            w = v ^ bit
            eidx = len(edge_list)
            edge_list.append((u, w))
            axis_vertex_to_eidx[(axis, v)] = eidx
    
    n_edges = len(edge_list)
    assert n_edges == 448
    
    # Precompute face data
    # Face is defined by (a, b, c) where a < b, c is 5-bit complement
    # Vertices: v00, v10, v01, v11
    # Edges: (a, v00), (b, v00), (a, v01), (b, v10)
    # Note: v01 has bit a=0, v10 has bit b=0.
    
    face_edge_indices = [] # List of 4 edge indices for each face
    edge_to_faces = [[] for _ in range(n_edges)] # List of face indices for each edge
    
    face_id = 0
    for a in range(d):
        for b in range(a+1, d):
            other_bits = [i for i in range(d) if i != a and i != b]
            for c_val in range(1 << 5):
                base = 0
                for k, bit_idx in enumerate(other_bits):
                    if c_val & (1 << k):
                        base |= (1 << bit_idx)
                
                v00 = base
                v10 = base | (1 << a)
                v01 = base | (1 << b)
                v11 = base | (1 << a) | (1 << b)
                
                # Get edge indices
                # Edge (a, v00)
                e0 = axis_vertex_to_eidx[(a, v00)]
                # Edge (b, v00)
                e1 = axis_vertex_to_eidx[(b, v00)]
                # Edge (a, v01)
                e2 = axis_vertex_to_eidx[(a, v01)]
                # Edge (b, v10)
                e3 = axis_vertex_to_eidx[(b, v10)]
                
                face_edges = [e0, e1, e2, e3]
                face_edge_indices.append(face_edges)
                
                for e in face_edges:
                    edge_to_faces[e].append(face_id)
                
                face_id += 1
    
    n_faces = len(face_edge_indices)
    
    # Convert to NumPy arrays
    face_edge_indices = np.array(face_edge_indices, dtype=np.int32) # (n_faces, 4)
    
    # Precompute neighbor coordinates for fast can_add check
    # For each edge e, we need the coordinates (axis, vertex) of the other 3 edges in each of its faces.
    # edge_to_faces[e] gives the face indices.
    # For a face f, the edges are face_edge_indices[f].
    # The neighbors of e in face f are the other 3.
    
    neighbor_coords = np.zeros((n_edges, 18, 2), dtype=np.int32) # (n_edges, 6 faces * 3 neighbors, 2 coords)
    
    # Map (axis, vertex) to flat index for M? Or just use 2D indexing.
    # M is (7, 128).
    # We need to extract M[axis, vertex].
    
    for e in range(n_edges):
        faces = edge_to_faces[e]
        # There should be exactly 6 faces
        assert len(faces) == 6, f"Edge {e} in {len(faces)} faces"
        
        for i, f in enumerate(faces):
            face_edges = face_edge_indices[f]
            # Find the position of e in face_edges
            # face_edges is a numpy array of 4 ints
            idx_in_face = np.where(face_edges == e)[0][0]
            
            # Get the other 3
            neighbors = []
            for j in range(4):
                if j == idx_in_face:
                    continue
                ne = face_edges[j]
                # Get axis and vertex for ne
                # We need a reverse map from edge index to (axis, vertex)
                # Let's build one
                pass
            
    # Build reverse map
    eidx_to_axis_vertex = {}
    for (axis, v), eidx in axis_vertex_to_eidx.items():
        eidx_to_axis_vertex[eidx] = (axis, v)
        
    for e in range(n_edges):
        faces = edge_to_faces[e]
        for i, f in enumerate(faces):
            face_edges = face_edge_indices[f]
            idx_in_face = np.where(face_edges == e)[0][0]
            
            for j in range(4):
                if j == idx_in_face:
                    continue
                ne = face_edges[j]
                ax, vt = eidx_to_axis_vertex[ne]
                # Store in neighbor_coords[e, i*3 + count, :]
                # We need a counter for j
                pass

    # Let's do this more carefully
    neighbor_coords = np.zeros((n_edges, 18, 2), dtype=np.int32)
    for e in range(n_edges):
        faces = edge_to_faces[e]
        for i, f in enumerate(faces):
            face_edges = face_edge_indices[f]
            idx_in_face = np.where(face_edges == e)[0][0]
            count = 0
            for j in range(4):
                if j == idx_in_face:
                    continue
                ne = face_edges[j]
                ax, vt = eidx_to_axis_vertex[ne]
                neighbor_coords[e, i*3 + count, 0] = ax
                neighbor_coords[e, i*3 + count, 1] = vt
                count += 1
            # count should be 3

    def check_valid(M):
        """Check if matrix M (7x128 bool/int) is C4-free."""
        M_flat = M.flatten()
        face_vals = M_flat[face_edge_indices]
        return np.all(face_vals.sum(axis=1) < 4)
    
    def can_add(M, e):
        """Check if edge e can be added to M without creating C4."""
        coords = neighbor_coords[e]
        axes = coords[:, 0]
        verts = coords[:, 1]
        vals = M[axes, verts]
        sums = vals.reshape(6, 3).sum(axis=1)
        return np.all(sums < 3)
    
    def count_edges(M):
        return int(np.sum(M))
    
    def greedy_tensor(seed=None):
        rng = random.Random(seed)
        M = np.zeros((d, n_vertices), dtype=np.int8)
        
        e_indices = list(range(n_edges))
        rng.shuffle(e_indices)
        
        for e in e_indices:
            if can_add(M, e):
                ax, vt = eidx_to_axis_vertex[e]
                M[ax, vt] = 1
        
        return M
    
    def local_search_tensor(M, seed=None, max_iter=2000):
        rng = random.Random(seed)
        M = M.copy()
        
        improved = True
        iter_count = 0
        consecutive_no_improve = 0
        
        while improved and time_left() > 20 and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # 1. Try to add any edge
            # Get list of unselected edges
            unselected = []
            for e in range(n_edges):
                ax, vt = eidx_to_axis_vertex[e]
                if not M[ax, vt]:
                    unselected.append(e)
            
            if unselected:
                rng.shuffle(unselected)
                for e in unselected:
                    if time_left() < 20:
                        break
                    if can_add(M, e):
                        ax, vt = eidx_to_axis_vertex[e]
                        M[ax, vt] = 1
                        improved = True
                        consecutive_no_improve = 0
                if improved:
                    continue
            
            # 2. Try swaps
            if time_left() > 30:
                selected = []
                for e in range(n_edges):
                    ax, vt = eidx_to_axis_vertex[e]
                    if M[ax, vt]:
                        selected.append(e)
                
                unselected = []
                for e in range(n_edges):
                    ax, vt = eidx_to_axis_vertex[e]
                    if not M[ax, vt]:
                        unselected.append(e)
                
                if selected and unselected:
                    # Try random swaps
                    swap_found = False
                    for _ in range(100):
                        if time_left() < 20:
                            break
                        e_add = rng.choice(unselected)
                        e_rem = rng.choice(selected)
                        
                        # Test swap
                        ax_add, vt_add = eidx_to_axis_vertex[e_add]
                        ax_rem, vt_rem = eidx_to_axis_vertex[e_rem]
                        
                        M[ax_rem, vt_rem] = 0
                        M[ax_add, vt_add] = 1
                        
                        if check_valid(M):
                            improved = True
                            consecutive_no_improve = 0
                            swap_found = True
                            break
                        else:
                            # Revert
                            M[ax_rem, vt_rem] = 1
                            M[ax_add, vt_add] = 0
                    
                    if not swap_found:
                        # Perturbation
                        if len(selected) > 10:
                            n_rem = rng.randint(3, 7)
                            to_remove = rng.sample(selected, min(n_rem, len(selected) // 5))
                            for e in to_remove:
                                ax, vt = eidx_to_axis_vertex[e]
                                M[ax, vt] = 0
                            improved = True
                            consecutive_no_improve = 0
                        else:
                            consecutive_no_improve += 1
                            if consecutive_no_improve > 10:
                                break
        
        return M
    
    def polish_tensor(M, seed=None):
        M = M.copy()
        changed = True
        while changed and time_left() > 25:
            changed = False
            for e in range(n_edges):
                ax, vt = eidx_to_axis_vertex[e]
                if M[ax, vt]:
                    continue
                if can_add(M, e):
                    M[ax, vt] = 1
                    changed = True
        return M
    
    best_count = 0
    best_M = None
    
    attempts = 0
    while time_left() > 30:
        attempts += 1
        seed = attempts * 104729 + 13
        
        if attempts % 4 == 1:
            M = greedy_tensor(seed)
        elif attempts % 4 == 2:
            M = greedy_tensor(seed + 99991)
        elif attempts % 4 == 3:
            if best_M is not None:
                M = best_M.copy()
                rng = random.Random(seed)
                selected = []
                for e in range(n_edges):
                    ax, vt = eidx_to_axis_vertex[e]
                    if M[ax, vt]:
                        selected.append(e)
                if selected:
                    to_remove = rng.sample(selected, min(5, len(selected) // 6))
                    for e in to_remove:
                        ax, vt = eidx_to_axis_vertex[e]
                        M[ax, vt] = 0
            else:
                M = greedy_tensor(seed)
        else:
            if best_M is not None:
                M = best_M.copy()
                rng = random.Random(seed + 555)
                selected = []
                for e in range(n_edges):
                    ax, vt = eidx_to_axis_vertex[e]
                    if M[ax, vt]:
                        selected.append(e)
                if selected:
                    to_remove = rng.sample(selected, min(8, len(selected) // 4))
                    for e in to_remove:
                        ax, vt = eidx_to_axis_vertex[e]
                        M[ax, vt] = 0
            else:
                M = greedy_tensor(seed)
        
        M = local_search_tensor(M, seed)
        M = polish_tensor(M, seed + 7777)
        
        cnt = count_edges(M)
        if check_valid(M):
            if cnt > best_count:
                best_count = cnt
                best_M = M.copy()
        
        if time_left() < 45:
            break
    
    if best_M is None or best_count < 250:
        M = greedy_tensor(1)
        best_M = M
        best_count = count_edges(M)
    
    result = []
    for e in range(n_edges):
        ax, vt = eidx_to_axis_vertex[e]
        if best_M[ax, vt]:
            w = vt ^ (1 << ax)
            result.append([vt, w])
    
    return result
