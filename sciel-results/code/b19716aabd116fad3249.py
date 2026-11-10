import time
import random

def construct():
    deadline = time.time() + 110.0
    d = 7
    n = 1 << d
    
    # 1. Precompute edges
    edges = []
    edge_index = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_index[(u, v, i)] = len(edges)
                edges.append((u, v, i))
    num_edges = len(edges)
    
    # 2. Precompute face IDs
    pair_index = {}
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            pair_index[(i, j)] = idx
            idx += 1
            
    def get_compact_rest(u, i, j):
        val = 0
        bit_pos = 0
        for k in range(d):
            if k == i or k == j:
                continue
            if u & (1 << k):
                val |= (1 << bit_pos)
            bit_pos += 1
        return val

    edge_faces = []
    for u, v, i in edges:
        faces = []
        for j in range(d):
            if j == i:
                continue
            if i < j:
                p_idx = pair_index[(i, j)]
            else:
                p_idx = pair_index[(j, i)]
            rest_compact = get_compact_rest(u, i, j)
            face_id = p_idx * (1 << (d - 2)) + rest_compact
            faces.append(face_id)
        edge_faces.append(faces)
        
    total_faces = len(pair_index) * (1 << (d - 2))
    
    face_counts = [0] * total_faces
    selected = set()
    
    def add_edge(e_idx):
        for fid in edge_faces[e_idx]:
            face_counts[fid] += 1
        selected.add(e_idx)
        
    def remove_edge(e_idx):
        for fid in edge_faces[e_idx]:
            face_counts[fid] -= 1
        selected.remove(e_idx)
        
    def can_add(e_idx):
        for fid in edge_faces[e_idx]:
            if face_counts[fid] >= 3:
                return False
        return True
        
    def greedy_fill():
        order = list(range(num_edges))
        random.shuffle(order)
        for e_idx in order:
            if can_add(e_idx):
                add_edge(e_idx)
                
    def try_add_missing():
        missing = [i for i in range(num_edges) if i not in selected]
        random.shuffle(missing)
        added_any = False
        for e_idx in missing:
            if can_add(e_idx):
                add_edge(e_idx)
                added_any = True
        return added_any
        
    def algebraic_init():
        """
        Deterministic algebraic construction using linear algebra over F_2^7.
        For each vertex, select edges based on a linear form to avoid C4s.
        """
        selected.clear()
        for i in range(total_faces):
            face_counts[i] = 0
        
        # Use a parity-based selection: for each vertex u and dimension i,
        # select edge (u, u^2^i) if the parity of certain bits is 0.
        # This creates a structured pattern that avoids 4-cycles.
        
        # Strategy: For each edge in dimension i at vertex u, include it
        # if the sum of bits at positions not in {i} has even parity.
        # This is a known construction for C4-free subgraphs of hypercubes.
        
        for e_idx in range(num_edges):
            u, v, i = edges[e_idx]
            # Check if all faces this edge belongs to have < 3 edges
            if can_add(e_idx):
                # Use a deterministic rule: include edge if parity of (u without bit i) is even
                u_mask = u & ~(1 << i)
                bit_count = bin(u_mask).count('1')
                if bit_count % 2 == 0:
                    add_edge(e_idx)
                # Also try including some edges with odd parity to maximize
                # We'll do a second pass with different criteria
        # Second pass: try to add more edges with different parity rules
        for e_idx in range(num_edges):
            if e_idx not in selected:
                if can_add(e_idx):
                    u, v, i = edges[e_idx]
                    u_mask = u & ~(1 << i)
                    bit_count = bin(u_mask).count('1')
                    # Use a different linear form: sum of bits at even positions
                    even_sum = sum((u_mask >> k) & 1 for k in range(d) if k % 2 == 0 and k != i)
                    if even_sum % 2 == 0:
                        add_edge(e_idx)
                        
    best_size = 0
    best_selected = set()
    
    # Start with algebraic construction
    algebraic_init()
    best_size = len(selected)
    best_selected = set(selected)
    
    # Try to improve with greedy fill
    try_add_missing()
    if len(selected) > best_size:
        best_size = len(selected)
        best_selected = set(selected)
    
    # Local search with simulated annealing-like acceptance
    current_size = len(selected)
    p_accept_worse = 0.1
    kick_size = 3  # Remove more edges for bigger kicks
    
    while time.time() < deadline:
        if not try_add_missing():
            # Stuck, do a kick
            if len(selected) < 2:
                # If very small, restart with algebraic init
                algebraic_init()
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
                continue
                
            # Save state
            old_face_counts = face_counts[:]
            old_selected = set(selected)
            
            # Remove random edges for kick
            to_remove = random.sample(list(selected), min(kick_size, len(selected)))
            for e_idx in to_remove:
                remove_edge(e_idx)
                
            # Greedy re-add
            missing = [i for i in range(num_edges) if i not in selected]
            random.shuffle(missing)
            for e_idx in missing:
                if can_add(e_idx):
                    add_edge(e_idx)
                    
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = set(selected)
            elif len(selected) < len(old_selected):
                # Accept worse with some probability to escape local optima
                if random.random() > p_accept_worse:
                    # Revert
                    selected.clear()
                    for i in range(total_faces):
                        face_counts[i] = 0
                    for e_idx in old_selected:
                        add_edge(e_idx)
                # Otherwise keep the worse state to explore
            
        # Periodic restart to escape deep local optima
        if random.random() < 0.005:
            algebraic_init()
            try_add_missing()
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = set(selected)

    # Restore best solution
    result = []
    for e_idx in best_selected:
        u, v, _ = edges[e_idx]
        result.append([u, v])
        
    return result
