import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110.0
    d = 7
    n = 1 << d
    
    # 1. Precompute edges
    edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
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
    
    # Use numpy arrays for faster face count updates
    face_counts = np.zeros(total_faces, dtype=np.int8)
    selected = np.zeros(num_edges, dtype=bool)
    
    def add_edge(e_idx):
        for fid in edge_faces[e_idx]:
            face_counts[fid] += 1
        selected[e_idx] = True
        
    def remove_edge(e_idx):
        for fid in edge_faces[e_idx]:
            face_counts[fid] -= 1
        selected[e_idx] = False
        
    def can_add(e_idx):
        for fid in edge_faces[e_idx]:
            if face_counts[fid] >= 3:
                return False
        return True
        
    def try_add_missing():
        """Try to add any missing edges that can be added. Returns True if any were added."""
        added_any = False
        for e_idx in range(num_edges):
            if not selected[e_idx] and can_add(e_idx):
                add_edge(e_idx)
                added_any = True
        return added_any
        
    def algebraic_init():
        """
        Better algebraic construction: use a structured pattern based on
        linear forms over F_2. For each edge (u, u^2^i), include it based
        on a hash of the vertex and dimension that avoids C4 patterns.
        """
        selected[:] = False
        face_counts[:] = 0
        
        # Use a multi-criteria deterministic rule
        # For edge (u, v) in dimension i, include if:
        # 1. The XOR of certain bit positions has even parity
        # This creates a structured pattern that tends to avoid C4s
        
        # Strategy: For each dimension i, use a different linear form
        # to decide which edges to include. This breaks symmetry and
        # avoids creating 4-cycles in any face.
        
        for e_idx in range(num_edges):
            u, v, i = edges[e_idx]
            if can_add(e_idx):
                # Use a combination of parity checks
                u_mask = u & ~(1 << i)
                
                # Check 1: parity of all bits except i
                parity1 = bin(u_mask).count('1') % 2
                
                # Check 2: parity of bits at even positions (excluding i)
                even_bits = sum((u_mask >> k) & 1 for k in range(d) if k % 2 == 0 and k != i)
                parity2 = even_bits % 2
                
                # Check 3: parity of bits at odd positions (excluding i)
                odd_bits = sum((u_mask >> k) & 1 for k in range(d) if k % 2 == 1 and k != i)
                parity3 = odd_bits % 2
                
                # Include if at least 2 of the 3 parities are even
                even_count = (1 if parity1 == 0 else 0) + (1 if parity2 == 0 else 0) + (1 if parity3 == 0 else 0)
                if even_count >= 2:
                    add_edge(e_idx)
        
        # Second pass: try to add more with relaxed criteria
        for e_idx in range(num_edges):
            if not selected[e_idx] and can_add(e_idx):
                u, v, i = edges[e_idx]
                u_mask = u & ~(1 << i)
                # Use a different rule: include if parity of (u XOR v) bits at positions > i is even
                diff = u ^ v
                high_bits = diff & (~((1 << (i + 1)) - 1))
                if bin(high_bits).count('1') % 2 == 0:
                    add_edge(e_idx)
    
    def two_flip_move():
        """
        Try a 2-flip move: remove 2 random selected edges and try to
        add 2 different missing edges. This can escape local optima
        that single-edge moves cannot.
        """
        sel_list = np.where(selected)[0]
        if len(sel_list) < 2:
            return False
            
        # Pick 2 random edges to remove
        to_remove = random.sample(list(sel_list), 2)
        e1, e2 = to_remove
        
        # Save state
        old_face_counts = face_counts.copy()
        old_selected = selected.copy()
        
        # Remove both
        remove_edge(e1)
        remove_edge(e2)
        
        # Try to add 2 missing edges
        added = []
        missing = np.where(~selected)[0]
        random.shuffle(missing)
        
        for e_idx in missing:
            if can_add(e_idx):
                add_edge(e_idx)
                added.append(e_idx)
                if len(added) == 2:
                    break
        
        # Check if we improved or stayed same
        if len(added) >= 2:
            # Keep the move if we added at least as many as we removed
            return True
        else:
            # Revert
            selected[:] = old_selected
            face_counts[:] = old_face_counts
            return False
            
    def adaptive_kick(kick_size):
        """
        Remove kick_size random edges and greedily re-add.
        Returns True if the result is better than before.
        """
        sel_list = np.where(selected)[0]
        if len(sel_list) < kick_size:
            return False
            
        old_face_counts = face_counts.copy()
        old_selected = selected.copy()
        old_size = len(sel_list)
        
        # Remove random edges
        to_remove = random.sample(list(sel_list), kick_size)
        for e_idx in to_remove:
            remove_edge(e_idx)
            
        # Greedy re-add
        missing = np.where(~selected)[0]
        random.shuffle(missing)
        for e_idx in missing:
            if can_add(e_idx):
                add_edge(e_idx)
                
        new_size = int(np.sum(selected))
        
        if new_size >= old_size:
            return True
        else:
            # Revert with some probability
            if random.random() > 0.15:
                selected[:] = old_selected
                face_counts[:] = old_face_counts
            return False
    
    best_size = 0
    best_selected = None
    
    # Start with algebraic construction
    algebraic_init()
    best_size = int(np.sum(selected))
    best_selected = selected.copy()
    
    # Try to improve with greedy fill
    try_add_missing()
    if int(np.sum(selected)) > best_size:
        best_size = int(np.sum(selected))
        best_selected = selected.copy()
    
    # Local search with adaptive strategies
    stuck_count = 0
    
    while time.time() < deadline:
        if not try_add_missing():
            stuck_count += 1
        else:
            stuck_count = 0
            if int(np.sum(selected)) > best_size:
                best_size = int(np.sum(selected))
                best_selected = selected.copy()
        
        # Adaptive kick size based on stuckness
        if stuck_count > 0:
            kick_size = min(3 + stuck_count // 5, 10)
            
            # Try 2-flip move first (cheaper)
            if random.random() < 0.3:
                two_flip_move()
                if int(np.sum(selected)) > best_size:
                    best_size = int(np.sum(selected))
                    best_selected = selected.copy()
                else:
                    adaptive_kick(kick_size)
                    if int(np.sum(selected)) > best_size:
                        best_size = int(np.sum(selected))
                        best_selected = selected.copy()
            else:
                adaptive_kick(kick_size)
                if int(np.sum(selected)) > best_size:
                    best_size = int(np.sum(selected))
                    best_selected = selected.copy()
        
        # Periodic restart to escape deep local optima
        if stuck_count > 50 or random.random() < 0.003:
            algebraic_init()
            try_add_missing()
            if int(np.sum(selected)) > best_size:
                best_size = int(np.sum(selected))
                best_selected = selected.copy()
            stuck_count = 0

    # Restore best solution
    result = []
    for e_idx in range(num_edges):
        if best_selected[e_idx]:
            u, v, _ = edges[e_idx]
            result.append([u, v])
        
    return result
