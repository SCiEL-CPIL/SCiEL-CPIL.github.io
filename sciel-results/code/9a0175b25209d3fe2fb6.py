import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    d = 7
    n_vertices = 1 << d
    n_edges_total = n_vertices * d // 2
    
    # 1. Precomputation
    # Generate all edges
    edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx
                
    # Generate all 2-faces (4-cycles)
    faces = []
    edge_to_faces = [[] for _ in range(n_edges_total)]
    
    for i in range(d):
        for j in range(i+1, d):
            for x in range(n_vertices):
                if (x & (1 << i)) == 0 and (x & (1 << j)) == 0:
                    v0 = x
                    v1 = x ^ (1 << i)
                    v2 = x ^ (1 << j)
                    v3 = x ^ (1 << i) ^ (1 << j)
                    
                    e01 = edge_to_idx[(min(v0, v1), max(v0, v1))]
                    e13 = edge_to_idx[(min(v1, v3), max(v1, v3))]
                    e32 = edge_to_idx[(min(v3, v2), max(v3, v2))]
                    e20 = edge_to_idx[(min(v2, v0), max(v2, v0))]
                    
                    face = [e01, e13, e32, e20]
                    faces.append(face)
                    for e in face:
                        edge_to_faces[e].append(len(faces) - 1)
                        
    n_faces = len(faces)
    
    # Build conflict graph: two edges conflict if they share a face AND are adjacent in that face
    # Actually, for C4-free: we cannot have ALL 4 edges of a face.
    # So conflicts are: if we have 3 edges of a face, the 4th is blocked.
    # This is not a simple pairwise conflict graph. Let's use the face-based approach.
    
    # State
    selected = [False] * n_edges_total
    face_counts = [0] * n_faces
    num_selected = 0
    
    def can_add(edge_idx):
        for f_idx in edge_to_faces[edge_idx]:
            if face_counts[f_idx] == 3:
                return False
        return True

    def add_edge(edge_idx):
        nonlocal num_selected
        if selected[edge_idx]:
            return False
        if not can_add(edge_idx):
            return False
        selected[edge_idx] = True
        num_selected += 1
        for f_idx in edge_to_faces[edge_idx]:
            face_counts[f_idx] += 1
        return True

    def remove_edge(edge_idx):
        nonlocal num_selected
        if not selected[edge_idx]:
            return False
        selected[edge_idx] = False
        num_selected -= 1
        for f_idx in edge_to_faces[edge_idx]:
            face_counts[f_idx] -= 1
        return True

    def greedy_fill(randomize=True):
        nonlocal num_selected
        unselected = [i for i in range(n_edges_total) if not selected[i]]
        if randomize:
            random.shuffle(unselected)
        
        added = 0
        for e in unselected:
            if can_add(e):
                add_edge(e)
                added += 1
        return added

    def get_conflict_edges():
        """Get edges that are in faces with count 3 (blocking further additions)"""
        conflict_edges = set()
        for f_idx in range(n_faces):
            if face_counts[f_idx] == 3:
                for e in faces[f_idx]:
                    if selected[e]:
                        conflict_edges.add(e)
        return conflict_edges

    def swap_move(kick_size=3):
        """
        Perform a swap move: remove k edges from conflict faces, 
        then greedily add back as many as possible.
        This is like a 2-opt move in TSP.
        """
        nonlocal num_selected, selected, face_counts
        
        conflict_edges = get_conflict_edges()
        if not conflict_edges:
            return 0
        
        # Pick k random conflict edges to remove
        to_remove = list(conflict_edges)
        random.shuffle(to_remove)
        to_remove = to_remove[:min(kick_size, len(to_remove))]
        
        # Save state
        old_selected = selected[:]
        old_face_counts = face_counts[:]
        old_num = num_selected
        
        # Remove the edges
        for e in to_remove:
            remove_edge(e)
        
        # Greedily add back
        greedy_fill(True)
        
        delta = num_selected - old_num
        
        if delta <= 0:
            # Revert
            selected = old_selected
            face_counts = old_face_counts
            num_selected = old_num
            return 0
        else:
            return delta

    def multi_swap_move(num_swaps=5, kick_size=2):
        """
        Perform multiple sequential swap moves.
        """
        total_delta = 0
        for _ in range(num_swaps):
            delta = swap_move(kick_size)
            total_delta += delta
            if delta == 0:
                break
        return total_delta

    # Initial Solution: Run greedy multiple times
    best_num = 0
    best_state = [False] * n_edges_total
    best_face_counts = [0] * n_faces
    
    for _ in range(20):
        # Reset
        selected = [False] * n_edges_total
        face_counts = [0] * n_faces
        num_selected = 0
        
        greedy_fill(True)
        
        if num_selected > best_num:
            best_num = num_selected
            best_state = selected[:]
            best_face_counts = face_counts[:]
            
    # Copy best state to current
    selected = best_state[:]
    face_counts = best_face_counts[:]
    num_selected = best_num
    
    # Local Search with swap moves
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        # Try to add edges first
        improved = False
        for _ in range(30):
            e = random.randint(0, n_edges_total - 1)
            if not selected[e]:
                if add_edge(e):
                    improved = True
                    if num_selected > best_num:
                        best_num = num_selected
                        best_state = selected[:]
                        best_face_counts = face_counts[:]
        
        if improved:
            no_improve_count = 0
            continue
        
        # Try swap moves
        delta = multi_swap_move(num_swaps=10, kick_size=2)
        
        if delta > 0:
            no_improve_count = 0
            if num_selected > best_num:
                best_num = num_selected
                best_state = selected[:]
                best_face_counts = face_counts[:]
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Larger kick
                conflict_edges = get_conflict_edges()
                if conflict_edges:
                    to_remove = list(conflict_edges)
                    random.shuffle(to_remove)
                    kick_size = min(10, len(to_remove))
                    to_remove = to_remove[:kick_size]
                    
                    old_selected = selected[:]
                    old_face_counts = face_counts[:]
                    old_num = num_selected
                    
                    for e in to_remove:
                        remove_edge(e)
                    
                    greedy_fill(True)
                    
                    if num_selected > best_num:
                        best_num = num_selected
                        best_state = selected[:]
                        best_face_counts = face_counts[:]
                        no_improve_count = 0
                    else:
                        selected = old_selected
                        face_counts = old_face_counts
                        num_selected = old_num
                else:
                    break

    # Final polish: try to add any remaining edges
    for e in range(n_edges_total):
        if not best_state[e]:
            # Check if can add
            can_add_check = True
            for f_idx in edge_to_faces[e]:
                if best_face_counts[f_idx] == 3:
                    can_add_check = False
                    break
            if can_add_check:
                best_state[e] = True
                best_num += 1
                for f_idx in edge_to_faces[e]:
                    best_face_counts[f_idx] += 1

    # Construct output
    out = []
    for i in range(n_edges_total):
        if best_state[i]:
            out.append(list(edges[i]))
            
    return out

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Edges found: {len(res)}")
