import time
import random

def construct():
    d = 7
    n = 1 << d  # 128 vertices
    deadline = time.time() + 115  # Safety margin

    # Precompute edges
    all_edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_idx[(x, y)] = idx
    m = len(all_edges)  # 448

    # Precompute faces
    face_edges = []
    edge_faces = [[] for _ in range(m)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_idx[(min(a, b), max(a, b))]
                    e2 = edge_idx[(min(b, d2), max(b, d2))]
                    e3 = edge_idx[(min(d2, c), max(d2, c))]
                    e4 = edge_idx[(min(c, a), max(c, a))]
                    
                    fid = len(face_edges)
                    face_edges.append([e1, e2, e3, e4])
                    for ei in (e1, e2, e3, e4):
                        edge_faces[ei].append(fid)
                        
    n_faces = len(face_edges)  # 672

    best_removed = None
    best_size = m  # Minimize removed edges

    # Precompute for each edge, the list of faces it belongs to
    # Already in edge_faces

    def greedy_hitting_set():
        """
        Greedy hitting set: pick edge covering most uncovered faces.
        Returns set of removed edges R.
        """
        R = set()
        uncovered = set(range(n_faces))
        
        while uncovered:
            # Count how many uncovered faces each edge covers
            counts = [0] * m
            for f in uncovered:
                for e in face_edges[f]:
                    counts[e] += 1
            
            max_count = max(counts)
            if max_count == 0:
                break
                
            candidates = [e for e in range(m) if counts[e] == max_count]
            best_e = random.choice(candidates)
            R.add(best_e)
            
            # Remove covered faces
            to_remove = []
            for f in edge_faces[best_e]:
                if f in uncovered:
                    to_remove.append(f)
            for f in to_remove:
                uncovered.remove(f)
        
        return R

    def redundancy_removal(R):
        """
        Remove redundant edges from R (edges not strictly needed to hit any face).
        """
        changed = True
        while changed:
            changed = False
            cover_count = [0] * n_faces
            for e in R:
                for f in edge_faces[e]:
                    cover_count[f] += 1
            
            for e in list(R):
                redundant = True
                for f in edge_faces[e]:
                    if cover_count[f] == 1:
                        redundant = False
                        break
                
                if redundant:
                    R.remove(e)
                    changed = True
                    break
        return R

    def local_search_minimize_R(R, time_limit):
        """
        Local search to minimize |R|.
        Phase 1: Greedy add-back of safe edges.
        Phase 2: Swap-based moves to escape local optima.
        """
        start_time = time.time()
        R = set(R)
        
        def is_safe_to_add(e, R_set):
            """Check if adding edge e back (removing from R) keeps all faces valid."""
            for f in edge_faces[e]:
                has_other_removed = False
                for e2 in face_edges[f]:
                    if e2 != e and e2 in R_set:
                        has_other_removed = True
                        break
                if not has_other_removed:
                    return False
            return True
        
        # Phase 1: Repeatedly add back safe edges
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            safe_edges = [e for e in R if is_safe_to_add(e, R)]
            if safe_edges:
                e = random.choice(safe_edges)
                R.remove(e)
                changed = True
        
        # Phase 2: Swap-based local search
        # Try swapping: remove e1 from R, add e2 to R
        # This is valid if after swap, all faces still hit.
        for _ in range(100):
            if time.time() > time_limit:
                break
            
            if not R:
                break
            
            # Pick a random edge to remove from R
            e1 = random.choice(list(R))
            
            # Find candidate e2 to add to R
            # e2 should help cover faces that e1 was covering
            # Simple heuristic: pick random edge not in R
            candidates = [e for e in range(m) if e not in R]
            if not candidates:
                break
            e2 = random.choice(candidates)
            
            # Check if swap is valid
            R_test = set(R)
            R_test.remove(e1)
            R_test.add(e2)
            
            # Verify all faces are still hit
            valid = True
            for f in range(n_faces):
                hit = False
                for e in face_edges[f]:
                    if e in R_test:
                        hit = True
                        break
                if not hit:
                    valid = False
                    break
            
            if valid:
                R = R_test
                # After swap, try to add back more
                changed = True
                while changed and time.time() < time_limit:
                    changed = False
                    safe_edges = [e for e in R if is_safe_to_add(e, R)]
                    if safe_edges:
                        e = random.choice(safe_edges)
                        R.remove(e)
                        changed = True
        
        return R

    # Run multiple iterations of greedy + redundancy removal + local search
    iterations = 80
    for it in range(iterations):
        if time.time() > deadline - 20:
            break
        
        # Greedy hitting set
        R = greedy_hitting_set()
        
        # Redundancy removal
        R = redundancy_removal(R)
        
        # Local search
        time_left = deadline - 25 - time.time()
        if time_left > 1:
            R = local_search_minimize_R(R, time.time() + min(time_left, 5))
        
        # Check size
        if len(R) < best_size:
            best_size = len(R)
            best_removed = R.copy()
            if best_size <= 140:
                break

    # Fallback
    if best_removed is None:
        best_removed = set(range(m))
        best_size = m

    # Construct final answer
    selected_edges = []
    for e in range(m):
        if e not in best_removed:
            x, y = all_edges[e]
            selected_edges.append([x, y])
            
    return selected_edges

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Edges: {len(edges)}")
    
    d = 7
    n = 1 << d
    all_edges_set = set()
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges_set.add((x, y))
                
    selected_set = set()
    for u, v in edges:
        selected_set.add((min(u, v), max(u, v)))
        
    valid = True
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(a, b), max(a, b))
                    e2 = (min(b, d2), max(b, d2))
                    e3 = (min(d2, c), max(d2, c))
                    e4 = (min(c, a), max(c, a))
                    
                    cnt = 0
                    if e1 in selected_set: cnt += 1
                    if e2 in selected_set: cnt += 1
                    if e3 in selected_set: cnt += 1
                    if e4 in selected_set: cnt += 1
                    
                    if cnt == 4:
                        print(f"Invalid face found at {x}, dims {i},{j}")
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        print("Valid!")
    else:
        print("Invalid!")
